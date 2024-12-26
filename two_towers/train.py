from typing import Dict, Any, Tuple, Text

import argparse
import json
import os

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pyarrow as pa
import ray
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
import tensorflow as tf

# Checkpoint saving and restoring
def _is_chief(task_type, task_id, cluster_spec):
    return (task_type is None
          or task_type == 'chief'
          or (task_type == 'worker'
              and task_id == 0
              and 'chief' not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id, cluster_spec):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)

def train_ds_function(input_context: tf.distribute.InputContext, path: str) -> tf.data.Dataset:
    return get_dataset(input_context, path, "train")

def test_ds_function(input_context: tf.distribute.InputContext, path: str) -> tf.data.Dataset:
    return get_dataset(input_context, path, "test")

def get_dataset(input_context: tf.distribute.InputContext, path: str, ds_type: str) -> ray.data.Dataset:
    batch_size = input_context.get_per_replica_batch_size(128)
    parquet_path = os.path.join(path, ds_type)
    
    dataset = (
            ray.data.read_parquet(parquet_path)
               .to_tf(feature_columns=["movieId", "userId"], label_columns = ["rating"])
               .map(lambda f,l: { "movieId": f["movieId"][0], "userId": f["userId"][0], "rating": l["rating"][0] })
               .shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
               .batch(batch_size)
               .cache()
    )
    return dataset

def get_movies(path: str) -> ray.data.Dataset:
    parquet_path = os.path.join(path, "movies.parquet")
    
    dataset = (
            ray.data.read_parquet(parquet_path)
               .to_tf(feature_columns=["movieId"], label_columns = ["title"])
               .map(lambda f,t: { "movieId": f["movieId"][0] })
               .batch(128)
               .cache()
    )
    return dataset

def get_metadata(path: str) -> Dict[str, Any]:
    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path, "r") as f:
        return json.loads(f.read())

class MovielensModel(tfrs.Model):

    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["userId"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movieId"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)


def build_two_tower_model(movies: tf.data.Dataset, metadata: Dict[str, Any], optimizer: tf.keras.optimizers.Optimizer) -> tf.keras.Model:

    embedding_dimension = 32
    unique_user_ids = metadata["unique_user_ids"]
    unique_movie_ids = metadata["unique_movie_ids"]
    user_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), name="userId", dtype=tf.int64),
        tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    movie_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), name="movieId", dtype=tf.int64),
        tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids),
        tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.map(movie_model)
    )
    task = tfrs.tasks.Retrieval(metrics=metrics)
    
    model = MovielensModel(user_model, movie_model, task)
    model.compile(optimizer=optimizer)

    return model


def train_func(config: dict):
    epochs = config.get("epochs", 3)
    learning_rate = config.get("lr", 0.001)
    path = config.get("path")
    checkpoint_dir = os.path.join(path, 'ckpt')

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        train_dataset = strategy.distribute_datasets_from_function(lambda input_context: train_ds_function(input_context, path))
        test_dataset = strategy.distribute_datasets_from_function(lambda input_context: test_ds_function(input_context, path))
        movies = get_movies(path)
        metadata = get_metadata(path)
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        model = build_two_tower_model(movies, metadata, optimizer)
    

    @tf.function
    def train_step(batch) -> Tuple[Any, Dict[str, Any]]:
        """Training step function."""

        def step_fn(inputs):
            """Per-Replica step function."""
            metrics = model.train_step(inputs)            
            return metrics

        per_replica_losses = strategy.run(step_fn, args=(batch,))

        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses["total_loss"], axis=None)
        return loss, per_replica_losses

    @tf.function
    def test_step(batch) -> Tuple[Any, Dict[str, Any]]:
        """Test step function."""

        def step_fn(inputs):
            """Per-Replica step function."""
            metrics = model.test_step(inputs)            
            return metrics

        per_replica_losses = strategy.run(step_fn, args=(batch,))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses["total_loss"], axis=None)
        return loss, per_replica_losses

    @tf.function
    def train_epoch(train_dataset, epoch):
        step_in_epoch = 0.0
        max_batches = tf.constant(100000.0, dtype=tf.dtypes.float32)
        total_loss = 0.0
        num_batches = 0.0
        per_replica_losses = {
            'loss': 0.0, 
            'factorized_top_k/top_1_categorical_accuracy': 0.0, 
            'factorized_top_k/top_5_categorical_accuracy': 0.0,
            'factorized_top_k/top_10_categorical_accuracy': 0.0,
            'factorized_top_k/top_50_categorical_accuracy': 0.0, 
            'factorized_top_k/top_100_categorical_accuracy': 0.0, 
            'regularization_loss': 0.0, 
            'total_loss': 0.0
        }

        while step_in_epoch < max_batches:
            batch = train_dataset.get_next_as_optional()
            if not batch.has_value():
                break
            loss, per_replica_losses = train_step(batch.get_value())
            total_loss += loss
            num_batches += 1.0
            step_in_epoch += 1.0
        tf.print("per_replica_losses: ", per_replica_losses)

        train_loss = total_loss / num_batches
        tf.print('Epoch: ', epoch, "train_loss:", train_loss)


    @tf.function
    def test_epoch(test_dataset, epoch):
        step_in_epoch = 0.0
        max_batches = tf.constant(100000.0, dtype=tf.dtypes.float32)
        total_loss = 0.0
        num_batches = 0.0
        per_replica_losses = {
            'loss': 0.0, 
            'factorized_top_k/top_1_categorical_accuracy': 0.0, 
            'factorized_top_k/top_5_categorical_accuracy': 0.0,
            'factorized_top_k/top_10_categorical_accuracy': 0.0,
            'factorized_top_k/top_50_categorical_accuracy': 0.0, 
            'factorized_top_k/top_100_categorical_accuracy': 0.0, 
            'regularization_loss': 0.0, 
            'total_loss': 0.0
        }

        while step_in_epoch < max_batches:
            batch = test_dataset.get_next_as_optional()
            if not batch.has_value():
                break
            loss, per_replica_losses = test_step(batch.get_value())
            total_loss += loss
            num_batches += 1.0
            step_in_epoch += 1.0
        tf.print("per_replica_losses: ", per_replica_losses)

        test_loss = total_loss / num_batches
        tf.print('Epoch: ', epoch, "test_loss:", test_loss)

    epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')

    task_type, task_id, cluster_spec = (strategy.cluster_resolver.task_type,
                                        strategy.cluster_resolver.task_id,
                                        strategy.cluster_resolver.cluster_spec())

    checkpoint = tf.train.Checkpoint(model=model, epoch=epoch)

    write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id, cluster_spec)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

    while epoch.numpy() < epochs:

        train_epoch(iter(train_dataset), epoch)
        test_epoch(iter(test_dataset), epoch)

        checkpoint_manager.save()
        if not _is_chief(task_type, task_id, cluster_spec):
            tf.io.gfile.rmtree(write_checkpoint_dir)

        epoch.assign_add(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--path", type=str, help="Path to Dataset."
    )

    args, _ = parser.parse_known_args()

    import ray

    ray.init(address=args.address)
    config = {"lr": 0.1, "batch_size": 128, "epochs": args.epochs, "path": args.path}


    scaling_config = ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu)
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        # run_config=ray.train.RunConfig(
        #     storage_path="s3://datasets/ml-latest-small/",
        #     name="two_tower",
        # )
    )

    result = trainer.fit()
    print(result.metrics)
    print(result.checkpoint)


