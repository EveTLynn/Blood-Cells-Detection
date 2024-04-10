import os
import matplotlib.pyplot as plt
import subprocess
import argparse
import tensorflow as tf
# Import required libraries from tensorflow models
import orbit
import tensorflow_models as tfm
from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib
from official.vision.ops.preprocess_ops import normalize_image
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder

def config_model(model_name: str ='retinanet_resnetfpn_coco',
                 working_dir: str = './working_dir',
                 input_path: str = './working_dir/augmented_data/bccd_coco_tfrecords/',
                 checkpoint_url: str = "https://storage.googleapis.com/tf_model_garden/vision/retinanet/retinanet-resnet50fpn.tar.gz",
                 HEIGHT: int = 256,
                 WIDTH: int = 256,
                 train_steps: int = 1000):
    """
    Configures the RetinaNet model for training.

    This function configures a RetinaNet model for training on a provided dataset.
    It defines paths, downloads the pre-trained checkpoint, adjusts model and
    dataset configurations, sets up the training strategy, and creates the task object.

    Args:
        model_name (str, optional): Name of the model architecture. Defaults to 'retinanet_resnetfpn_coco'.
        working_dir (str, optional): Base directory containing the trained model directory structure. Defaults to './working_dir'.
        input_path (str, optional): Path to the directory containing TFRecord files for training, validation, and testing. Defaults to './working_dir/augmented_data/bccd_coco_tfrecords/'.
        checkpoint_url (str, optional): URL to download the pre-trained checkpoint file. Defaults to "https://storage.googleapis.com/tf_model_garden/vision/retinanet/retinanet-resnet50fpn.tar.gz".
        HEIGHT (int, optional): Height of the input image for training. Defaults to 256.
        WIDTH (int, optional): Width of the input image for training. Defaults to 256.
        train_steps (int, optional): Number of training steps. Defaults to 1000.

    Returns:
        tuple[str, Any, Any, Any]: A tuple containing the following elements:
            - model_dir (str): Path to the directory containing the trained model.
            - task (Any): The task object for training the model.
            - exp_config (Any): The experiment configuration object.
            - distribution_strategy (Any): The distribution strategy for training.
    """


    # define paths
    train_data_input_path = os.path.join(input_path, 'train-00000-of-00001.tfrecord')
    valid_data_input_path = os.path.join(input_path, 'valid-00000-of-00001.tfrecord')
    test_data_input_path = os.path.join(input_path, 'test-00000-of-00001.tfrecord')
    model_dir = os.path.join(working_dir, model_name, 'trained_model')
    ckpt_path = os.path.join(working_dir, model_name, 'base_ckpt')

    # get the model architecture
    exp_config = exp_factory.get_exp_config(model_name)

    # get resnet checkpoint from tensorflow model garden
    tmp_dir = "/tmp/"
    # Download the checkpoint file
    subprocess.run(["wget", checkpoint_url, "-P", tmp_dir])
    # Create the destination directory if it doesn't exist
    os.makedirs(ckpt_path, exist_ok=True)
    # Extract the checkpoint file
    subprocess.run(["tar", "-xvzf", f"{tmp_dir}/retinanet-resnet50fpn.tar.gz", "-C", ckpt_path])
    # Remove the downloaded checkpoint file
    os.remove(f"{tmp_dir}/retinanet-resnet50fpn.tar.gz")

    # Adjust the model and dataset configurations
    batch_size = 8
    num_classes = 3

    IMG_SIZE = [HEIGHT, WIDTH, 3]

    # Backbone config.
    exp_config.task.init_checkpoint = os.path.join(working_dir, model_name, 'base_ckpt', 'ckpt-33264')  # path to pretrained checkpoints
    exp_config.task.init_checkpoint_modules = 'backbone'
    exp_config.task.freeze_backbone = True  # freeze backbone
    exp_config.task.annotation_file = ''

    # Model config.
    exp_config.task.model.input_size = IMG_SIZE
    exp_config.task.model.num_classes = num_classes + 1
    exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

    # Training data config.
    exp_config.task.train_data.input_path = train_data_input_path
    exp_config.task.train_data.dtype = 'float32'
    exp_config.task.train_data.global_batch_size = batch_size
    exp_config.task.train_data.parser.aug_scale_max = 1.0
    exp_config.task.train_data.parser.aug_scale_min = 1.0
    exp_config.task.train_data.parser.skip_crowd_during_training = False

    # Validation data config.
    exp_config.task.validation_data.input_path = valid_data_input_path
    exp_config.task.validation_data.dtype = 'float32'
    exp_config.task.validation_data.global_batch_size = batch_size
    exp_config.task.validation_data.parser.skip_crowd_during_training = False

    ### Adjust the trainer configuration

    logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

    if 'GPU' in ''.join(logical_device_names):
        print('This may be broken in Colab.')
        device = 'GPU'
    elif 'TPU' in ''.join(logical_device_names):
        print('This may be broken in Colab.')
        device = 'TPU'
    else:
        print('Running on CPU is slow, so only train for a few steps.')
        device = 'CPU'

    exp_config.trainer.steps_per_loop = 100 # steps_per_loop = num_of_training_examples // train_batch_size

    # the trainer by default will save the 5 lastest checkpoints
    exp_config.trainer.best_checkpoint_eval_metric = 'APm' # export checkpoint with highest mAP
    exp_config.trainer.best_checkpoint_export_subdir = './best_checkpoints'
    exp_config.trainer.summary_interval = 100
    exp_config.trainer.checkpoint_interval = 100
    exp_config.trainer.validation_interval = 100
    exp_config.trainer.validation_steps =  100 # validation_steps = num_of_validation_examples // eval_batch_size
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
    exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

    ### Set up the distribution strategy
    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Checks for available hardware devices (GPUs or TPUs) and create appropriate strategy.
    if 'GPU' in ''.join(logical_device_names):
        distribution_strategy = tf.distribute.MirroredStrategy()
    elif 'TPU' in ''.join(logical_device_names):
        tf.tpu.experimental.initialize_tpu_system()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
        distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        print('Warning: this will be really slow.')
        # One single device in this case
        distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

    print('Done setting up the distribution strategy.')


    #### Create the Task object
    with distribution_strategy.scope():
        task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)

    return model_dir, task, exp_config, distribution_strategy


def export_saved_model(exp_config,
                       ckpt_path: str = 'best_ckpt',
                       working_dir: str ='./working_dir',
                       model_name: str='retinanet_resnetfpn_coco',
                       HEIGHT: int = 256,
                       WIDTH: int = 256):
    """
    Exports a trained RetinaNet model for inference.

    Args:
        exp_config (object): Configuration object specific to the model
            architecture.
        ckpt_path (str): Path to the checkpoint file to use for exporting the model.
          This can be one of the following:
              *  **'latest_ckpt'**: The function will automatically locate the latest
                  checkpoint in the trained model directory
                  (`working_dir/model_name/trained_model`).
              *  **'best_ckpt'**:  The function will use the checkpoint located in the
                  `best_checkpoints` subdirectory within the trained model directory
                  (`working_dir/model_name/trained_model/best_checkpoints`).
              *  **Path to a specific checkpoint file**: You can provide the full path
                  to a desired checkpoint file.
        working_dir (str, optional): Base directory containing the trained model
            directory structure. Defaults to `'./working_dir'`.
        model_name (str, optional): Name of the subdirectory within `working_dir`
            containing the trained model. Defaults to `'retinanet_resnetfpn_coco'`.
        HEIGHT (int, optional): Height of the input image for inference. Defaults to 256.
        WIDTH (int, optional): Width of the input image for inference. Defaults to 256.

    Returns:
        None. This function saves the exported model to the specified `export_dir`.
    """

    # Construct path to the export directory
    export_dir = os.path.join(working_dir, model_name, 'exported_model')

    # Determine checkpoint path based on user input
    if ckpt_path == 'latest_ckpt':
        ckpt_path = tf.train.latest_checkpoint(os.path.join(working_dir, model_name, 'trained_model'))
    elif ckpt_path == 'best_ckpt':
        ckpt_path = os.path.join(working_dir, model_name, 'trained_model', 'best_checkpoints')

    # Export the saved model
    export_saved_model_lib.export_inference_graph(
        input_type='image_tensor',
        batch_size=1,
        input_image_size=[HEIGHT, WIDTH],
        params=exp_config,
        checkpoint_path=ckpt_path,
        export_dir=export_dir)


def main():
    """
    Main function for training a RetinaNet model on COCO and exporting it for inference.

    This function parses command-line arguments, configures the model, runs training
    and evaluation, and exports a saved model suitable for inference.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train RetinaNet on COCO and export a model for inference.')
    parser.add_argument('--model_name', type=str, default='retinanet_resnetfpn_coco',
                        help='Name of the model architecture.')
    parser.add_argument('--working_dir', type=str, default='./working_dir',
                        help='Base directory for model working directory.')
    parser.add_argument('--input_path', type=str, default='./working_dir/augmented_data/bccd_coco_tfrecords/',
                        help='Path to input TFRecord files.')
    parser.add_argument('--checkpoint_url', type=str, default="https://storage.googleapis.com/tf_model_garden/vision/retinanet/retinanet-resnet50fpn.tar.gz",
                        help='URL of the pre-trained model checkpoint.')
    parser.add_argument('--HEIGHT', type=int, default=256,
                        help='Height of input images for training.')
    parser.add_argument('--WIDTH', type=int, default=256,
                        help='Width of input images for training.')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='Number of training steps.')
    parser.add_argument('--ckpt_path', type=str, default='best_ckpt',
                        help='Path to the checkpoint to use for export (relative to model directory).')

    args = parser.parse_args()

    # Configure the model
    model_dir, task, exp_config, distribution_strategy = config_model(args.model_name, args.working_dir,
                                                                       args.input_path, args.checkpoint_url,
                                                                       args.HEIGHT, args.WIDTH, args.train_steps)

    # Run training and evaluation
    print('Starting training and evaluation...')
    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True)

    #Export the saved model
    print('Exporting the saved model...')
    export_saved_model(exp_config=exp_config,
                       ckpt_path=args.ckpt_path,
                       working_dir=args.working_dir,
                       model_name=args.model_name,
                       HEIGHT=args.HEIGHT,
                       WIDTH=args.WIDTH)

    print('Training, evaluation, and model export completed successfully!')

if __name__ == '__main__':
  main()
