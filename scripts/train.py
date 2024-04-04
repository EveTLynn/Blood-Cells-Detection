import os
import matplotlib.pyplot as plt
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

# Define model name and paths
model_name ='retinanet_resnetfpn_coco'

train_data_input_path = './augmented_data/bccd_coco_tfrecords/train-00000-of-00001.tfrecord'
valid_data_input_path = './augmented_data/bccd_coco_tfrecords/valid-00000-of-00001.tfrecord'
test_data_input_path = './augmented_data/bccd_coco_tfrecords/test-00000-of-00001.tfrecord'
model_dir = os.path.join(model_name, 'trained_model')
export_dir = os.path.join(model_name, 'exported_model')
ckpt_path = os.path.join(model_name, 'base_ckpt')

# get the model architecture
exp_config = exp_factory.get_exp_config(model_name)

# get resnet checkpoint from tensorflow model garden
checkpoint_url = "https://storage.googleapis.com/tf_model_garden/vision/retinanet/retinanet-resnet50fpn.tar.gz"
tmp_dir = "/tmp/"
# Download the checkpoint file
subprocess.run(["wget", checkpoint_url, "-P", tmp_dir])
# Create the destination directory if it doesn't exist
os.makedirs(ckpt_path, exist_ok=True)
# Extract the checkpoint file
subprocess.run(["tar", "-xvzf", f"{tmp_dir}/retinanet-resnet50fpn.tar.gz", "-C", ckpt_path])
# Remove the downloaded checkpoint file
os.remove(f"{tmp_dir}/retinanet-resnet50fpn.tar.gz")

####### Adjust the model and dataset configurations
batch_size = 8
num_classes = 3

HEIGHT, WIDTH = 256, 256
IMG_SIZE = [HEIGHT, WIDTH, 3]
path = './working_dir/'

# Backbone config.
exp_config.task.init_checkpoint = os.path.join(path, model_name, 'base_ckpt', 'ckpt-33264')  # path to pretrained checkpoints
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


train_steps = 10000
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

pp.pprint(exp_config.as_dict())
display.Javascript('google.colab.output.setIframeHeight("500px");')

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

# Create category index dictionary to map the labels to coressponding label names.
category_index={
    1: {
        'id': 1,
        'name': 'RBC'
       },
    2: {
        'id': 2,
        'name': 'WBC'
       },
    3: {
        'id': 3,
        'name': 'Platelets'
       }
}
tf_ex_decoder = TfExampleDecoder()

# Helper function for visualizing the results from TFRecords.
def show_batch(records, num_of_examples, cols=6, min_score_thresh=0.4):
  # some hyperparameters
  rows = num_of_examples//cols + 1
  fig_height = rows*10 + (rows-1)*2
  fig_width = cols*10 + (cols-1)*2
  plt.figure(figsize=(fig_width, fig_height))
  use_normalized_coordinates=True

  # visualize predictions
  for i, serialized_example in enumerate(records):
    plt.subplot(rows, cols, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = decoded_tensors['image'].numpy().astype('uint8')
    scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        decoded_tensors['groundtruth_boxes'].numpy(),
        decoded_tensors['groundtruth_classes'].numpy().astype('int'),
        scores,
        category_index=category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Image-{i+1}', fontsize=30)
  plt.show()

