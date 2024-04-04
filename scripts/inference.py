import matplotlib.pyplot as plt
import tensorflow as tf
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils

def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, _ = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  return image

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

export_dir = './working_dir/retinanet_resnetfpn_coco/exported_model'
# import pretrained model
imported = tf.saved_model.load(export_dir)

# retrieve the inference function from the loaded SavedModel
model_fn = imported.signatures['serving_default']

# show some images from test set with their ground truth boxes
num_of_examples = 72 # Change this to see more images (also for inference below)

test_ds = tf.data.TFRecordDataset(
    './working_dir/augmented_data/bccd_coco_tfrecords/test-00000-of-00001.tfrecord').take(
        num_of_examples)
show_batch(test_ds, num_of_examples)

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


# some hyperparameters
HEIGHT = 256
WIDTH = 256
number_of_images = 72  # Change this to see more images
cols = 6
rows = number_of_images // cols + 1
input_image_size = (HEIGHT, WIDTH)
fig_height = rows * 10 + (rows - 1) * 4
fig_width = cols * 10 + (cols - 1) * 4
plt.figure(figsize=(fig_width, fig_height))
min_score_thresh = 0.3  # Change minimum score for threshold

# visualize predictions
for i, serialized_example in enumerate(test_ds):
    plt.subplot(rows, cols, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = build_inputs_for_object_detection(decoded_tensors['image'], input_image_size)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.uint8)
    image_np = image[0].numpy()
    # predictions
    result = model_fn(image)

    # Get detections and scores
    boxes = result['detection_boxes'][0].numpy()
    classes = result['detection_classes'][0].numpy().astype(int)
    scores = result['detection_scores'][0].numpy()

    # Filter detections based on score threshold
    filtered_boxes = boxes[scores >= min_score_thresh]
    filtered_classes = classes[scores >= min_score_thresh]

    # Count detections by category
    category_counts = {}
    for category in filtered_classes:
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

    # Generate count text based on category_index
    count_text = ""
    for category, count in category_counts.items():
        label = category_index[category]['name']
        count_text += f"{label}: {count}, "

    # Truncate trailing comma and space
    count_text = count_text[:-2]

    # Visualize predictions
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        filtered_boxes,
        filtered_classes,
        scores,
        category_index=category_index,
        use_normalized_coordinates=False,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4
    )

    # Display image and count text
    plt.imshow(image_np)
    plt.title(f'Image-{i+1}. {count_text}', fontsize=30)
    plt.axis('off')

plt.show()
