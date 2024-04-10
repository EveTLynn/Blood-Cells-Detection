import matplotlib.pyplot as plt
import tensorflow as tf
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils
import numpy as np
import argparse

def build_inputs_for_object_detection(image: tf.Tensor, input_image_size: int = [256, 256]) -> tf.Tensor:
    """
    Preprocesses an image for object detection model inference during serving.

    This function takes an image tensor and resizes and crops it to the dimensions
    expected by the object detection model. It uses the `resize_and_crop_image`
    function from the official TensorFlow Vision library with specific arguments:

    * `padded_size`: Set to `input_image_size` to ensure the image is resized
                       to a square shape with the specified size.
    * `aug_scale_min` and `aug_scale_max`: Set to 1.0 since data augmentation
                       (e.g., scaling) is not performed during serving.

    Args:
        image: A TensorFlow tensor representing the input image.
        input_image_size: The target size (width and height) for the preprocessed image.

    Returns:
        A TensorFlow tensor representing the preprocessed image ready for model input.
    """

    # Resize and crop the image for object detection model input
    image, _ = resize_and_crop_image(
        image,
        input_image_size,
        padded_size=input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0
    )

    return image

def visualize_results(test_ds: tf.data.Dataset, category_index: dict[int, dict], 
                      tf_ex_decoder: TfExampleDecoder,
                      num_of_examples: int,  model_fn: tf.function = None,
                      visualize_ground_truth: bool = True, cols: int = 6,
                      min_score_thresh: float = 0.4,
                      HEIGHT: int = 256, WIDTH: int = 256) -> None:
    """
    Visualizes either ground truth bounding boxes or model predictions on images
    from a TFRecord dataset.

    This function iterates through a provided TFRecord dataset, decodes each
    example. It then either visualizes the ground truth bounding boxes and class
    labels (if `visualize_ground_truth` is True) or performs inference using the
    provided model function and visualizes the predicted bounding boxes and class
    labels (if `visualize_ground_truth` is False) on a grid layout using matplotlib.
    Finally, it saves the resulting visualization as an image.

    Args:
        test_ds: A TFRecord dataset containing serialized examples.
        category_index: A dictionary mapping class IDs to human-readable names.
        tf_ex_decoder: A TfExampleDecoder instance for decoding TFRecords.
        num_of_examples: The number of examples to visualize from the dataset.
        cols: The number of columns in the grid layout for visualization (default: 6).
        min_score_thresh: Minimum score threshold for visualization (default: 0.4).
        visualize_ground_truth: Boolean flag indicating whether to visualize ground truth
                                (True) or predictions (False).
    """
    # Create category index dictionary for label to name mapping
    category_index = {
        1: {'id': 1, 'name': 'RBC'},
        2: {'id': 2, 'name': 'WBC'},
        3: {'id': 3, 'name': 'Platelets'}
    }

    # Calculate layout dimensions based on number of examples and columns
    rows = num_of_examples // cols + 1
    fig_height = rows * 10 + (rows - 1) * 2
    fig_width = cols * 10 + (cols - 1) * 2
    input_image_size = (HEIGHT, WIDTH)

    # Create a matplotlib figure with appropriate size
    plt.figure(figsize=(fig_width, fig_height))
    use_normalized_coordinates = True

    # Loop through each record in the dataset
    for i, serialized_example in enumerate(test_ds):
        plt.subplot(rows, cols, i + 1)
        # Decode the serialized example
        decoded_tensors = tf_ex_decoder.decode(serialized_example)

        # Visualize ground truth or predictions based on the argument
        if visualize_ground_truth:
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
        else:
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

    # Save the visualization as a PNG image
    if visualize_ground_truth:
        plt.savefig('ground_truth_image.png')
    else:
        plt.savefig('inference_image.png')


def import_and_run_inference(export_dir: str, test_ds_path: str, num_of_examples: int,
                             HEIGHT: int = 256, WIDTH: int = 256, min_score_thresh: float = 0.4) -> None:
    """
    Imports a pre-trained object detection model, runs inference on a test dataset,
    and visualizes both ground truth and predicted bounding boxes.

    This function performs the following steps:

    1. Creates a TfExampleDecoder instance for decoding TFRecords.
    2. Loads the pre-trained object detection model from the provided export directory.
    3. Retrieves the inference function ('serving_default') from the loaded SavedModel.
    4. Creates a TFRecord dataset from the test dataset path and limits it to the specified number of examples.
    5. Visualizes ground truth bounding boxes using the `visualize_results` function.
    6. Runs inference using the loaded model and visualizes the predicted bounding boxes
       using the `visualize_results` function.

    Args:
        export_dir: Path to the directory containing the exported SavedModel.
        test_ds_path: Path to the TFRecord dataset containing test images.
        num_of_examples: The number of examples to visualize from the dataset.
        HEIGHT: Target image height for pre-processing (default: 256).
        WIDTH: Target image width for pre-processing (default: 256).
        min_score_thresh: Minimum score threshold for visualization (default: 0.4).
    """

    # Create TFRecord decoder
    tf_ex_decoder = TfExampleDecoder()

    # Load the pre-trained object detection model
    imported = tf.saved_model.load(export_dir)
    model_fn = imported.signatures['serving_default'] # Retrieve inference function

    # Load the test dataset (limited to specified number of examples)
    test_ds = tf.data.TFRecordDataset(test_ds_path).take(num_of_examples)

    # visualize ground truth boxes
    gt_test_ds = test_ds
    visualize_results(gt_test_ds, category_index, tf_ex_decoder, num_of_examples, visualize_ground_truth=True)

    # Visualize inference (call repeat to create a new iterator)
    visualize_results(test_ds.repeat(), category_index, tf_ex_decoder, num_of_examples, model_fn, visualize_ground_truth=False)

def main():
    """
    This function parses command-line arguments specifying the paths to the exported
    model directory, test dataset, and other visualization parameters. Then, it calls
    the `import_and_run_inference` function to perform inference and visualization.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, default='./working_dir/retinanet_resnetfpn_coco/exported_model',
                        help='Directory to save the exported model.')
    parser.add_argument('--test_ds_path', type=str, default='./working_dir/augmented_data/bccd_coco_tfrecords/test-00000-of-00001.tfrecord',
                        help='Path to the test dataset in TFRecord format.')
    parser.add_argument('--number_of_images', type=int, default=18,
                        help='Number of images to visualize.')
    parser.add_argument('--HEIGHT', type=int, default=256)
    parser.add_argument('--WIDTH', type=int, default=256)
    parser.add_argument('--min_score_thresh', type=float, default=0.4)

    args = parser.parse_args()

    # Run inference and visualization
    import_and_run_inference(args.export_dir, args.test_ds_path, args.number_of_images, args.HEIGHT, args.WIDTH, args.min_score_thresh)


if __name__ == '__main__':
  main()
