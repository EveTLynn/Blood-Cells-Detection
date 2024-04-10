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

def visualize_gt_boxes(test_ds_path: str, num_of_examples: int, cols: int = 6,
                       min_score_thresh: float = 0.4, HEIGHT: int = 256, WIDTH: int = 256) -> None:
    """
    Visualizes ground truth bounding boxes and class labels on images from a TFRecord dataset.
    
    This function loads a TFRecord dataset containing ground truth information
    (`test_ds_path`), decodes examples, and visualizes the ground truth bounding
    boxes and class labels on a grid layout using matplotlib. It assumes the
    dataset contains images and corresponding ground truth annotations (boxes and classes).
    
    Args:
        test_ds_path: Path to the test dataset in TFRecord format.
        num_of_examples: The number of examples to visualize from the dataset.
        cols: The number of columns in the grid layout for visualization (default: 6).
        min_score_thresh: Minimum score threshold for visualization (default: 0.4, not used here).
        HEIGHT: Target image height for visualization (default: 256).
        WIDTH: Target image width for visualization (default: 256).
    
    Returns:
        None. This function saves the visualization as an image file ("ground_truth_image.png").
    """
    # Create category index dictionary for label to name mapping
    category_index = {
        1: {'id': 1, 'name': 'RBC'},
        2: {'id': 2, 'name': 'WBC'},
        3: {'id': 3, 'name': 'Platelets'}
    }
    
    # Create TFRecord decoder
    tf_ex_decoder = TfExampleDecoder()

    # Load the test dataset (limited to specified number of examples)
    test_ds = tf.data.TFRecordDataset(test_ds_path).take(num_of_examples)

    rows = num_of_examples//cols + 1
    fig_height = (num_of_examples // cols + 1) * 10 + ((num_of_examples // cols + 1) - 1) * 2
    fig_width = cols * 10 + (cols - 1) * 2
    figure = plt.figure(figsize=(fig_width, fig_height))

    use_normalized_coordinates = True

    # visualize predictions
    for i, serialized_example in enumerate(test_ds):
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

    plt.savefig('ground_truth_image.png')


def visualize_predicted_boxes(export_dir: str, test_ds_path: str, num_of_examples: int,
                             cols: int = 6, min_score_thresh: float = 0.4,
                             HEIGHT: int = 256, WIDTH: int = 256) -> None:
    """
    Visualizes predicted bounding boxes and class labels on images from a TFRecord dataset.

    This function loads a pre-trained object detection model from a saved model
    directory (`export_dir`), decodes examples from a TFRecord dataset
    (`test_ds_path`), performs inference using the loaded model, and visualizes
    the predicted bounding boxes and class labels on a grid layout using matplotlib.

    Args:
        export_dir: Path to the directory containing the exported saved model.
        test_ds_path: Path to the test dataset in TFRecord format.
        num_of_examples: The number of examples to visualize from the dataset.
        cols: The number of columns in the grid layout for visualization (default: 6).
        min_score_thresh: Minimum score threshold for visualization (default: 0.4).
        HEIGHT: Target image height for pre-processing (default: 256).
        WIDTH: Target image width for pre-processing (default: 256).

    Returns:
        None. This function saves the visualization as an image file ("inference_image.png").
    """
    # Create category index dictionary for label to name mapping
    category_index = {
        1: {'id': 1, 'name': 'RBC'},
        2: {'id': 2, 'name': 'WBC'},
        3: {'id': 3, 'name': 'Platelets'}
    }
    # Create TFRecord decoder
    tf_ex_decoder = TfExampleDecoder()

    # Load the pre-trained object detection model
    imported = tf.saved_model.load(export_dir)
    model_fn = imported.signatures['serving_default'] # Retrieve inference function

    # Load the test dataset (limited to specified number of examples)
    test_ds = tf.data.TFRecordDataset(test_ds_path).take(num_of_examples)

    # calculate figure height and width
    fig_height = (num_of_examples // cols + 1) * 10 + ((num_of_examples // cols + 1) - 1) * 2
    fig_width = cols * 10 + (cols - 1) * 2
    figure = plt.figure(figsize=(fig_width, fig_height))

    use_normalized_coordinates = False
    input_image_size = (HEIGHT, WIDTH)

    for i, serialized_example in enumerate(test_ds):
        plt.subplot(num_of_examples // cols + 1, cols, i + 1)
        print(i)
        decoded_tensors = tf_ex_decoder.decode(serialized_example)

        image = build_inputs_for_object_detection(decoded_tensors['image'], input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        image_np = image[0].numpy()

        result = model_fn(image)

        boxes = result['detection_boxes'][0].numpy()
        classes = result['detection_classes'][0].numpy().astype(int)
        scores = result['detection_scores'][0].numpy()

        filtered_boxes = boxes[scores >= min_score_thresh]
        filtered_classes = classes[scores >= min_score_thresh]

        category_counts = {}
        for category in filtered_classes:
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

        count_text = ""
        for category, count in category_counts.items():
            label = category_index[category]['name']
            count_text += f"{label}: {count}, "

        count_text = count_text[:-2]

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            filtered_boxes,
            filtered_classes,
            scores,
            category_index=category_index,
            use_normalized_coordinates=use_normalized_coordinates,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=4
        )

        plt.imshow(image_np)
        plt.title(f'Image-{i+1}. {count_text}', fontsize=30)
        plt.axis('off')

    plt.savefig('inference_image.png')

def main():
    """
    This function parses command-line arguments specifying the paths to the exported
    model directory, test dataset, and other visualization parameters. Then, it calls
    the `import_and_run_inference` function to perform inference and visualization.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, default='./working_dir/retinanet_resnetfpn_coco/exported_model',
                        help='Directory containing the exported model.')
    parser.add_argument('--test_ds_path', type=str, default='./working_dir/augmented_data/bccd_coco_tfrecords/test-00000-of-00001.tfrecord',
                        help='Path to the test dataset in TFRecord format.')
    parser.add_argument('--num_of_examples', type=int, default=18,
                        help='Number of images to visualize.')
    parser.add_argument('--min_score_thresh', type=float, default=0.4,
                        help='Minimum score threshold for visualization.')
    parser.add_argument('--HEIGHT', type=int, default=256,
                        help='Height of the input images for preprocessing.')
    parser.add_argument('--WIDTH', type=int, default=256,
                        help='Width of the input images for preprocessing.')

    args = parser.parse_args()

    # Run inference and visualization
    visualize_gt_boxes(args.test_ds_path, num_of_examples=args.num_of_examples,
                       min_score_thresh=args.min_score_thresh, HEIGHT=args.HEIGHT, WIDTH=args.WIDTH)
    visualize_predicted_boxes(args.export_dir, args.test_ds_path, num_of_examples=args.num_of_examples,
                              min_score_thresh=args.min_score_thresh, HEIGHT=args.HEIGHT, WIDTH=args.WIDTH)


if __name__ == '__main__':
  main()
