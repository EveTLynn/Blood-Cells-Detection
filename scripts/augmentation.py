import os
import shutil
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import albumentations as alb
import argparse

working_dir = './working_dir'
def create_image_df(label_path: str) -> pd.DataFrame:
  """
  Parses a PASCAL VOC formatted XML annotation file and creates a Pandas DataFrame.

  Args:
      label_path (str): Path to the XML annotation file.

  Returns:
      pd.DataFrame: A DataFrame containing image metadata (filename, width, height)
       and bounding box information (class, xmin, ymin, xmax, ymax).

  Raises:
      FileNotFoundError: If the specified XML file is not found.
      RuntimeError: If the XML file structure is not compatible with PASCAL VOC format.
  """

  # Check if the XML file exists
  if not os.path.exists(label_path):
    raise FileNotFoundError(f"XML annotation file not found at: {label_path}")

  xml_list = []

  try:
    # Parse the XML file
    tree = ET.parse(label_path)
    root = tree.getroot()

    # Extract information for each object element
    for member in root.findall('object'):
      # Extract filename, width, height, class name, and bounding box coordinates
      value = (
          root.find('filename').text,
          int(root.find('size')[0].text),  # Image width
          int(root.find('size')[1].text),  # Image height
          member[0].text,                  # Object class
          int(member[4][0].text),  # xmin
          int(member[4][1].text),  # ymin
          int(member[4][2].text),  # xmax
          int(member[4][3].text)   # ymax
      )
      xml_list.append(value)

  except ET.ParseError as e:
    raise RuntimeError(f"Invalid XML file structure: {e}")

  column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list, columns=column_names)

  return xml_df


def write_xml_annotation(folder_path: str, image_name: str, width: int, height: int,
                         bboxes: list[list], class_labels: list[str]) -> None:
  """
  Creates and saves an XML annotation file in PASCAL VOC format for a single image.

  This function takes information about an image (filename, dimensions) and a list of
  bounding boxes (coordinates and class labels) to generate an XML annotation file
  adhering to the PASCAL VOC format. The file is saved in the specified folder.

  Args:
      folder_path (str): Path to the folder where the XML file will be saved.
      image_name (str): Name of the image file (without extension).
      width (int): Width of the image in pixels.
      height (int): Height of the image in pixels.
      bboxes (list of list): List of bounding boxes in Pascal VOC format ([[x_min, y_min, x_max, y_max]]).
      class_labels (list of str): List of class labels for each bounding box.
  """

  # Create the root element for the XML annotation
  annotation = ET.Element("annotation")
  # Sub-element for filename with proper extension
  ET.SubElement(annotation, "filename").text = f"{image_name}.jpg"

  # Sub-elements for source information (replace with your dataset details)
  source = ET.SubElement(annotation, "source")
  ET.SubElement(source, "database").text = "BCCD_augmented"  
  ET.SubElement(source, "annotation").text = "Object Detection"

  # Sub-element for image size
  size = ET.SubElement(annotation, "size")
  ET.SubElement(size, "width").text = str(width)
  ET.SubElement(size, "height").text = str(height)
  ET.SubElement(size, "depth").text = str(3)  # Assuming RGB image (3 channels)

  # Create object elements and bounding boxes for each entry in the lists
  for bbox, class_label in zip(bboxes, class_labels):
    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = class_label
    ET.SubElement(obj, "pose").text = "Unspecified"

    # Extract bounding box coordinates (assuming already in PASCAL VOC format)
    x_min, y_min, x_max, y_max = bbox

    # Sub-element for bounding box details
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(x_min))
    ET.SubElement(bndbox, "ymin").text = str(int(y_min))
    ET.SubElement(bndbox, "xmax").text = str(int(x_max))
    ET.SubElement(bndbox, "ymax").text = str(int(y_max))

  # Create the XML tree structure
  tree = ET.ElementTree(annotation)

  # Create the folder path if it doesn't exist
  os.makedirs(folder_path, exist_ok=True)
  # Create the complete file path with extension
  xml_file_path = os.path.join(folder_path, f"{image_name}.xml")

  # Write the XML tree to the file with UTF-8 encoding
  with open(xml_file_path, "wb") as f:
    tree.write(f, encoding="utf-8")


def augment_images(input_path: str,
                   output_path: str,
                   num_augmentations: int = 3,
                   aug_width: int = 256,
                   aug_height: int = 256,
                   ) -> None:
  """
  Augments images in a directory and creates corresponding bounding box annotations.

  This function reads images and their associated bounding box annotations (in PASCAL VOC format)
  from a specified directory. It then applies a series of image augmentation techniques defined
  by an Albumentations augmentation pipeline. For each image-annotation pair, the function creates
  `num_augmentations` augmented versions and saves them along with their corresponding augmented
  bounding boxes to the specified output directory. The directory structure will mirror the input.

  **Input Directory Structure:**
    input_path/
    - images/ (folder containing original images)
    - annotations/ (folder containing PASCAL VOC format annotation files)

  **Output Directory Structure:**
    output_path/
    - images/ (folder containing augmented images)
    - annotations/ (folder containing augmented annotation files)

  Args:
    input_path (str): Path to the directory containing the original images and annotations.
    output_path (str): Path to the directory where the augmented images and annotations will be saved.
    num_augmentations (int, optional): The number of augmented versions to create for each image. Defaults to 3.
    aug_width (int, optional): Target width for resizing augmented images. Defaults to 256.
    aug_height (int, optional): Target height for resizing augmented images. Defaults to 256.
  """

  # TODO: Allow customization of augmentation parameters through arguments
  # Define Albumentations augmentation pipeline (adjust hyperparameters as needed)
  augmentor = alb.Compose([alb.RandomCrop(width=aug_width, height=aug_height),   # Adjust crop size
                           alb.HorizontalFlip(p=0.5),               # Flips images horizontally with 50% probability
                           alb.RandomBrightnessContrast(p=0.2),     # Adjust brightness/contrast randomly
                           alb.RandomGamma(p=0.2),                  # Adjust gamma values
                           alb.RGBShift(p=0.2),                     # Shift RGB channels randomly
                           alb.VerticalFlip(p=0.5)],                # Flips images vertically with 50% probability
                           bbox_params=alb.BboxParams(format='pascal_voc',
                                                      min_visibility=0.5,  # drop if new bbox is less than 50% visible
                                                      label_fields=['class_labels']))

  # Process each image in the input directory
  for image in os.listdir(os.path.join(input_path, 'images')):
    img = cv2.imread(os.path.join(input_path, 'images', image))
    label_path = os.path.join(input_path, 'annotations', f'{image.split(".")[0]}.xml')

    # if the annotations exist check the validity of the coordinates
    if os.path.exists(label_path):
      # parse the annotations to get the coordinates and labels
      xml_df = create_image_df(label_path)

      # Filter invalid bounding boxes during parsing (replace with your specific checks)
      valid_bboxes = []
      valid_labels = []
      for idx, row in xml_df.iterrows():
          bbox = row.iloc[-4:].tolist()  # Assuming coordinates are the last 4 columns
          if (len(bbox) != 4 or  # Ensure 4 coordinates
              not all(isinstance(val, (int, float)) for val in bbox) or  # Check data types
              any(np.isnan(bbox)) or  # Check for NaNs
              bbox[2] <= bbox[0] or   # Check x_max > x_min (assuming x is at index 0 and 2)
              bbox[3] <= bbox[1]):    # Check y_max > y_min (assuming y is at index 1 and 3)
              print(f"Skipping invalid bbox: {bbox} for image: {xml_df.iloc[idx, 0]}.")
              continue  # Skip to the next row in the DataFrame

          valid_bboxes.append(bbox)
          valid_labels.append(row["class"])  

      # Use the filtered bounding boxes and labels for augmentation
      bboxes = valid_bboxes
      class_labels = valid_labels

      # Generate and save augmented images and annotations
      for x in range(num_augmentations):
        # run through the augmentation pipline
        augmented = augmentor(image=img, bboxes=bboxes, class_labels=class_labels)

        # # Save augmented image to desninated folder
        os.makedirs(os.path.join(output_path,'images'), exist_ok=True)
        image_name = f'{image.split(".")[0]}{x}'
        image_file_path = os.path.join(output_path, 'images', f"{image_name}.jpg")
        cv2.imwrite(image_file_path, augmented['image'])

        # write augmented annotations to desninated folder
        xml_path = os.path.join(output_path, 'annotations')
        write_xml_annotation(folder_path = xml_path, image_name = image_name, 
                             width = aug_height, height = aug_height,
                             bboxes = augmented["bboxes"], class_labels = augmented["class_labels"])

  # Genrate a text file with all the image names (without the extension) for voc2coco script
  # Get a list of image file names (without extensions)
  image_folder = os.path.join(output_path, 'images')
  # files = os.listdir(image_folder)
  image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.lower().endswith('.jpg')]

  # Write the file names to a text file
  with open('filenames.txt', 'w') as f:
      for image_name in image_files:
          f.write(image_name + '\n')
  shutil.move("./filenames.txt", output_path)


def main():
  parser = argparse.ArgumentParser(
      description="Augment images and annotations for object detection.")
  parser.add_argument("--input_path", type=str, default=None,
                      help="Path to the input directory.")
  parser.add_argument("--output_path", type=str, default=None,
                      help="Path to the output directory.")
  parser.add_argument("--num_augmentations", type=int, default=3,
                      help="Number of augmentations to create.")
  args = parser.parse_args()

  # Check if input and output paths are provided
  if None in [args.input_path, args.output_path]:
      parser.error("Both input_path and output_path must be provided.")

  # Create subfolders train, val, and test in the output path
  train_output_path = os.path.join(args.output_path, 'train')
  val_output_path = os.path.join(args.output_path, 'val')
  test_output_path = os.path.join(args.output_path, 'test')
  for folder_path in [train_output_path, val_output_path, test_output_path]:
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)

  # Perform augmentation for train set
  try:
      augment_images(os.path.join(args.input_path, 'train'), train_output_path, args.num_augmentations)
      print("Train set augmentation completed successfully.")
  except Exception as e:
      print(f"An error occurred during train set augmentation: {e}")

  # Perform augmentation for validation set
  try:
      augment_images(os.path.join(args.input_path, 'val'), val_output_path, args.num_augmentations)
      print("Validation set augmentation completed successfully.")
  except Exception as e:
      print(f"An error occurred during validation set augmentation: {e}")

  # Perform augmentation for test set
  try:
      augment_images(os.path.join(args.input_path, 'test'), test_output_path, args.num_augmentations)
      print("Test set augmentation completed successfully.")
  except Exception as e:
      print(f"An error occurred during test set augmentation: {e}")


if __name__ == "__main__":
  main()
