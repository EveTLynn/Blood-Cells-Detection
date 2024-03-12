# -*- coding: utf-8 -*-
"""custom_preprocess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V18xSZnCE7uPJlUD3C4xS8bgqvZoJGGM
"""

# Some custom functions to incoporate Albumentations, voc2coco with TF Model Garden

import os
import glob
import shutil
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import albumentations as alb


def split_images(images_path, train_images_folder, val_images_folder, test_images_folder,
                 train_size=0.85, val_size=0.1, test_size=0.05):

  """
  Splits a folder of images into training, validation, and test sets.

  Args:
      images_path (str): Path to the folder containing the images.
      train_images_folder (str): Path to the folder where the training images will be saved.
      val_images_folder (str): Path to the folder where the validation images will be saved.
      test_images_folder (str): Path to the folder where the test images will be saved.
      train_size (float): The proportion of the dataset to include in the training set.
      val_size (float): The proportion of the dataset to include in the validation set.
      test_size (float): The proportion of the dataset to include in the test set.
  """
  # Create a list of image filenames in 'images_path'
  image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']   #  a list of image extensions
  imgs_list = [filename for filename in os.listdir(images_path) if os.path.splitext(filename)[-1] in image_extensions]
  print(f'Number of images: {len(imgs_list)}')

  # Sets the random seed
  random.seed(42)
  # Shuffle the list of image filenames
  random.shuffle(imgs_list)

  # determine the number of images for each set
  train_size = int(len(imgs_list) * 0.85)
  val_size = int(len(imgs_list) * 0.10)
  test_size = int(len(imgs_list) * 0.05)
  print(f"Train set: {train_size}, Val set: {val_size}, Test set: {test_size}")

  # Copy image files to destination folders
  for i, f in enumerate(imgs_list):
      if i < train_size:
          dest_folder = train_images_folder
      elif i < train_size + val_size:
          dest_folder = val_images_folder
      else:
          dest_folder = test_images_folder
      shutil.copy(os.path.join(images_path, f), os.path.join(dest_folder, f))


def create_filenamelist_and_move_annotations(image_folder, output_file_name, annotation_folder):
  """
  Creates a text file containing a list of image filenames (without extensions) and
  copies corresponding XML annotation files to a specified output folder.

  Args:
      image_folder (str): Path to the folder containing the image files.
      output_file_name (str): Name of the text file to create, containing image filenames.
      annotation_folder (str): Path to the folder containing the XML annotation files.
  """
  # Get a list of all the image files in the directory
  files = os.listdir(image_folder)

  # Get a list of image file names (without extensions)
  image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.lower().endswith('.jpg')]

  # Write the file names to a text file
  with open(output_file_name, 'w') as f:
      for image_name in image_files:
          f.write(image_name + '\n')
          # Match XML files with corresponding image names
          xml_file = os.path.join(annotation_path, f'{image_name}.xml')
          if os.path.exists(xml_file):
              # Move the matched XML file to the output folder
              shutil.copy(xml_file, os.path.join(annotation_folder, f'{image_name}.xml'))


def create_image_df(label_path):
  """
  Parses a PASCAL VOC formatted XML annotation file and creates a Pandas DataFrame.

  Args:
      label_path (str): Path to the XML annotation file.
  Returns:
      pandas.DataFrame: A DataFrame containing image metadata and bounding box information.
  """

  xml_list = []
  tree = ET.parse(label_path)
  root = tree.getroot()
  for member in root.findall('object'):
    value = (root.find('filename').text ,
             int(root.find('size')[0].text),
             int(root.find('size')[1].text),
             member[0].text,
             int(member[4][0].text),
             int(member[4][1].text),
             int(member[4][2].text),
             int(member[4][3].text)
             )
    xml_list.append(value)
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list, columns=column_name)
  return xml_df


def write_xml_annotation(folder_path, image_name, width, height, bboxes, class_labels):
  """
  Writes an XML annotation file for a single image in PASCAL VOC format.

  Args:
      folder_path (str): Path to the folder where the XML file will be saved.
      image_name (str): Name of the image file (without extension).
      width (int): Width of the image in pixels.
      height (int): Height of the image in pixels.
      bboxes (list of lists): List of bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max).
      class_labels (list of str): List of class labels for each bounding box.
  """

  annotation = ET.Element("annotation")
  ET.SubElement(annotation, "filename").text = f"{image_name}.jpg"

  source = ET.SubElement(annotation, "source")
  ET.SubElement(source, "database").text = "BCCD_augmented"  # Replace with your dataset name
  ET.SubElement(source, "annotation").text = "Object Detection"

  size = ET.SubElement(annotation, "size")
  ET.SubElement(size, "width").text = str(width)
  ET.SubElement(size, "height").text = str(height)
  ET.SubElement(size, "depth").text = str(3)  # Assuming RGB image (3 channels)

  for bbox, class_label in zip(bboxes, class_labels):
    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = class_label
    ET.SubElement(obj, "pose").text = "Unspecified"

    # No need to convert coordinates since they're already in pixels (Pascal VOC format)
    x_min, y_min, x_max, y_max = bbox

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(x_min))
    ET.SubElement(bndbox, "ymin").text = str(int(y_min))
    ET.SubElement(bndbox, "xmax").text = str(int(x_max))
    ET.SubElement(bndbox, "ymax").text = str(int(y_max))

  tree = ET.ElementTree(annotation)

  # Create the folder path if it doesn't exist
  os.makedirs(folder_path, exist_ok=True)
  xml_file_path = os.path.join(folder_path, f"{image_name}.xml")

  with open(xml_file_path, "wb") as f:
    tree.write(f, encoding="utf-8")


def augment_images(input_path, output_path, num_augmentations=3):
  """
  Augments images in a directory and creates corresponding bounding box annotations.

  This function reads images and their associated bounding box annotations (in PASCAL VOC format)
  from a specified input directory. It then applies a series of image augmentation techniques
  defined by an Albumentations augmentation pipeline. For each image, the function creates
  `num_augmentations` augmented versions and saves them along with their corresponding
  augmented bounding boxes to the specified output directory.

  Args:
      input_path (str): Path to the directory containing the original images and annotations.
          The directory structure should be:
          - input_path/
              - images/ (folder containing original images)
              - annotations/ (folder containing PASCAL VOC format annotation files)
      output_path (str): Path to the directory where the augmented images and annotations will be saved.
          The same directory structure as the input will be created here.
      num_augmentations (int, optional): The number of augmented versions to create for each image. Defaults to 3.
  """

  # TODO: Add args for custom augmentation
  # Specify the hyperparameters for augmentation using Albumentations
  augmentor = alb.Compose([alb.RandomCrop(width=480, height=480),
                           alb.HorizontalFlip(p=0.5),
                           alb.RandomBrightnessContrast(p=0.2),
                           alb.RandomGamma(p=0.2),
                           alb.RGBShift(p=0.2),
                           alb.VerticalFlip(p=0.5)],
                           bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['class_labels']))

  # Process each image in the input directory
  for image in os.listdir(os.path.join(input_path, 'images')):
    img = cv2.imread(os.path.join(input_path, 'images', image))
    label_path = os.path.join(input_path, 'annotations', f'{image.split(".")[0]}.xml')

    # if the annotations exist
    if os.path.exists(label_path):
      # parse the annotations to get the coordinates and labels
      xml_df = create_image_df(label_path)
      bboxes = xml_df.iloc[:, -4:].values
      class_labels = xml_df["class"].to_list()

      # # Generate and save augmented images and annotations
      for x in range(num_augmentations):
        # run through the augmentation pipline
        augmented = augmentor(image=img, bboxes=bboxes, class_labels=class_labels)

        # save the image to desninated folder
        os.makedirs(os.path.join(output_path,'images'), exist_ok=True)
        image_name = f'{image.split(".")[0]}_{x}'
        image_file_path = os.path.join(output_path, 'images', f"{image_name}.jpg")
        cv2.imwrite(image_file_path, augmented['image'])

        # write annotations to desninated folder
        xml_path = os.path.join(output_path, 'annotations')
        write_xml_annotation(folder_path = xml_path, image_name = f'{image.split(".")[0]}_{x}', # no idea why image_name=image_name wasn't working
                             width = 480, height = 480,
                             bboxes = augmented["bboxes"], class_labels = augmented["class_labels"])

  # Genrate a text file with all the image names (without the extension) for voc2coco script
  # Get a list of image file names (without extensions)
  image_folder = os.path.join(output_path, 'images')
  files = os.listdir(image_folder)
  image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.lower().endswith('.jpg')]

  # Write the file names to a text file
  with open('filenames.txt', 'w') as f:
      for image_name in image_files:
          f.write(image_name + '\n')
  shutil.move("./filenames.txt", output_path)