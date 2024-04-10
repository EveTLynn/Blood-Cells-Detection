import os
import shutil
import random
import argparse

def split_images(images_path: str, output_path: str, train_size: float = 0.85, val_size: float = 0.1, test_size: float = 0.05) -> None:
  """
  Splits a folder of images into training, validation, and test sets based on user-specified proportions.
  
  This function takes a directory containing images and splits them into three separate folders
  for training, validation, and testing purposes. The user can define the desired proportions
  for each set using the `train_size`, `val_size`, and `test_size` parameters.
  
  Args:
  images_path (str): Path to the folder containing the original images.
  output_path (str): Path to the folder where the training, validation, and test images will be saved.
  train_size (float, optional): Proportion of the dataset for the training set (defaults to 0.85).
  val_size (float, optional): Proportion of the dataset for the validation set (defaults to 0.1).
  test_size (float, optional): Proportion of the dataset for the test set (defaults to 0.05).
  """

  # Create destination folders if they don't exist
  train_images_folder = os.path.join(output_path, 'train', 'images')
  val_images_folder = os.path.join(output_path, 'val', 'images')
  test_images_folder = os.path.join(output_path, 'test', 'images')
  
  for folder_path in [train_images_folder, val_images_folder, test_images_folder]:
    os.makedirs(folder_path, exist_ok=True)
  
  # Create a list of image filenames in 'images_path'
  image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']   #  a list of image extensions
  imgs_list = [filename for filename in os.listdir(images_path) if os.path.splitext(filename)[-1] in image_extensions]
  print(f'Number of images: {len(imgs_list)}')
  
  # Shuffle the list of image filenames
  random.seed(42)
  random.shuffle(imgs_list)
  
  # Calculate the number of images for each set based on proportions
  train_count = int(len(imgs_list) * train_size)
  val_count = int(len(imgs_list) * val_size)
  test_count = len(imgs_list) - train_count - val_count
  print(f"Train set: {train_count}, Val set: {val_count}, Test set: {test_count}")
  
  # Copy image files to destination folders
  for i, f in enumerate(imgs_list):
  # Determine destination folder based on index
    if i < train_count:
      dest_folder = train_images_folder
    elif i < train_count + val_count:
      dest_folder = val_images_folder
    else:
      dest_folder = test_images_folder
    shutil.copy(os.path.join(images_path, f), os.path.join(dest_folder, f))


def create_filenamelist_and_move_annotations(image_folder: str, output_file_name: str,
                               input_path: str, output_path: str) -> None:
  """
  Creates a text file containing a list of image filenames (without extensions) and
  moves corresponding XML annotation files from the input path to the output path.

  This function processes a directory containing image files (`.jpg` extension assumed),
  generates a text file listing the image filenames, and moves any corresponding XML
  annotation files (with the same base filename as the image) from a specified input
  directory to a specified output directory.

  Args:
  image_folder (str): Path to the folder containing the image files.
  output_file_name (str): Name of the text file to create, containing image filenames.
  input_path (str): Path to the directory containing the XML annotation files.
  output_path (str): Path to the directory where the XML annotation files will be moved.
  """

  # Get a list of image file names (without extensions)
  image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.lower().endswith('.jpg')]

  # Write the file names to the text file
  with open(output_file_name, 'w') as f:
    for image_name in image_files:
      f.write(image_name + '\n')
      # Match XML files with corresponding image names
      xml_file = os.path.join(input_path, f'{image_name}.xml')
      if os.path.exists(xml_file):
        # Move the matched XML file to the output folder
        shutil.copy(xml_file, os.path.join(output_path, f'{image_name}.xml'))

  # Note: Using shutil.move instead of shutil.copy above would permanently remove
  # the files from the input_path directory. If you don't want to remove them,
  # keep the code as shutil.copy(xml_file, ...)


def main():
  parser = argparse.ArgumentParser(description="Split images into train, val, and test sets.")
  parser.add_argument("--images_path", type=str, required=True,
                      help="Path to the folder containing the original images.")
  parser.add_argument("--annotation_path", type=str, required=True,
                      help="Path to the folder containing the XML annotation files.")
  parser.add_argument("--output_path", type=str, required=True,
                      help="Path to the output directory.")
  parser.add_argument("--train_size", type=float, default=0.85,
                      help="Proportion of the dataset for the training set (defaults to 0.85).")
  parser.add_argument("--val_size", type=float, default=0.1,
                      help="Proportion of the dataset for the validation set (defaults to 0.1).")
  parser.add_argument("--test_size", type=float, default=0.05,
                      help="Proportion of the dataset for the test set (defaults to 0.05).")
  args = parser.parse_args()

  # Create output subfolders
  train_images_folder = os.path.join(args.output_path, "train", "images")
  val_images_folder = os.path.join(args.output_path, "val", "images")
  test_images_folder = os.path.join(args.output_path, "test", "images")

  train_anno_folder = os.path.join(args.output_path, "train", "annotations")
  val_anno_folder = os.path.join(args.output_path, "val", "annotations")
  test_anno_folder = os.path.join(args.output_path, "test", "annotations")

  for folder_path in [train_images_folder, val_images_folder, test_images_folder,
                      train_anno_folder, val_anno_folder, test_anno_folder]:
      os.makedirs(folder_path, exist_ok=True)

  # Split images into train, val, and test sets
  split_images(images_path=args.images_path, output_path=args.output_path, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size)
  # Create train, val, test txt files and move annotations
  create_filenamelist_and_move_annotations(train_images_folder, 'train.txt', args.annotation_path, train_anno_folder)
  create_filenamelist_and_move_annotations(val_images_folder, 'val.txt', args.annotation_path, val_anno_folder)
  create_filenamelist_and_move_annotations(test_images_folder, 'test.txt', args.annotation_path, test_anno_folder)

  # Move txt files to corresponding folders
  shutil.move("train.txt", os.path.join(args.output_path, "train", "train.txt"))
  shutil.move("val.txt", os.path.join(args.output_path, "val", "val.txt"))
  shutil.move("test.txt", os.path.join(args.output_path, "test", "test.txt"))

  print("Image splitting, annotation files creation, and txt files moving completed successfully.")


if __name__ == "__main__":
  main()
