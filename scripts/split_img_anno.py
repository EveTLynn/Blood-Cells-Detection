import os
import shutil
import random

def split_images(images_path: str, train_images_folder: str, val_images_folder: str, test_images_folder: str,
  train_size: float = 0.85, val_size: float = 0.1, test_size: float = 0.05) -> None:
  """
  Splits a folder of images into training, validation, and test sets based on user-specified proportions.
  
  This function takes a directory containing images and splits them into three separate folders
  for training, validation, and testing purposes. The user can define the desired proportions
  for each set using the `train_size`, `val_size`, and `test_size` parameters.
  
  Args:
  images_path (str): Path to the folder containing the original images.
  train_images_folder (str): Path to the folder where the training images will be saved.
  val_images_folder (str): Path to the folder where the validation images will be saved.
  test_images_folder (str): Path to the folder where the test images will be saved.
  train_size (float, optional): Proportion of the dataset for the training set (defaults to 0.85).
  val_size (float, optional): Proportion of the dataset for the validation set (defaults to 0.1).
  test_size (float, optional): Proportion of the dataset for the test set (defaults to 0.05).
  """

  # Create a list of image filenames in 'images_path'
  image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']   #  a list of image extensions
  imgs_list = [filename for filename in os.listdir(images_path) if os.path.splitext(filename)[-1] in image_extensions]
  print(f'Number of images: {len(imgs_list)}')
  
  # Sets the random seed
  random.seed(42)
  # Shuffle the list of image filenames
  random.shuffle(imgs_list)
  
  # Calculate the number of images for each set based on proportions
  train_size = int(len(imgs_list) * 0.85)
  val_size = int(len(imgs_list) * 0.10)
  test_size = int(len(imgs_list) * 0.05)
  print(f"Train set: {train_size}, Val set: {val_size}, Test set: {test_size}")
  
  # Copy image files to destination folders
  for i, f in enumerate(imgs_list):
  # Determine destination folder based on index
    if i < train_size:
      dest_folder = train_images_folder
    elif i < train_size + val_size:
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


  # Get a list of all the image files in the directory
  # files = os.listdir(image_folder)
  
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
  
### define paths and create folders for train, val, test ###
images_path = './BCCD_Dataset/BCCD/JPEGImages'
annotation_path = './BCCD_Dataset/BCCD/Annotations'
working_dir ='./working_dir'

# path to new images folders
train_images_folder = os.path.join(working_dir, 'bccd_dataset', 'train', 'images')
val_images_folder = os.path.join(working_dir, 'bccd_dataset', 'val', 'images')
test_images_folder = os.path.join(working_dir, 'bccd_dataset', 'test', 'images')
# path to new annotation folders
train_anno_folder = os.path.join(working_dir, 'bccd_dataset', 'train', 'annotations')
val_anno_folder = os.path.join(working_dir, 'bccd_dataset', 'val', 'annotations')
test_anno_folder = os.path.join(working_dir, 'bccd_dataset', 'test', 'annotations')

# Create destination folders if they don't exist
for folder_path in [train_images_folder, val_images_folder, test_images_folder,
                    train_anno_folder, val_anno_folder, test_anno_folder]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

#Spilt images into train, val, test (ratio = 0.85, 0.1, 0.05)
split_images(images_path, train_images_folder, val_images_folder, test_images_folder)

# create train, val, test txt files and move annotations
create_filenamelist_and_move_annotations(train_images_folder, 'train.txt', annotation_path, train_anno_folder)
create_filenamelist_and_move_annotations(val_images_folder, 'val.txt', annotation_path, val_anno_folder)
create_filenamelist_and_move_annotations(test_images_folder, 'test.txt', annotation_path, test_anno_folder)

# move txt files (these can be use for voc2coco & tfrecords generation if you don't want augmenting data)
shutil.move("train.txt", "./working_dir/bccd_dataset/train")
shutil.move("val.txt", "./working_dir/bccd_dataset/val")
shutil.move("test.txt", "./working_dir/bccd_dataset/test")