#!/bin/bash

# split image and annotation
python ./scripts/split_img_anno.py \
  --images_path ./BCCD_Dataset/BCCD/JPEGImages \
  --annotation_path ./BCCD_Dataset/BCCD/Annotations \
  --output_path ./working_dir/bccd_dataset 

# augmentation
python ./scripts/augmentation.py \
    --input_path ./working_dir/bccd_dataset \
    --output_path ./working_dir/augmented_data \
    --num_augmentations 10

# Convert the annotions from PASCAL VOC to Coco format
## train
python ./scripts/voc2coco.py \
  --ann_dir ./working_dir/augmented_data/train/annotations \
  --ann_ids ./working_dir/augmented_data/train/filenames.txt \
  --labels ./scripts/labels.txt \
  --output ./working_dir/augmented_data/train/train_cocoformat.json \
  --ext xml

## val
python ./scripts/voc2coco.py \
  --ann_dir ./working_dir/augmented_data/val/annotations \
  --ann_ids ./working_dir/augmented_data/val/filenames.txt \
  --labels ./scripts/labels.txt \
  --output ./working_dir/augmented_data/val/val_cocoformat.json \
  --ext xml

## test
python ./scripts/voc2coco.py \
  --ann_dir ./working_dir/augmented_data/test/annotations \
  --ann_ids ./working_dir/augmented_data/test/filenames.txt \
  --labels ./scripts/labels.txt \
  --output ./working_dir/augmented_data/test/test_cocoformat.json \
  --ext xml

# generate tfrecords
## train
TRAIN_DATA_DIR='./working_dir/augmented_data/train/images'
TRAIN_ANNOTATION_FILE_DIR='./working_dir/augmented_data/train/train_cocoformat.json'
OUTPUT_TFRECORD_TRAIN='./working_dir/augmented_data/bccd_coco_tfrecords/train'

python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=${TRAIN_DATA_DIR} \
  --object_annotations_file=${TRAIN_ANNOTATION_FILE_DIR} \
  --output_file_prefix=${OUTPUT_TFRECORD_TRAIN} \
  --num_shards=1

## validation
VALID_DATA_DIR='./working_dir/augmented_data/val/images'
VALID_ANNOTATION_FILE_DIR='./working_dir/augmented_data/val/val_cocoformat.json'
OUTPUT_TFRECORD_VALID='./working_dir/augmented_data/bccd_coco_tfrecords/valid'

python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=$VALID_DATA_DIR \
  --object_annotations_file=$VALID_ANNOTATION_FILE_DIR \
  --output_file_prefix=$OUTPUT_TFRECORD_VALID \
  --num_shards=1

# test
TEST_DATA_DIR='./working_dir/augmented_data/test/images'
TEST_ANNOTATION_FILE_DIR='./working_dir/augmented_data/test/test_cocoformat.json'
OUTPUT_TFRECORD_TEST='./working_dir/augmented_data/bccd_coco_tfrecords/test'

python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=$TEST_DATA_DIR \
  --object_annotations_file=$TEST_ANNOTATION_FILE_DIR \
  --output_file_prefix=$OUTPUT_TFRECORD_TEST \
  --num_shards=1

# train and evaluate
python ./scripts/train.py \
  --model_name=retinanet_resnetfpn_coco \
  --working_dir=./working_dir \
  --input_path=./working_dir/augmented_data/bccd_coco_tfrecords/ \
  --checkpoint_url="https://storage.googleapis.com/tf_model_garden/vision/retinanet/retinanet-resnet50fpn.tar.gz" \
  --HEIGHT=256 \
  --WIDTH=256 \
  --train_steps=1000 \
  --ckpt_path=best_ckpt

# run inference
python ./scripts/inference.py \
  --export_dir=./working_dir/retinanet_resnetfpn_coco/exported_model \
  --test_ds_path=./working_dir/augmented_data/bccd_coco_tfrecords/test-00000-of-00001.tfrecord \
  --number_of_images=18 \
  --HEIGHT=256 \
  --WIDTH=256 \
  --min_score_thresh=0.4