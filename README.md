# Blood Cells Detection on BCCD Dataset
This project implements fine-tuning of a RetinaNet model with a ResNet-50 backbone from the [TensorFlow Model Garden (TFM)](https://github.com/tensorflow/models) for detecting three types of blood cells (Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets) in microscopic images from the [BCCD dataset](https://github.com/Shenggan/BCCD_Dataset).

## Project Overview
This project was run on Colab free GPU with Tensorflow 2.15.0, it will take approximately 1 hour to run for 10,000 epochs. A copy of the notebook is available in this repository but since its size is too large for rendering on Github, please use [nbviewer](https://nbviewer.org/)
or visit [this Colab link](https://colab.research.google.com/drive/1PsVqMfThRWEhOG1w2HDbs7OgICSIMd2d?usp=drive_link). 

The notebook wil guide you through the following steps:

### 1 . Data Preparation
- Clone the BCCD dataset and this github repo for `custom_preprocessing.py`, `voc2coco.py`, and `labels.txt` scripts. The voc2coco scripts is from the [Roboflow github](https://github.com/roboflow/voc2coco) and a copy of it is stored in this repository for convenience.
- Use functions from `custom_preprocessing.py` to split images and annotations to three separate folders: train, val, test
- Augment images and annotations (random cropping, flipping, etc.) with Albumentations library, also utilize the functions from `custom_preprocessing.py`
- Convert annotations from PASCAL VOC format to COCO format using `voc2coco.py` script
- Generate [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) with `create_coco_tf_record.py` from TFM package, it is a TensorFlow simple format for storing a sequence of binary records.

After this step the structure of the working directory will have structure like below
```
Blood-Cells-Detection
├── augmented_data                      # Augemented data
│  ├── bccd_coco_tfrecords              # This folder contains tfrecords for train, val and test data 
│  ├── test
│  ├── train
│  └── val                              # The structure of test, train, val folder is the same
│       ├── annotations
│       ├── images
│       ├── filenames.txt               # Names of the annotation files without the extension
│       └── val_cocoformat.json         # Annotation after convert from PASCAL VOC to COCO format
|── bccd_dataset                        # Splited bccd dataset
|       ├── test
|       ├── train
|       └── val                         # The structure of test, train, val folder is the same
|            ├── annotations
|            ├── images
|            └── val.txt                # Names of the annotation files without the extension
└── scripts
        ├── custom_preprocessing.py     # A script to preprocess data for voc2coco script
        |── labels.txt                  # A text file contain name of each blood type for encoding
        └── voc2coco.py                 # A script to convert xml file to coco format
```

### 2. Model Configuration
Leverage the TFM package to access the pre-trained RetinaNet model with a ResNet-50 backbone.
- Adjust the model, dataset and trainer configuration, including:
  - Adjust the model and dataset configurations
    - Backbone: Get the pretrained checkpoint from [TFM official vision model github](https://github.com/tensorflow/models/blob/master/official/vision/MODEL_GARDEN.md), initiate pretrained checkpoint and freeze backbone (keep the backbone's weight from being updated)
    - Define image size, number of class, input paths (train and validation)
  - Adjust the trainer configuration (train - val steps, optimizer, saving checkpoints..)
- Set up the distribution stratergy takes full advantage of available hardware
- Create the Task object: an easy way to build, train, and evaluate models 

### 3. Training and Evaluation
- Train the fine-tuned model on the prepared TFRecord data.
- Monitor training progress using TensorBoard: the Tensorboard read `events.out.tfevents` and visualize as charts, graphs, and images which can help us understand how the model is performing during training and diagnose any issues.
- Evaluate the model's performance on a held-out validation set using [COCO detection evaluation metrics](https://cocodataset.org/#detection-eval), but as TF allow to export best model based on only one metric, I chose mean average precision for simplicity.
- 5 newest models are saved by default and I also keep the model with the highest mAP (this will be used for inference later)

Here is the output directory tree
```
retina_resnetfpn_coco
├── base_ckpt                          # Contains pretrained checkpoint from TFM
├── trained_model
│  ├── best_checkpoints                # Checkpoint with the highest mAP
│  ├── train                           # Contains train tfevent file for Tensorboard
│  ├── val                             # Contains validation tfevent file for Tensorboard
│  ├── ckpt-100.data-00000-of-00000    # 5 newest checkpoints 
│  └── ckpt-100.index
└── exported_model                     # Stores exported model used for inference
```

Due to the random nature of data augmentation, each training run with the same model architecture will produce slightly different results. 

I performed 4-5 training runs each with 10,000 epochs to get a broader picture of the model's performance, it seems that the mean Average Precision (mAP) and validation loss stopped improving significantly after approximately 4,000 epochs. The best mAP achieved across these runs fell within a range of 52% to 58%. 

Below is the last training and evaluation results.

![](results/APm.png) | ![](results/ARm.png) | ![](results/model_loss.png)|
:------------------: | :------------------: | :------------------:|
Mean Average Precison |  Mean Average Recall | Model Loss

It's clear that after roughly 3500 epochs, the mAP, mAR and also validation loss stagnated or even started to deteriorate. This behavior suggests that the model might be overfitting the training data.

Since the dataset size is relatively small, with only 364 images, overfitting is a potential concern. Though, indeed, the results could be improved, I decided to settle to the current model for now. I saved the checkpoint at 3500 epochs (the highest mAP) for generating predictions on new data.

### 4. Inference:
- Load the trained model weights.
- Apply the model to detect blood cells in new, unseen images. 
- The model will output bounding boxes and class labels for identified cells with the counting for each blood cells type.

![](results/test_gt.png)  |
:-------------------------:
Visualization of groundtruth bounding boxes

![](results/test_pred.png)|
:-------------------------:
Visualization of predicted bounding boxes

The model actually detects the blood cells accurately despite not having very good mAP!   

<!---
**PS:** This project has been an incredible learning journey, and I couldn't have made it without the amazing Ms. Tyna as my mentor from the PyLadies Vienna program! Her support made this project a total joy to work on.
Huge thanks for all the support and guidance!
-->

**PPS: I'm Always Learning!**

This repository is a work in progress, and I welcome your contributions! If you have any suggestions for improvement, feel free to open an issue or submit a pull request. I'm always looking for ways to enhance this project and make it more valuable for the community.
