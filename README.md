# Plant Disease Classification using CNN and Transfer Learning

This repository contains code and resources for the classification of multiple plant diseases using Convolutional Neural Networks (CNN) and Transfer Learning techniques.

![plant](https://github.com/adewoleaj/-Plant-Disease-Classification-using-CNN-and-Transfer-Learning/blob/main/plant%20disease.png?raw=true)


## Introduction

The primary objective of this study is to build a customized CNN model tailored specifically for the classification of plant diseases. We aim to evaluate its performance by comparing it against seven pre-trained architectures: VGG16, VGG19, ResNet50, DenseNet121, InceptionV4, EfficientNet, and MobileNet.

## Objectives

1. Designing a specialized CNN architecture focused on achieving high accuracy in plant disease classification.
2. Conducting a comparative analysis to assess the performance of the custom-designed CNN architecture against pre-trained models.
3. Evaluating the effectiveness of various optimization algorithms when applied to the top-performing models.

## Dataset

We utilized a subset of the Plant Village dataset, comprising tomato, potato, and pepper plants. This subset contains 20,638 images distributed across 15 distinct plant classes. The dataset was sourced from Kaggle ([Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)).

## Data Preprocessing and Augmentation

The images were resized to 64x64x3 dimensions and normalized by dividing the pixel values by 255. Data augmentation techniques were applied to address the issue of imbalanced datasets, including Horizontal Flip, Vertical Flip, Height Shift, Width Shift, Rotation, Shear, and Zoom.

## Methodology

The Adam optimizer was selected to train the CNN model due to its ability to automatically adjust learning rates, facilitating faster convergence and improved stability during training. Subsequently, both the custom and pre-trained models underwent training and evaluation using a variety of metrics. To further optimize performance, multiple optimizers including SGD, RMSprop, and Adamax were employed for both custom and pre-trained models. Each training session comprised 50 epochs, with early stopping implemented using a patience value of four to mitigate overfitting.

Transfer learning was utilized by freezing the top layer of the pre-trained model and then connecting it to the fully connected layers of our top-performing custom model. This approach leveraged the learned features from the pre-trained model while allowing for fine-tuning on our specific dataset.

![Research Flow](https://github.com/adewoleaj/-Plant-Disease-Classification-using-CNN-and-Transfer-Learning/blob/main/flow%20chat.png?raw=true)


For detailed information about the methodology, please refer to the attached PDF and Jupyter Notebook.

## Model Evaluation

Evaluation metrics include Accuracy, Precision, Recall, and F1 Score.

## Results

### Custom Model
- Optimizer: Adam
- Number of Convolutional Layers: 4
- Learning Rate: 0.001

### Pre-trained Model (ResNet50)
- Optimizer: Adamax

Overall, both the pre-trained and custom models performed similarly, with a slight 1% advantage observed for the pre-trained model in terms of accuracy. Notably, both models outperformed previous work conducted on the same dataset.


![compare](https://github.com/adewoleaj/-Plant-Disease-Classification-using-CNN-and-Transfer-Learning/blob/main/Presentation1.png?raw=true)


For more details, please refer to the attached documentation and code files.
