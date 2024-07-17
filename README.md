# Neural-Networks-and-Deep-Learning-Course-Projects

- [Neural-Networks-and-Deep-Learning-Course-Projects](#neural-networks-and-deep-learning-course-projects)
  - [HW1: Fully Connected Neural Networks](#hw1-fully-connected-neural-networks)
    - [1. McCulloch-Pitts Network](#1-mcculloch-pitts-network)
    - [2. Adversarial Attacks](#2-adversarial-attacks)
    - [3. Adeline and Madeline](#3-adeline-and-madeline)
    - [4. Optimised Neural Network](#4-optimised-neural-network)
  - [HW2: Convolutional Neural Networks](#hw2-convolutional-neural-networks)
    - [1. Alzheimers Classification](#1-alzheimers-classification)
    - [2. Fine-Tuning VGG16 and ResNet-50 and Data Augmentation effect](#2-fine-tuning-vgg16-and-resnet-50-and-data-augmentation-effect)
  - [HW3: Region Based Convolutional Neural Networks](#hw3-region-based-convolutional-neural-networks)
    - [1. Semantic Segmentation using U-Net](#1-semantic-segmentation-using-u-net)
    - [2. Underwater Object Detection using Faster R-CNN](#2-underwater-object-detection-using-faster-r-cnn)

## HW1: Fully Connected Neural Networks

This project contains 4 parts:

### 1. McCulloch-Pitts Network

In this part, we implemented a twos complement adder using a simple 2 layer network with 7 input neurons and 4 output neurons. The Network takes a 4-bit binary number as input and produces the twos complement of the number as output. The input neurons are binary neurons, which means that they can only take values 0 or 1. The output neurons are also binary neurons and only the corresponding output neuron for the digit should be 1 and the rest should be 0. For example if the input is 1100 the output should be 0100.

### 2. Adversarial Attacks

In this part, we implemented the Fast Gradient Sign Method (FGSM) and the Projected Gradient Descent (PGD) attacks on a neural network trained on the MNIST dataset. The FGSM attack is a white-box attack that generates adversarial examples by adding a small perturbation to the input data in the direction of the gradient of the loss function with respect to the input data. The PGD attack is a white-box attack that generates adversarial examples by iteratively applying the FGSM attack with a small step size and clipping the perturbation to ensure that the resulting adversarial example is within a small L-infinity norm ball around the original input data. We evaluated the attacks on the MNIST dataset and measured the success rate of the attacks.

### 3. Adeline and Madeline

In this part, we implemented the Adeline and Madaline algorithms for classification. The Adeline algorithm is a single-layer neural network that uses the least mean squares (LMS) algorithm to learn the weights of the network. The Madaline algorithm is a multi-layer neural network that uses the perceptron learning rule to learn the weights of the network. We evaluated the algorithms on the Iris dataset and measured the accuracy of the algorithms.

### 4. Optimised Neural Network

In this part, we tried to see how overfitting and other things can be avoided in a neural network. We see different models for two problems one is classification and the other is regression. We see how the model performs on the training and validation data and how the model can be optimised.

## HW2: Convolutional Neural Networks

In this assignment we designed and worked with Convolutional Neural Networks (CNNs). The assignment is divided into 2 parts:

### 1. Alzheimers Classification

In this part, we implemented a CNN to classify brain MRI images as normal or Alzheimer's. We used the Alzheimer's dataset from Kaggle which contains brain MRI images of patients with Alzheimer's and normal patients. We preprocessed the images by resizing them and normalizing the pixel values to be between 0 and 1. We then trained a CNN on the preprocessed images and evaluated the performance of the CNN on the test set. Three different models were implemented and tested and the results were compared. This part was based on [this paper](https://www.researchgate.net/publication/349874169_A_CNN_based_framework_for_classification_of_Alzheimer's_disease).

### 2. Fine-Tuning VGG16 and ResNet-50 and Data Augmentation effect

In this part, we fine-tuned the VGG16 and ResNet-50 models on the cat-dog dataset from Kaggle. We used the pre-trained models and fine-tuned them on the cat-dog dataset by replacing the last layer of the models with a new layer that has 2 output neurons for the cat and dog classes. We then trained the models on the cat-dog dataset and evaluated the performance of the models on the test set. For both models, we trained them with and without data augmentation and compared the results. This part was based on [this paper](https://pdfs.semanticscholar.org/6086/30604cf7b62579930425ab57cc4191c034c9.pdf).

## HW3: Region Based Convolutional Neural Networks

In this assignment we designed and worked with Region Based Convolutional Neural Networks (R-CNNs). The assignment is divided into 2 parts:

### 1. Semantic Segmentation using U-Net

In this part, we implemented a U-Net model for semantic segmentation of the [this dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data) which contains brain MRI images and their corresponding masks. We preprocessed the images and used `Albumentations` library for data augmentation. We then trained the U-Net model on the preprocessed images and evaluated the performance of the model on the test set. We also visualized the predictions of the model on the test set. This part was based on [this paper](https://arxiv.org/pdf/2210.13336.pdf).

### 2. Underwater Object Detection using Faster R-CNN

In this part, we implemented a Faster R-CNN model for object detection in underwater images. We used the [this dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots/data) which contains underwater images and their corresponding annotations. First we analyzed the dataset and visualized the images and annotations and their distribution and used the extracted information for RPN and ROI generation. Then we preprocessed the images and used `Albumentations` library for data augmentation, we also used OHEM (Online Hard Example Mining) to improve the performance of the model and used mosaic augmentation to improve the performance of the model. We then trained the Faster R-CNN model on the preprocessed images and evaluated the performance of the model on the test set. We also visualized the predictions of the model on the test set. This part was based on [this paper](https://arxiv.org/abs/1506.01497), and we also add some modifications to it (like using GIoU loss instead of the original loss function) and OHEM to improve the performance of the model. Also the backbone of the model was ResNet-101 unlike the original paper which used VGG-16.
