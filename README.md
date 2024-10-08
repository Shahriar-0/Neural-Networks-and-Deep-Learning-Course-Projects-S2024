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
  - [HW4: Recurrent Neural Networks](#hw4-recurrent-neural-networks)
    - [1. Semantic Segmentation using a hybrid CNN-LSTM model](#1-semantic-segmentation-using-a-hybrid-cnn-lstm-model)
    - [2. Remaining Useful Life Prediction using a hybrid CNN-LSTM model](#2-remaining-useful-life-prediction-using-a-hybrid-cnn-lstm-model)
  - [HW5: Transformers](#hw5-transformers)
    - [1. Fake News Detection](#1-fake-news-detection)
    - [2. Image Classification](#2-image-classification)
  - [HW6: Deep Generative Models](#hw6-deep-generative-models)
    - [1. Variational Auto-Encoder](#1-variational-auto-encoder)
    - [2. Image Translation](#2-image-translation)
  - [HW Extra](#hw-extra)
    - [1. Labelling Using Clustering](#1-labelling-using-clustering)
    - [2. Data Augmentation in FaBERT](#2-data-augmentation-in-fabert)
    - [3. Wake Word Detection](#3-wake-word-detection)
    - [4. Image Segmentation](#4-image-segmentation)

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

## HW4: Recurrent Neural Networks

In this assignment we designed and worked with Recurrent Neural Networks (RNNs). The assignment is divided into 2 parts:

### 1. Semantic Segmentation using a hybrid CNN-LSTM model

For this part we used different RNN models to using [this dataset](https://www.kaggle.com/datasets/behdadkarimi/persian-tweets-emotional-dataset) to do a semantic analysis. We used normal preprocessing method at first, then we used `ParsBERT` for tokenizing, and after that we first used a CNN and LSTM model to do the analysis then we used a proposed hybrid CNN-LSTM model based on [this paper](https://arxiv.org/ftp/arxiv/papers/2307/2307.07740.pdf) to improve the results.

### 2. Remaining Useful Life Prediction using a hybrid CNN-LSTM model

In this part, we designed and worked with Recurrent Neural Networks (RNNs). The goal of this challenge is to predict the remaining useful life of a machine based on the data collected from it. The data is collected from a machine in a factory. We will use multiple DL models such as CNN, LSTM and a hybrid model to predict the remaining useful life of the machine. This part is based on [this paper](https://www.researchgate.net/publication/358360497_A_hybrid_deep_learning_framework_for_intelligent_predictive_maintenance_of_Cyber-Physical_Systems) and the data can be found [here](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data), which is NASA's dataset for a jet engine.

## HW5: Transformers

In this part, we designed and worked with Transformers. The assignment is divided into 2 parts:

### 1. Fake News Detection

In this part, we wanted to check how much of the news related to COVID-19 are real. For this purpose, we use two methods of transform learning, namely fine-tuning and feature-based, to design a model that can detect fake news, for more information about these you check check [this paper](https://www.sciencedirect.com/science/article/pii/S0950705123003921). We used these two methods on two models of BERT and CT-BERT based on [this paper](https://www.sciencedirect.com/science/article/pii/S0950705123003921). Using these two methods we can detect fake news from real news, with a high accuracy. The data for this part was taken from youtube and twitter and for that it took a lot of preprocessing.

### 2. Image Classification

For this part, we wanted to classify the images of the `CIFAR-10` dataset. This part is inspired by the famous [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) which describes the ViT model. Since training a transformer from scratch is a tedious and time consuming task, we wanted to use another method called transfer learning, for doing that we followed the instructions of [this paper](https://arxiv.org/ftp/arxiv/papers/2110/2110.05270.pdf) and trained different transformers and CNN networks and fine-tuned them on `CIFAR-10`, using the pre-trained `imagenet1k` weights.

## HW6: Deep Generative Models

This part is designed to work with Deep Generative Models. It consists of two parts:

### 1. Variational Auto-Encoder

This part focuses on working with different kind of VAEs (Variational Auto-Encoders) such as, VAE, CVAE, VQ-VAE, etc. Datasets used in this part are [Anime Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) and [Cartoon Face Dataset](https://www.kaggle.com/datasets/brendanartley/cartoon-faces-googles-cartoon-set). The VQ-VAE model and VQ-VAE 2 models are respectively based on [this paper](https://arxiv.org/abs/1711.00937), and [this paper](https://arxiv.org/abs/1906.00446).

### 2. Image Translation

Image Translation is the process of creating an image from another image. In this assignment we used Pix2Pix model which is based on [this paper](https://arxiv.org/abs/1611.07004). Then we used the implemented model on [this dataset](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) and evaluated the performance of the model on the test set.

## HW Extra

This was an extra assignment, a part to access our general knowledge.

### 1. Labelling Using Clustering

In this part we wanted to carry out a semisupervised learning to label and cluster the two famous `MNIST` and `Fashion MNIST` datasets. At first we a fully convolutional encoder to get the features of the images, and then we used KMEANS clustering to label them. The encoder was trained as a part of an auto-encoder, designed to reconstruct the images.

### 2. Data Augmentation in FaBERT

In this part, we were assigned to learn about data augmentation methods in NLP. The dataset we used for this part was [DeepSentiPers](https://github.com/JoyeBright/DeepSentiPers). For data augmentation we used a back translation technique and doubled our data size, then we used bert model for doing a semantic analysis on this dataset.

### 3. Wake Word Detection

In this part, we wanted to detect if an audio contains a wake word or not. The data for this part was produced and gathered by ourselves and can be replicated using the methods in the source. After that, we used different methods of preprocessing and data augmentation to make the data more suitable for training, and then we used different models to detect the wake word.

### 4. Image Segmentation

For last part, we wanted to segment the images. The data for this part was [SUIM](https://irvlab.cs.umn.edu/resources/suim-dataset) dataset and we used U-Net and after that a more advanced Ta-U-Net model. The train dataset contains about 1500 RGB images and the test dataset contains about 100 RGB images. For better results we normalized the images to be between 0 and 1 and used data augmentation to make the data more suitable for training. Most of the implementation of the model was based on [this paper](https://www.mdpi.com/1424-8220/22/12/4438), although some changes were made to make the model more efficient.