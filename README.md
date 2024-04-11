# Neural-Networks-and-Deep-Learning-Course-Projects

## Outline

- [Neural-Networks-and-Deep-Learning-Course-Projects](#neural-networks-and-deep-learning-course-projects)
  - [Outline](#outline)
  - [HW1: Fully Connected Neural Networks](#hw1-fully-connected-neural-networks)
    - [1. McCulloch-Pitts Network](#1-mcculloch-pitts-network)
    - [2. Adversarial Attacks](#2-adversarial-attacks)
    - [3. Adeline and Madeline](#3-adeline-and-madeline)
    - [4. Optimised Neural Network](#4-optimised-neural-network)

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
