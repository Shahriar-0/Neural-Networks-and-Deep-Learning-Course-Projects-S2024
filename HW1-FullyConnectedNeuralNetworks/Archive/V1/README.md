# Neural Network and Deep Learning Course - Fully Connected Neural Networks
## Project Description
This project contains 4 parts:
### 1. **McCulloch-Pitts Network**: 
In this part, We are going to implement a simple 2 layer network with 7 input neurons and 4 output neurons. This network is designed to recognize the digits 6, 7, 8 and 9. The input neurons are binary neurons, which means that they can only take values 0 or 1. The output neurons are also binary neurons and only the corresponding output neuron for the digit should be 1 and the rest should be 0. For example, if the input is 6, then the output should be [1, 0, 0, 0]. for other combinations (other digits and invalid combinations) the output should be [0, 0, 0, 0].
### 2. **Adaline and Madaline**: 
In this part, we are going to implement the Adaline and Madaline models. \
The Adaline model is a single layer network that takes continuous inputs and produces continuous outputs. The Madaline model is a multi-layer network that takes continuous inputs and produces binary outputs. \
We are going to use Adaline model to classify the Iris dataset. The Iris dataset contains 150 samples of iris flowers. Each sample contains 4 features: sepal length, sepal width, petal length, and petal width. The samples are labeled with 3 classes: setosa, versicolor, and virginica. \
We are going to use Madaline model to classify the moons dataset. The moons dataset contains 500 samples of 2D points. The samples are labeled with 2 classes. The classes are not linearly separable. We will train 3 different Madaline models with different number of hidden neurons (3, 5, and 8) and compare the results.
### 3. **DAC: Deep Autoencoder-based Clustering**:
In this part, we are going to implement the Deep Autoencoder-based Clustering (DAC) model based on the paper "DAC: Deep Autoencoder-based Clustering, a General Deep Learning Framework of Representation Learning" by Si Lu and Ruisi Li. \
The DAC model is a deep learning model that is used for clustering. The model is trained to learn a good representation of the input data and then the representation is used to cluster the data. \
We are going to use the DAC model to cluster the MNIST dataset. The MNIST dataset contains 60,000 samples of 28x28 images of handwritten digits. The samples are labeled with 10 classes (0-9). We are going to use the DAC model to cluster the samples into 10 clusters and compare the results with the true labels. We will use the Adjusted Rand Index (ARI) to measure the performance of the clustering.
### 4. **Multi-Layer Perceptron (MLP) and Knowledge Distillation**:
In this part, we are going to implement and train two different Multi-Layer Perceptron (MLP) models. The first model is a teacher model and the second model is a student model. We are going to use the teacher model to train the student model using the Knowledge Distillation technique based on the paper "Distilling the Knowledge in a Neural Network" by Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. \
The Knowledge Distillation technique is used to transfer the knowledge from a large model (teacher model) to a smaller model (student model) by training the student model to mimic the outputs of the teacher model. \
We are going to use the MNIST dataset to train the models. The MNIST dataset contains 60,000 samples of 28x28 images of handwritten digits. The samples are labeled with 10 classes (0-9). We are going to use the teacher model to train the student model to classify the samples into the 10 classes. We will compare the performance of the student model with and without the Knowledge Distillation technique. We will use the accuracy to measure the performance of the models.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.