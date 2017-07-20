# Convolutional Neural Network on Cucumber 9

#### Brief Description TL;DR

The goal of this repository is to gain knowledge in implementing Convolutional Neural Network on the Cucumber dataset.

---
#### Description

In advancing knowledge about convolutional neural networks, we must first understand the basic parts of a CNN. In this project we will create 3 different models and see how they classify the images, and learn which produces higher accuracy in classifying cucumbers

---
### Models Used
1. Perceptron

- The perceptron is the simplest example of a neural network. Representing a single neuron, the perceptron takes in all 1024 pixels of the first channel of the cucumber dataset with their corresponding weights and biases. Then computes its logits value using the function below with softmax as its activation function. 

2. 3 Layer Feed Forward Neural Network

- The next model created is a 3 Layer Feed Forward Neural Network, This model is composed of an input layer, 3 Hidden Layers, and an Output Layer (naming a neural net is still quite ambiguous, we can either name it using the number of input layers or the total number of layers ). For this model the it takes in all 1024 pixels of the r channel and outputs a 9 classification that are one hot encoded. For the middle layers, we used ReLU as our activation function and softmax for our output layer.

3. Convolutional Neural Network

- The final model created is a Convolutional Neural Network, the model is built with 2 convolutional layers and 2 fully connected layer to finish of the classification function. For the convolutional layer tensorflow pads the output of the convolution to make it the same width and height of the starting image while making it possible to increase the depth of each layer to make more convolutions. For each middle layer a ReLU activation function is used and a softmax for the output layer.

---
### About the Dataset
The dataset used is the Cucumber9 Image Dataset. The dataset is patterned just like the CIFAR10 dataset. You can learn more about the dataset <a href="">here</a>
