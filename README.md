# Handwritten Digit Recognition with a Simple Neural Network

This project implements a simple two-layer neural network for recognizing handwritten digits from the MNIST dataset. The network is trained using backpropagation and gradient descent.

## Table of Contents

- [Handwritten Digit Recognition with a Simple Neural Network](#handwritten-digit-recognition-with-a-simple-neural-network)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Theory](#theory)
    - [Neural Networks](#neural-networks)
    - [Forward Propagation](#forward-propagation)
    - [Backpropagation](#backpropagation)
    - [Gradient Descent](#gradient-descent)
    - [Activation Functions](#activation-functions)
      - [ReLU (Rectified Linear Unit)](#relu-rectified-linear-unit)
      - [Softmax](#softmax)
    - [Loss Function](#loss-function)
  - [Implementation Details](#implementation-details)
    - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    - [Network Architecture](#network-architecture)
    - [Training](#training)
    - [Testing](#testing)
  - [How to Run the Code](#how-to-run-the-code)
  - [License](#license)

## Introduction

The MNIST dataset is a collection of 70,000 handwritten digits (0-9) that is commonly used for training and testing image classification models. This project uses a simple neural network to classify these digits.

## Theory

### Neural Networks

Neural networks are a class of machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes (neurons) organized in layers. Each connection between neurons has a weight associated with it, which determines the strength of the connection.

### Forward Propagation

Forward propagation is the process of passing input data through the network to obtain the output. The input data is multiplied by the weights of each connection and summed with a bias term. This weighted sum is then passed through an activation function to produce the output of the neuron. This process is repeated for each layer until the final output is obtained.

### Backpropagation

Backpropagation is the process of calculating the gradients of the loss function with respect to the weights and biases of the network. These gradients are then used to update the weights and biases during training.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the loss function. It works by iteratively updating the weights and biases in the direction of the negative gradient.

### Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.

#### ReLU (Rectified Linear Unit)

ReLU is a simple activation function that returns the input if it is positive, and 0 otherwise. It is defined as:

```
ReLU(z) = max(0, z)
```

#### Softmax

Softmax is an activation function used in the output layer for multi-class classification problems. It converts the output scores into probabilities that sum to 1. It is defined as:

```
Softmax(z_i) = exp(z_i) / sum(exp(z_j) for j in all_classes)
```

### Loss Function

The loss function measures the difference between the predicted output and the true output. In this project, the cross-entropy loss is used, which is defined as:

```
Loss = -sum(y_i * log(a_i) for i in all_classes)
```

where:
- `y_i` is the true label (one-hot encoded)
- `a_i` is the predicted probability

## Implementation Details

### Data Loading and Preprocessing

- The `data` function loads the MNIST dataset from CSV files.
- The `one_hot_Y` function converts the labels to one-hot encoding.

### Network Architecture

- The network has two layers:
  - A hidden layer with 15 neurons and ReLU activation.
  - An output layer with 10 neurons and Softmax activation.

### Training

- The `neuralnet` function trains the network using backpropagation and gradient descent.
- The `train` function loads the training data and calls the `neuralnet` function to train the network.

### Testing

- The `test` function loads a random test image and performs inference using the trained model.

## How to Run the Code

1. **Dependencies:** Install the required libraries: `numpy`, `matplotlib`, and `pandas`.
2. **Dataset:** Download the MNIST dataset and place the `train.csv` and `test.csv` files in the `dataset` folder.
3. **Training:** Run the `main.py` script to train the network. The trained weights will be saved in a file named `weights.npz`.
4. **Testing:** After training, run the `main.py` script again to test the network on a random test image.

## License

This project is licensed under the MIT License.
