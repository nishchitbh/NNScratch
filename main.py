import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

# This code implements a simple neural network for classifying handwritten digits from the MNIST dataset. It includes functions for data loading, preprocessing, forward propagation, backward propagation, and training. 
def data(location):
    '''Loads data from a CSV file.
    Args:
        location: Path to the CSV file.
    Returns:
        tuple: A tuple containing the training data and labels.
    '''
    df = pd.read_csv(location)
    label = df["label"].to_numpy()
    data_ = df.iloc[:, 1:]
    data_ = data_.to_numpy()
    return data_.T, label


def one_hot_Y(Y):
    '''Converts labels to one-hot encoding.
    Args:
        Y: Array of labels.
    Returns:
        ndarray: One-hot encoded labels.
    '''
    one_hot_array = np.eye(10)[Y]
    return one_hot_array.T


def Z(w, x, b):
    '''Calculates the weighted sum of inputs and bias.
    Args:
        w: Weights matrix.
        x: Input vector.
        b: Bias vector.
    Returns:
        ndarray: Weighted sum of inputs and bias.
    '''
    dot = np.dot(w, x)
    return dot + b


def ReLU(Z):
    '''Applies the ReLU activation function.
    Args:
        Z: Weighted sum of inputs and bias.
    Returns:
        ndarray: Output after applying ReLU.
    '''
    return np.maximum(0, Z)


def Softmax(Z):
    '''Applies the Softmax activation function.
    Args:
        Z: Weighted sum of inputs and bias.
    Returns:
        ndarray: Output after applying Softmax.
    '''
    Z_shifted = Z - np.max(Z, axis=0)  # Shift values for numerical stability
    exp_Z = np.exp(Z_shifted)
    softmax_output = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return softmax_output


def ComputeLoss(a, Y):
    '''Calculates the cross-entropy loss.
    Args:
        a: Predicted probabilities.
        Y: True labels.
    Returns:
        ndarray: Cross-entropy loss.
    '''
    values = a[Y, np.arange(a.shape[1])]
    negative_log = -np.log(values)
    return negative_log


def deriv_ReLU(z):
    '''Calculates the derivative of the ReLU activation function.
    Args:
        z: Input to ReLU.
    Returns:
        ndarray: Derivative of ReLU.
    '''
    return z > 0


def ForwardProp(w1, b1, w2, b2, x):
    '''Performs forward propagation through the neural network.
    Args:
        w1: Weights of the first layer.
        b1: Bias of the first layer.
        w2: Weights of the second layer.
        b2: Bias of the second layer.
        x: Input vector.
    Returns:
        tuple: A tuple containing the outputs of each layer (z1, a1, a2).
    '''
    z1 = Z(w1, x, b1)
    a1 = ReLU(z1)
    z2 = Z(w2, a1, b2)
    a2 = Softmax(z2)

    return z1, a1, a2


def BackwardProp(X, w2, z1, a1, a2, one_hot, m):
    '''Performs backward propagation to calculate gradients.
    Args:
        X: Input data.
        w2: Weights of the second layer.
        z1: Output of the first layer before activation.
        a1: Output of the first layer after activation.
        a2: Output of the second layer after activation.
        one_hot: One-hot encoded labels.
        m: Number of training examples.
    Returns:
        tuple: Gradients of the weights and biases (dJ_dw1, dJ_db1, dJ_dw2, dJ_db2).
    '''
    dL_dz2 = a2 - one_hot
    dJ_dw2 = np.dot(dL_dz2, a1.T) / m
    dJ_db2 = np.sum(dL_dz2, axis=1).reshape(10, 1) / m
    dL_dz1 = np.dot(w2.T, dL_dz2) * deriv_ReLU(z1)
    dJ_dw1 = np.dot(dL_dz1, X.T) / m
    dJ_db1 = np.sum(dL_dz1, axis=1).reshape(15, 1) / m
    return dJ_dw1, dJ_db1, dJ_dw2, dJ_db2


def neuralnet(X, Y, iterations, alpha):
    '''Trains a two-layer neural network.
    Args:
        X: Input data.
        Y: True labels.
        iterations: Number of training iterations.
        alpha: Learning rate.
    Returns:
        tuple: Weights and biases of the trained network, and a list of training costs.
    '''
    m = Y.shape[0]
    w1 = np.random.randn(15, X.shape[0]) * 0.01
    w2 = np.random.randn(10, w1.shape[0]) * 0.01
    one_hot = one_hot_Y(Y)
    b1 = np.zeros((15, 1))
    b2 = np.zeros((10, 1))
    costList = []
    for i in range(iterations):
        z1, a1, a2 = ForwardProp(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = BackwardProp(X, w2, z1, a1, a2, one_hot, m)
        w1 -= alpha * dw1
        b1 -= alpha * db1
        w2 -= alpha * dw2
        b2 -= alpha * db2
        cost = np.sum(ComputeLoss(a2, Y)) / m
        costList.append(cost)
    return w1, b1, w2, b2, costList


def train(plot: bool):
    '''Loads training data and trains the neural network.
    Args:
        plot: Flag to indicate whether to plot the training cost.
    '''
    trainX, trainY = data('./dataset/train.csv')
    w1, b1, w2, b2, costList = neuralnet(trainX, trainY, 100, 0.001)
    np.savez('weights.npz', w1=w1, b1=b1, w2=w2, b2=b2)
    if plot:
        plot_x = list(range(len(costList)))
        plt.plot(plot_x, costList, marker='o')
        plt.title('Cost over iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()


def test():
    '''Loads a test image and performs inference using the trained model.
    '''
    df = pd.read_csv("./dataset/test.csv")
    data_ = df.to_numpy()[random.randint(0, len(df))]
    data_ = data_.reshape(data_.shape[0], 1)
    to_plot = data_.reshape((28, 28))
    plt.imshow(to_plot)
    plt.show()
    weights = np.load("weights.npz")
    print(weights['w1'].shape)
    z1, a1, a2 = ForwardProp(
        weights['w1'], weights['b1'], weights['w2'], weights['b2'], data_)
    print(a2)
    print("Test result:", np.argmax(a2))


def main():
    '''Main function to execute training or testing based on the availability of weights.
    '''
    if not 'weights.npz' in os.listdir():
        train(plot=True)
    test()


if __name__ == "__main__":
    main()
