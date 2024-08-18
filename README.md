# Simple Handwritten Digit Classifier

This project implements a simple two-layer neural network for classifying handwritten digits from the MNIST dataset. It includes functionalities for data loading, preprocessing, training, and testing.

## Features

* **Data Loading and Preprocessing:** Loads data from CSV files and converts labels to one-hot encoding.
* **Forward Propagation:** Implements forward propagation through the network using ReLU activation for the hidden layer and Softmax for the output layer.
* **Backward Propagation:** Calculates gradients using backpropagation to update weights and biases.
* **Training:** Trains the network using gradient descent with a specified learning rate and number of iterations.
* **Testing:** Loads a random test image and performs inference using the trained model.
* **Visualization:** Plots the training cost over iterations.

## Requirements

* **Python:** Ensure you have Python installed.
* **Libraries:** Install the necessary libraries using the following command:

```bash
pip install numpy matplotlib pandas
```

## Dataset

The project expects the MNIST dataset to be available in the following format:

* **`train.csv`:** Contains the training data with labels.
* **`test.csv`:** Contains the test data.

You can download the MNIST dataset from various sources, including the official website or Kaggle.

## Usage

1. **Training:** Run the script to train the neural network. The trained weights will be saved in a file named `weights.npz`.
   ```bash
   python main.py
   ```

2. **Testing:** After training, the script will automatically load a random test image and perform inference using the saved weights.

## Functionality Breakdown

* **`data(location)`:** Loads data from a CSV file and returns the data and labels.
* **`one_hot_Y(Y)`:** Converts labels to one-hot encoding.
* **`Z(w, x, b)`:** Calculates the weighted sum of inputs and bias.
* **`ReLU(Z)`:** Applies the ReLU activation function.
* **`Softmax(Z)`:** Applies the Softmax activation function.
* **`ComputeLoss(a, Y)`:** Calculates the cross-entropy loss.
* **`deriv_ReLU(z)`:** Calculates the derivative of the ReLU activation function.
* **`ForwardProp(w1, b1, w2, b2, x)`:** Performs forward propagation through the network.
* **`BackwardProp(X, w2, z1, a1, a2, one_hot, m)`:** Performs backward propagation to calculate gradients.
* **`neuralnet(X, Y, iterations, alpha)`:** Trains the neural network.
* **`train(plot)`:** Loads training data and trains the network.
* **`test()`:** Loads a test image and performs inference.
* **`main()`:** Main function to execute training or testing.

## Notes

* The network architecture, hyperparameters (learning rate, iterations), and activation functions can be modified for experimentation.
* The code provides a basic implementation and can be further improved with techniques like regularization, optimization algorithms, and more complex network architectures.

## License

This project is licensed under the MIT License.
