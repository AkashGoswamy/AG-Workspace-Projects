"""
Neural Network from Scratch with NumPy
========================================
A simple feedforward neural network (multilayer perceptron) implemented
using only NumPy, demonstrating forward propagation, backpropagation,
and gradient descent training.
"""

import numpy as np


class NeuralNetwork:
    """A feedforward neural network with ReLU hidden layers and a
    sigmoid output layer, trained with full-batch gradient descent."""

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # He initialization works well with ReLU activations
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self.z_values.append(z)
            a = self.sigmoid(z) if i == len(self.weights) - 1 else self.relu(z)
            self.activations.append(a)
        return a

    def backward(self, y, output, learning_rate):
        m = y.shape[0]
        deltas = [None] * len(self.weights)

        # Combined derivative of binary cross-entropy + sigmoid output
        deltas[-1] = output - y

        for i in range(len(self.weights) - 2, -1, -1):
            deltas[i] = (deltas[i + 1] @ self.weights[i + 1].T) * self.relu_derivative(self.z_values[i])

        for i in range(len(self.weights)):
            grad_w = self.activations[i].T @ deltas[i] / m
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b

    def train(self, X, y, epochs=2000, learning_rate=0.1, verbose=True):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-9) + (1 - y) * np.log(1 - output + 1e-9))
            losses.append(loss)
            self.backward(y, output, learning_rate)
            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch:5d} | loss = {loss:.4f}")
        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


if __name__ == "__main__":
    # XOR problem: classic test for non-linear decision boundaries
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # Architecture: 2 inputs -> 4 hidden neurons -> 1 output
    nn = NeuralNetwork(layer_sizes=[2, 4, 1])
    nn.train(X, y, epochs=5000, learning_rate=0.5)

    print("\nPredictions:")
    preds = nn.predict(X)
    for inputs, pred, target in zip(X, preds, y):
        print(f"{inputs} -> predicted {pred[0]}, actual {int(target[0])}")
