# Neural Network from Scratch with NumPy

A simple feedforward neural network (multilayer perceptron) implemented using **only NumPy** — no PyTorch, no TensorFlow, no ML frameworks. Demonstrates forward propagation, backpropagation, and gradient descent from first principles.

---

## Features

- Fully connected feedforward architecture (configurable layers)
- ReLU activations for hidden layers
- Sigmoid activation for the output layer
- Binary cross-entropy loss
- He initialization for weights
- Full-batch gradient descent
- Trains and solves the XOR problem out of the box

---

## Requirements

- Python 3.7+
- NumPy

Install NumPy if you don't have it:

```bash
pip install numpy
```

---

## Usage

### Run the built-in XOR demo

```bash
python neural_network.py
```

Expected output:

```
Epoch     0 | loss = 0.7193
Epoch   500 | loss = 0.6190
...
Epoch  4500 | loss = 0.0142

Predictions:
[0. 0.] -> predicted 0, actual 0
[0. 1.] -> predicted 1, actual 1
[1. 0.] -> predicted 1, actual 1
[1. 1.] -> predicted 0, actual 0
```

### Use in your own code

```python
import numpy as np
from neural_network import NeuralNetwork

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Define architecture: [input_size, hidden_size, ..., output_size]
nn = NeuralNetwork(layer_sizes=[2, 4, 1])

# Train
losses = nn.train(X, y, epochs=5000, learning_rate=0.5)

# Predict
predictions = nn.predict(X)
```

---

## API Reference

### `NeuralNetwork(layer_sizes, seed=42)`

Creates a new neural network.

| Parameter | Type | Description |
|---|---|---|
| `layer_sizes` | `list[int]` | Number of neurons per layer, e.g. `[2, 4, 1]` |
| `seed` | `int` | Random seed for reproducibility (default: `42`) |

---

### `train(X, y, epochs=2000, learning_rate=0.1, verbose=True)`

Trains the network using full-batch gradient descent.

| Parameter | Type | Description |
|---|---|---|
| `X` | `ndarray` | Input features, shape `(n_samples, n_features)` |
| `y` | `ndarray` | Binary labels, shape `(n_samples, 1)` |
| `epochs` | `int` | Number of training iterations |
| `learning_rate` | `float` | Step size for gradient descent |
| `verbose` | `bool` | Print loss every 10% of epochs |

Returns a list of loss values recorded at each epoch.

---

### `predict(X)`

Returns binary predictions (0 or 1) for input `X`.

---

### `forward(X)`

Runs a forward pass and returns the raw output activations.

---

## Architecture & Implementation Details

```
Input Layer  →  Hidden Layer(s) [ReLU]  →  Output Layer [Sigmoid]
```

- **Weight initialization:** He initialization (`sqrt(2 / fan_in)`) — well-suited for ReLU networks
- **Loss function:** Binary cross-entropy
- **Output gradient:** Analytically combined BCE + sigmoid derivative → `output - y`
- **Backpropagation:** Reverse-mode autodiff implemented manually using stored activations and pre-activation values (`z`)

---

## Project Structure

```
.
├── neural_network.py   # Neural network implementation + XOR demo
└── README.md
```

---

## License

MIT License. Free to use, modify, and distribute.
