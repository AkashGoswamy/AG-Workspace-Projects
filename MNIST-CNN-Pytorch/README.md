# MNIST Digit Classifier

A convolutional neural network trained on the MNIST dataset to classify handwritten digits (0–9). Built with PyTorch. Reaches **99.49% test accuracy** over 15 epochs on CPU.

---

## What it does

The notebook walks through the full pipeline — data loading with augmentation, a two-block CNN, training with cosine LR annealing, evaluation with a confusion matrix, and a live interactive canvas where you can draw a digit and get an instant prediction.

---

## Model

Two convolutional blocks followed by a fully connected classifier.

```
Input (1 × 28 × 28)
  └─ Conv2d(1→32) + BatchNorm + ReLU + MaxPool  →  (32 × 14 × 14)
  └─ Conv2d(32→64) + BatchNorm + ReLU + MaxPool →  (64 × 7 × 7)
  └─ Flatten                                    →  (3136,)
  └─ Linear(3136→128) + ReLU + Dropout(0.5)    →  (128,)
  └─ Linear(128→10)                             →  logits
```

**Parameters:** ~420K total

---

## Results

| Epochs | Final Test Accuracy |
|--------|-------------------|
| 15     | **99.49%**        |

Training progression (CPU, ~13m 30s):

| Epoch | Train Loss | Test Acc |
|-------|-----------|----------|
| 1     | 0.4965    | 98.48%   |
| 5     | 0.1607    | 99.16%   |
| 10    | 0.1072    | 99.38%   |
| 15    | 0.0825    | **99.49%** |

Per-class accuracy is strong across all digits; the hardest classes are 5, 6, and 7 — consistent with MNIST literature.

---

## Stack

- Python 3.11
- PyTorch 2.11 (CPU)
- Torchvision 0.26
- scikit-learn (confusion matrix)
- ipywidgets + ipycanvas (live drawing widget)

---

## Running it

1. Clone the repo and install dependencies:
   ```bash
   pip install torch torchvision scikit-learn matplotlib ipywidgets ipycanvas pillow
   ```

2. Open `MNIST.ipynb` in VS Code or JupyterLab and run all cells in order.

3. MNIST data is downloaded automatically on first run into a `./data/` folder.

4. After training, the weights are saved as `mnist_cnn.pth` in the project root.

5. The last cell loads the canvas widget — draw a digit, hit **Predict**.

> `num_workers=0` is set in the DataLoaders for Windows compatibility. Linux/macOS users can increase this for faster loading.

---

## Structure

```
├── MNIST.ipynb       # full notebook
├── mnist_cnn.pth     # saved weights (generated after training)
└── data/             # MNIST raw files (auto-downloaded)
```
