# Handwritten Digit Recognition with MLP (MNIST)

This project implements a simple **Multilayer Perceptron (MLP)** for handwritten digit classification using the **MNIST** dataset. The goal is to demonstrate foundational neural network concepts including **forward propagation**, **backpropagation**, and **SGD optimization**, with experiments on various network configurations and hyperparameters.

> 🔬 This project was developed as part of a university course assignment for hands-on practice with neural networks.

## 🔧 Features

- Built from scratch using **NumPy** (no external ML frameworks).
- Implements:
  - Custom forward and backward propagation
  - Mini-batch stochastic gradient descent
  - Tanh activation and squared loss
- Supports multiple configurations:
  - Variable hidden layer sizes
  - Learning rate comparison
  - Training on small vs. full dataset
- Evaluation metrics:
  - Prediction accuracy
  - Mean squared loss
  - Visualizations of misclassified and correctly classified digits

---

## 📁 Project Structure

├── main_code.py # Main training/testing script
├── mlp_functions.py # Core MLP functions (forward, backprop, predict, test)
├── utils.py # Data loading and preprocessing (not provided here)
└── MNIST_data/ # MNIST dataset files (download from official source)
* mlp_function_pytorch - is a similar version only using Pytorch -
* the course excercise requested us not to implement the assaigment without Pytorch


---

## 📊 Example Experiments

You can control which experiment to run using the `q` variable inside `main_code.py`:

- `q = 1` – Display best and worst predictions (by loss)
- `q = 2` – Compare learning rates: 0.01, 0.1, and 1
- `q = 3` – Train on 100 balanced samples (10 per class) and track test performance
- `q = 4` – Train on full dataset with same network and compare to small-sample results

Training results include accuracy, loss curves, and optional visualizations.

---

📜 License
This project is for academic demonstration purposes. No commercial use without permission.


## 🧪 Getting Started

### 1. Setup

Ensure you have Python 3.6+ with `numpy` and `matplotlib` installed:

```bash
pip install numpy matplotlib
2. Download MNIST Data
Place the following files inside the MNIST_data/ directory:

train-images.idx3-ubyte

train-labels.idx1-ubyte

t10k-images.idx3-ubyte

t10k-labels.idx1-ubyte

Edit main_code.py to choose the experiment you want (q = 1, q = 2, etc.), then:
python main_code.py

