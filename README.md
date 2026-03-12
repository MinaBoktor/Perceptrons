# Perceptrons

A specialized visual laboratory for comparing the architectural behaviors of **Single-Layer Perceptrons (SLP)** and **Adaline (Adaptive Linear Neuron)**. This project implements these fundamental neural models from scratch to demonstrate how weight optimization strategies influence the convergence and placement of linear decision boundaries.

---

## 🚀 Overview

This framework provides an interactive environment to study binary classification. Unlike high-level libraries that hide the training process, this tool exposes the internal mechanics of weight updates, bias integration, and error minimization through a real-time Streamlit dashboard.

### Key Technical Features

- **Manual Implementations:** Core algorithms (SLP and Adaline) developed using NumPy and pure Python logic.
- **Interactive Hyperparameter Tuning:** Adjust learning rates, epoch counts, and MSE thresholds on the fly.
- **Dynamic Decision Boundaries:** Real-time visualization of how the model separates data points in a 2D feature space.
- **Performance Analytics:** Automated generation of Confusion Matrices and Learning Curves (MSE/Misclassifications per epoch).

---

## 🧠 Algorithms Implemented

### 1. Single-Layer Perceptron (SLP)

An error-driven learning model that updates weights only when a misclassification occurs. It utilizes a **Heaviside step function (signum)** for binary output. It is highly effective for perfectly linearly separable datasets.

### 2. Adaline (Adaptive Linear Neuron)

An improvement over the basic Perceptron that uses a **linear activation function**. It employs **Gradient Descent** to minimize the Mean Squared Error (MSE) across all training samples, typically resulting in a more stable and optimally centered decision boundary compared to the standard SLP.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `app.py` | The main Streamlit dashboard and visualization engine. |
| `perceptrons.py` | The backend engine containing the custom SLP and Adaline classes/functions. |
| `preprocessing.py` | Data pipeline for handling missing values, label encoding, and feature scaling (StandardScaler). |
| `requirements.txt` | List of necessary Python dependencies. |
| `data.csv` | The dataset used for training and validation. |

---

## 🛠️ Installation & Usage

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/Perceptrons.git
cd Perceptrons
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the Dashboard:**

```bash
streamlit run app.py
```

---

## 📊 Results & Analysis

The framework demonstrates that while both models succeed on well-separated data, **Adaline consistently outperforms the Perceptron on overlapping feature sets**. By minimizing cost rather than just reacting to errors, Adaline finds a boundary that maximizes the "safety margin" between clusters, proving more robust against noise in non-linearly separable environments.
