# Machine Learning Model Development

This project demonstrates the implementation and application of foundational machine learning algorithms using Python, NumPy, Pandas, Matplotlib, and PyTorch. It covers supervised learning (Classification), unsupervised learning (Clustering), and dimensionality reduction (PCA).

## Table of Contents
- [Logistic Regression](#logistic-regression)
- [K-Means Clustering](#k-means-clustering)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Deep Learning and KNN](#deep-learning-and-knn)
- [Requirements](#requirements)

---

## Logistic Regression
This section focuses on building and evaluating logistic regression models from scratch using NumPy.

### 1. Simple Logistic Regression
- **Dataset:** `ex2data1.txt` - Historical data of university applicants, containing scores from two exams and the admission decision (binary: 0 or 1).
- **Goal:** Predict university admission based on exam scores.
- **Key Implementations:** Sigmoid function, Cost function, and Stochastic Gradient Descent (SGD).
- **Evaluation:** Decision boundary visualization, Accuracy, and ROC/AUC curves.

### 2. Regularized Logistic Regression
- **Dataset:** `ex2data2.txt` - Quality assurance test results for microchips from a fabrication plant, featuring two different test scores and a status (pass/fail). The data is non-linearly separable.
- **Goal:** Predict microchip quality using a non-linear decision boundary.
- **Key Implementations:** Feature mapping to 6th-degree polynomial terms, Regularized cost function, and Gradient Descent with Momentum.

---

## K-Means Clustering
This notebook focuses on unsupervised clustering using the K-means algorithm implemented from scratch.

- **Datasets:** 
  - **Convex Dataset:** Synthetic 2D data generated with four distinct Gaussian "blobs" to test standard clustering.
  - **Non-Convex Dataset:** Synthetic "Moons" dataset (using `sklearn.datasets.make_moons`) to demonstrate the limitations of standard K-Means on non-spherical clusters.
- **Goal:** Explore how K-means identifies natural groupings in data.
- **Key Implementations:** Manual NumPy implementation of centroid initialization, assignment, and update steps.
- **Evaluation:** Analysis of clustering cost (inertia) using the Elbow method.

---

## Principal Component Analysis (PCA)
This section covers dimensionality reduction and data reconstruction techniques.

- **Dataset:** High-dimensional image data (often using the MNIST or similar grayscale digit representations).
- **Goal:** Reduce the dimensionality of the data while preserving as much variance as possible.
- **Key Implementations:** Computation of the covariance matrix and eigenvalue decomposition to find principal components.
- **Analysis:** Performance evaluation by reconstructing images from reduced subspaces (e.g., 3, 10, and 100 dimensions).

---

## Deep Learning and KNN
This section explores advanced classification algorithms applied to complex image datasets.

### 1. Feedforward & Convolutional Neural Networks (PyTorch)
- **Datasets:** 
  - **MNIST:** 70,000 grayscale images of handwritten digits (0-9), size-normalized to 28x28 pixels.
  - **Fashion-MNIST:** 70,000 grayscale images of Zalando's article images (e.g., T-shirts, trousers, sneakers) across 10 classes.
- **Goal:** Classify complex image data using deep learning architectures.
- **Key Implementations:** 
  - **FFN:** Multi-layer perceptron with ReLU activations and Log-Softmax output.
  - **CNN:** Convolutional layers with 5x5 kernels, Max Pooling, and fully connected layers for spatial feature extraction.
- **Training:** Negative Log-Likelihood Loss and SGD optimizer.

### 2. K-Nearest Neighbors (KNN)
- **Dataset:** Fashion-MNIST images.
- **Goal:** Apply the KNN algorithm for supervised classification.
- **Implementation:** Evaluation of model accuracy and the impact of the 'K' hyperparameter.

---

## Requirements
- Python 3.7+
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- PyTorch
- Torchvision
- Scikit-learn (for dataset generation)

You can install the dependencies using pip:
```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn
```
