<p align="center">
  <img src="assets/scikit-jax-logo-1-transparent.jpg" alt="Alt text" width="300"/>
</p>

# Scikit-JAX: Classical Machine Learning on the GPU

Welcome to **Scikit-JAX**, a machine learning library designed to leverage the power of GPUs through JAX for efficient and scalable classical machine learning algorithms. Our library provides implementations for a variety of classical machine learning techniques, optimized for performance and ease of use.

## Features

- **Linear Regression**: Implemented with options for different weight initialization methods and dropout regularization.
- **KMeans**: Clustering algorithm to group data points into clusters.
- **Principal Component Analysis (PCA)**: Dimensionality reduction technique to simplify data while preserving essential features.
- **Multinomial Naive Bayes**: Classifier suitable for discrete data, such as text classification tasks.
- **Gaussian Naive Bayes**: Classifier for continuous data with a normal distribution assumption.

## Installation

To install Scikit-JAX, you can use pip. The package is available on PyPI:

```terminal
pip install scikit-jax==0.0.3dev1
```

## Usage

Here is a quick guide on how to use the key components of Scikit-JAX.

### Linear Regression
```py
from skjax.linear_model import LinearRegression

# Initialize the model
model = LinearRegression(weights_init='xavier', epochs=100, learning_rate=0.01)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Plot losses
model.plot_losses()
```

### K-Means
```python
from skjax.clustering import KMeans

# Initialize the model
kmeans = KMeans(num_clusters=3)

# Fit the model
kmeans.fit(X_train)
```

### Gaussian Naive Bayes
```python
from skjax.naive_bayes import GaussianNaiveBayes

# Initialize the model
nb = GaussianNaiveBayes()

# Fit the model
nb.fit(X_train, y_train)

# Make predictions
predictions = nb.predict(X_test)
```

### License

Scikit-JAX is licensed under the [MIT License](LICENSE.txt).
