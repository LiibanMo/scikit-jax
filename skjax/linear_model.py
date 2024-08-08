""" Model(s) """

from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ._utils._helper_functions import calculate_loss_gradients, compute_mse
from ._utils.config import (DROPOUT, LAMBDA_, LEARNING_RATE, EPOCHS_lr,
                            MAX_PATIENCE_lr, P, RANDOM_STATE_lr)

# ================================================================================================================ #


class LinearRegression:
    """
    Linear Regression model with options for various weight initialization methods and dropout regularization.

    Attributes:
        weights (str): Initialization method for weights ('zero', 'random', 'lecun', 'xavier', 'he').
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        p (int): Number of features in the input data.
        lambda_ (float): Regularization parameter for L2 regularization.
        max_patience (int): Number of epochs to wait for improvement before early stopping.
        dropout (float): Dropout rate to prevent overfitting.
        random_state (int): Seed for random number generation.
        losses_in_training_data (np.ndarray): Array to store training losses for each epoch.
        losses_in_validation_data (np.ndarray): Array to store validation losses for each epoch.
        stopped_at (int): Epoch at which training stopped, either due to completion or early stopping.
    """

    def __init__(
        self,
        weights_init: str = "zero",
        epochs: int = EPOCHS_lr,
        learning_rate: float = LEARNING_RATE,
        p: int = P,
        lambda_: float = LAMBDA_,
        max_patience: int = MAX_PATIENCE_lr,
        dropout: float = DROPOUT,
        random_state: int = RANDOM_STATE_lr,
    ):
        """
        Initialize the LinearRegressionModel.

        Args:
            weights_init (str): Method to initialize weights ('zero', 'random', 'lecun', 'xavier', 'he'). Default is 'zero'.
            epochs (int): Number of epochs for training. Default is 2000.
            learning_rate (float): Learning rate for optimization. Default is 0.0005.
            p (int): Number of features in the dataset. Default is 2.
            lambda_ (float): Regularization parameter for L2 regularization. Default is 0.
            max_patience (int): Maximum number of epochs to wait for improvement before early stopping. Default is 200.
            dropout (float): Dropout rate to prevent overfitting. Default is 0.
            random_state (int): Seed for random number generation. Default is 41.
        """
        self.weights = weights_init
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.p = p
        self.lambda_ = lambda_
        self.max_patience = max_patience
        self.dropout = dropout
        self.random_state = random_state
        self.losses_in_training_data = np.zeros(epochs)
        self.losses_in_validation_data = np.zeros(epochs)
        self.stopped_at = epochs  # Records epoch stopped at. First assumes training using every epoch.

    def fit(
        self,
        X_train: jax.Array,
        y_train: jax.Array,
        X_val: Optional[jax.Array] = None,
        y_val: Optional[jax.Array] = None,
    ):
        """
        Fit the model to the training data.

        Args:
            X_train (jax.Array): Training data features.
            y_train (jax.Array): Training data labels.
            X_val (jax.Array, optional): Validation data features. Default is None.
            y_val (jax.Array, optional): Validation data labels. Default is None.

        Returns:
            self: The instance of the LinearRegression object with fitted weights.
        """
        # Initializing parameters
        best_beta: Optional[jax.Array] = None

        n = len(X_train)

        best_mse = jnp.inf

        patience_counter = 0

        key = jax.random.key(self.random_state)

        # Defining the weights initializers
        weights_init_dict = {
            "zero": jnp.zeros(X_train.shape[1]),
            "random": jax.random.normal(key, shape=(X_train.shape[1],)),
            "lecun": jax.random.normal(key, shape=(X_train.shape[1],))
            * jnp.sqrt(1 / X_train.shape[0]),
            "xavier": jax.random.normal(key, shape=(X_train.shape[1],))
            * jnp.sqrt(2 / (X_train.shape[0] + y_train.shape[0])),
            "he": jax.random.normal(key, shape=(X_train.shape[1],))
            * jnp.sqrt(2 / X_train.shape[0]),
        }

        self.weights = weights_init_dict[self.weights]

        # Training Loop
        for epoch in range(self.epochs):
            # Dropout
            key, subkey = jax.random.split(key)
            dropout_mask = jax.random.bernoulli(
                subkey, p=(1 - self.dropout), shape=(n, 1)
            )
            X_train_dropout = X_train * dropout_mask

            # Calculating loss on training data
            mse_train = compute_mse(
                y_pred=jnp.dot(X_train_dropout, self.weights), y_true=y_train
            )
            self.losses_in_training_data[epoch] = mse_train

            # Calculate loss gradients
            loss_gradient_wrt_beta = calculate_loss_gradients(
                self.weights, X_train_dropout, y_train, self.p, self.lambda_
            )

            # Optimiser step
            self.weights -= self.learning_rate * loss_gradient_wrt_beta

            if X_val is not None and y_val is not None:
                # Validation step
                mse_val = compute_mse(y_val, jnp.dot(X_val, self.weights))
                self.losses_in_validation_data[epoch] = mse_val

                # Potential early stopping
                if mse_val < best_mse:
                    best_mse = mse_val
                    patience_counter = 0
                    best_beta = self.weights
                else:
                    patience_counter += 1

                if patience_counter >= self.max_patience:
                    print(f"Stopped at epoch {epoch+1}.")
                    self.stopped_at = epoch + 1
                    break

        if X_val is not None and y_val is not None:
            self.weights = best_beta

        return self

    def predict(self, X_test: jax.Array):
        """
        Predict the labels for the given test data.

        Args:
            X_test (jax.Array): Test data features.

        Returns:
            jax.Array: Predicted labels for the test data.
        """
        return jnp.dot(X_test, self.weights)

    def plot_losses(self):
        """
        Plot training and validation losses over epochs.

        Displays a plot of Mean Squared Error (MSE) for both training and validation data across epochs.
        """
        plt.figure(figsize=(10, 5))
        # Plotting training losses
        plt.title("MSE vs Epochs")
        plt.plot(
            range(self.stopped_at),
            self.losses_in_training_data[: self.stopped_at],
            c="blue",
            label="Training",
        )
        # Plotting validation losses
        plt.plot(
            range(self.stopped_at),
            self.losses_in_validation_data[: self.stopped_at],
            c="orange",
            label="Valdation",
        )
        plt.legend()
        plt.show()
