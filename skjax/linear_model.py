""" Model(s) """

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# ================================================================================================================ #
# Pure/Functional Implementations for Linear Regression
# ================================================================================================================ #


@partial(jax.jit, static_argnames=("add_intercept",))
def _fit_lstsq(
    X: jax.Array, y: jax.Array, add_intercept: bool = True
) -> jax.Array:
    """Solves for linear regression coefficients using least squares."""
    if add_intercept:
        X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
    # More numerically stable than forming the normal equations
    coeffs, _, _, _ = jnp.linalg.lstsq(X, y, rcond=None)
    return coeffs


@partial(jax.jit, static_argnames=("add_intercept",))
def _predict(X: jax.Array, coeffs: jax.Array, add_intercept: bool = True) -> jax.Array:
    """Makes predictions using linear regression coefficients."""
    if add_intercept:
        X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
    return X @ coeffs


class LinearRegression:
    """
    Linear Regression model solved via Normal Equations using SVD.
    This implementation is a wrapper around a JIT-compiled JAX implementation.
    """

    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
        self.coeff: Optional[jax.Array] = None

    def fit(self, X: jax.Array, y: jax.Array) -> "LinearRegression":
        """
        Fit the linear model.

        Args:
            X (jax.Array): Training data of shape (n_samples, n_features).
            y (jax.Array): Target values of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        self.coeff = _fit_lstsq(X, y, self.add_intercept)
        return self

    def predict(self, X: jax.Array) -> jax.Array:
        """
        Predict using the linear model.

        Args:
            X (jax.Array): Data to predict on, shape (n_samples, n_features).

        Returns:
            jax.Array: Predicted values.
        """
        if self.coeff is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return _predict(X, self.coeff, self.add_intercept)


# ================================================================================================================ #
# Pure/Functional Implementations for Batch Gradient Descent
# ================================================================================================================ #


def _loss_fn(
    weights: jax.Array, X: jax.Array, y: jax.Array, lambda_: float
) -> jax.Array:
    """Computes MSE loss with L2 regularization."""
    residuals = X @ weights - y
    mse_loss = jnp.mean(residuals**2)
    l2_penalty = lambda_ * jnp.sum(weights[1:] ** 2)  # Regularize weights, not bias
    return mse_loss + l2_penalty


@jax.jit
def _compute_mse(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Computes Mean Squared Error."""
    return jnp.mean((y_pred - y_true) ** 2)


@partial(jax.jit, static_argnames=("dropout_rate", "lambda_"))
def _train_step(
    weights: jax.Array,
    X: jax.Array,
    y: jax.Array,
    key: jax.Array,
    learning_rate: float,
    dropout_rate: float,
    lambda_: float,
) -> Tuple[jax.Array, jax.Array]:
    """Performs a single training step (batch gradient descent)."""
    key, subkey = jax.random.split(key)

    # Apply dropout
    if dropout_rate > 0.0:
        dropout_mask = jax.random.bernoulli(
            subkey, p=1 - dropout_rate, shape=X.shape
        )
        X_dropout = X * dropout_mask
    else:
        X_dropout = X

    # Compute gradients and update weights
    grad = jax.grad(_loss_fn)(weights, X_dropout, y, lambda_)
    new_weights = weights - learning_rate * grad
    return new_weights, key


@jax.jit
def _val_step(weights: jax.Array, X_val: jax.Array, y_val: jax.Array) -> jax.Array:
    """Computes validation loss."""
    y_pred = _predict(X_val, weights, add_intercept=False) # X already has intercept
    return _compute_mse(y_val, y_pred)


def _train_scan(
    epochs: int,
    learning_rate: float,
    lambda_: float,
    dropout_rate: float,
    max_patience: int,
    X_train: jax.Array,
    y_train: jax.Array,
    X_val: jax.Array,
    y_val: jax.Array,
    initial_weights: jax.Array,
    key: jax.Array,
):
    """JAX-compiled training loop with early stopping."""

    @jax.jit
    def scan_body(carry, _):
        """Body of the scan function for one epoch."""
        (
            weights,
            best_weights,
            best_val_mse,
            patience_counter,
            key,
        ) = carry

        # Training step
        new_weights, new_key = _train_step(
            weights, X_train, y_train, key, learning_rate, dropout_rate, lambda_
        )

        # Validation step
        val_mse = _val_step(new_weights, X_val, y_val)
        train_mse = _compute_mse(y_train, _predict(X_train, new_weights, add_intercept=False))


        # Early stopping logic
        is_better = val_mse < best_val_mse
        patience_counter = jax.lax.cond(
            is_better, lambda: 0, lambda: patience_counter + 1
        )
        best_val_mse = jnp.where(is_better, val_mse, best_val_mse)
        best_weights = jax.lax.cond(is_better, lambda: new_weights, lambda: best_weights)

        # If patience is exceeded, we stop updating weights
        weights = jax.lax.cond(patience_counter >= max_patience, lambda: weights, lambda: new_weights)
        
        return (weights, best_weights, best_val_mse, patience_counter, new_key), (train_mse, val_mse)

    initial_carry = (
        initial_weights,
        initial_weights,
        jnp.inf,
        0,
        key,
    )
    (final_weights, best_weights, best_val_mse, final_patience, _), (train_losses, val_losses) = jax.lax.scan(
        scan_body, initial_carry, None, length=epochs
    )
    
    # Find the epoch where we stopped
    stopped_at = jnp.sum(val_losses != 0.0)

    return best_weights, train_losses, val_losses, stopped_at


class LinearRegressionBGD:
    """
    Linear Regression with Batch Gradient Descent, implemented in JAX.

    This model is a wrapper around a JIT-compiled training loop (`jax.lax.scan`)
    for high performance. It supports L2 regularization, dropout, and early
    stopping.
    """

    def __init__(
        self,
        weights_init: str = "random",
        epochs: int = 2000,
        learning_rate: float = 5e-3,
        lambda_: float = 0.0,
        max_patience: int = 200,
        dropout: float = 0.0,
        random_state: int = 41,
    ):
        self.weights_init_method = weights_init
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.max_patience = max_patience
        self.dropout = dropout
        self.random_state = random_state

        self.weights: Optional[jax.Array] = None
        self.losses_in_training_data: np.ndarray = np.zeros(epochs)
        self.losses_in_validation_data: np.ndarray = np.zeros(epochs)
        self.stopped_at = epochs

    def fit(
        self,
        X_train: jax.Array,
        y_train: jax.Array,
        X_val: Optional[jax.Array] = None,
        y_val: Optional[jax.Array] = None,
    ) -> "LinearRegressionBGD":
        """
        Fit the model to the training data.
        """
        # Add intercept term
        X_train_b = jnp.hstack([jnp.ones((X_train.shape[0], 1)), X_train])
        if X_val is not None:
            X_val_b = jnp.hstack([jnp.ones((X_val.shape[0], 1)), X_val])
        else:
             # Create dummy data if no validation set is provided
            X_val_b, y_val = X_train_b, y_train


        key = jax.random.key(self.random_state)
        n_features = X_train_b.shape[1]
        
        # Weight initialization
        key, subkey = jax.random.split(key)
        if self.weights_init_method == "zero":
            initial_weights = jnp.zeros(n_features)
        elif self.weights_init_method == "random":
            initial_weights = jax.random.normal(subkey, shape=(n_features,))
        elif self.weights_init_method == "lecun":
            initial_weights = jax.random.normal(subkey, shape=(n_features,)) * jnp.sqrt(1 / (n_features-1))
        elif self.weights_init_method == "xavier":
            initial_weights = jax.random.normal(subkey, shape=(n_features,)) * jnp.sqrt(1 / (n_features-1))
        elif self.weights_init_method == "he":
             initial_weights = jax.random.normal(subkey, shape=(n_features,)) * jnp.sqrt(2 / (n_features-1))
        else:
            raise ValueError(f"Unknown weights_init: {self.weights_init_method}")

        
        best_weights, train_losses, val_losses, stopped_at = _train_scan(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            lambda_=self.lambda_,
            dropout_rate=self.dropout,
            max_patience=self.max_patience,
            X_train=X_train_b,
            y_train=y_train,
            X_val=X_val_b,
            y_val=y_val,
            initial_weights=initial_weights,
            key=key,
        )

        self.weights = best_weights
        self.stopped_at = int(stopped_at)
        self.losses_in_training_data = np.array(train_losses)
        self.losses_in_validation_data = np.array(val_losses)

        return self

    def predict(self, X_test: jax.Array) -> jax.Array:
        """
        Predict the labels for the given test data.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        # `add_intercept=True` is handled inside the pure predict function
        return _predict(X_test, self.weights, add_intercept=True)

    def plot_losses(self):
        """
        Plot training and validation losses over epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.title("MSE vs Epochs")
        
        epochs_ran = range(self.stopped_at)
        plt.plot(
            epochs_ran,
            self.losses_in_training_data[: self.stopped_at],
            c="blue",
            label="Training",
        )
        if jnp.any(self.losses_in_validation_data != 0):
             plt.plot(
                epochs_ran,
                self.losses_in_validation_data[: self.stopped_at],
                c="orange",
                label="Validation",
            )
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.show()

