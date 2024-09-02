import jax
import jax.numpy as jnp
from jax import jit

# ------------------------------------------------------------------------------------------ #
### LinearRegression
# ------------------------------------------------------------------------------------------ #


@jit
def compute_mse(y_true, y_pred):
    return jnp.mean((y_pred - y_true) ** 2)


# ------------------------------------------------------------------------------------------ #


@jit
def calculate_loss_gradients(
    beta: jax.Array, X: jax.Array, y: jax.Array, p: int = 2, lambda_: float = 0.01
):
    """
    Forward pass on data X.

    Args:
        beta (jax.Array): weights and bias.
        X (jax.Array): data
    """
    return (2 / len(y)) * X.T @ (jnp.dot(X, beta) - y) + lambda_ * p * jnp.sign(
        beta
    ) * jnp.insert(beta[1:] ** (p - 1), 0, 0)
