import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skjax.linear_model import LinearRegression

np.random.seed(42)


@pytest.fixture
def generate_basic_linear():
    X = np.linspace(0, 10, 100)
    y = 2 * X + 1 + np.random.normal(0, 0.5, X.shape)
    return X, y


def data_for_higher_dimensional_fitting():
    pass
