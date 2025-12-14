import jax.numpy as jnp
import numpy as np
import pytest

from skjax.linear_model import LinearRegression

np.random.seed(42)


@pytest.fixture
def generate_basic_linear():
    X = np.random.rand(100, 1)
    # y = 1 * x_0 + 2 * x_1 + ...
    y = 2 * X[:, 0] + 1 + np.random.normal(0, 0.5, 100)
    return X, y


def test_fit_predict_basic_linear(generate_basic_linear):
    """Test fitting a simple linear regression model."""
    X, y = generate_basic_linear
    model = LinearRegression()
    model.fit(X, y)

    assert model.coeff is not None
    # Coeffs should be [intercept, slope] -> [1, 2]
    assert jnp.allclose(model.coeff, jnp.array([1.0, 2.0]), atol=0.5)

    # Test predict
    predictions = model.predict(X)
    assert predictions.shape == y.shape


def test_predict_new_data(generate_basic_linear):
    """Test that predict works on new data."""
    X, y = generate_basic_linear
    model = LinearRegression().fit(X, y)

    X_new = jnp.array([[0.5], [2.0]])
    y_pred = model.predict(X_new)

    # Expected: y = 2*x + 1 => 2*0.5+1=2, 2*2+1=5
    assert y_pred.shape == (2,)
    assert jnp.allclose(y_pred, jnp.array([2.0, 5.0]), atol=0.5)


def test_multiple_features():
    """Test linear regression with multiple features."""
    X = np.random.rand(100, 3)
    # y = 1*x_0 + 2*x_1 + 3*x_2 + 4
    y = 1 * X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 4 + np.random.normal(0, 0.1, 100)
    model = LinearRegression()
    model.fit(X, y)

    assert model.coeff is not None
    assert model.coeff.shape == (4,)  # 3 features + intercept
    # Coeffs are [intercept, w1, w2, w3] -> [4, 1, 2, 3]
    assert jnp.allclose(model.coeff, jnp.array([4.0, 1.0, 2.0, 3.0]), atol=0.5)
