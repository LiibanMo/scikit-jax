import jax.numpy as jnp
import numpy as np
import pytest

from skjax.linear_model import LinearRegressionSGD


@pytest.fixture
def setup_data():
    """Fixture to set up test data and LinearRegression instance."""
    X_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_train = jnp.array([1.0, 2.0, 3.0])
    X_val = jnp.array([[7.0, 8.0], [9.0, 10.0]])
    y_val = jnp.array([4.0, 5.0])

    model = LinearRegressionSGD(
        weights_init="random",
        epochs=10,
        learning_rate=0.01,
        p=2,
        lambda_=0.1,
        max_patience=2,
        dropout=0.1,
        random_state=42,
    )
    return X_train, y_train, X_val, y_val, model


def test_initialization(setup_data):
    """Test initialization of LinearRegression instance."""
    _, _, _, _, model = setup_data

    assert model.weights == "random"
    assert model.epochs == 10
    assert model.learning_rate == 0.01
    assert model.p == 2
    assert model.lambda_ == 0.1
    assert model.max_patience == 2
    assert model.dropout == 0.1
    assert model.random_state == 42

    assert isinstance(model.weights, str)
    assert isinstance(model.epochs, int)
    assert isinstance(model.learning_rate, float)
    assert isinstance(model.p, int)
    assert isinstance(model.dropout, float)
    assert isinstance(model.random_state, int)
    assert isinstance(model.losses_in_training_data, np.ndarray)
    assert isinstance(model.losses_in_validation_data, np.ndarray)
    assert isinstance(model.stopped_at, int)

    assert model.epochs >= 1
    assert model.dropout <= 1
    assert model.lambda_ >= 0
    assert model.stopped_at >= 1


def test_fit(setup_data):
    """Test the fit method."""
    X_train, y_train, _, _, model = setup_data

    model = model.fit(X_train, y_train)

    assert isinstance(model, LinearRegressionSGD)
    assert len(model.losses_in_training_data) == model.epochs
    assert len(model.losses_in_validation_data) == model.epochs
    assert model.stopped_at >= model.epochs


def test_predict(setup_data):
    """Test the predict method."""
    X_train, y_train, X_val, y_val, model = setup_data

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    assert predictions.shape == y_val.shape
    assert jnp.all(jnp.isfinite(predictions))


def test_losses_plot(setup_data):
    """Test if plotting method runs without error."""
    X_train, y_train, X_val, y_val, model = setup_data

    model.fit(X_train, y_train, X_val, y_val)
    try:
        model.plot_losses()
    except Exception as e:
        pytest.fail(f"plot_losses() raised an exception: {e}")
