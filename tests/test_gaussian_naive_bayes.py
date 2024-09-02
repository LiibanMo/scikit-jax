import pytest

import jax.numpy as jnp

from skjax.naive_bayes import GaussianNaiveBayes


@pytest.fixture
def setup():
    model = GaussianNaiveBayes(seed=42)
    X = jnp.array([[1.0, 2.0], [1.5, 1.8], [2.0, 2.2], [3.0, 3.0], [3.5, 3.5]])
    y = jnp.array([0, 0, 0, 1, 1])
    return X, y, model

def test_initialization(setup):
    """Test initialization of GaussianNaiveBayes instance."""
    _, _, model = setup

    assert model.priors is None
    assert model.means is None
    assert model.stds is None

def test_fit(setup):
    """Test the fit method of GaussianNaiveBayes."""
    X, y, model = setup
    model.fit(X, y)
    assert model.priors is not None
    assert model.means is not None
    assert model.stds is not None

    # Check that priors, means, and stds are not empty
    assert len(model.priors) > 0
    assert len(model.means) > 0
    assert len(model.stds) > 0

def test_predict(setup):
    """Test the predict method of GaussianNaiveBayes."""
    X, y, model = setup
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert jnp.all(predictions >= 0)
    assert jnp.all(predictions < len(jnp.unique(y)))

