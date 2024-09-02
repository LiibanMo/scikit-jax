import jax.numpy as jnp
import pytest

from skjax.naive_bayes import MultinomialNaiveBayes


@pytest.fixture
def setup():
    """Set up test data and MultinomialNaiveBayes instance."""
    X = jnp.array([[1, 2, 0], [2, 1, 1], [0, 1, 3], [1, 0, 2], [2, 2, 1]])
    y = jnp.array([0, 1, 0, 1, 0])

    model = MultinomialNaiveBayes(alpha=1.0)

    return X, y, model


def test_initialization(setup):
    """Test initialization of MultinomialNaiveBayes instance."""
    _, _, model = setup

    assert model.alpha == 1.0
    assert model.priors is None
    assert model.likelihoods is None


def test_fit(setup):
    """Test the fit method of MultinomialNaiveBayes."""
    X, y, model = setup

    model.fit(X, y)
    assert model.priors is not None
    assert model.likelihoods is not None
    assert model.priors.shape[0] == len(jnp.unique(y))
    for class_label in model.likelihoods.keys():
        assert len(model.likelihoods[class_label]) == X.shape[1]


def test_predict(setup):
    """Test the predict method of MultinomialNaiveBayes."""
    X, y, model = setup

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert jnp.all(predictions >= 0)
    assert jnp.all(predictions < len(jnp.unique(y)))


def test_smoothing(setup):
    """Test the effect of smoothing parameter on fit."""
    X, y, model = setup

    model = MultinomialNaiveBayes(alpha=0.5)
    model.fit(X, y)

    assert model.likelihoods is not None


def test_no_smoothing(setup):
    """Test fitting without smoothing."""
    X, y, model = setup

    model = MultinomialNaiveBayes(alpha=0)
    model.fit(X, y)
    assert model.likelihoods is not None
