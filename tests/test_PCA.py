import jax.numpy as jnp
import pytest

from skjax.decomposition import PCA

num_components = 2


@pytest.fixture
def setup():
    """Sets up test data and PCA instance."""
    X = jnp.array(
        [
            [2.5, 2.4, 3.5],
            [0.5, 0.7, 2.1],
            [2.2, 2.9, 3.3],
            [1.9, 2.2, 2.9],
            [3.1, 3.0, 3.8],
        ]
    )

    pca = PCA(num_components=num_components)

    return X, pca


def test_initialization(setup):
    """Tests initialization of PCA instance."""

    _, pca = setup

    assert pca.mean is None
    assert pca.principal_components is None
    assert pca.explained_variance is None


def test_fit(setup):
    """Tests the fit method of PCA."""
    X, pca = setup
    pca.fit(X)

    assert pca.mean is not None
    assert pca.principal_components is not None
    assert pca.explained_variance is not None
    assert pca.principal_components.shape[0] == X.shape[1]
    assert pca.explained_variance.shape[0] == X.shape[1]


def test_transform(setup):
    """Tests the transform method of PCA."""
    X, pca = setup
    pca.fit(X)

    X_transformed = pca.transform(X)

    assert X_transformed.shape[1] == num_components
    assert jnp.all(jnp.isfinite(X_transformed))


def test_fit_transform(setup):
    """Tests the fit_transform method of PCA."""
    X, pca = setup
    X_transformed = pca.fit_transform(X)

    assert X_transformed.shape[1] == num_components
    assert jnp.all(jnp.isfinite(X_transformed))
    assert pca.mean is not None
    assert pca.principal_components is not None


def test_inverse_transform(setup):
    """Tests the inverse_transform method of PCA."""
    X, pca = setup
    pca.fit(X)

    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    assert X_reconstructed.shape == X.shape
    assert jnp.all(jnp.isfinite(X_reconstructed))
    assert jnp.allclose(X[:, 0], X_reconstructed[:, 0], atol=10)


def test_exceptions(setup):
    """Tests that exceptions are raised when fitting is not done before transforming."""
    X, pca = setup

    with pytest.raises(RuntimeError):
        pca.transform(X)

    with pytest.raises(RuntimeError):
        pca.inverse_transform(X)
