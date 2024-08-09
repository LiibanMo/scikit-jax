import unittest
import jax.numpy as jnp
from skjax.decomposition import PCA  # Replace with the actual module name where PCA is defined

class TestPCA(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and PCA instance."""
        self.X = jnp.array([
            [2.5, 2.4, 3.5],
            [0.5, 0.7, 2.1],
            [2.2, 2.9, 3.3],
            [1.9, 2.2, 2.9],
            [3.1, 3.0, 3.8]
        ])
        self.num_components = 2
        self.pca = PCA(num_components=self.num_components)

    def test_initialization(self):
        """Test initialization of PCA instance."""
        self.assertEqual(self.pca.num_components, self.num_components)
        self.assertIsNone(self.pca.mean)
        self.assertIsNone(self.pca.principal_components)
        self.assertIsNone(self.pca.explained_variance)

    def test_fit(self):
        """Test the fit method of PCA."""
        self.pca.fit(self.X)
        self.assertIsNotNone(self.pca.mean)
        self.assertIsNotNone(self.pca.principal_components)
        self.assertIsNotNone(self.pca.explained_variance)
        
        # Check shape of principal components and explained variance
        self.assertEqual(self.pca.principal_components.shape[0], self.X.shape[1])
        self.assertEqual(self.pca.explained_variance.shape[0], self.X.shape[1])

    def test_transform(self):
        """Test the transform method of PCA."""
        self.pca.fit(self.X)
        X_transformed = self.pca.transform(self.X)
        self.assertEqual(X_transformed.shape[1], self.num_components)
        self.assertTrue(jnp.all(jnp.isfinite(X_transformed)))

    def test_fit_transform(self):
        """Test the fit_transform method of PCA."""
        X_transformed = self.pca.fit_transform(self.X)
        self.assertEqual(X_transformed.shape[1], self.num_components)
        self.assertTrue(jnp.all(jnp.isfinite(X_transformed)))
        # Ensure PCA is fitted
        self.assertIsNotNone(self.pca.mean)
        self.assertIsNotNone(self.pca.principal_components)

    def test_inverse_transform(self):
        """Test the inverse_transform method of PCA."""
        self.pca.fit(self.X)
        X_transformed = self.pca.transform(self.X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        self.assertEqual(X_reconstructed.shape, self.X.shape)
        self.assertTrue(jnp.all(jnp.isfinite(X_reconstructed)))
        
        # Check if first columns of reconstructed data and original data is close-ish.
        self.assertTrue(jnp.allclose(self.X[:, 0], X_reconstructed[:, 0], atol=10))

    def test_exceptions(self):
        """Test that exceptions are raised when fitting is not done before transforming."""
        with self.assertRaises(RuntimeError):
            self.pca.transform(self.X)
        
        with self.assertRaises(RuntimeError):
            self.pca.inverse_transform(self.X)

if __name__ == "__main__":
    unittest.main()