import unittest

import jax
import jax.numpy as jnp
import pandas as pd

from skjax.clustering import KMeans

# Example data
data = pd.read_csv('tests/files/basic5.csv')

class TestKMeans(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and KMeans instance."""
        self.X = jnp.asarray(pd.read_csv('tests/files/basic5.csv').drop(columns='color').to_numpy())
        self.model = KMeans(
            num_clusters=3,
            epochs=10,
            init="random",
            max_patience=3,
            seed=42,
        )

    def test_initialization(self):
        """Test initialization of KMeans instance."""
        self.assertEqual(self.model.num_clusters, 3)
        self.assertEqual(self.model.epochs, 10)
        self.assertEqual(self.model.max_patience, 3)
        self.assertEqual(self.model.init, "random")
        self.assertEqual(self.model.seed, 42)
        self.assertIsNone(self.model.centroids)

    def test_fit(self):
        """Test the fit method of KMeans."""
        self.model.fit(self.X)
        self.assertIsNotNone(self.model.centroids)
        self.assertEqual(self.model.centroids.shape[0], self.model.num_clusters)
        self.assertEqual(self.model.centroids.shape[1], self.X.shape[1])

    def test_fit_early_stopping(self):
        """Test early stopping based on patience."""
        model = KMeans(
            num_clusters=2,
            epochs=10,
            init="random",
            max_patience=1,
            seed=42,
        )
        model.fit(self.X)
        self.assertTrue(model.centroids is not None)

    def test_no_early_stopping(self):
        """Test no early stopping with patience larger than required."""
        model = KMeans(
            num_clusters=2,
            epochs=10,
            init="random",
            max_patience=None,
            seed=42,
        )
        model.fit(self.X)
        self.assertTrue(model.centroids is not None)

if __name__ == "__main__":
    unittest.main()
