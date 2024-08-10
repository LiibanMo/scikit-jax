import unittest
import time

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

    def test_01_fit_time(self):
        model = KMeans(num_clusters=3)
        start_time = time.time()
        model.fit(self.X)
        end_time = time.time()
        
        time_taken = end_time - start_time
        print(f"Time taken to fit the model: {time_taken:.4f} seconds")
        # Assert that time taken is within a reasonable limit, e.g., 1 second
        self.assertLess(time_taken, 1.0, "Model fitting took too long")

    def test_02_fit(self):
        """Test the fit method of KMeans."""
        self.model.fit(self.X)
        self.assertIsNotNone(self.model.centroids)
        self.assertEqual(self.model.centroids.shape[0], self.model.num_clusters)
        self.assertEqual(self.model.centroids.shape[1], self.X.shape[1])


    def test_03_fit_early_stopping(self):
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

    def test_04_no_early_stopping(self):
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
