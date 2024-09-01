import time
import unittest

import jax.numpy as jnp
import pandas as pd

from skjax.clustering import KMeans


class TestKMeans(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and KMeans instance."""
        self.X1 = pd.read_csv("tests/files/basic5.csv").drop(columns="color").to_numpy(jnp.float32)
        self.X2 = jnp.array([
            [1, 2], [1, 3], [2, 2], [2, 3],  # Cluster 1
            [8, 9], [8, 8], [9, 9], [9, 8]   # Cluster 2
        ])
        self.X3 = jnp.array([
            [1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 1
            [5, 5], [5, 6], [6, 5], [6, 6],  # Cluster 2
            [8, 1], [8, 2], [9, 1], [9, 2]   # Cluster 3
        ])

    def test_01_fit_time(self):
        model = KMeans(num_clusters=3)
        start_time = time.time()
        model.fit(self.X1)
        end_time = time.time()

        time_taken = end_time - start_time
        print(f"Time taken to fit the model: {time_taken:.4f} seconds")
        # Assert that time taken is within a reasonable limit, e.g., 1 second
        self.assertLess(time_taken, 1.0, "Model fitting took too long")

    def test_02_fit(self):
        """Test the fit method of KMeans."""
        model = KMeans(num_clusters=2).fit(self.X2)
        true_centroids = jnp.array([
            [1.5, 2.5],
            [8.5, 8.5]
            ]).sort(axis=0)
        pred_centroids = model.centroids.sort(axis=0)
        self.assertTrue(jnp.allclose(true_centroids, pred_centroids))

    def test_03_fit(self):
        """Test the fit method of KMeans."""
        model = KMeans(num_clusters=3).fit(self.X3)
        true_centroids = jnp.array([
            [8.5, 1.5],
            [1.5, 1.5],
            [5.5, 5.5]
            ]).sort(axis=0)
        pred_centroids = model.centroids.sort(axis=0)
        self.assertTrue(jnp.allclose(true_centroids, pred_centroids))


if __name__ == "__main__":
    unittest.main()
