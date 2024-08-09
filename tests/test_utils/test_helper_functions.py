import unittest
import jax.numpy as jnp
from jax import random
import numpy as np
from skjax._utils._helper_functions import (
    split_data, compute_mse, calculate_loss_gradients,
    compute_euclidean_distance, kmeans_plus_initialization,
    initialize_k_centroids, calculating_distances_between_centroids_and_points,
    calculate_new_centroids, calculate_stds_in_each_cluster,
    compute_priors, compute_likelihoods, compute_posteriors,
    gaussian_pdf, compute_means, compute_stds
)


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        # Set up any state needed for the tests
        self.key = random.PRNGKey(0)
        self.X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.y = jnp.array([0, 1, 0])
    
    def test_split_data(self):
        data = jnp.arange(100).reshape(50, 2)
        train, val, test = split_data(data, val_size=0.2, test_size=0.2)
        self.assertEqual(len(train), 32)
        self.assertEqual(len(val), 8)
        self.assertEqual(len(test), 10)

    def test_compute_mse(self):
        y_true = jnp.array([1, 2, 3])
        y_pred = jnp.array([1, 2, 2])
        self.assertTrue(jnp.isclose(compute_mse(y_true, y_pred), 0.33333334))

    def test_calculate_loss_gradients(self):
        beta = jnp.array([1.0, 0.5])
        X = jnp.array([[1, 2], [3, 4]])
        y = jnp.array([1, 2])
        gradients = calculate_loss_gradients(beta, X, y, p=2, lambda_=0.01)
        self.assertEqual(gradients.shape, beta.shape)

    def test_compute_euclidean_distance(self):
        x1 = jnp.array([1.0, 2.0])
        x2 = jnp.array([4.0, 6.0])
        self.assertTrue(jnp.isclose(compute_euclidean_distance(x1, x2), 5.0))

    def test_kmeans_plus_initialization(self):
        centroids = kmeans_plus_initialization(self.key, num_clusters=2, X=self.X)
        self.assertEqual(centroids.shape, (2, 2))

    def test_initialize_k_centroids(self):
        centroids = initialize_k_centroids(num_clusters=2, X=self.X, init="random", seed=12)
        self.assertEqual(centroids.shape, (2, 2))

    def test_calculating_distances_between_centroids_and_points(self):
        centroids = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        distances = calculating_distances_between_centroids_and_points(centroids, self.X)
        self.assertEqual(distances.shape, (2, 3))

    def test_calculate_new_centroids(self):
        centroids = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        new_centroids = calculate_new_centroids(centroids, self.X)
        self.assertEqual(new_centroids.shape, (2, 2))

    def test_calculate_stds_in_each_cluster(self):
        centroids = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        stds = calculate_stds_in_each_cluster(centroids, self.X)
        self.assertTrue(stds >= 0)  # Standard deviation should be non-negative

    def test_compute_priors(self):
        priors = compute_priors(self.y)
        self.assertEqual(priors.shape[0], len(jnp.unique(self.y)))

    def test_compute_likelihoods(self):
        alpha = 1
        likelihoods = compute_likelihoods(self.X, self.y, alpha)
        self.assertTrue(isinstance(likelihoods, dict))

    def test_compute_posteriors(self):
        priors = compute_priors(self.y)
        likelihoods = compute_likelihoods(self.X, self.y)
        posteriors = compute_posteriors(self.X, priors, likelihoods)
        self.assertEqual(posteriors.shape[0], self.X.shape[0])

    def test_gaussian_pdf(self):
        x = jnp.array([1.0, 2.0])
        mean = jnp.array([0.0, 1.0])
        std = jnp.array([1.0, 1.0])
        pdf = gaussian_pdf(x, mean, std)
        self.assertEqual(pdf.shape, x.shape)

    def test_compute_means(self):
        means = compute_means(self.X, self.y)
        self.assertEqual(len(means), len(jnp.unique(self.y)))

    def test_compute_stds(self):
        stds = compute_stds(self.X, self.y)
        self.assertEqual(len(stds), len(jnp.unique(self.y)))


if __name__ == "__main__":
    unittest.main()