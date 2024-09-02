from typing import Optional

import jax
import jax.numpy as jnp

from skjax._utils.helpers._clustering import (
    initialize_centroids, assign_clusters_to_data, calculate_new_centroids
    )

# ------------------------------------------------------------------------------------------ #


class KMeans:
    def __init__(
        self,
        num_clusters: int,
        epochs: int = 5,
        random_state: int = 5,
    ):
        """
        Initialize the KMeans clustering algorithm.

        Args:
            num_clusters (int): The number of clusters to form.
            epochs (int, optional): The number of iterations to run. Default is 25.
            init (str, optional): Method for initializing centroids ('random' or other methods). Default is 'random'.
            max_patience (int, optional): The maximum number of epochs to wait for improvement before stopping early. Default is 5.
            seed (int, optional): Random seed for reproducibility. Default is 12.
        """
        self.num_clusters: int = num_clusters
        self.epochs: int = epochs
        self.random_state: int = random_state

    def fit(self, X: jax.Array) -> None:
        """
        Compute the KMeans clustering.

        Args:
            X (jax.Array): Input data, where each row is a data point.

        Returns:
            self: The instance of the KMeans object with fitted centroids.
        """
        self.init_centroids = initialize_centroids(
            X, num_clusters=self.num_clusters
        )

        centroids_for_each_data_point = assign_clusters_to_data(X, self.init_centroids)
        print(centroids_for_each_data_point)

        for epoch in range(self.epochs):
            centroids, centroids_for_each_data_point = calculate_new_centroids(
                X, centroids_for_each_data_point, self.num_clusters
            )

        self.centroids = jnp.asarray(list(centroids.values()))

        return self
