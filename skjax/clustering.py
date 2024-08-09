from typing import Optional

import jax
import jax.numpy as jnp

from ._utils._helper_functions import (
    calculate_new_centroids, calculate_stds_in_each_cluster,
    calculating_distances_between_centroids_and_points, initialize_k_centroids)
from ._utils.config import EPOCHS_k, MAX_PATIENCE_k, RANDOM_STATE_k

# ------------------------------------------------------------------------------------------ #


class KMeans:
    def __init__(
        self,
        num_clusters: int,
        epochs: int = EPOCHS_k,
        init: str = "random",
        max_patience: Optional[int] = MAX_PATIENCE_k,
        seed: int = RANDOM_STATE_k,
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
        self.max_patience: Optional[int] = max_patience
        self.init: str = init
        self.seed: int = seed
        self.centroids = None

    def fit(self, X: jax.Array) -> None:
        """
        Compute the KMeans clustering.

        Args:
            X (jax.Array): Input data, where each row is a data point.

        Returns:
            self: The instance of the KMeans object with fitted centroids.
        """
        best_std = jnp.inf
        patience = 0
        centroids = initialize_k_centroids(
            num_clusters=self.num_clusters, X=X, init=self.init, seed=self.seed
        )

        print("Training model...")
        for epoch in range(self.epochs):
            centroids = calculate_new_centroids(centroids, X)
            current_std = calculate_stds_in_each_cluster(centroids, X)

            if current_std < best_std:
                self.centroids = centroids
                best_std = current_std
                patience = 0
            else:
                patience += 1

            if self.max_patience is not None and patience >= self.max_patience:
                print(f"Terminated at epoch:{epoch}")
                break
