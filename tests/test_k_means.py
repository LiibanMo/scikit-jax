import time
import pytest

import jax.numpy as jnp
import pandas as pd

from skjax.clustering import KMeans

X1 = pd.read_csv("tests/files/basic5.csv").drop(columns="color").to_numpy(jnp.float32)
X2 = jnp.array([
        [1, 2], [1, 3], [2, 2], [2, 3],  # Cluster 1
        [8, 9], [8, 8], [9, 9], [9, 8]   # Cluster 2
    ])
X3 = jnp.array([
        [1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 1
        [5, 5], [5, 6], [6, 5], [6, 6],  # Cluster 2
        [8, 1], [8, 2], [9, 1], [9, 2]   # Cluster 3
    ])

def test_01_fit_time():
    model = KMeans(num_clusters=3)
    start_time = time.time()
    model.fit(X1)
    end_time = time.time()

    time_taken = end_time - start_time
    # Assert that time taken is within a reasonable limit, e.g., 1 second
    assert time_taken < 5.0, f"Model fitting took too long: {time_taken:.4f}s"

def test_02_fit():
    """Test the fit method of KMeans."""
    model = KMeans(num_clusters=2).fit(X2)
    true_centroids = jnp.array([
        [1.5, 2.5],
        [8.5, 8.5]
        ]).sort(axis=0)
    pred_centroids = model.centroids.sort(axis=0)
    assert jnp.allclose(true_centroids, pred_centroids) 

def test_03_fit():
    """Test the fit method of KMeans."""
    model = KMeans(num_clusters=3).fit(X3)
    true_centroids = jnp.array([
        [8.5, 1.5],
        [1.5, 1.5],
        [5.5, 5.5]
        ]).sort(axis=0)
    pred_centroids = model.centroids.sort(axis=0)
    assert jnp.allclose(true_centroids, pred_centroids) 
