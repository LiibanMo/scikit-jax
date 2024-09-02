import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

# ------------------------------------------------------------------------------------------ #
### KMeans
# ------------------------------------------------------------------------------------------ #


def initialize_centroids(X: jax.Array, num_clusters: int, random_state: int = 12):
    # k-means++ initialisation

    assert (
        num_clusters > 0
    ), f"num_clusters should be a natural number greater than 0. Instead got {num_clusters}"

    X_without_centroids = X.copy()

    key = jax.random.PRNGKey(random_state)
    init_index = jax.random.choice(key, X.shape[0])

    initialised_centroids = {cluster: None for cluster in range(num_clusters)}

    init_centroid = X[init_index]
    initialised_centroids[0] = init_centroid

    if num_clusters == 1:
        return init_centroid
    else:
        for cluster in range(1, num_clusters):
            X_without_centroids = jnp.vstack(
                [
                    X_without_centroids[:init_index],
                    X_without_centroids[init_index + 1 :],
                ]
            )
            squared_distances_from_init_centroid = (
                norm(X_without_centroids - init_centroid, axis=1) ** 2
            )
            prob_dist_of_centroid_chosen = (
                squared_distances_from_init_centroid
                / squared_distances_from_init_centroid.sum()
            )
            key, subkey = jax.random.split(key)
            init_index = jax.random.choice(
                subkey, X_without_centroids.shape[0], p=prob_dist_of_centroid_chosen
            )
            init_centroid = X_without_centroids[init_index]
            initialised_centroids[cluster] = init_centroid

    return initialised_centroids


# ------------------------------------------------------------------------------------------ #


def assign_clusters_to_data(X: jax.Array, centroids: dict):

    distances_matrix = jnp.zeros(shape=(len(centroids), X.shape[0]))

    for cluster, centroid in centroids.items():
        distances_from_cluster = norm(X - centroid, axis=1)
        distances_matrix = distances_matrix.at[cluster].set(distances_from_cluster)

    assigned_clusters = distances_matrix.argmin(axis=0)

    return assigned_clusters


# ------------------------------------------------------------------------------------------ #


def calculate_new_centroids(
    X: jax.Array, assigned_centroids: jax.Array, num_clusters: int
):

    centroids = {cluster: None for cluster in range(num_clusters)}
    new_distances = jnp.zeros(shape=(num_clusters, X.shape[0]))

    for cluster in range(num_clusters):
        indices_for_cluster = jnp.where(assigned_centroids == cluster)[0]
        X_at_cluster = X[indices_for_cluster]
        new_centroid = jnp.mean(X_at_cluster, axis=0)
        centroids[cluster] = new_centroid
        new_distances_for_cluster = norm(X - new_centroid, axis=1)
        new_distances = new_distances.at[cluster].set(new_distances_for_cluster)

    updated_centroids_for_X = new_distances.argmin(axis=0)
    return centroids, updated_centroids_for_X
