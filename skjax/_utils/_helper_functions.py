import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

# ------------------------------------------------------------------------------------------ #


def split_data(data, val_size=0.1, test_size=0.2):
    """
    Splits the data into training, validation, and test sets.

    Args:
        data (jax.Array): The dataset to split.
        val_size (float, optional): Proportion of data to use for validation. Defaults to 0.1.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the training data, validation data, and test data as jax.Array.
    """
    split_index_test = int(len(data) * (1 - test_size))

    data_non_test = data[:split_index_test]
    data_test = data[split_index_test:]

    split_index_val = int(len(data_non_test) * (1 - val_size))

    data_train = data_non_test[:split_index_val]
    data_val = data_non_test[split_index_val:]

    return jnp.asarray(data_train), jnp.asarray(data_val), jnp.asarray(data_test)


# ------------------------------------------------------------------------------------------ #
### LINEAR REGRESSION
# ------------------------------------------------------------------------------------------ #


@jit
def compute_mse(y_true, y_pred):
    return jnp.mean((y_pred - y_true) ** 2)


# ------------------------------------------------------------------------------------------ #


@jit
def calculate_loss_gradients(
    beta: jax.Array, X: jax.Array, y: jax.Array, p: int = 2, lambda_: float = 0.01
):
    """
    Forward pass on data X.

    Args:
        beta (jax.Array): weights and bias.
        X (jax.Array): data
    """
    return (2 / len(y)) * X.T @ (jnp.dot(X, beta) - y) + lambda_ * p * jnp.sign(
        beta
    ) * jnp.insert(beta[1:] ** (p - 1), 0, 0)


# ------------------------------------------------------------------------------------------ #
### K-MEANS
# ------------------------------------------------------------------------------------------ #


def compute_euclidean_distance(x1, x2):
    """
    Computes the Euclidean distance between two vectors.

    Args:
        x1 (jax.Array): First vector.
        x2 (jax.Array): Second vector.

    Returns:
        jax.Array: Euclidean distance between the two vectors.
    """
    return jnp.sqrt(jnp.sum((x1 - x2) ** 2))


# ------------------------------------------------------------------------------------------ #


def kmeans_plus_initialization(key: jax.Array, num_clusters: int, X: jax.Array):
    """
    Initializes centroids using the k-means++ algorithm.

    Args:
        key (jax.Array): Random key for initialization.
        num_clusters (int): Number of clusters (centroids) to initialize.
        X (jax.Array): Data points to initialize centroids from.

    Returns:
        jax.Array: Initialized centroids as jax.Array.
    """
    indices_of_available_centroids = np.arange(
        len(X)
    )  # Used to keep track which centroid has been chosen.

    key, subkey = jax.random.split(key)

    init_index = jax.random.choice(key, indices_of_available_centroids)
    init_centroids = [X[init_index]]
    indices_of_available_centroids = np.delete(
        indices_of_available_centroids, init_index
    )

    index = 0
    while len(init_centroids) < num_clusters:
        distance_between_points_and_init_centroid = jnp.asarray(
            [compute_euclidean_distance(init_centroids[index], x) for x in X]
        )
        # Computing probabilities of choosing centroids proportional to distance
        probabilities = distance_between_points_and_init_centroid[
            indices_of_available_centroids
        ] / jnp.sum(
            distance_between_points_and_init_centroid[indices_of_available_centroids]
        )
        index_of_centroid_chosen = jax.random.choice(
            subkey, indices_of_available_centroids, p=probabilities, replace=False
        )
        init_centroids.append(X[index_of_centroid_chosen])
        indices_of_available_centroids = np.delete(
            indices_of_available_centroids, index_of_centroid_chosen
        )
        index += 1

    return jnp.asarray(init_centroids)


# ------------------------------------------------------------------------------------------ #


def initialize_k_centroids(num_clusters, X, init: str = "random", seed: int = 12):
    """
    Initializes centroids for k-means clustering.

    Args:
        num_clusters (int): Number of clusters (centroids) to initialize.
        X (jax.Array): Data points to initialize centroids from.
        init (str, optional): Initialization method ('random' or 'k-means++'). Defaults to 'random'.
        seed (int, optional): Random seed for initialization. Defaults to 12.

    Returns:
        jax.Array: Initialized centroids as jax.Array.
    """
    key = jax.random.key(seed)
    initialization = {
        "random": jax.random.choice(key, X, shape=(num_clusters,), replace=False),
        "k-means++": kmeans_plus_initialization(key, num_clusters, X),
    }
    return initialization[init]


# ------------------------------------------------------------------------------------------ #


def calculating_distances_between_centroids_and_points(centroids, X):
    """
    Calculates the distance between each centroid and each data point.

    Args:
        centroids (jax.Array): Centroids for the k-means algorithm.
        X (jax.Array): Data points.

    Returns:
        jax.Array: Distance matrix where each entry (i, j) represents the distance between the i-th centroid and the j-th data point.
    """
    return jnp.asarray(
        [[compute_euclidean_distance(centroid, x) for x in X] for centroid in centroids]
    )


# ------------------------------------------------------------------------------------------ #


def calculate_new_centroids(centroids, X):
    """
    Computes the new centroids based on the current centroids and data points.

    Args:
        centroids (jax.Array): Current centroids.
        X (jax.Array): Data points.

    Returns:
        jax.Array: New centroids computed as the mean of the data points assigned to each centroid.
    """
    distances_between_centroids_and_points = (
        calculating_distances_between_centroids_and_points(centroids, X)
    )
    labels_of_each_point = jnp.argmin(distances_between_centroids_and_points.T, axis=1)
    indices_of_each_cluster = [
        jnp.where(labels_of_each_point == label)
        for label in jnp.unique(labels_of_each_point)
    ]
    new_centroids = jnp.asarray(
        [
            jnp.mean(X[collection_of_indices].T, axis=1)
            for collection_of_indices in indices_of_each_cluster
        ]
    )
    return new_centroids


# ------------------------------------------------------------------------------------------ #


def calculate_stds_in_each_cluster(centroids, X):
    """
    Calculates the standard deviation of data points in each cluster.

    Args:
        centroids (jax.Array): Centroids for the k-means algorithm.
        X (jax.Array): Data points.

    Returns:
        jax.Array: Sum of the standard deviations of data points in each cluster.
    """
    distances_between_centroids_and_points = (
        calculating_distances_between_centroids_and_points(centroids, X)
    )
    labels_of_each_point = jnp.argmin(distances_between_centroids_and_points.T, axis=1)
    indices_of_each_cluster = [
        jnp.where(labels_of_each_point == label)
        for label in jnp.unique(labels_of_each_point)
    ]
    return jnp.sum(
        jnp.asarray(
            [
                jnp.std(X[collection_of_indices])
                for collection_of_indices in indices_of_each_cluster
            ]
        )
    )


# ------------------------------------------------------------------------------------------ #
### NAIVE BAYES
# ------------------------------------------------------------------------------------------ #


def compute_priors(y: jax.Array) -> jax.Array:
    """
    Computes the prior probabilities of each class.

    Args:
        y (jax.Array): Array of class labels.

    Returns:
        jax.Array: Array of prior probabilities for each class.
    """
    unique_classes = jnp.unique(y)
    prior_probabilities = []

    for class_ in unique_classes.tolist():
        prior_probabilities.append(jnp.mean(jnp.where(y == class_, 1, 0)))

    return jnp.asarray(prior_probabilities)


# ------------------------------------------------------------------------------------------ #


def compute_likelihoods(X: jax.Array, y: jax.Array, alpha: int = 0) -> dict:
    """
    Computes the likelihoods of each feature given each class.

    Args:
        X (jax.Array): Feature matrix.
        y (jax.Array): Array of class labels.
        alpha (int, optional): Laplace smoothing parameter. Defaults to 0.

    Returns:
        dict: Dictionary of likelihoods where each key is a class label and each value is a list of dictionaries,
              each representing the probability of each category given the class.
    """
    unique_classes = jnp.unique_values(y)
    unique_categories_in_every_feature = [jnp.unique(x).tolist() for x in X.T]
    collection_of_indices_of_each_class = [
        jnp.where(y == class_) for class_ in unique_classes.tolist()
    ]
    likelihoods_of_each_class_per_category = {
        class_: [] for class_ in unique_classes.tolist()
    }

    for class_, collection_of_indices in zip(
        unique_classes.tolist(), collection_of_indices_of_each_class
    ):
        for j, categories in enumerate(unique_categories_in_every_feature):
            likelihoods_per_feature = [
                (
                    jnp.sum(jnp.where(X[collection_of_indices][:, j] == category, 1, 0))
                    + alpha
                )
                / (len(X[collection_of_indices][:, j]) + alpha * X.shape[1])
                for category in categories
            ]
            likelihoods_of_each_class_per_category[class_].append(
                {
                    category: likelihoods_per_feature[ith_category].item()
                    for ith_category, category in enumerate(categories)
                }
            )

    return likelihoods_of_each_class_per_category


# ------------------------------------------------------------------------------------------ #


def compute_posteriors(X: jax.Array, priors: jax.Array, likelihoods: dict) -> jax.Array:
    """
    Computes the posterior probabilities for each class given the feature matrix.

    Args:
        X (jax.Array): Feature matrix.
        priors (jax.Array): Array of prior probabilities for each class.
        likelihoods (dict): Dictionary of likelihoods for each class and feature.

    Returns:
        jax.Array: Matrix of posterior probabilities where each row corresponds to a data point
    """
    vector_of_posteriors_for_data_point_i = jnp.zeros(len(likelihoods))
    matrix_of_posteriors = []

    for x in X:
        for i in range(len(likelihoods)):
            posterior = jnp.log(priors[i]) + jnp.sum(
                jnp.log(
                    jnp.asarray(
                        [likelihoods[i][j][x_ij.item()] for j, x_ij in enumerate(x)]
                    )
                )
            )
            vector_of_posteriors_for_data_point_i = (
                vector_of_posteriors_for_data_point_i.at[i].set(posterior)
            )
        matrix_of_posteriors.append(vector_of_posteriors_for_data_point_i)

    return jnp.asarray(matrix_of_posteriors)


# ------------------------------------------------------------------------------------------ #


def gaussian_pdf(x, mean, std) -> jax.Array:
    """
    Computes the probability density function of a Gaussian distribution.

    Args:
        x (jax.Array): Data points for which to compute the probability density.
        mean (jax.Array): Mean of the Gaussian distribution.
        std (jax.Array): Standard deviation of the Gaussian distribution.

    Returns:
        jax.Array: Probability density values for the given data points.
    """
    return jnp.exp(-0.5 * ((x - mean) / std) ** 2) / (std * jnp.sqrt(2 * jnp.pi))


# ------------------------------------------------------------------------------------------ #


def compute_means(X: jax.Array, y: jax.Array, random_state=12) -> dict:
    """
    Computes the mean of each feature for each class.

    Args:
        X (jax.Array): Feature matrix where rows represent samples and columns represent features.
        y (jax.Array): Array of class labels corresponding to each sample in X.
        random_state (int, optional): Seed for the random number generator. Defaults to 12.

    Returns:
        dict: A dictionary where keys are class labels and values are lists of means of features for each class.
    """
    np.random.seed(random_state)

    unique_classes = jnp.unique(y).tolist()
    indices_for_each_class = [jnp.where(y == class_) for class_ in unique_classes]

    dictionary_of_means = dict(
        zip(
            unique_classes,
            [
                [
                    jnp.mean(X[collection_of_indices][:, j]).item()
                    for j in range(X.shape[1])
                ]
                for collection_of_indices in indices_for_each_class
            ],
        )
    )

    return dictionary_of_means


# ------------------------------------------------------------------------------------------ #


def compute_stds(X: jax.Array, y: jax.Array, random_state=12) -> dict:
    """
    Computes the standard deviation of each feature for each class.

    Args:
        X (jax.Array): Feature matrix where rows represent samples and columns represent features.
        y (jax.Array): Array of class labels corresponding to each sample in X.
        random_state (int, optional): Seed for the random number generator. Defaults to 12.

    Returns:
        dict: A dictionary where keys are class labels and values are lists of standard deviations of features for each class.
    """
    np.random.seed(random_state)

    unique_classes = jnp.unique(y).tolist()
    indices_for_each_class = [jnp.where(y == class_) for class_ in unique_classes]

    dictionary_of_stds = dict(
        zip(
            unique_classes,
            [
                [
                    jnp.std(X[collection_of_indices][:, j]).item()
                    for j in range(X.shape[1])
                ]
                for collection_of_indices in indices_for_each_class
            ],
        )
    )

    return dictionary_of_stds
