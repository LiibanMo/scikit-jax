import jax
import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------------------------------------ #
### MultinomialNB
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
### GaussianNB
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
