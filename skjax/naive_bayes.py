""" Implementing Multinomial Naive Bayes """

from typing import Optional

import jax
import jax.numpy as jnp
from jax import tree

from _utils.helpers._helper_functions import (compute_likelihoods,
                                               compute_means,
                                               compute_posteriors,
                                               compute_priors, compute_stds,
                                               gaussian_pdf)

# ------------------------------------------------------------------------------------------ #


class MultinomialNaiveBayes:
    """
    A class implementing the Multinomial Naive Bayes classifier.

    This classifier is suitable for discrete data, such as text classification tasks where features represent word counts or frequencies. It calculates prior probabilities and feature likelihoods from the training data and uses these to predict the class of new data points.

    Attributes:
        alpha (Optional[jax.Array]): The smoothing parameter for the likelihood estimates. If None, no smoothing is applied. Smoothing is used to handle cases where some feature values might not appear in the training data for a given class.
        priors (jax.Array): The prior probabilities of each class, computed from the training labels.
        likelihoods (dict): A dictionary where each key is a class label and each value is a JAX array of feature likelihoods for that class.

    Methods:
        fit(X: jax.Array, y: jax.Array):
            Trains the Multinomial Naive Bayes model on the provided feature matrix and labels.

        predict(X: jax.Array):
            Predicts class labels for the provided feature matrix using the trained model.

    """

    def __init__(self, alpha: int = 0):
        """
        Initializes the MultinomialNaiveBayes classifier.

        Args:
            alpha (Optional[jax.Array]): The smoothing parameter for likelihood estimation. If None, no smoothing is applied. Defaults to None.
        """
        self.alpha = alpha
        self.priors = None
        self.likelihoods = None

    def fit(self, X: jax.Array, y: jax.Array):
        """
        Fits the model to the training data.

        Args:
            X (jax.Array): The feature matrix for the training data, where each feature represents discrete counts or frequencies.
            y (jax.Array): The vector of class labels corresponding to the training data.

        Returns:
            self: The fitted MultinomialNaiveBayes instance.
        """
        ### CALCULATING PRIORS
        self.priors = compute_priors(y)

        ### CALCULATING LIKELIHOODS
        self.likelihoods = compute_likelihoods(X, y, self.alpha)

        return self

    def predict(self, X: jax.Array):
        """
        Predicts class labels for the given data using the trained model.

        Args:
            X (jax.Array): The feature matrix for which predictions are to be made. Each feature represents discrete counts or frequencies.

        Returns:
            jax.Array: An array of predicted class labels for each sample in X.
        """
        return jnp.argmax(compute_posteriors(X, self.priors, self.likelihoods), axis=1)


# ------------------------------------------------------------------------------------------ #


class GaussianNaiveBayes:
    """
    A class implementing the Gaussian Naive Bayes classifier.

    This classifier assumes that the features follow a Gaussian distribution and is used for classification tasks. It calculates the priors, means, and standard deviations of each class from the training data and uses these statistics to predict the class of new data points.

    Attributes:
        priors (jax.Array): The prior probabilities of each class, computed from the training labels.
        means (dict): A dictionary where each key is a class label and each value is a JAX array of feature means for that class.
        stds (dict): A dictionary where each key is a class label and each value is a JAX array of feature standard deviations for that class.
        seed (int): The random seed for reproducibility.

    Methods:
        fit(X: jax.Array, y: jax.Array):
            Trains the Gaussian Naive Bayes model on the provided feature matrix and labels.

        predict(X: jax.Array):
            Predicts class labels for the provided feature matrix using the trained model.

    """

    def __init__(self, seed: int = 12):
        """
        Initializes the GaussianNaiveBayes classifier.

        Args:
            seed (int): The random seed for initializing random number generators used in the computations. Defaults to 12.
        """
        self.priors: jax.Array = None
        self.means: dict = None
        self.stds: dict = None
        self.seed: int = seed

    def fit(self, X: jax.Array, y: jax.Array):
        """
        Fits the model to the training data.

        Args:
            X (jax.Array): The feature matrix for the training data.
            y (jax.Array): The vector of class labels corresponding to the training data.

        Returns:
            self: The fitted GaussianNaiveBayes instance.
        """
        self.priors = compute_priors(y)
        self.means = compute_means(X, y, self.seed)
        self.stds = compute_stds(X, y, self.seed)

        return self

    def predict(self, X: jax.Array):
        """
        Predicts class labels for the given data using the trained model.

        Args:
            X (jax.Array): The feature matrix for which predictions are to be made.

        Returns:
            jax.Array: An array of predicted class labels for each sample in X.
        """
        posteriors = []
        for x in X:
            likelihoods = jnp.array(
                [
                    gaussian_pdf(x, jnp.array(means), jnp.array(stds))
                    for means, stds in zip(self.means.values(), self.stds.values())
                ]
            )
            vector_of_posteriors = jnp.log(jnp.dot(likelihoods, self.priors))
            posteriors.append(vector_of_posteriors)

        return jnp.argmax(jnp.array(posteriors), axis=1)
