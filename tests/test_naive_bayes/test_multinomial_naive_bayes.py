import unittest
import jax.numpy as jnp
from skjax.naive_bayes import MultinomialNaiveBayes  # Replace with the actual module and class name

class TestMultinomialNaiveBayes(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and MultinomialNaiveBayes instance."""
        self.X = jnp.array([
            [1, 2, 0],
            [2, 1, 1],
            [0, 1, 3],
            [1, 0, 2],
            [2, 2, 1]
        ])
        self.y = jnp.array([0, 1, 0, 1, 0])
        
        self.model = MultinomialNaiveBayes(alpha=1.0)

    def test_initialization(self):
        """Test initialization of MultinomialNaiveBayes instance."""
        self.assertEqual(self.model.alpha, 1.0)
        self.assertIsNone(self.model.priors)
        self.assertIsNone(self.model.likelihoods)
    
    def test_fit(self):
        """Test the fit method of MultinomialNaiveBayes."""
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.priors)
        self.assertIsNotNone(self.model.likelihoods)
        # Check that priors and likelihoods are of the expected shapes
        self.assertEqual(self.model.priors.shape[0], len(jnp.unique(self.y)))
        for class_label in self.model.likelihoods.keys():
            self.assertEqual(len(self.model.likelihoods[class_label]), self.X.shape[1])
    
    def test_predict(self):
        """Test the predict method of MultinomialNaiveBayes."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)
        self.assertTrue(jnp.all(predictions >= 0))
        self.assertTrue(jnp.all(predictions < len(jnp.unique(self.y))))
    
    def test_smoothing(self):
        """Test the effect of smoothing parameter on fit."""
        model = MultinomialNaiveBayes(alpha=0.5)
        model.fit(self.X, self.y)
        self.assertIsNotNone(model.likelihoods)
        # You might want to add more specific assertions based on the behavior of smoothing
    
    def test_no_smoothing(self):
        """Test fitting without smoothing."""
        model = MultinomialNaiveBayes(alpha=0)
        model.fit(self.X, self.y)
        self.assertIsNotNone(model.likelihoods)
        # You might want to add more specific assertions based on the behavior without smoothing

if __name__ == "__main__":
    unittest.main()