import unittest
import jax.numpy as jnp
from skjax.naive_bayes import GaussianNaiveBayes  # Replace with the actual module and class name

class TestGaussianNaiveBayes(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and GaussianNaiveBayes instance."""
        self.X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [2.0, 2.2],
            [3.0, 3.0],
            [3.5, 3.5]
        ])
        self.y = jnp.array([0, 0, 0, 1, 1])
        
        self.model = GaussianNaiveBayes(seed=42)

    def test_initialization(self):
        """Test initialization of GaussianNaiveBayes instance."""
        self.assertEqual(self.model.seed, 42)
        self.assertIsNone(self.model.priors)
        self.assertIsNone(self.model.means)
        self.assertIsNone(self.model.stds)

    def test_fit(self):
        """Test the fit method of GaussianNaiveBayes."""
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.priors)
        self.assertIsNotNone(self.model.means)
        self.assertIsNotNone(self.model.stds)
        
        # Check that priors, means, and stds are not empty
        self.assertGreater(len(self.model.priors), 0)
        self.assertGreater(len(self.model.means), 0)
        self.assertGreater(len(self.model.stds), 0)
    
    def test_predict(self):
        """Test the predict method of GaussianNaiveBayes."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)
        self.assertTrue(jnp.all(predictions >= 0))
        self.assertTrue(jnp.all(predictions < len(jnp.unique(self.y))))

if __name__ == "__main__":
    unittest.main()