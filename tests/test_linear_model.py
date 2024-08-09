import unittest

import jax
import jax.numpy as jnp
import numpy as np

from skjax.linear_model import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data and LinearRegression instance."""
        self.X_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.y_train = jnp.array([1.0, 2.0, 3.0])
        self.X_val = jnp.array([[7.0, 8.0], [9.0, 10.0]])
        self.y_val = jnp.array([4.0, 5.0])

        self.model = LinearRegression(
            weights_init="random",
            epochs=10,
            learning_rate=0.01,
            p=2,
            lambda_=0.1,
            max_patience=2,
            dropout=0.1,
            random_state=42,
        )

    def test_initialization(self):
        """Test initialization of LinearRegression instance."""
        # Checking correct initialization value
        self.assertEqual(self.model.weights, "random")
        self.assertEqual(self.model.epochs, 10)
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(self.model.p, 2)
        self.assertEqual(self.model.lambda_, 0.1)
        self.assertEqual(self.model.max_patience, 2)
        self.assertEqual(self.model.dropout, 0.1)
        self.assertEqual(self.model.random_state, 42)

        # Checking instances
        self.assertIsInstance(self.model.weights, str)
        self.assertIsInstance(self.model.epochs, int)
        self.assertIsInstance(self.model.learning_rate, float)
        self.assertIsInstance(self.model.p, int)
        self.assertIsInstance(self.model.dropout, float)
        self.assertIsInstance(self.model.random_state, int)
        self.assertIsInstance(self.model.losses_in_training_data, np.ndarray)
        self.assertIsInstance(self.model.losses_in_validation_data, np.ndarray)
        self.assertIsInstance(self.model.stopped_at, int)

        # Checking inequalities
        self.assertLessEqual(1, self.model.epochs)
        self.assertLessEqual(self.model.dropout, 1)
        self.assertLessEqual(0, self.model.lambda_)
        self.assertLessEqual(1, self.model.stopped_at)

    def test_fit(self):
        """Test the fit method."""
        model = self.model.fit(self.X_train, self.y_train)

        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(len(model.losses_in_training_data), model.epochs)
        self.assertEqual(len(model.losses_in_validation_data), model.epochs)
        self.assertGreaterEqual(model.stopped_at, model.epochs)

    def test_predict(self):
        """Test the predict method."""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_val)
        self.assertEqual(predictions.shape, self.y_val.shape)
        self.assertTrue(jnp.all(jnp.isfinite(predictions)))

    def test_losses_plot(self):
        """Test if plotting method runs without error."""
        try:
            self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            self.model.plot_losses()
        except Exception as e:
            self.fail(f"plot_losses() raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
