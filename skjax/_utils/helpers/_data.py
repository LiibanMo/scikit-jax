import jax
import jax.numpy as jnp


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
