import numpy as np


def is_increasing(array: np.ndarray) -> bool:
    """
    Check if array values are strictly increasing.

    Args:
        array: Array to check.

    Returns:
        True if each element is strictly greater than the previous one,
        False otherwise.

    Examples:
        >>> is_increasing(np.array([1, 2, 5, 10]))
        True
        >>> is_increasing(np.array([1, 2, 2, 5]))
        False
    """
    return bool(np.all(np.diff(array) > 0))
