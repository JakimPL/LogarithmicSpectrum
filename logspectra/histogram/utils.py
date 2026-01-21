import numpy as np


def is_increasing(array: np.ndarray) -> bool:
    return bool(np.all(np.diff(array) > 0))
