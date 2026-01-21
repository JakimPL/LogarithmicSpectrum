from typing import NamedTuple, Union

import numpy as np


class Wave(NamedTuple):
    """
    A discrete wave signal with time and amplitude values.

    Attributes:
        x: Time values (sample indices or timestamps).
        y: Amplitude values (signal samples).
    """

    x: np.ndarray
    y: np.ndarray


def get_wave_array(wave: Union[np.ndarray, Wave]) -> np.ndarray:
    """
    Extract the amplitude array from a wave.

    Args:
        wave: Wave object or numpy array.

    Returns:
        1D array of amplitude values.

    Raises:
        TypeError: If wave is neither numpy array nor Wave instance.
        ValueError: If wave array is not one-dimensional.
    """
    if isinstance(wave, Wave):
        wave = wave.y

    elif not isinstance(wave, np.ndarray):
        raise TypeError("wave must be either a numpy array or a Wave instance")

    if wave.ndim != 1:
        raise ValueError("wave must be a one-dimensional array")

    return wave
