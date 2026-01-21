from typing import NamedTuple, Union

import numpy as np


class Wave(NamedTuple):
    x: np.ndarray
    y: np.ndarray


def get_wave_array(wave: Union[np.ndarray, Wave]) -> np.ndarray:
    if isinstance(wave, Wave):
        wave = wave.y
    elif not isinstance(wave, np.ndarray):
        raise TypeError("wave must be either a numpy array or a Wave instance")

    if wave.ndim != 1:
        raise ValueError("wave must be a one-dimensional array")

    return wave
