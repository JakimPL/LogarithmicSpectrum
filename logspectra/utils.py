from typing import TypeAlias, Union

import numpy as np

from logspectra.config import FFTConfig
from logspectra.constants import BINS_PER_OCTAVE

Float: TypeAlias = Union[float, np.floating]


def is_increasing(array: np.ndarray) -> bool:
    return bool(np.all(np.diff(array) > 0))


def rectangle(length: int) -> np.ndarray:
    return np.ones(length, dtype=float)


def to_log_even_bands(
    bands: np.ndarray,
    fft_config: FFTConfig,
) -> np.ndarray:
    size: int = fft_config.log_even_components or len(bands)
    log_even_bands: np.ndarray = np.exp(np.linspace(np.log(fft_config.cutoff), np.log(bands[-1]), size + 1))
    return log_even_bands


def get_number_of_log_components(
    sample_rate: int,
    cutoff: Float,
    bins_per_octave: int = BINS_PER_OCTAVE,
) -> int:
    """
    Calculate the number of music notes (logarithmically spaced frequency components)
    that fit within the frequency range, from cutoff to Nyquist frequency
    (0.5 * sample_rate), given a specific number of bins per octave (12 by default).
    """
    notes: np.floating = np.log2(0.5 * sample_rate / cutoff) * bins_per_octave
    return int(np.ceil(notes))
