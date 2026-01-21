import numpy as np

from logspectra.config import FFTConfig


def rectangle_window(length: int) -> np.ndarray:
    """
    Create a rectangular (uniform) window.

    Args:
        length: Window length.

    Returns:
        Array of ones with the specified length.
    """
    return np.ones(length, dtype=float)


def to_log_even_bands(
    bands: np.ndarray,
    fft_config: FFTConfig,
) -> np.ndarray:
    """
    Generate logarithmically-spaced frequency band edges.

    Creates evenly-spaced bins on a logarithmic scale from cutoff frequency
    to the maximum frequency in the input bands.

    Args:
        bands: Original frequency band edges.
        fft_config: FFT configuration (cutoff, log_even_components).

    Returns:
        Array of log-spaced frequency edges.
    """
    size: int = fft_config.log_even_components or len(bands)
    log_even_bands: np.ndarray = np.exp(np.linspace(np.log(fft_config.cutoff), np.log(bands[-1]), size + 1))
    return log_even_bands
