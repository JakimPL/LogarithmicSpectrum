from typing import Union

import numpy as np
from scipy.fft import rfft, rfftfreq

from logspectra.config import FFTConfig, Sampling
from logspectra.histogram import Histogram
from logspectra.spectrum.utils import to_log_even_bands
from logspectra.wave import Wave, get_wave_array


def calculate_spectrum(
    wave: Union[np.ndarray, Wave],
    fft_config: FFTConfig,
    sampling: Sampling,
) -> Histogram:
    """
    Calculate the power spectrum of a wave using FFT.

    Computes the real FFT and returns the power spectrum (squared magnitude)
    as a histogram with linearly-spaced frequency bins.

    DC component is excluded.

    Args:
        wave: Input wave as array or Wave object.
        fft_config: FFT configuration (size).
        sampling: Sampling configuration (rate).

    Returns:
        Histogram with frequency edges and power spectrum values.

    Raises:
        TypeError: If fft_config or sampling have incorrect types.
    """
    if not isinstance(fft_config, FFTConfig):
        raise TypeError("fft_config must be an instance of FFTConfig")

    if not isinstance(sampling, Sampling):
        raise TypeError("sampling must be an instance of Sampling")

    wave = get_wave_array(wave)
    fft_size: int = fft_config.size
    sample_rate: float = sampling.rate

    fft: np.ndarray = rfft(wave, n=fft_size)[1:]
    energy = np.square(np.abs(fft) / fft_size)
    bands: np.ndarray = rfftfreq(fft_size, d=1.0 / sample_rate)
    return Histogram(edges=bands, values=energy)


def calculate_log_spectrum(
    wave: Union[np.ndarray, Wave],
    fft_config: FFTConfig,
    sampling: Sampling,
) -> Histogram:
    """
    Calculate the power spectrum with logarithmically-spaced frequency bins.

    Computes the linear FFT spectrum and then rebins it to logarithmic scale
    using the configuration parameters.

    Args:
        wave: Input wave as array or Wave object.
        fft_config: FFT configuration (size, cutoff, log_even_components).
        sampling: Sampling configuration (rate).

    Returns:
        Histogram with log-spaced frequency edges and rebinned power values.

    Raises:
        TypeError: If fft_config or sampling have incorrect types.
    """
    if not isinstance(fft_config, FFTConfig):
        raise TypeError("fft_config must be an instance of FFTConfig")

    if not isinstance(sampling, Sampling):
        raise TypeError("sampling must be an instance of Sampling")

    linear_spectrum: Histogram = calculate_spectrum(wave, fft_config, sampling)
    log_even_bands: np.ndarray = to_log_even_bands(linear_spectrum.edges, fft_config)
    return linear_spectrum.rebin(log_even_bands)
