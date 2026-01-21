from typing import Union

import numpy as np
from scipy.fft import rfft, rfftfreq

from logspectra.config import FFTConfig, Sampling
from logspectra.histogram import Histogram
from logspectra.utils import to_log_even_bands
from logspectra.wave import Wave, get_wave_array


def calculate_spectrum(
    wave: Union[np.ndarray, Wave],
    fft_config: FFTConfig,
    sampling: Sampling,
) -> Histogram:
    if not isinstance(fft_config, FFTConfig):
        raise TypeError("fft_config must be an instance of FFTConfig")

    if not isinstance(sampling, Sampling):
        raise TypeError("sampling must be an instance of Sampling")

    wave = get_wave_array(wave)
    fft_size: int = fft_config.size
    sample_rate: float = sampling.rate

    fft: np.ndarray = rfft(wave, n=fft_size)[1:]  # we don't care about DC offset
    energy = np.square(np.abs(fft) / fft_size)  # power spectrum
    bands: np.ndarray = rfftfreq(fft_size, d=1.0 / sample_rate)
    return Histogram(edges=bands, values=energy)


def calculate_log_even_spectrum(
    wave: Union[np.ndarray, Wave],
    fft_config: FFTConfig,
    sampling: Sampling,
) -> Histogram:
    if not isinstance(fft_config, FFTConfig):
        raise TypeError("fft_config must be an instance of FFTConfig")

    if not isinstance(sampling, Sampling):
        raise TypeError("sampling must be an instance of Sampling")

    linear_spectrum: Histogram = calculate_spectrum(wave, fft_config, sampling)
    log_even_bands: np.ndarray = to_log_even_bands(linear_spectrum.edges, fft_config)
    return linear_spectrum.rebin(log_even_bands)
