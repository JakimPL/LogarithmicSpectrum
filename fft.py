import numpy as np
from scipy.fft import rfft, rfftfreq

from histogram import Histogram


def calculate_spectrum(wave: np.ndarray, fft_size: int, sample_rate: float) -> Histogram:
    fft: np.ndarray = rfft(wave, n=fft_size)[1:]  # we don't care about DC offset
    energy = np.square(np.abs(fft) / fft_size)  # power spectrum
    bands: np.ndarray = rfftfreq(fft_size, d=1.0 / sample_rate)
    return Histogram(edges=bands, values=energy)
