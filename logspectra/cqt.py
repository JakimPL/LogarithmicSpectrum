from typing import Union

import librosa
import numpy as np

from logspectra.config import FFTConfig, Sampling
from logspectra.constants import BINS_PER_OCTAVE
from logspectra.histogram import Histogram
from logspectra.utils import Float, rectangle
from logspectra.wave import Wave, get_wave_array


def calculate_nbins(
    cutoff: Float,
    sample_rate: int,
    bins_per_octave: int = BINS_PER_OCTAVE,
) -> int:
    """Calculate the number of CQT bins needed to cover the frequency range up to Nyquist."""
    nyquist = sample_rate / 2.0
    n_octaves = np.log2(nyquist / cutoff)
    return int(np.ceil(n_octaves * bins_per_octave))


def convert_midpoints_to_edges(midpoints: np.ndarray) -> np.ndarray:
    edges: np.ndarray = np.empty(len(midpoints) + 1)
    edges[1:-1] = np.sqrt(midpoints[:-1] * midpoints[1:])
    edges[0] = midpoints[0] / np.sqrt(midpoints[1] / midpoints[0])
    edges[-1] = midpoints[-1] * np.sqrt(midpoints[-1] / midpoints[-2])
    return edges


def normalize_cqt_energy(
    energy: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: int,
    bins_per_octave: int = BINS_PER_OCTAVE,
) -> np.ndarray:
    """A rough CQT energy normalization by the wavelet lengths."""
    q = 1 / (2 ** (1 / bins_per_octave) - 1)
    wavelet_lengths = np.ceil(q * sample_rate / frequencies)
    energy_scaled: np.ndarray = 2.0 * energy / wavelet_lengths
    return energy_scaled


def calculate_cqt_spectrum(
    wave: Union[np.ndarray, Wave],
    fft_config: FFTConfig,
    sampling: Sampling,
) -> Histogram:
    """Calculate the Constant-Q Transform (CQT) spectrum of a wave."""
    if not isinstance(fft_config, FFTConfig):
        raise TypeError("fft_config must be an instance of FFTConfig")

    if not isinstance(sampling, Sampling):
        raise TypeError("sampling must be an instance of Sampling")

    wave = get_wave_array(wave)
    sample_rate: int = sampling.rate
    cutoff: float = float(fft_config.cutoff)
    bins_per_octave: int = fft_config.bins_per_octave

    n_bins = calculate_nbins(cutoff, sample_rate, bins_per_octave)
    hop_length = len(wave) + 1  # a single frame
    cqt = librosa.cqt(
        wave,
        sr=sample_rate,
        fmin=cutoff,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        window=rectangle,
    )

    energy: np.ndarray = np.mean(np.square(np.abs(cqt)), axis=1)
    frequencies = librosa.cqt_frequencies(
        n_bins=n_bins,
        fmin=cutoff,
        bins_per_octave=bins_per_octave,
    )

    energy_scaled = normalize_cqt_energy(energy, frequencies, sample_rate, bins_per_octave)
    bands: np.ndarray = convert_midpoints_to_edges(frequencies)
    return Histogram(edges=bands, values=energy_scaled)
