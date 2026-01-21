import numpy as np

from logspectra.config import Sampling, WaveDefinition
from logspectra.types import Float
from logspectra.wave.wave import Wave


def sine(
    x: np.ndarray,
    frequency: Float,
    amplitude: Float,
    phase: Float,
    k: int = 1,
) -> np.ndarray:
    """
    Generate a sine wave component.

    Computes amplitudes * sin(2π * k * frequency * x + phase) for each x value.

    Args:
        x: Time domain values.
        frequency: Base frequency in Hz.
        amplitudes: Amplitude multiplier.
        phase: Phase shift in radians.
        k: Harmonic multiplier (1 for fundamental, 2 for first overtone, etc.).

    Returns:
        Array of sine wave values.
    """
    factor = 2.0 * k * np.pi * frequency
    return np.sin(x * factor + phase) * amplitude


def get_domain(sampling: Sampling) -> np.ndarray:
    """
    Generate the time domain for wave synthesis.

    Args:
        sampling: Sampling configuration (duration, rate).

    Returns:
        Array of time values from 0 to duration (exclusive).
    """
    return np.linspace(0.0, sampling.duration, sampling.samples, endpoint=False)


def compose_wave(
    x: np.ndarray,
    wave_definition: WaveDefinition,
) -> np.ndarray:
    """
    Compose a wave from multiple harmonic sine components.

    Sums sine waves at frequencies f, 2f, 3f, ... where f is the base frequency,
    each with its own amplitude and phase from the wave definition.

    Args:
        x: Time domain values.
        wave_definition: Wave specification (base_frequency, amplitudes, phases).
    Returns:
        Array of composed wave amplitude values.
    """
    f = wave_definition.base_frequency
    amplitudes = wave_definition.amplitudes
    phases = wave_definition.phases
    y: np.ndarray = np.zeros_like(x)
    for i, (a, φ) in enumerate(zip(amplitudes, phases)):
        k = i + 1
        y += sine(x, f, a, φ, k)

    return y


def synthesize_wave(
    wave_definition: WaveDefinition,
    sampling: Sampling,
) -> Wave:
    """
    Generate a synthetic wave from harmonic components.

    Creates time domain and composes multiple sine harmonics according to
    the wave definition.

    Args:
        wave_definition: Wave specification (base_frequency, amplitudes, phases).
        sampling: Sampling configuration (duration, rate).

    Returns:
        Wave object with time (x) and amplitude (y) arrays.
    """
    x: np.ndarray = get_domain(sampling)
    y: np.ndarray = compose_wave(x, wave_definition)
    return Wave(x, y)
