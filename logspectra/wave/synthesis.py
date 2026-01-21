import numpy as np

from logspectra.config import Sampling, WaveDefinition
from logspectra.types import Float
from logspectra.wave.wave import Wave


def sine(
    x: np.ndarray,
    frequency: Float,
    coefficient: Float,
    phase: Float,
    k: int = 1,
) -> np.ndarray:
    factor = 2.0 * k * np.pi * frequency
    return np.sin(x * factor + phase) * coefficient


def get_domain(sampling: Sampling) -> np.ndarray:
    """Get the time domain for the given sampling configuration."""
    x = np.linspace(0.0, sampling.duration, sampling.samples, endpoint=False)
    return x


def compose_wave(
    x: np.ndarray,
    wave_definition: WaveDefinition,
) -> np.ndarray:
    f = wave_definition.base_frequency
    coefficients = wave_definition.coefficients
    phases = wave_definition.phases
    y: np.ndarray = np.zeros_like(x)
    for i, (c, φ) in enumerate(zip(coefficients, phases)):
        k = i + 1
        y += sine(x, f, c, φ, k)

    return y


def synthesize_wave(
    wave_definition: WaveDefinition,
    sampling: Sampling,
) -> Wave:
    """Generate a synthetic wave composed of multiple sine components."""
    x: np.ndarray = get_domain(sampling)
    y: np.ndarray = compose_wave(x, wave_definition)
    return Wave(x, y)
