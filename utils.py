from typing import Optional, Tuple, TypeAlias, Union

import numpy as np

from constants import C1_FREQUENCY

Float: TypeAlias = Union[float, np.floating]


def is_increasing(array: np.ndarray) -> bool:
    return bool(np.all(np.diff(array) > 0))


def sine(
    x: np.ndarray,
    frequency: Float,
    coefficient: Float,
    phase: Float,
    k: int = 1,
) -> np.ndarray:
    factor = 2.0 * k * np.pi * frequency
    return np.sin(x * factor + phase) * coefficient


def compose(
    x: np.ndarray,
    base_frequency: Float,
    coefficents: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    wave: np.ndarray = np.zeros_like(x)
    for i, (c, φ) in enumerate(zip(coefficents, phases)):
        k = i + 1
        wave += sine(x, base_frequency, c, φ, k)

    return wave


def synthesize_wave(
    duration: Float,
    sample_rate: Float,
    base_frequency: Float,
    coefficients: np.ndarray,
    phases: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic wave composed of multiple sine components."""
    samples = int(duration * sample_rate)
    x = np.linspace(0.0, duration, samples, endpoint=False)
    wave: np.ndarray = compose(x, base_frequency, coefficients, phases)
    return x, wave


def rectangle(length: int) -> np.ndarray:
    return np.ones(length, dtype=float)


def to_log_even_bands(
    bands: np.ndarray,
    size: Optional[int] = None,
    cutoff: Float = C1_FREQUENCY,
) -> np.ndarray:
    size = size or len(bands)
    log_even_bands: np.ndarray = np.exp(np.linspace(np.log(cutoff), np.log(bands[-1]), size))
    return log_even_bands
