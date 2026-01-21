from __future__ import annotations

from pydantic import BaseModel, Field

from logspectra.constants import BINS_PER_OCTAVE
from logspectra.types import FFTSize


class FFTConfig(BaseModel):
    """
    Configuration for FFT/CQT and frequency spectrum transformations.

    Used to specify FFT parameters and how to transform the linear frequency
    spectrum into logarithmic bands.

    Attributes:
        size: FFT size (power of 2, from 64 to 65536).
        cutoff: Cutoff frequency in Hz below which log-even bands are used.
        log_even_components: Number of log-even frequency bins.
        bins_per_octave: Number of bins per octave for CQT (Constant-Q Transform).
            12 by default (semitone resolution).
    """

    size: FFTSize = Field(default=4096, description="Size of the FFT")
    cutoff: float = Field(default=220.0, gt=0.0, le=22050.0, description="Cutoff frequency in Hz for lower bands")
    log_even_components: int = Field(default=80, gt=0, le=1024, description="Number of log-even frequency components")
    bins_per_octave: int = Field(default=BINS_PER_OCTAVE, gt=0, le=64, description="Number of bins per octave for CQT")
