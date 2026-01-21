from __future__ import annotations


from pydantic import BaseModel, Field

from logspectra.constants import BINS_PER_OCTAVE
from logspectra.types import FFTSize


class FFTConfig(BaseModel):
    size: FFTSize = Field(default=4096, description="Size of the FFT")
    cutoff: float = Field(default=220.0, gt=0.0, le=22050.0, description="Cutoff frequency in Hz for lower bands")
    log_even_components: int = Field(default=80, gt=0, le=1024, description="Number of log-even frequency components")
    bins_per_octave: int = Field(default=BINS_PER_OCTAVE, gt=0, le=64, description="Number of bins per octave for CQT")
