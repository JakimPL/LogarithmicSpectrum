from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from logspectra.constants import BINS_PER_OCTAVE

SampleRate: TypeAlias = Literal[16000, 22050, 32000, 44100, 48000, 96000, 192000]
FFTSize: TypeAlias = Literal[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


class FFTConfig(BaseModel):
    size: FFTSize = Field(default=4096, description="Size of the FFT")
    cutoff: float = Field(default=220.0, gt=0.0, le=22050.0, description="Cutoff frequency in Hz for lower bands")
    log_even_components: int = Field(default=80, gt=0, le=1024, description="Number of log-even frequency components")
    bins_per_octave: int = Field(default=BINS_PER_OCTAVE, gt=0, le=64, description="Number of bins per octave for CQT")


class Sampling(BaseModel):
    duration: float = Field(default=0.1, gt=0.0, le=10.0, description="Duration of the signal in seconds")
    rate: SampleRate = Field(default=44100, description="Sampling rate in Hz")

    @property
    def samples(self) -> int:
        return int(np.ceil(self.duration * self.rate))


class WaveDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    coefficients: np.ndarray = Field(..., description="Array of coefficients for each sine component")
    phases: np.ndarray = Field(..., description="Array of phases for each sine component")
    base_frequency: float = Field(default=440.0, gt=0.0, le=22050.0, description="Base frequency of the wave in Hz")

    @model_validator(mode="after")
    def validate_arrays(self) -> WaveDefinition:
        if len(self.coefficients) == 0:
            raise ValueError("coefficients must be non-empty")
        if len(self.phases) == 0:
            raise ValueError("phases must be non-empty")
        if len(self.coefficients) != len(self.phases):
            raise ValueError(
                f"coefficients and phases must have the same length, got {len(self.coefficients)} and {len(self.phases)}"
            )

        return self

    def __len__(self) -> int:
        return len(self.coefficients)
