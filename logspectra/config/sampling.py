from __future__ import annotations


import numpy as np
from pydantic import BaseModel, Field

from logspectra.types import SampleRate


class Sampling(BaseModel):
    duration: float = Field(default=0.1, gt=0.0, le=10.0, description="Duration of the signal in seconds")
    rate: SampleRate = Field(default=44100, description="Sampling rate in Hz")

    @property
    def samples(self) -> int:
        return int(np.ceil(self.duration * self.rate))
