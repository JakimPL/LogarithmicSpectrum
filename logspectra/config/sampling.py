from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from logspectra.types import SampleRate


class Sampling(BaseModel):
    """
    Audio sampling configuration.

    Defines the sampling rate and duration for audio signal generation and processing.

    Attributes:
        duration: Duration of the signal in seconds (0 to 10).
        rate: Sampling rate in Hz (standard rates like 44100, 48000, etc.).
    """

    duration: float = Field(default=0.1, gt=0.0, le=10.0, description="Duration of the signal in seconds")
    rate: SampleRate = Field(default=44100, description="Sampling rate in Hz")

    @property
    def samples(self) -> int:
        """
        Total number of samples for the configured duration and rate.

        Returns:
            Ceiling of duration * rate.
        """
        return int(np.ceil(self.duration * self.rate))
