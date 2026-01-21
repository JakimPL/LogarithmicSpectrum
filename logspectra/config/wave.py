from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class WaveDefinition(BaseModel):
    """
    Definition of a composite waveform from multiple sine components.

    Specifies a waveform as a sum of sine waves with different frequencies,
    each with its own coefficient (amplitude) and phase. The frequency of
    component i is base_frequency * (i + 1).

    Attributes:
        amplitudes: Amplitude coefficients for each harmonic component.
        phases: Phase shifts (in radians) for each harmonic component.
        base_frequency: Fundamental frequency in Hz (up to Nyquist frequency).

    Examples:
        >>> # Pure A440 tone
        >>> wave = WaveDefinition(amplitudes=np.array([1.0]), phases=np.array([0.0]), base_frequency=440.0)
        >>> len(wave)
        1
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    amplitudes: np.ndarray = Field(..., description="Array of amplitudes for each sine component")
    phases: np.ndarray = Field(..., description="Array of phases for each sine component")
    base_frequency: float = Field(default=440.0, gt=0.0, le=22050.0, description="Base frequency of the wave in Hz")

    @model_validator(mode="after")
    def validate_arrays(self) -> WaveDefinition:
        """
        Validate that coefficients and phases are non-empty and equal length.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If arrays are empty or have mismatched lengths.
        """
        if len(self.amplitudes) == 0:
            raise ValueError("coefficients must be non-empty")
        if len(self.phases) == 0:
            raise ValueError("phases must be non-empty")
        if len(self.amplitudes) != len(self.phases):
            raise ValueError(
                f"coefficients and phases must have the same length, got {len(self.amplitudes)} and {len(self.phases)}"
            )

        return self

    def __len__(self) -> int:
        """
        Number of harmonic components in the waveform.

        Returns:
            Length of the coefficients array.
        """
        return len(self.amplitudes)
