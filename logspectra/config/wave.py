from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
