from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from logspectra.config import FFTConfig, Sampling, WaveDefinition
from logspectra.cqt import calculate_cqt_spectrum
from logspectra.fft import calculate_log_even_spectrum, calculate_spectrum
from logspectra.histogram import Histogram
from logspectra.wave import Wave, synthesize_wave


class Example(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wave_definition: WaveDefinition = Field(..., description="Definition of the synthetic wave")
    fft_config: FFTConfig = Field(default_factory=FFTConfig, description="FFT configuration")
    sampling: Sampling = Field(default_factory=Sampling, description="Sampling configuration")

    def wave(self) -> Wave:
        return synthesize_wave(self.wave_definition, self.sampling)

    def spectrum(self, wave: Optional[Wave] = None) -> Histogram:
        return calculate_spectrum(
            wave or self.wave(),
            self.fft_config,
            self.sampling,
        )

    def log_even_spectrum(self, wave: Optional[Wave] = None) -> Histogram:
        return calculate_log_even_spectrum(
            wave or self.wave(),
            self.fft_config,
            self.sampling,
        )

    def cqt_spectrum(self, wave: Optional[Wave] = None) -> Histogram:
        return calculate_cqt_spectrum(
            wave or self.wave(),
            self.fft_config,
            self.sampling,
        )
