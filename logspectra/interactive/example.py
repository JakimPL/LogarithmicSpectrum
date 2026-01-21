from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from logspectra.config import FFTConfig, Sampling, WaveDefinition
from logspectra.histogram import Histogram
from logspectra.plots import compare_spectra
from logspectra.spectrum import calculate_cqt_spectrum, calculate_log_spectrum, calculate_spectrum
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
        return calculate_log_spectrum(
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

    def compare(
        self,
        wave: Optional[Wave] = None,
    ) -> None:
        wave = wave or self.wave()
        spectrum = self.spectrum(wave)
        log_even_spectrum = self.log_even_spectrum(wave)
        cqt_spectrum = self.cqt_spectrum(wave)
        compare_spectra(
            spectra=(spectrum, log_even_spectrum, cqt_spectrum),
            sample_rate=self.sampling.rate,
            cutoff=self.fft_config.cutoff,
            titles=("Regular Spectral Density", "Log-Even Spectral Density", "CQT Spectral Density"),
            colors=("green", "gray", "blue"),
            figsize=(12, 3),
            draw_verticals=False,
        )
