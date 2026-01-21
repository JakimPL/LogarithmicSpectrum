from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from logspectra.config import FFTConfig, Sampling, WaveDefinition
from logspectra.histogram import Histogram
from logspectra.plots import compare_spectra
from logspectra.spectrum import calculate_cqt_spectrum, calculate_log_spectrum, calculate_spectrum
from logspectra.wave import Wave, synthesize_wave


class Example(BaseModel):
    """
    Example demonstrating different spectrum computation methods.

    Provides convenient methods to generate a synthetic wave and compute its
    spectrum using different approaches: regular FFT, log-even rebinning, and CQT.

    Attributes:
        wave_definition: Definition of the synthetic wave (harmonics, phases, frequency).
        fft_config: FFT configuration (size, cutoff, log components, bins per octave).
        sampling: Sampling configuration (duration, rate).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wave_definition: WaveDefinition = Field(..., description="Definition of the synthetic wave")
    fft_config: FFTConfig = Field(default_factory=FFTConfig, description="FFT configuration")
    sampling: Sampling = Field(default_factory=Sampling, description="Sampling configuration")

    def wave(self) -> Wave:
        """
        Generate the synthetic wave from the wave definition.

        Returns:
            Synthesized wave with time and amplitude arrays.
        """
        return synthesize_wave(self.wave_definition, self.sampling)

    def spectrum(self, wave: Optional[Wave] = None) -> Histogram:
        """
        Calculate the regular FFT power spectrum.

        Args:
            wave: Optional wave to analyze. If None,
                generates from wave_definition.

        Returns:
            Histogram with linearly-spaced frequency bins.
        """
        return calculate_spectrum(
            wave or self.wave(),
            self.fft_config,
            self.sampling,
        )

    def log_even_spectrum(self, wave: Optional[Wave] = None) -> Histogram:
        """
        Calculate the spectrum with logarithmically-spaced bins via rebinning.

        Args:
            wave: Optional wave to analyze. If None,
                generates from wave_definition.

        Returns:
            Histogram with log-spaced frequency bins from FFT rebinning.
        """
        return calculate_log_spectrum(
            wave or self.wave(),
            self.fft_config,
            self.sampling,
        )

    def cqt_spectrum(self, wave: Optional[Wave] = None) -> Histogram:
        """
        Calculate the Constant-Q Transform spectrum.

        Args:
            wave: Optional wave to analyze. If None,
                generates from wave_definition.

        Returns:
            Histogram with log-spaced frequency bins from CQT computation.
        """
        return calculate_cqt_spectrum(
            wave or self.wave(),
            self.fft_config,
            self.sampling,
        )

    def compare(
        self,
        wave: Optional[Wave] = None,
    ) -> None:
        """
        Compare all three spectrum computation methods side by side.

        Generates a figure with three subplots showing regular FFT,
        log-even rebinned, and CQT spectra for visual comparison.

        Args:
            wave: Optional wave to analyze. If None,
                generates from wave_definition.
        """
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
