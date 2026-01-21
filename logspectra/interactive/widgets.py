import warnings
from typing import get_args

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import FloatSlider, IntSlider, SelectionSlider, interact

from logspectra.config import FFTConfig, Sampling, WaveDefinition
from logspectra.interactive.example import Example
from logspectra.types import FFTSize, SampleRate


def interactive_example(wave_definition: WaveDefinition) -> None:
    """
    Create interactive widgets to control Example parameters and redraw comparison.

    Args:
        wave_definition: The wave definition to use for all comparisons
    """
    warnings.filterwarnings("ignore")

    valid_sample_rates = get_args(SampleRate)
    valid_fft_sizes = get_args(FFTSize)

    output = widgets.Output()

    def update_plot(
        sample_rate: SampleRate,
        fft_size: FFTSize,
        cutoff: float,
        log_even_components: int,
        bins_per_octave: int,
    ) -> None:
        """Update the comparison plot with new parameters."""
        with output:
            output.clear_output(wait=True)
            sampling = Sampling(rate=sample_rate)
            fft_config = FFTConfig(
                size=fft_size,
                cutoff=cutoff,
                log_even_components=log_even_components,
                bins_per_octave=bins_per_octave,
            )
            example = Example(
                wave_definition=wave_definition,
                fft_config=fft_config,
                sampling=sampling,
            )
            example.compare()
            plt.close("all")

    sample_rate_widget = SelectionSlider(
        options=valid_sample_rates,
        value=44100,
        description="Sample Rate (Hz):",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px"),
    )

    fft_size_widget = SelectionSlider(
        options=valid_fft_sizes,
        value=4096,
        description="FFT Size:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px"),
    )

    cutoff_widget = FloatSlider(
        min=20.0,
        max=2000.0,
        step=10.0,
        value=220.0,
        description="Cutoff (Hz):",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px"),
    )

    log_even_widget = IntSlider(
        min=8,
        max=128,
        step=1,
        value=80,
        description="Log-Even Bins (Log-Even only):",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px"),
    )

    bins_per_octave_widget = IntSlider(
        min=1,
        max=36,
        step=1,
        value=12,
        description="Bins per Octave (CQT only):",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px"),
    )

    display(output)  # type: ignore

    interact(
        update_plot,
        sample_rate=sample_rate_widget,
        fft_size=fft_size_widget,
        cutoff=cutoff_widget,
        log_even_components=log_even_widget,
        bins_per_octave=bins_per_octave_widget,
    )
