from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes

from logspectra.histogram import Histogram
from logspectra.types import Float


def get_spectrum_limits(sample_rate: int, cutoff: Float) -> Tuple[Tuple[float, float], List[float]]:
    """
    Calculate plot limits and tick positions for logarithmic frequency axis.

    Generates octave-based tick positions from cutoff to Nyquist frequency.

    Args:
        sample_rate: Sampling rate in Hz.
        cutoff: Minimum frequency in Hz.

    Returns:
        Tuple of (xrange, xticks) where xrange is (min, max) frequency range
        and xticks is a list of frequencies at powers of 2 times the cutoff.
    """
    xticks = [float(2.0**k * cutoff) for k in range(int(np.log2(sample_rate / cutoff)))]
    xrange = float(cutoff), 0.5 * sample_rate
    return xrange, xticks


def plot_spectrum(
    ax: Axes,
    spectrum: Histogram,
    xrange: Tuple[float, float],
    xticks: List[float],
    title: str,
    color: str = "blue",
    edge_color: Optional[str] = None,
    draw_verticals: bool = True,
) -> None:
    """
    Plot a frequency spectrum on a logarithmic scale.

    Args:
        ax: Matplotlib axes to plot on.
        spectrum: Spectrum histogram (edges in Hz, values as energy density).
        xrange: Frequency range (min, max) for x-axis limits.
        xticks: Frequency values for x-axis tick positions.
        title: Plot title.
        color: Fill color for the spectrum.
        edge_color: Color for vertical bin edge lines. Defaults to color.
        draw_verticals: Whether to draw vertical lines at bin edges.
    """
    if edge_color is None:
        edge_color = color

    ax.stairs(spectrum.densities, spectrum.edges, fill=True, color=color, alpha=0.7)
    if draw_verticals:
        for edge in spectrum.edges:
            ax.axvline(edge, color=edge_color, linewidth=0.5, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Energy² / Hz")
    ax.set_xlim(*xrange)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    ax.set_title(title)


def compare_spectra(
    spectra: Sequence[Histogram],
    sample_rate: int,
    cutoff: Float,
    titles: Sequence[str],
    colors: Optional[Sequence[str]] = None,
    edge_colors: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    draw_verticals: bool = True,
) -> None:
    """
    Compare multiple spectra side by side.

    Creates a figure with subplots for each spectrum, using consistent y-axis
    scaling for easy comparison.

    Args:
        spectra: Sequence of spectrum histograms to compare.
        sample_rate: Sampling rate in Hz.
        cutoff: Minimum frequency in Hz for x-axis range.
        titles: Title for each subplot (must match length of spectra).
        colors: Fill colors for each spectrum. Defaults to blue.
        edge_colors: Edge line colors for each spectrum. Defaults to colors.
        figsize: Figure size (width, height) in inches.
        draw_verticals: Whether to draw vertical lines at bin edges.

    Raises:
        ValueError: If spectra is empty, or if lengths of titles, colors,
            or edge_colors don't match the number of spectra.
    """
    if not len(spectra):
        raise ValueError("At least one spectrum is required")

    if len(spectra) != len(titles):
        raise ValueError("Number of spectra and titles must match")

    if colors is not None and len(colors) != len(spectra):
        raise ValueError("Number of colors must match number of spectra")

    if edge_colors is not None and len(edge_colors) != len(spectra):
        raise ValueError("Number of edge colors must match number of spectra")

    _, axes = plt.subplots(
        1,
        len(spectra),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    xrange, xticks = get_spectrum_limits(sample_rate, cutoff)

    y_max = max(np.max(spectrum.densities) for spectrum in spectra if len(spectrum.densities) > 0)
    y_lim = (0, y_max * 1.1)

    for i, spectrum in enumerate(spectra):
        ax = axes[0, i]
        title = titles[i]
        color = colors[i] if colors is not None and i < len(colors) else "blue"
        edge_color = edge_colors[i] if edge_colors is not None and i < len(edge_colors) else color
        plot_spectrum(
            ax,
            spectrum,
            xrange,
            xticks,
            title,
            color=color,
            edge_color=edge_color,
            draw_verticals=draw_verticals,
        )
        ax.set_ylim(*y_lim)

    plt.show()
