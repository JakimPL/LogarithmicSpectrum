from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes

from logspectra.histogram import Histogram
from logspectra.interval import Interval
from logspectra.utils import Float


def get_spectrum_limits(sample_rate: int, cutoff: Float) -> Tuple[Tuple[float, float], List[float]]:
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
    """Plot a spectrum histogram with vertical boundaries."""
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


def compare_histogram_and_interval(
    histogram: Histogram,
    interval: Interval,
    annotation: Optional[str] = None,
    color_histogram: str = "gray",
    color_interval: str = "yellow",
) -> None:
    if not interval:
        raise ValueError("Interval must be non-empty")

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.stairs(histogram.values, histogram.edges, fill=True, color=color_histogram, alpha=0.7, label="histogram")
    for i, edge in enumerate(histogram.edges):
        ax.axvline(edge, color=color_histogram, linewidth=0.5, alpha=0.5)
        if i < len(histogram):
            x = (edge + histogram.edges[i + 1]) / 2
            y = histogram.values[i]
            plt.annotate(f"{y:2g}", (x, y), ha="center", color="#404040")

    rebin = histogram.rebin(interval)
    value = rebin.values[0]
    interval_string = ", ".join(map(str, interval))
    ax.stairs(
        rebin.values,
        rebin.edges,
        fill=True,
        color=color_interval,
        alpha=0.7,
        label=f"interval $[{interval_string}]$",
    )

    midpoint = interval.midpoint
    assert midpoint is not None

    if annotation is None:
        annotation = f"{value:2g}"

    if annotation:
        xy = (float(midpoint), float(value))
        plt.annotate(annotation, xy, ha="center", color="#C0C040")

    plt.title("Matching a single interval")
    plt.legend()
    plt.show()


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
    """Compare two spectra side by side."""
    if not len(spectra):
        raise ValueError("At least one spectrum is required")

    if len(spectra) != len(titles):
        raise ValueError("Number of spectra and titles must match")

    if colors is not None and len(colors) != len(spectra):
        raise ValueError("Number of colors must match number of spectra")

    if edge_colors is not None and len(edge_colors) != len(spectra):
        raise ValueError("Number of edge colors must match number of spectra")

    _, axes = plt.subplots(1, len(spectra), figsize=figsize, squeeze=False)
    xrange, xticks = get_spectrum_limits(sample_rate, cutoff)

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

    plt.show()
