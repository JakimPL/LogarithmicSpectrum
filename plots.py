from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes

from histogram import Histogram
from interval import Interval


def get_spectrum_limits(sample_rate: int, cutoff: float) -> Tuple[Tuple[float, float], List[float]]:
    xticks = [2.0**k * cutoff for k in range(int(np.log2(sample_rate / cutoff)))]
    xrange = cutoff, 0.5 * sample_rate
    return xrange, xticks


def plot_spectrum(
    ax: Axes,
    spectrum: Histogram,
    xrange: Tuple[float, float],
    xticks: List[float],
    title: str,
    color: str = "blue",
    edge_color: Optional[str] = None,
) -> None:
    """Plot a spectrum histogram with vertical boundaries."""
    if edge_color is None:
        edge_color = color

    ax.stairs(spectrum.densities, spectrum.edges, fill=True, color=color, alpha=0.7)
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
    spectrum1: Histogram,
    spectrum2: Histogram,
    sample_rate: int,
    cutoff: float,
    title1: str,
    title2: str,
    color1: str = "green",
    color2: str = "gray",
    edge_color1: str = "darkgreen",
    edge_color2: str = "black",
) -> None:
    """Compare two spectra side by side."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    xrange, xticks = get_spectrum_limits(sample_rate, cutoff)

    plot_spectrum(ax1, spectrum1, xrange, xticks, title1, color=color1, edge_color=edge_color1)
    plot_spectrum(ax2, spectrum2, xrange, xticks, title2, color=color2, edge_color=edge_color2)

    plt.tight_layout()
    plt.show()
