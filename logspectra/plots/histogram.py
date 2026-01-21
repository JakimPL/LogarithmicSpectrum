from typing import Optional

import matplotlib.pyplot as plt

from logspectra.histogram import Histogram, Interval


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
