from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Iterator, List, Tuple, Union

import numpy as np

from logspectra.histogram.interval import Interval
from logspectra.histogram.utils import is_increasing


@dataclass
class Histogram:
    edges: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.values) + 1:
            raise ValueError("edges should have exactly |values| + 1 elements")

        if len(self.edges) < 2:
            raise ValueError("At least two edges are required to create a histogram")

        if not is_increasing(self.edges):
            raise ValueError("edges need to be strictly increasing")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Histogram):
            return False

        edges_equal = np.array_equal(self.edges, other.edges)
        values_equal = np.array_equal(self.values, other.values)
        return edges_equal and values_equal

    def __hash__(self) -> int:
        return hash((tuple(self.edges), tuple(self.values)))

    def __iter__(self) -> Iterator[Tuple[Interval, np.floating]]:
        for i in range(len(self)):
            yield self.interval(i), self.values[i]

    def __len__(self) -> int:
        return len(self.values)

    def interval(self, i: int) -> Interval:
        """i-th interval of the histogram."""
        if not 0 <= i < len(self):
            raise IndexError(f"Index {i} out of bounds")

        return Interval(self.edges[i], self.edges[i + 1])

    def rebin(
        self,
        target_bins: Union[Interval, np.ndarray, Histogram],
    ) -> Histogram:
        """Rebin the histogram to the target bins."""
        edges: np.ndarray
        if isinstance(target_bins, Interval):
            edges = np.array([target_bins.left, target_bins.right])

        elif isinstance(target_bins, np.ndarray):
            if not is_increasing(target_bins):
                raise ValueError("array of edges need to be strictly increasing")

            edges = target_bins.copy()

        elif isinstance(target_bins, Histogram):
            histogram: Histogram = target_bins
            edges = histogram.edges.copy()

        else:
            raise TypeError(
                f"Unsupported target_bins, expected Interval, np.ndarray, or Histogram, got {type(target_bins)}"
            )

        self.validate_overlap(edges)
        return self._rebin(edges)

    def validate_overlap(self, edges: np.ndarray) -> None:
        """Check if the target edges overlap with the histogram range."""
        edges_range: Interval = Interval(edges[0], edges[-1])
        if not edges_range.contains(self.range):
            warnings.warn(
                "Rebinning to intervals outside of the histogram range may lead to unexpected results",
                RuntimeWarning,
            )

    def _rebin(self, target_bins: np.ndarray) -> Histogram:
        """A more efficient way to rebin using numpy interpolation."""
        if not isinstance(target_bins, np.ndarray):
            raise TypeError(f"target_bins must be a numpy array, got {type(target_bins)}")

        if not is_increasing(target_bins):
            raise ValueError("array of edges need to be strictly increasing")

        cumsum: np.ndarray = np.concatenate([[0], np.cumsum(self.values)])
        interpolation: np.ndarray = np.interp(
            target_bins,
            self.edges,
            cumsum,
            left=0,
            right=cumsum[-1],
        )
        values: np.ndarray = np.diff(interpolation)
        return Histogram(edges=target_bins, values=values)

    @cached_property
    def range(self) -> Interval:
        """Overall range of the histogram."""
        return Interval(self.edges[0], self.edges[-1])

    @cached_property
    def widths(self) -> np.ndarray:
        """Widths of the histogram bins."""
        return np.diff(self.edges)

    @cache
    def width(self, i: int) -> np.floating:
        """Width of the i-th bin."""
        width: np.floating = self.widths[i]
        return width

    @cache
    def density(self, i: int) -> np.floating:
        """Density in the i-th bin: value / interval length."""
        interval = self.interval(i)
        if not interval:
            zero: np.floating = self.values.dtype.type(0.0)
            return zero

        density: np.floating = self.values[i] / interval.length
        return density

    @cached_property
    def densities(self) -> np.ndarray:
        """Densities for all bins."""
        densities: List[np.floating] = [self.density(i) for i in range(len(self))]
        return np.array(densities, dtype=self.values.dtype)
