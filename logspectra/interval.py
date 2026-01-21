from __future__ import annotations

from typing import List, NamedTuple, Optional, Self

import numpy as np

from logspectra.utils import Float, is_increasing


class Interval(NamedTuple):
    left: Float
    right: Float

    def __bool__(self) -> bool:
        return bool(self.left < self.right)

    @property
    def length(self) -> Float:
        if not bool(self):
            return 0.0

        return self.right - self.left

    @property
    def midpoint(self) -> Optional[Float]:
        if not bool(self):
            return None

        return 0.5 * (self.left + self.right)

    def intersection(self, other: Self) -> Self:
        if not isinstance(other, Interval):
            raise TypeError(f"Expected Interval, got {type(other)}")

        left = np.maximum(self.left, other.left)
        right = np.minimum(self.right, other.right)
        return self.__class__(left, right)

    def contains(self, other: Self) -> bool:
        return bool(self.left <= other.left and self.right >= other.right)

    def relative_measure(self, other: Self) -> Float:
        if not isinstance(other, Interval):
            raise TypeError(f"Expected Interval, got {type(other)}")

        if not bool(self):
            return 0.0

        return self.intersection(other).length / self.length

    @classmethod
    def unit(cls) -> Self:
        """Create a unit interval [0, 1]."""
        return cls(0.0, 1.0)

    @classmethod
    def from_edges(cls, edges: np.ndarray) -> List[Self]:
        """Create a list of Intervals from an array of edges."""
        if len(edges) < 2:
            raise ValueError("At least two edges are required to create intervals")

        if not is_increasing(edges):
            raise ValueError("Edges need to be strictly increasing")

        return [cls(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
