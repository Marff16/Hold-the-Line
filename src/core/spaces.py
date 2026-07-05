"""Small compatibility layer for Gymnasium spaces.

The environment exposes Gymnasium ``Box`` spaces when Gymnasium is installed.
For lightweight local smoke tests, this module provides a tiny fallback with the
subset of behavior used by the MVP.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


try:  # pragma: no cover - exercised only when Gymnasium is installed.
    from gymnasium import spaces

    Box = spaces.Box
except ModuleNotFoundError:  # pragma: no cover - fallback is covered instead.

    class Box:
        """Minimal Box space compatible with this project's tests and demos."""

        def __init__(self, low, high, shape: Iterable[int] | None = None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            if shape is None:
                low_arr = np.array(low, dtype=self.dtype)
                high_arr = np.array(high, dtype=self.dtype)
                self.shape = low_arr.shape
            else:
                self.shape = tuple(shape)
                low_arr = np.full(self.shape, low, dtype=self.dtype)
                high_arr = np.full(self.shape, high, dtype=self.dtype)

            self.low = low_arr
            self.high = high_arr

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x) -> bool:
            arr = np.asarray(x, dtype=self.dtype)
            return arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high)

