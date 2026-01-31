"""
One Euro Filter for smooth, low-latency signal filtering.

The One Euro Filter is specifically designed for noisy signals from motion tracking.
It adapts its cutoff frequency based on signal speed:
- When signal is stable: applies more smoothing (low cutoff)
- When signal changes rapidly: allows more signal through (high cutoff)

This provides better smoothing than simple moving averages while maintaining
responsiveness to real movements.

Reference: Casiez, Roussel, Vogel. "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems" (CHI 2012)
"""

import math
from collections import defaultdict


class LowPassFilter:
    """Simple first-order low-pass filter."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.y = None
        self.initialized = False

    def filter(self, x):
        if not self.initialized:
            self.y = x
            self.initialized = True
            return x

        self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y

    def set_alpha(self, alpha):
        self.alpha = max(0.0, min(1.0, alpha))

    def reset(self):
        self.y = None
        self.initialized = False


class OneEuroFilter:
    """
    One Euro Filter for adaptive signal smoothing.

    Parameters:
        min_cutoff: Minimum cutoff frequency in Hz (lower = more smoothing when stable)
        beta: Speed coefficient (higher = more responsive to fast changes)
        d_cutoff: Cutoff frequency for derivative (usually 1.0)

    Typical values:
        - For smooth, slow movements: min_cutoff=0.5, beta=0.0001
        - For responsive tracking: min_cutoff=1.0, beta=0.5
        - For pose landmarks: min_cutoff=0.5, beta=0.1 (balanced)
    """

    def __init__(self, rate=30.0, min_cutoff=0.5, beta=0.1, d_cutoff=1.0):
        """
        Initialize filter.

        Args:
            rate: Expected sample rate in Hz (frames per second)
            min_cutoff: Minimum cutoff frequency (Hz) when signal is stable
            beta: Speed coefficient - higher values make filter more responsive
            d_cutoff: Cutoff frequency for the derivative filter
        """
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

        self.last_time = None
        self.last_value = None

    def _alpha(self, cutoff):
        """Compute alpha from cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, timestamp=None):
        """
        Filter a value.

        Args:
            x: The value to filter
            timestamp: Optional timestamp (for variable frame rate)

        Returns:
            Filtered value
        """
        # Handle variable frame rate
        if timestamp is not None:
            if self.last_time is not None:
                dt = timestamp - self.last_time
                if dt > 0:
                    self.rate = 1.0 / dt
            self.last_time = timestamp

        # Estimate derivative
        if self.last_value is None:
            dx = 0.0
        else:
            dx = (x - self.last_value) * self.rate

        self.last_value = x

        # Filter the derivative
        self.dx_filter.set_alpha(self._alpha(self.d_cutoff))
        edx = self.dx_filter.filter(dx)

        # Adaptive cutoff based on signal speed
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the signal
        self.x_filter.set_alpha(self._alpha(cutoff))
        return self.x_filter.filter(x)

    def reset(self):
        """Reset filter state."""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
        self.last_value = None


class LandmarkFilter:
    """
    One Euro Filter applied to MediaPipe landmarks.

    Filters each landmark's x, y, z coordinates independently.
    Maintains separate filter state for each landmark index.
    """

    def __init__(self, rate=30.0, min_cutoff=0.5, beta=0.1, d_cutoff=1.0):
        """
        Initialize landmark filter.

        Args:
            rate: Expected frame rate
            min_cutoff: Smoothing when stable
            beta: Responsiveness to fast movement
            d_cutoff: Derivative cutoff
        """
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Filters for each landmark, keyed by (landmark_idx, coordinate)
        self.filters = defaultdict(
            lambda: OneEuroFilter(rate, min_cutoff, beta, d_cutoff)
        )

    def filter_landmark(self, landmark_idx, x, y, z=None, timestamp=None):
        """
        Filter a single landmark's coordinates.

        Args:
            landmark_idx: MediaPipe landmark index
            x, y, z: Coordinates (z is optional)
            timestamp: Optional timestamp for variable frame rate

        Returns:
            Tuple of filtered (x, y) or (x, y, z)
        """
        fx = self.filters[(landmark_idx, 'x')].filter(x, timestamp)
        fy = self.filters[(landmark_idx, 'y')].filter(y, timestamp)

        if z is not None:
            fz = self.filters[(landmark_idx, 'z')].filter(z, timestamp)
            return (fx, fy, fz)

        return (fx, fy)

    def filter_landmarks(self, landmarks, indices=None, timestamp=None, use_3d=False):
        """
        Filter multiple landmarks at once.

        Args:
            landmarks: MediaPipe landmark list
            indices: List of landmark indices to filter (None = all)
            timestamp: Optional timestamp
            use_3d: Whether to include z coordinate

        Returns:
            Dict mapping landmark_idx to filtered (x, y) or (x, y, z) tuples
        """
        if indices is None:
            indices = range(len(landmarks))

        filtered = {}
        for idx in indices:
            lm = landmarks[idx]
            if use_3d:
                filtered[idx] = self.filter_landmark(
                    idx, lm.x, lm.y, lm.z, timestamp
                )
            else:
                filtered[idx] = self.filter_landmark(
                    idx, lm.x, lm.y, timestamp=timestamp
                )

        return filtered

    def reset(self):
        """Reset all filters."""
        for f in self.filters.values():
            f.reset()
        self.filters.clear()


# Pre-configured filter profiles
def create_smooth_filter(rate=30.0):
    """Create a filter optimized for smooth, stable output."""
    return LandmarkFilter(rate=rate, min_cutoff=0.3, beta=0.05, d_cutoff=1.0)


def create_responsive_filter(rate=30.0):
    """Create a filter optimized for responsive tracking."""
    return LandmarkFilter(rate=rate, min_cutoff=1.0, beta=0.5, d_cutoff=1.0)


def create_balanced_filter(rate=30.0):
    """Create a balanced filter for general pose tracking."""
    return LandmarkFilter(rate=rate, min_cutoff=0.5, beta=0.1, d_cutoff=1.0)
