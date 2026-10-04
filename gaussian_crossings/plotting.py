"""Optional plotting helpers (requires matplotlib).

Install with ``pip install "gaussian-crossings[plot]"``. The rest of the package
does not import matplotlib.
"""

from typing import Optional, Union

import numpy as np

try:
    import matplotlib.colors as mcolors
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "gaussian_crossings.plotting requires matplotlib; install it with "
        'pip install "gaussian-crossings[plot]".'
    ) from exc

__all__ = ["MidpointNormalize"]


class MidpointNormalize(mcolors.Normalize):
    """Matplotlib normalizer with a specified midpoint for diverging colormaps.

    A color normalizer that maps an in-range midpoint to 0.5 in the colormap,
    producing a diverging color scale.  This is used for Fano factor heatmaps
    where the Poisson reference value (F=1) should appear at the center of
    the colormap, with sub-Poissonian (F<1) and super-Poissonian (F>1) regions
    mapped to opposite ends.

    When the entire range is above or below the midpoint, it occupies the
    upper or lower half of the colormap, respectively. Values outside the
    bounds are linearly extrapolated unless ``clip=True``. Input masks are
    preserved.

    For equal bounds, all values map to 0.5 if the bound equals the midpoint,
    or to the center of the appropriate half (0.25 below, 0.75 above).
    The inverse then returns the sole bound for every color coordinate.

    Attributes:
        midpoint: The data value that maps to 0.5 when it lies within the bounds;
            otherwise it selects which half of the colormap is used.
    """

    def __init__(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        midpoint: Optional[float] = None,
        clip: bool = False,
    ) -> None:
        """Initialize the normalizer.

        Args:
            vmin: Minimum data value.
            vmax: Maximum data value.
            midpoint: Center value, or the reference that selects the upper/lower
                half of the colormap when it lies outside the bounds.
            clip: Whether to clip values outside [vmin, vmax].

        Raises:
            ValueError: If a bound or midpoint is missing or nonfinite, or
                vmin exceeds vmax.
        """
        if vmin is None or vmax is None or midpoint is None:
            raise ValueError("vmin, vmax, and midpoint must all be specified")
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)
        self._knots()

    def _knots(self):
        """Return the data and color coordinates of the piecewise linear map."""
        if not np.isfinite([self.vmin, self.midpoint, self.vmax]).all():
            raise ValueError("vmin, vmax, and midpoint must be finite")
        if self.vmin > self.vmax:
            raise ValueError("vmin must not exceed vmax")
        if self.vmin == self.vmax:
            color = 0.5
            if self.vmin > self.midpoint:
                color = 0.75
            elif self.vmin < self.midpoint:
                color = 0.25
            return [self.vmin], [color]
        if self.midpoint <= self.vmin:
            return [self.vmin, self.vmax], [0.5, 1.0]
        if self.midpoint >= self.vmax:
            return [self.vmin, self.vmax], [0.0, 0.5]
        return [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]

    @staticmethod
    def _interpolate(values, x, y):
        """Interpolate internally and extend the first/last segment outside."""
        if len(x) == 1:
            return np.full_like(values, y[0], dtype=float)
        result = np.interp(values, x, y)
        result = np.where(
            values < x[0], y[0] + (values - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), result
        )
        return np.where(
            values > x[-1],
            y[-1] + (values - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]),
            result,
        )

    def __call__(
        self, value: Union[float, np.ndarray], clip: Optional[bool] = None
    ) -> np.ma.MaskedArray:
        """Normalize the value(s).

        Args:
            value: Value(s) to normalize.
            clip: Whether to clip (overrides instance setting if provided).

        Returns:
            Normalized value(s), preserving masks and scalar inputs. Values
            may lie outside [0, 1] when clipping is disabled.
        """
        data, colors = self._knots()
        if np.ndim(value) == 0 and np.ma.is_masked(value):
            return np.ma.masked
        values, scalar = self.process_value(value)
        clip = self.clip if clip is None else clip
        raw = np.clip(values.data, self.vmin, self.vmax) if clip else values.data
        result = np.ma.array(self._interpolate(raw, data, colors), mask=values.mask)
        return result[0] if scalar else result

    def inverse(self, value):
        """Map color coordinates back to data, preserving masks and shape.

        With distinct bounds, this inverts the unclipped map, including
        extrapolation. Clipping discards out-of-range information, so clipped
        values invert to a bound. Equal bounds map every input to the sole bound.
        """
        data, colors = self._knots()
        if np.ndim(value) == 0 and np.ma.is_masked(value):
            return np.ma.masked
        values, scalar = self.process_value(value)
        result = np.ma.array(self._interpolate(values.data, colors, data), mask=values.mask)
        return result[0] if scalar else result
