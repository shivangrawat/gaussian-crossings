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

    A color normalizer that maps a chosen midpoint to 0.5 in the colormap,
    producing a diverging color scale.  This is used for Fano factor heatmaps
    where the Poisson reference value (F=1) should appear at the center of
    the colormap, with sub-Poissonian (F<1) and super-Poissonian (F>1) regions
    mapped to opposite ends.

    Attributes:
        midpoint: The data value that maps to 0.5 in the normalized range.
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
            midpoint: Value that maps to 0.5 in the normalized range.
            clip: Whether to clip values outside [vmin, vmax].

        Raises:
            ValueError: If vmin, vmax, or midpoint is not specified.
        """
        if vmin is None or vmax is None or midpoint is None:
            raise ValueError("vmin, vmax, and midpoint must all be specified")
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(
        self, value: Union[float, np.ndarray], clip: Optional[bool] = None
    ) -> np.ma.MaskedArray:
        """Normalize the value(s).

        Args:
            value: Value(s) to normalize.
            clip: Whether to clip (overrides instance setting if provided).

        Returns:
            Normalized value(s) in the range [0, 1].
        """
        # If the entire dataset is above the midpoint
        if self.vmin > self.midpoint:
            return np.ma.masked_array(np.interp(value, [self.vmin, self.vmax], [0.5, 1]))
        # If the entire dataset is below the midpoint
        elif self.vmax < self.midpoint:
            return np.ma.masked_array(np.interp(value, [self.vmin, self.vmax], [0, 0.5]))
        # Otherwise, use a diverging normalization with the midpoint in the center
        else:
            return np.ma.masked_array(
                np.interp(value, [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1])
            )
