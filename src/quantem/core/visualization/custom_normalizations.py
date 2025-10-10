from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from matplotlib import colors
from numpy.typing import NDArray

"""
Custom normalization based on astropy's visualization routines.

Original implementation:
https://github.com/astropy/astropy/blob/main/astropy/visualization/mpl_normalize.py

Licensed under a 3-clause BSD style license.
"""


class BaseInterval(ABC):
    """
    Base class for the interval classes, which when called with an array of values,
    return an interval clipped to the [0:1] range.
    """

    @abstractmethod
    def get_limits(self, values: NDArray) -> tuple[float, float]:
        """
        Get the minimum and maximum values for the interval.
        This method must be implemented by subclasses.

        Parameters
        ----------
        values : array-like
            The input values.

        Returns
        -------
        vmin, vmax : float
            The minimum and maximum values.
        """
        raise NotImplementedError("Subclasses must implement get_limits")

    def __call__(self, values: NDArray) -> NDArray:
        """
        Transform values using this interval.

        Parameters
        ----------
        values : array-like
            The input values.

        Returns
        -------
        result : ndarray
            The transformed values.
        """
        vmin, vmax = self.get_limits(values)

        # subtract vmin
        values = np.subtract(values, vmin)
        if np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.float64)
        # divide by interval
        if (vmax - vmin) != 0.0:
            np.true_divide(values, vmax - vmin, out=values)

        # clip to [0:1]
        np.clip(values, 0.0, 1.0, out=values)
        return values

    def inverse(self, values: NDArray) -> NDArray:
        """
        Pseudo-inverse interval transform. Note this does not recover
        the original range due to clipping. Used for colorbars.

        Parameters
        ----------
        values : array-like
            The input values.

        Returns
        -------
        result : ndarray
            The transformed values.
        """
        vmin, vmax = self.get_limits(values)

        values = np.multiply(values, vmax - vmin)
        np.add(values, vmin, out=values)
        return values


@dataclass
class ManualInterval(BaseInterval):
    """
    Interval based on user-specified values.

    Parameters
    ----------
    vmin : float, optional
        The minimum value in the scaling.
    vmax : float, optional
        The maximum value in the scaling.
    """

    vmin: float | None = None
    vmax: float | None = None

    def get_limits(self, values: NDArray) -> tuple[float, float]:
        # Avoid overhead of preparing array if both limits have been specified
        # manually, for performance.

        if self.vmin is not None and self.vmax is not None:
            return self.vmin, self.vmax

        # Make sure values is a Numpy array
        values = np.asarray(values).ravel()

        # Filter out invalid values (inf, nan)
        values = values[np.isfinite(values)]
        vmin = np.min(values) if self.vmin is None else self.vmin
        vmax = np.max(values) if self.vmax is None else self.vmax

        return vmin, vmax


@dataclass
class CenteredInterval(BaseInterval):
    """
    Centered interval based on user-specified halfrange.

    Parameters
    ----------
    vcenter : float
        The center value in the scaling.
    half_range : float, optional
        The half range in the scaling.
    """

    vcenter: float = 0.0
    half_range: float | None = None

    def get_limits(self, values: NDArray) -> tuple[float, float]:
        if self.half_range is not None:
            return self.vcenter - self.half_range, self.vcenter + self.half_range

        values = np.asarray(values).ravel()
        values = values[np.isfinite(values)]
        vmin = np.min(values)
        vmax = np.max(values)

        half_range = np.maximum(np.abs(vmin - self.vcenter), np.abs(vmax - self.vcenter))

        return self.vcenter - half_range, self.vcenter + half_range


@dataclass
class QuantileInterval(BaseInterval):
    """
    Interval based on a keeping a specified fraction of pixels.

    Parameters
    ----------
    lower_quantile : float or None
        The lower quantile below which to ignore pixels. If None, then
        defaults to 0.
    upper_quantile : float or None
        The upper quantile above which to ignore pixels. If None, then
        defaults to 1.
    """

    lower_quantile: float = 0.02
    upper_quantile: float = 0.98

    def get_limits(self, values: NDArray) -> tuple[float, float]:
        # Make sure values is a Numpy array
        values = np.asarray(values).ravel()

        # Filter out invalid values (inf, nan)
        values = values[np.isfinite(values)]
        vmin, vmax = np.quantile(values, (self.lower_quantile, self.upper_quantile))  # type: ignore

        return vmin, vmax


@dataclass
class LinearStretch:
    r"""
    A linear stretch with a slope and offset.

    The stretch is given by:

    .. math::
        y = slope * x + intercept

    Parameters
    ----------
    slope : float, optional
        The ``slope`` parameter used in the above formula.  Default is 1.
    intercept : float, optional
        The ``intercept`` parameter used in the above formula.  Default is 0.
    """

    slope: float = 1.0
    intercept: float = 0.0

    def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
        if self.slope == 1.0 and self.intercept == 0.0:
            return values

        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        if self.slope != 1.0:
            np.multiply(values, self.slope, out=values)
        if self.intercept != 0.0:
            np.add(values, self.intercept, out=values)
        return values

    @property
    def inverse(self) -> "LinearStretch":
        return LinearStretch(1 / self.slope, -self.intercept / self.slope)


@dataclass
class PowerLawStretch:
    r"""
    A power stretch.

    The stretch is given by:

    .. math::
        y = x^{power}

    Parameters
    ----------
    power : float
        The power index (see the above formula).  ``power`` must be greater
        than 0.
    """

    power: float = 1.0

    def __post_init__(self) -> None:
        if self.power <= 0.0:
            raise ValueError("power must be > 0")

    def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
        if self.power == 1.0:
            return values

        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        np.power(values, self.power, out=values)
        return values

    @property
    def inverse(self) -> "PowerLawStretch":
        return PowerLawStretch(1.0 / self.power)


@dataclass
class LogarithmicStretch:
    r"""
    A logarithmic stretch.

    The stretch is given by:

    .. math::
        y = \frac{\log{(a x + 1)}}{\log{(a + 1)}}

    Parameters
    ----------
    a : float
        The ``a`` parameter used in the above formula.  ``a`` must be
        greater than 0.  Default is 1000.
    """

    a: float = 1000.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        np.multiply(values, self.a, out=values)
        np.add(values, 1.0, out=values)
        np.log(values, out=values)
        np.true_divide(values, np.log(self.a + 1.0), out=values)
        return values

    @property
    def inverse(self) -> "InverseLogarithmicStretch":
        return InverseLogarithmicStretch(self.a)


@dataclass
class InverseLogarithmicStretch:
    r"""
    Inverse transformation for `LogarithmicStretch`.

    The stretch is given by:

    .. math::
        y = \frac{e^{y \log{a + 1}} - 1}{a} \\
        y = \frac{e^{y} (a + 1) - 1}{a}

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula.  ``a`` must be
        greater than 0.  Default is 1000.
    """

    a: float = 1000.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        np.multiply(values, np.log(self.a + 1.0), out=values)
        np.exp(values, out=values)
        np.subtract(values, 1.0, out=values)
        np.true_divide(values, self.a, out=values)
        return values

    @property
    def inverse(self) -> "LogarithmicStretch":
        return LogarithmicStretch(self.a)


@dataclass
class InverseHyperbolicSineStretch:
    r"""
    An asinh stretch.

    The stretch is given by:

    .. math::
        y = \frac{{\rm asinh}(x / a)}{{\rm asinh}(1 / a)}.

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula. The value of this
        parameter is where the asinh curve transitions from linear to
        logarithmic behavior, expressed as a fraction of the normalized
        image. The stretch becomes more linear as the ``a`` value is
        increased. ``a`` must be greater than 0. Default is 0.1.
    """

    a: float = 0.1

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)
        # map to [-1,1]
        np.multiply(values, 2.0, out=values)
        np.subtract(values, 1.0, out=values)

        np.true_divide(values, self.a, out=values)
        np.arcsinh(values, out=values)

        # map from [-1,1]
        np.true_divide(values, np.arcsinh(1.0 / self.a) * 2.0, out=values)
        np.add(values, 0.5, out=values)
        return values

    @property
    def inverse(self) -> "HyperbolicSineStretch":
        return HyperbolicSineStretch(1.0 / np.arcsinh(1.0 / self.a))


@dataclass
class HyperbolicSineStretch:
    r"""
    A sinh stretch.

    The stretch is given by:

    .. math::
        y = \frac{{\rm sinh}(x / a)}{{\rm sinh}(1 / a)}

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula. The stretch
        becomes more linear as the ``a`` value is increased. ``a`` must
        be greater than 0. Default is 1/3.
    """

    a: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be > 0")

    def __call__(self, values: NDArray, copy: bool = True) -> NDArray:
        values = np.array(values, copy=copy)
        np.clip(values, 0.0, 1.0, out=values)

        # map to [-1,1]
        np.subtract(values, 0.5, out=values)
        np.multiply(values, 2.0, out=values)

        np.true_divide(values, self.a, out=values)
        np.sinh(values, out=values)

        # map from [-1,1]
        np.true_divide(values, np.sinh(1.0 / self.a) * 2.0, out=values)
        np.add(values, 0.5, out=values)
        return values

    @property
    def inverse(self) -> "InverseHyperbolicSineStretch":
        return InverseHyperbolicSineStretch(1.0 / np.sinh(1.0 / self.a))


class CustomNormalization(colors.Normalize):
    """A flexible normalization class that combines interval and stretch operations.

    This class extends matplotlib's Normalize class to provide more sophisticated
    normalization options for visualization. It combines an interval operation
    (which maps data to a [0,1] range) with a stretch operation (which applies
    a transformation to the normalized data).

    Parameters
    ----------
    interval_type : str, default="quantile"
        Type of interval to use. Options are "quantile", "manual", or "centered".
    stretch_type : str, default="linear"
        Type of stretch to apply. Options are "linear", "power", "logarithmic", or "asinh".
    data : ndarray, optional
        Data array to use for setting limits if not explicitly provided.
    lower_quantile : float, default=0.02
        Lower quantile for "quantile" interval type.
    upper_quantile : float, default=0.98
        Upper quantile for "quantile" interval type.
    vmin : float, optional
        Minimum value for "manual" interval type.
    vmax : float, optional
        Maximum value for "manual" interval type.
    vcenter : float, default=0.0
        Center value for "centered" interval type.
    half_range : float, optional
        Half range for "centered" interval type.
    power : float, default=1.0
        Power for "power" stretch type.
    logarithmic_index : float, default=1000.0
        Index for "logarithmic" stretch type.
    asinh_linear_range : float, default=0.1
        Linear range for "asinh" stretch type.
    """

    def __init__(
        self,
        interval_type: str = "quantile",
        stretch_type: str = "linear",
        *,
        data: NDArray | None = None,
        lower_quantile: float = 0.02,
        upper_quantile: float = 0.98,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float = 0.0,
        half_range: float | None = None,
        power: float = 1.0,
        logarithmic_index: float = 1000.0,
        asinh_linear_range: float = 0.1,
    ) -> None:
        """Initialize the CustomNormalization object."""
        super().__init__(vmin=vmin, vmax=vmax, clip=False)
        if interval_type == "quantile":
            self.interval = QuantileInterval(
                lower_quantile=lower_quantile, upper_quantile=upper_quantile
            )
        elif interval_type == "manual":
            self.interval = ManualInterval(vmin=vmin, vmax=vmax)
        elif interval_type == "centered":
            self.interval = CenteredInterval(
                vcenter=vcenter,
                half_range=half_range,
            )
        else:
            raise ValueError("unrecognized interval_type.")

        if stretch_type == "power" or power != 1.0:
            self.stretch = PowerLawStretch(power)
        elif stretch_type == "linear":
            self.stretch = LinearStretch()
        elif stretch_type == "logarithmic":
            self.stretch = LogarithmicStretch(logarithmic_index)
        elif stretch_type == "asinh":
            self.stretch = InverseHyperbolicSineStretch(asinh_linear_range)
        else:
            raise ValueError("unrecognized stretch_type.")

        self.vmin = vmin
        self.vmax = vmax

        if data is not None:
            self._set_limits(data)

    def _set_limits(self, data: NDArray) -> None:
        """Set the normalization limits based on the provided data.

        Parameters
        ----------
        data : ndarray
            The data array to use for setting limits.

        Returns
        -------
        None
        """
        self.vmin, self.vmax = self.interval.get_limits(data)
        self.interval = ManualInterval(self.vmin, self.vmax)  # set explicitly with ManualInterval
        return None

    def __call__(self, value: NDArray, clip: bool | None = None) -> NDArray:  # type: ignore[override]
        """Apply the normalization to the input values.

        Parameters
        ----------
        value : array-like
            The input values to normalize.
        clip : bool, optional
            If True, clip the normalized values to [0, 1].

        Returns
        -------
        ndarray
            The normalized values, with invalid values masked.
        """
        values = self.interval(value)
        self.stretch(values, copy=False)
        return np.ma.masked_invalid(values)

    def inverse(self, value: NDArray) -> NDArray:  # type: ignore[override]
        """Apply the inverse normalization to the input values.

        Parameters
        ----------
        value : array-like
            The input values to inverse normalize.

        Returns
        -------
        ndarray
            The inverse normalized values.
        """
        values = self.stretch.inverse(value)
        values = self.interval.inverse(values)
        return values


@dataclass
class NormalizationConfig:
    """Configuration for CustomNormalization.

    This dataclass provides a convenient way to specify normalization parameters
    for the CustomNormalization class.

    Parameters
    ----------
    interval_type : str, default="quantile"
        Type of interval to use. Options are "quantile", "manual", or "centered".
    stretch_type : str, default="linear"
        Type of stretch to apply. Options are "linear", "power", "logarithmic", or "asinh".
    lower_quantile : float, default=0.02
        Lower quantile for "quantile" interval type.
    upper_quantile : float, default=0.98
        Upper quantile for "quantile" interval type.
    vmin : float, optional
        Minimum value for "manual" interval type.
    vmax : float, optional
        Maximum value for "manual" interval type.
    vcenter : float, default=0.0
        Center value for "centered" interval type.
    half_range : float, optional
        Half range for "centered" interval type.
    power : float, default=1.0
        Power for "power" stretch type.
    logarithmic_index : float, default=1000.0
        Index for "logarithmic" stretch type.
    asinh_linear_range : float, default=0.1
        Linear range for "asinh" stretch type.
    """

    interval_type: str = "quantile"
    stretch_type: str = "linear"
    lower_quantile: float = 0.02
    upper_quantile: float = 0.98
    vmin: float | None = None
    vmax: float | None = None
    vcenter: float = 0.0
    half_range: float | None = None
    power: float = 1.0
    logarithmic_index: float = 1000.0
    asinh_linear_range: float = 0.1


NORMALIZATION_PRESETS = {
    "linear_auto": lambda: NormalizationConfig(),
    "quantile": lambda: NormalizationConfig(),
    "linear_minmax": lambda: NormalizationConfig(interval_type="manual"),
    "minmax": lambda: NormalizationConfig(interval_type="manual"),
    "linear_centered": lambda: NormalizationConfig(interval_type="centered"),
    "log_auto": lambda: NormalizationConfig(stretch_type="logarithmic"),
    "log_minmax": lambda: NormalizationConfig(stretch_type="logarithmic", interval_type="manual"),
    "power_squared": lambda: NormalizationConfig(stretch_type="power", power=2.0),
    "power_sqrt": lambda: NormalizationConfig(stretch_type="power", power=0.5),
    "asinh_centered": lambda: NormalizationConfig(stretch_type="asinh", interval_type="centered"),
}


def _resolve_normalization(norm, **kwargs) -> NormalizationConfig:
    """Resolve various input types to a NormalizationConfig object.

    This function takes different input types and converts them to a
    NormalizationConfig object that can be used with CustomNormalization.

    Parameters
    ----------
    norm : None or dict or str or NormalizationConfig
        The normalization configuration to resolve.

    Returns
    -------
    NormalizationConfig
        The resolved normalization configuration.

    Raises
    ------
    ValueError
        If norm is a string that doesn't match any preset.
    TypeError
        If norm is not one of the supported types.
    """
    if norm is None:
        if "vmin" in kwargs or "vmax" in kwargs:
            return NormalizationConfig(
                interval_type="manual",
                stretch_type=kwargs.get("stretch_type", "linear"),
                vmin=kwargs.get("vmin"),
                vmax=kwargs.get("vmax"),
            )
        elif "lower_quantile" in kwargs or "upper_quantile" in kwargs:
            return NormalizationConfig(
                interval_type="quantile",
                lower_quantile=kwargs.get("lower_quantile", 0.02),
                upper_quantile=kwargs.get("upper_quantile", 0.98),
            )
        else:
            return NormalizationConfig()
    elif isinstance(norm, dict):
        return NormalizationConfig(**norm)
    elif isinstance(norm, str):
        if norm not in NORMALIZATION_PRESETS:
            raise ValueError(f"Unknown normalization preset: {norm}")
        return NORMALIZATION_PRESETS[norm]()
    elif isinstance(norm, NormalizationConfig):
        return norm
    else:
        raise TypeError("norm must be None, dict, str, or NormalizationConfig")
