import warnings
from dataclasses import dataclass, fields
from typing import Literal

from quantem.core.utils.validators import validate_gt, validate_lt
from quantem.core.visualization.custom_normalizations import NormalizationConfig
from quantem.core.visualization.visualization_utils import ScalebarConfig


class ShowParams:
    """
    Container for ``show_2d`` parameter dataclasses.

    Nested classes
    --------------
    Norm
        Normalization configuration (interval + stretch).
    Scalebar
        Scale bar overlay configuration.

    Examples
    --------
    >>> show_2d(img, norm=ShowParams.Norm(power=0.5))
    >>> show_2d(img, scalebar=ShowParams.Scalebar(sampling=0.5, units="Å"))
    >>> show_2d(dp, norm=ShowParams.Norm.log_auto(), cbar=True, cmap="turbo")
    """

    @dataclass
    class Norm:
        """
        Normalization parameters for ``show_2d``.

        Controls how pixel values are mapped to the [0, 1] display range via
        an *interval* (which values to keep) and a *stretch* (non-linear
        transfer function).

        If ``vmin`` or ``vmax`` is set and ``interval_type`` is left as the
        default ``"quantile"``, it is automatically changed to ``"manual"``.
        Likewise, setting ``vcenter`` to a non-zero value or providing
        ``half_range`` auto-selects ``"centered"``.

        Parameters
        ----------
        interval_type : ``"quantile"`` | ``"manual"`` | ``"centered"``
            How to determine the data range.
        stretch_type : ``"linear"`` | ``"power"`` | ``"logarithmic"`` | ``"asinh"``
            Transfer function applied after interval mapping.
        lower_quantile : float
            Lower quantile for ``"quantile"`` interval. Default 0.02.
        upper_quantile : float
            Upper quantile for ``"quantile"`` interval. Default 0.98.
        vmin : float or None
            Explicit minimum for ``"manual"`` interval.
        vmax : float or None
            Explicit maximum for ``"manual"`` interval.
        vcenter : float
            Centre value for ``"centered"`` interval. Default 0.0.
        half_range : float or None
            Symmetric half-range for ``"centered"`` interval.
        power : float
            Exponent for ``"power"`` stretch (e.g. 0.5 = sqrt). Default 1.0.
        logarithmic_index : float
            Index *a* for ``"logarithmic"`` stretch: ``log(a*x+1)/log(a+1)``.
            Default 1000.
        asinh_linear_range : float
            Transition parameter *a* for ``"asinh"`` stretch. Default 0.1.

        Examples
        --------
        >>> ShowParams.Norm()                        # quantile + linear (default)
        >>> ShowParams.Norm(power=0.5)               # quantile + sqrt stretch
        >>> ShowParams.Norm(vmin=0, vmax=1000)       # auto → manual range
        >>> ShowParams.Norm.log_auto()               # quantile + log stretch
        >>> ShowParams.Norm.centered(half_range=5)   # centered ± 5, linear
        """

        interval_type: Literal["quantile", "manual", "centered"] | None = None
        stretch_type: Literal["linear", "power", "logarithmic", "asinh"] = "linear"
        lower_quantile: float = 0.02
        upper_quantile: float = 0.98
        vmin: float | None = None
        vmax: float | None = None
        vcenter: float = 0.0
        half_range: float | None = None
        power: float = 1.0
        logarithmic_index: float = 1000.0
        asinh_linear_range: float = 0.1

        def __post_init__(self) -> None:
            manual_set = self.vmin is not None or self.vmax is not None
            centered_set = self.vcenter != 0.0 or self.half_range is not None
            quantile_set = self.lower_quantile != 0.02 or self.upper_quantile != 0.98
            user_chose = self.interval_type is not None

            # --- auto-infer interval_type when not explicitly provided ---
            if not user_chose:
                if manual_set and centered_set:
                    warnings.warn(
                        "Both vmin/vmax and vcenter/half_range were set; "
                        "defaulting to interval_type='manual'.",
                        stacklevel=2,
                    )
                    self.interval_type = "manual"
                elif manual_set:
                    self.interval_type = "manual"
                elif centered_set:
                    self.interval_type = "centered"
                else:
                    self.interval_type = "quantile"

            # --- warn about ignored interval fields ---
            if self.interval_type == "manual" and quantile_set:
                warnings.warn(
                    "lower_quantile/upper_quantile are ignored when interval_type='manual'.",
                    stacklevel=2,
                )
            if self.interval_type == "manual" and centered_set:
                warnings.warn(
                    "vcenter/half_range are ignored when interval_type='manual'.",
                    stacklevel=2,
                )
            if self.interval_type == "quantile" and manual_set:
                warnings.warn(
                    "vmin/vmax are ignored when interval_type='quantile'.",
                    stacklevel=2,
                )
            if self.interval_type == "quantile" and centered_set:
                warnings.warn(
                    "vcenter/half_range are ignored when interval_type='quantile'.",
                    stacklevel=2,
                )
            if self.interval_type == "centered" and manual_set:
                warnings.warn(
                    "vmin/vmax are ignored when interval_type='centered'.",
                    stacklevel=2,
                )
            if self.interval_type == "centered" and quantile_set:
                warnings.warn(
                    "lower_quantile/upper_quantile are ignored when interval_type='centered'.",
                    stacklevel=2,
                )

            # --- warn about ignored stretch fields ---
            if self.power != 1.0 and self.stretch_type not in ("power", "linear"):
                warnings.warn(
                    f"power={self.power} is ignored when stretch_type='{self.stretch_type}'.",
                    stacklevel=2,
                )
            if self.logarithmic_index != 1000.0 and self.stretch_type != "logarithmic":
                warnings.warn(
                    f"logarithmic_index={self.logarithmic_index} is ignored "
                    f"when stretch_type='{self.stretch_type}'.",
                    stacklevel=2,
                )
            if self.asinh_linear_range != 0.1 and self.stretch_type != "asinh":
                warnings.warn(
                    f"asinh_linear_range={self.asinh_linear_range} is ignored "
                    f"when stretch_type='{self.stretch_type}'.",
                    stacklevel=2,
                )

            # --- invalid value checks ---
            if self.vmin is not None and self.vmax is not None:
                validate_gt(self.vmax, self.vmin, "vmax", geq=False)
            validate_gt(self.lower_quantile, 0, "lower_quantile", geq=True)
            validate_gt(self.upper_quantile, self.lower_quantile, "upper_quantile")
            validate_lt(self.upper_quantile, 1.0, "upper_quantile", leq=True)
            if self.upper_quantile > 1.0:
                raise ValueError(f"upper_quantile must be <= 1, got {self.upper_quantile}.")
            if self.half_range is not None:
                validate_gt(self.half_range, 0, "half_range", geq=True)
            validate_gt(self.power, 0, "power")
            validate_gt(self.logarithmic_index, 0, "logarithmic_index")
            validate_gt(self.asinh_linear_range, 0, "asinh_linear_range")

        def to_config(self) -> NormalizationConfig:
            """Convert to a ``NormalizationConfig``."""
            return NormalizationConfig(**{f.name: getattr(self, f.name) for f in fields(self)})

        # ---- presets (mirror NORMALIZATION_PRESETS) ----

        @classmethod
        def linear_auto(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + linear stretch (the default)."""
            return cls(**kw)

        @classmethod
        def minmax(cls, **kw) -> "ShowParams.Norm":
            """Full min/max interval + linear stretch."""
            return cls(interval_type="manual", **kw)

        @classmethod
        def centered(
            cls, vcenter: float = 0.0, half_range: float | None = None, **kw
        ) -> "ShowParams.Norm":
            """Centered interval + linear stretch."""
            return cls(interval_type="centered", vcenter=vcenter, half_range=half_range, **kw)

        @classmethod
        def log_auto(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + logarithmic stretch."""
            return cls(stretch_type="logarithmic", **kw)

        @classmethod
        def log_minmax(cls, **kw) -> "ShowParams.Norm":
            """Full min/max interval + logarithmic stretch."""
            return cls(interval_type="manual", stretch_type="logarithmic", **kw)

        @classmethod
        def power_sqrt(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + square-root (power=0.5) stretch."""
            return cls(stretch_type="power", power=0.5, **kw)

        @classmethod
        def power_squared(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + squared (power=2.0) stretch."""
            return cls(stretch_type="power", power=2.0, **kw)

        @classmethod
        def asinh_centered(cls, vcenter: float = 0.0, **kw) -> "ShowParams.Norm":
            """Centered interval + asinh stretch."""
            return cls(interval_type="centered", stretch_type="asinh", vcenter=vcenter, **kw)

    @dataclass
    class Scalebar:
        """
        Scale bar parameters for ``show_2d``.

        Parameters
        ----------
        sampling : float
            Physical units per pixel. Default 1.0.
        units : str
            Unit label displayed on the scale bar (e.g. ``"Å"``, ``"nm"``,
            ``"1/Å"``). Default ``"pixels"``.
        length : float or None
            Fixed scale bar length in physical units. ``None`` auto-estimates
            a "nice" length.
        width_px : float
            Thickness of the bar in image pixels. Default 1.
        pad_px : float
            Padding between bar and plot edge in image pixels. Default 0.5.
        color : str
            Bar and label colour. Default ``"white"``.
        loc : ``"lower right"`` | ``"lower left"`` | ``"upper right"`` | ``"upper left"``
            Anchor location. Default ``"lower right"``.
        fontsize : int
            Font size of the scale bar label in points. Default 12.
        bold : bool
            Whether to render the label in bold. Default True.

        Examples
        --------
        >>> ShowParams.Scalebar(sampling=0.5, units="Å")
        >>> ShowParams.Scalebar(sampling=0.02, units="1/Å", color="black", fontsize=16)
        """

        sampling: float = 1.0
        units: str = "pixels"
        length: float | None = None
        width_px: float = 1
        pad_px: float = 0.5
        color: str = "white"
        loc: Literal["lower right", "lower left", "upper right", "upper left"] = "lower right"
        fontsize: int = 12
        bold: bool = False

        def __post_init__(self) -> None:
            validate_gt(self.sampling, 0, "sampling")
            if self.length is not None:
                validate_gt(self.length, 0, "length")
            validate_gt(self.width_px, 0, "width_px")
            validate_gt(self.fontsize, 0, "fontsize")

        def to_config(self) -> ScalebarConfig:
            """Convert to a ``ScalebarConfig``."""
            return ScalebarConfig(**{f.name: getattr(self, f.name) for f in fields(self)})
