"""
Show3DSlices: ptycho-oriented orthogonal slice viewer.

Displays orthogonal top/row/column slices with interactive sliders plus a
contextual 3D orientation view. All slicing happens in JavaScript for instant
response. This widget is intentionally focused on single-object iterative
ptychography volumes; comparison and tomography-specific workflows belong in
Show3DVolume.
"""
import json
import math
import pathlib
from numbers import Real
from typing import Self, Sequence

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.show2d import _reject_unknown_kwargs
from quantem.widget.state import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)


# Names that JS bundle's GPUColormapEngine knows. Keep in sync with
# js/colormaps.ts COLORMAPS table.
_VALID_CMAPS = frozenset({
    "inferno", "viridis", "plasma", "magma", "hot", "gray", "hsv", "turbo",
    "cividis", "RdBu", "RdBu_r", "seismic", "twilight", "twilight_shifted",
})


class Show3DSlices(anywidget.AnyWidget):
    """Ptycho multislice viewer with three orthogonal slice planes.

    Parameters
    ----------
    data : array_like
        3D array of shape (nz, ny, nx).
    title : str, optional
        Title displayed above the viewer.
    cmap : str, default "inferno"
        Colormap name. One of {valid set above}.
    pixel_size : float or sequence of 3 floats, optional
        Voxel sampling in angstroms. Pass a scalar for isotropic data, or a
        3-tuple `(pz, py, px)` for anisotropic data (e.g. multislice ptycho
        with z-thickness >> xy-sampling). Per-axis values flow to JS via the
        `pixel_size_axes` trait for correct scale bars on each panel.
    show_stats : bool, default False
        Compute per-slice statistics traits on each slice change (`widget.stats_mean`,
        `stats_min`, `stats_max`, `stats_std`, each a list of 3 floats: XY/XZ/YZ).
        Python-side only; the JS widget does not render a stats bar. Set False to
        skip 12 reductions per slice scrub on multi-MB volumes when you don't
        need the values.
    show_controls : bool, default True
        Show secondary controls for color, colorbar, smoothing, crosshair,
        z-stretch, contrast, and playback. The slice toolbar keeps only FFT
        and Reset Zoom visible.
    show_crosshair : bool, default True
        Show slice intersection guides across orthogonal panels.
    show_fft : bool, default False
        Toggle FFT panel for the active plane.
    fft_window : bool, default False
        Apply a 2D Hann window to each displayed slice before FFT. Useful for
        suppressing edge leakage/streaking in reciprocal-space panels.
    log_scale : bool, default False
        Use signed log1p for intensity mapping.
    auto_contrast : bool, default True
        Use percentile-based contrast (2nd-98th). On for ptycho phase data
        (long-tailed histogram - manual contrast usually crushes the signal).
    vmin, vmax : float, optional
        Manual contrast limits.
    fps : float, default 5.0
        Playback speed when scrubbing one axis.
    play_axis : int, default 0
        Which axis to animate (0=Z, 1=Y, 2=X, 3=cycle all).
    dim_labels : list of str, optional
        Labels for data axes 0, 1, 2 in that order. Default ["slice", "row", "col"]
        matches the project-wide detector-plane convention (axis 0 = multislice
        depth, axis 1 = row, axis 2 = col). Pass any 3-string list to override.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show3DSlices
    >>> volume = np.random.rand(64, 64, 64).astype(np.float32)
    >>> Show3DSlices(volume, title="My Volume", cmap="viridis")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3dslices.js"
    _widget_name = "Show3DSlices"
    _viewer_kind = "slices"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    viewer_kind = traitlets.Unicode("slices").tag(sync=True)

    # Volume dimensions
    nx = traitlets.Int(1).tag(sync=True)
    ny = traitlets.Int(1).tag(sync=True)
    nz = traitlets.Int(1).tag(sync=True)
    # Slice positions
    slice_x = traitlets.CInt(0).tag(sync=True)
    slice_y = traitlets.CInt(0).tag(sync=True)
    slice_z = traitlets.CInt(0).tag(sync=True)
    # Raw volume data (sent once)
    volume_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Scale bar. `pixel_size` is a scalar (lateral sampling, used by XY/XZ/YZ width-axis
    # scale bars). `pixel_size_axes` is the full per-axis triple [z, y, x] in the same
    # units - populated from tuple/list input; defaults to [pixel_size]*3 for scalar input.
    # Both sync to JS; JS uses pixel_size_axes when present for depth-axis scale bars
    # and falls back to pixel_size for the lateral axes.
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_size_axes = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # Depth-axis display stretch for non-cubic volumes (CSS-only, zero memory).
    # Scales XZ/YZ panel display height. Useful when nz << nxy (e.g. multislice
    # ptycho with nz=14, nxy=730 → set z_stretch high to make depth panels readable).
    z_stretch = traitlets.Float(1.0).tag(sync=True)
    # UI
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(False).tag(sync=True)
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(False).tag(sync=True)
    orthographic = traitlets.Bool(False).tag(sync=True)
    smooth = traitlets.Bool(False).tag(sync=True)
    # Deprecated compatibility no-op. The JS widget always renders the compact layout.
    compact = traitlets.Bool(True).tag(sync=True)
    flip = traitlets.Bool(False).tag(sync=True)
    # Axis labels (dim 0, 1, 2). Use detector-plane convention: axis 0 = slice
    # (multislice depth), axis 1 = row, axis 2 = col. Panel headers display as
    # "<dl[1]><dl[2]> (<dl[0]>=...)" so default reads as e.g. "row col (slice=7)".
    dim_labels = traitlets.List(traitlets.Unicode(), default_value=["slice", "row", "col"]).tag(sync=True)
    # Stats (3 values: xy, xz, yz)
    # stats_*: programmatic Python access only (no JS consumer). Don't sync.
    stats_mean = traitlets.List(traitlets.Float())
    stats_min = traitlets.List(traitlets.Float())
    stats_max = traitlets.List(traitlets.Float())
    stats_std = traitlets.List(traitlets.Float())
    # Playback
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)
    boomerang = traitlets.Bool(False).tag(sync=True)
    fps = traitlets.Float(5.0).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    play_axis = traitlets.Int(0).tag(sync=True)  # 0=Z, 1=Y, 2=X, 3=All
    # Validators (consistent with Show3D)

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal):
        val = str(proposal["value"])
        if val not in _VALID_CMAPS:
            raise traitlets.TraitError(
                f"Unknown cmap {val!r}. Valid: {sorted(_VALID_CMAPS)}"
            )
        return val

    @traitlets.validate("fps")
    def _validate_fps(self, proposal):
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"fps must be finite, got {val}")
        if val <= 0:
            raise traitlets.TraitError(f"fps must be > 0, got {val}")
        return val

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal):
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"pixel_size must be finite, got {val}")
        if val < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {val}")
        return val

    @traitlets.validate("pixel_size_axes")
    def _validate_pixel_size_axes(self, proposal):
        val = [float(v) for v in proposal["value"]]
        if len(val) != 3:
            raise traitlets.TraitError(
                f"pixel_size_axes must have length 3, got {len(val)}"
            )
        for v in val:
            if not math.isfinite(v):
                raise traitlets.TraitError(f"pixel_size_axes values must be finite, got {val}")
            if v < 0:
                raise traitlets.TraitError(f"pixel_size_axes values must be >= 0, got {val}")
        return val

    @traitlets.validate("play_axis")
    def _validate_play_axis(self, proposal):
        val = int(proposal["value"])
        if val not in (0, 1, 2, 3):
            raise traitlets.TraitError(f"play_axis must be 0/1/2/3, got {val}")
        return val

    @traitlets.validate("z_stretch")
    def _validate_z_stretch(self, proposal):
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"z_stretch must be finite, got {val}")
        return max(1.0, min(val, 30.0))

    @traitlets.validate("dim_labels")
    def _validate_dim_labels(self, proposal):
        val = list(proposal["value"])
        if len(val) != 3:
            raise traitlets.TraitError(
                f"dim_labels must have length 3, got {len(val)}"
            )
        return val

    @traitlets.validate("slice_z")
    def _validate_slice_z(self, proposal):
        return max(0, min(int(proposal["value"]), max(0, int(self.nz) - 1)))

    @traitlets.validate("slice_y")
    def _validate_slice_y(self, proposal):
        return max(0, min(int(proposal["value"]), max(0, int(self.ny) - 1)))

    @traitlets.validate("slice_x")
    def _validate_slice_x(self, proposal):
        return max(0, min(int(proposal["value"]), max(0, int(self.nx) - 1)))

    @traitlets.validate("vmax")
    def _validate_vmax_ge_vmin(self, proposal):
        new_vmax = proposal["value"]
        if new_vmax is not None:
            if not math.isfinite(new_vmax):
                raise traitlets.TraitError(f"vmax must be finite, got {new_vmax}")
            if self.vmin is not None and new_vmax < self.vmin:
                raise traitlets.TraitError(
                    f"vmax ({new_vmax}) must be >= vmin ({self.vmin})"
                )
        return new_vmax

    @traitlets.validate("vmin")
    def _validate_vmin_le_vmax(self, proposal):
        new_vmin = proposal["value"]
        if new_vmin is not None:
            if not math.isfinite(new_vmin):
                raise traitlets.TraitError(f"vmin must be finite, got {new_vmin}")
            if self.vmax is not None and new_vmin > self.vmax:
                raise traitlets.TraitError(
                    f"vmin ({new_vmin}) must be <= vmax ({self.vmax})"
                )
        return new_vmin

    def __init__(
        self,
        data,
        data_b=None,
        *,
        title: str = "",
        title_b: str = "",
        cmap: str = "inferno",
        pixel_size: float | Sequence[float] | None = 0.0,
        scale_bar_visible: bool = True,
        z_stretch: float | None = None,
        show_controls: bool = True,
        show_stats: bool = False,
        show_crosshair: bool = True,
        show_fft: bool = False,
        fft_window: bool = False,
        orthographic: bool = False,
        smooth: bool = False,
        flip: bool = False,
        show_diff: bool = False,
        log_scale: bool = False,
        auto_contrast: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        fps: float = 5.0,
        loop: bool = True,
        reverse: bool = False,
        boomerang: bool = False,
        linked_contrast: bool = True,
        play_axis: int = 0,
        dim_labels: list[str] | None = None,
        state=None,
        **kwargs,
    ):
        _reject_unknown_kwargs(type(self), kwargs)
        if data_b is not None:
            raise ValueError(
                "Show3DSlices is a single-object ptycho slice viewer. "
                "Use Show3DVolume for dual-volume comparison workflows."
            )
        if show_diff:
            raise ValueError(
                "Show3DSlices does not support difference/dual mode. "
                "Pass a single 3D object and inspect its orthogonal slices."
            )
        if title_b:
            raise ValueError(
                "Show3DSlices accepts only one title. "
                "Use the title= argument for the single ptycho object."
            )
        if linked_contrast is not True:
            raise ValueError(
                "Show3DSlices does not support linked_contrast; it has no dual-volume mode."
            )
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self.viewer_kind = self._viewer_kind
        # Pre-seed so free() / __repr__ / summary() are safe even if a validator
        # raises before _data is assigned below (e.g. wrong ndim or complex data).
        self._data: np.ndarray | None = None

        # Duck-typed Dataset3d extraction (matches Show2D / Show3D pattern).
        # `array` is the required payload; title/sampling/units are optional
        # metadata so lightweight Dataset3d-like wrappers work naturally.
        if hasattr(data, "array"):
            name = getattr(data, "name", "")
            if not title and name:
                title = name
            pixel_size_is_default = pixel_size is None or (
                np.isscalar(pixel_size) and float(pixel_size) == 0.0
            )
            if pixel_size_is_default and hasattr(data, "sampling"):
                try:
                    units = list(getattr(data, "units", []) or [])
                    samp = list(data.sampling)
                    # Unit conversion to Å per axis. If units are absent, assume
                    # sampling is already in Å so Dataset3d-like objects remain
                    # easy to use. A single unit applies to every sampling axis.
                    def unit_scale(unit):
                        if not unit:
                            return 1.0
                        u = str(unit).strip().lower()
                        if u == "nm":
                            return 10.0
                        if u in ("a", "å", "angstrom", "angstroms"):
                            return 1.0
                        raise ValueError(f"unsupported Dataset3d unit: {unit!r}")

                    def axis_unit(axis_index: int):
                        if not units:
                            return ""
                        if len(units) == 1:
                            return units[0]
                        return units[axis_index]

                    if len(samp) >= 3:
                        pixel_size = [
                            float(samp[i]) * unit_scale(axis_unit(i))
                            for i in (-3, -2, -1)
                        ]
                    elif len(samp) >= 1:
                        pixel_size = float(samp[-1]) * unit_scale(axis_unit(-1))
                except (IndexError, TypeError, ValueError):
                    pass
            data = data.array

        data = to_numpy(data)
        if data.ndim != 3:
            raise ValueError(f"Show3DSlices requires 3D data, got {data.ndim}D")
        if 0 in data.shape:
            raise ValueError(f"Empty volume: shape {data.shape}. All dims must be >= 1.")
        if not np.isfinite(data).all():
            raise ValueError(
                "Data contains NaN or inf. Clean first: "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3DSlices does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            self._data = data.astype(np.float32, copy=False)
        if not np.isfinite(self._data).all():
            raise ValueError(
                "Data exceeds float32 range (|value| > 3.4e38) after cast; "
                "rescale first before passing to Show3DSlices."
            )
        self.nz, self.ny, self.nx = self._data.shape

        # Default to middle slices
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2

        self.title = title
        self.cmap = cmap
        # pixel_size accepts: None → 0 (no scale bar), scalar (isotropic), or
        # 3-tuple/list/ndarray (anisotropic: [pz, py, px] in the same units).
        # For 3-tuple input the scalar trait is set to the lateral mean (py+px)/2
        # so existing scale-bar code keeps working; the full triple is published
        # via pixel_size_axes for per-axis scale bars.
        if pixel_size is None:
            pixel_size = 0.0
        if isinstance(pixel_size, Real) or np.isscalar(pixel_size):
            ps_scalar = float(pixel_size)
            ps_axes = [ps_scalar, ps_scalar, ps_scalar]
        else:
            try:
                ps_axes = [float(v) for v in pixel_size]
            except TypeError:
                raise TypeError(
                    f"pixel_size must be a scalar or 3-element sequence (Å/pixel), "
                    f"got {type(pixel_size).__name__}."
                )
            if len(ps_axes) != 3:
                raise ValueError(
                    f"pixel_size as a sequence must have exactly 3 elements [pz, py, px], "
                    f"got {len(ps_axes)}."
                )
            for v in ps_axes:
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"pixel_size_axes must be finite and >= 0, got {ps_axes}.")
            ps_scalar = (ps_axes[1] + ps_axes[2]) / 2.0  # lateral mean
        self.pixel_size = ps_scalar
        self.pixel_size_axes = ps_axes
        self.scale_bar_visible = scale_bar_visible
        # Default z_stretch = 15 for thin-Z multislice ptycho (nz~14, nxy~700+).
        # Renders depth panels at a readable height without hiding lateral detail.
        # User can override via constructor or runtime trait. For near-cubic data
        # (nxy/nz <= 4) keep 1.0 so XZ/YZ panels stay square.
        thin_z_ratio = min(self.nx, self.ny) / max(self.nz, 1)
        if z_stretch is None:
            z_stretch = 15.0 if thin_z_ratio > 4 else 1.0
        self.z_stretch = float(z_stretch)
        # Slices viewer is always compact. The 3D panel is an orientation/context
        # view, while detailed comparison/tomography workflows stay in Show3DVolume.
        self.compact = True
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.show_crosshair = show_crosshair
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.orthographic = orthographic
        self.smooth = smooth
        self.flip = flip
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.vmin = vmin
        self.vmax = vmax
        self.fps = fps
        self.loop = loop
        self.reverse = reverse
        self.boomerang = boomerang
        self.play_axis = play_axis
        if dim_labels is not None:
            self.dim_labels = dim_labels

        self._compute_stats()
        self.volume_bytes = self._data.tobytes()
        self.observe(self._on_slice_change, names=["slice_x", "slice_y", "slice_z"])
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_show_stats_change, names=["show_stats"])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                    expected_widget=self._widget_name,
                )
            else:
                state = unwrap_state_payload(state, expected_widget=self._widget_name)
            self.load_state_dict(state)

    def __repr__(self) -> str:
        return (
            f"{self._widget_name}({self.nz}×{self.ny}×{self.nx}, "
            f"slices=({self.slice_z},{self.slice_y},{self.slice_x}), cmap={self.cmap})"
        )

    def state_dict(self) -> dict:
        return {
            "title": self.title,
            "viewer_kind": self.viewer_kind,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_crosshair": self.show_crosshair,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "orthographic": self.orthographic,
            "smooth": self.smooth,
            "flip": self.flip,
            "pixel_size": self.pixel_size,
            "pixel_size_axes": list(self.pixel_size_axes),
            "scale_bar_visible": self.scale_bar_visible,
            "z_stretch": self.z_stretch,
            "slice_x": self.slice_x,
            "slice_y": self.slice_y,
            "slice_z": self.slice_z,
            "fps": self.fps,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            "play_axis": self.play_axis,
            "dim_labels": list(self.dim_labels),
        }

    def save(self, path: str) -> None:
        save_state_file(path, self._widget_name, self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        # Surface validator errors. Warn on unknown keys (typo / wrong widget version).
        if state.get("dual_mode") or state.get("show_diff"):
            raise ValueError(
                "Show3DSlices only supports a single 3D object. "
                "Use Show3DVolume for saved dual/diff comparison states."
            )
        allowed = {
            "title", "cmap", "log_scale", "auto_contrast", "vmin", "vmax",
            "viewer_kind",
            "show_stats", "show_controls", "show_crosshair", "show_fft",
            "fft_window", "orthographic", "smooth", "flip", "pixel_size", "pixel_size_axes",
            "scale_bar_visible", "z_stretch", "compact", "slice_x",
            "slice_y", "slice_z", "fps", "loop", "reverse", "boomerang",
            "play_axis", "dim_labels",
        }
        unknown = [k for k in state if k not in allowed and k not in {"dual_mode", "show_diff", "title_b", "linked_contrast"}]
        if unknown:
            import warnings
            warnings.warn(
                f"load_state_dict ignored unknown keys: {unknown}. "
                "Likely typo or saved by a different widget version.",
                stacklevel=2,
            )
        state = {k: v for k, v in state.items() if k in allowed}
        state.pop("viewer_kind", None)
        # Saved states from older versions may include compact=False. The current
        # widget intentionally ignores it and always uses the compact layout.
        state.pop("compact", None)
        vmin_marker = object()
        vmax_marker = object()
        vmin = state.pop("vmin", vmin_marker)
        vmax = state.pop("vmax", vmax_marker)
        if vmin is not vmin_marker or vmax is not vmax_marker:
            new_vmin = self.vmin if vmin is vmin_marker else vmin
            new_vmax = self.vmax if vmax is vmax_marker else vmax
            if new_vmin is not None and new_vmax is not None and float(new_vmin) > float(new_vmax):
                raise traitlets.TraitError(f"vmin ({new_vmin}) must be <= vmax ({new_vmax})")
            # Clear first so either half of a valid saved pair can be loaded
            # regardless of the widget's current contrast limits.
            self.vmin = None
            self.vmax = None
            if new_vmin is not None:
                self.vmin = float(new_vmin)
            if new_vmax is not None:
                self.vmax = float(new_vmax)
        for key, val in state.items():
            if self.has_trait(key):
                setattr(self, key, val)
        # Forward-compat: state saved before pixel_size_axes existed only has the
        # scalar pixel_size. Mirror it across all three axes so depth scale bars
        # don't desync from the lateral one after a load.
        if "pixel_size" in state and "pixel_size_axes" not in state:
            ps = float(state["pixel_size"])
            self.pixel_size_axes = [ps, ps, ps]

    def free(self) -> None:
        """Release RAM held by this widget. `del widget` won't free
        memory because traitlets observers pin the refcount."""
        if self._data is None:
            return
        self._data = None
        for trait in ("volume_bytes",):
            setattr(self, trait, b"")
        import gc
        gc.collect()

    def summary(self) -> None:
        lines = [self.title or self._widget_name, "═" * 32]
        lines.append(f"Volume:   {self.nz}×{self.ny}×{self.nx}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            unit = f"{ps / 10:.2f} nm/px" if ps >= 10 else f"{ps:.2f} Å/px"
            lines[-1] += f" ({unit})"
        labels = list(self.dim_labels)
        lines.append(
            f"Slices:   {labels[0]}={self.slice_z}  {labels[1]}={self.slice_y}  {labels[2]}={self.slice_x}"
        )
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(
                f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}"
            )
        scale = "log" if self.log_scale else "linear"
        if self.vmin is not None and self.vmax is not None:
            contrast = f"vmin={self.vmin:.4g}, vmax={self.vmax:.4g}"
        elif self.auto_contrast:
            contrast = "auto contrast"
        else:
            contrast = "manual contrast"
        display = f"{self.cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if self.fft_window:
                display += " Hann"
        lines.append(f"Display:  {display}")
        print("\n".join(lines))

    def _compute_stats(self) -> None:
        """Compute statistics for the 3 current slices.

        Skipped when show_stats is False to avoid 12 reductions
        per slice movement on multi-MB volumes (JS does not render a stats bar; this
        is for programmatic access only when the caller has opted in).
        """
        if not self.show_stats or self._data is None:
            return
        slices = [
            self._data[self.slice_z, :, :],
            self._data[:, self.slice_y, :],
            self._data[:, :, self.slice_x],
        ]
        with self.hold_sync():
            self.stats_mean = [float(np.mean(s, dtype=np.float64)) for s in slices]
            self.stats_min = [float(np.min(s)) for s in slices]
            self.stats_max = [float(np.max(s)) for s in slices]
            self.stats_std = [float(np.std(s, dtype=np.float64)) for s in slices]

    def _on_slice_change(self, change) -> None:
        if self.playing:
            return
        self._compute_stats()

    def _on_playing_change(self, change) -> None:
        if not self.playing:
            self._compute_stats()

    def _on_show_stats_change(self, change) -> None:
        if change.get("new"):
            self._compute_stats()

    def play(self) -> Self:
        self.playing = True
        return self

    def pause(self) -> Self:
        self.playing = False
        return self

    def stop(self) -> Self:
        self.playing = False
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2
        return self

    def _normalize_slice(self, slc: np.ndarray) -> np.ndarray:
        if self.log_scale:
            slc = np.sign(slc) * np.log1p(np.abs(slc))
        # Mirror JS path: when flip=True the on-screen renderer negates the data
        # and flips the contrast range (min<->max with sign). Python-side saved
        # images should match what the user sees on screen.
        if self.flip:
            slc = -slc
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
            if self.flip:
                vmin, vmax = -vmax, -vmin
        elif self.auto_contrast:
            vmin = float(np.percentile(slc, 2))
            vmax = float(np.percentile(slc, 98))
        else:
            vmin = float(slc.min())
            vmax = float(slc.max())
        if vmax > vmin:
            return np.clip((slc - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(slc.shape, dtype=np.uint8)

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        plane: str | None = None,
        slice_idx: int | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save a volume slice as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        plane : str, optional
            One of 'xy', 'xz', 'yz'. Defaults to 'xy'.
        slice_idx : int, optional
            Slice index along the chosen axis. Defaults to current position.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from extension.
        dpi : int, default 150
            Output DPI metadata.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        plane = (plane or "xy").lower()
        if plane == "xy":
            idx = slice_idx if slice_idx is not None else self.slice_z
            max_idx = self.nz
        elif plane == "xz":
            idx = slice_idx if slice_idx is not None else self.slice_y
            max_idx = self.ny
        elif plane == "yz":
            idx = slice_idx if slice_idx is not None else self.slice_x
            max_idx = self.nx
        else:
            raise ValueError(f"Unknown plane: {plane!r}. Use 'xy', 'xz', or 'yz'.")

        if idx < 0 or idx >= max_idx:
            raise IndexError(f"Slice index {idx} out of range [0, {max_idx}) for plane '{plane}'")

        if plane == "xy":
            slc = self._data[idx]
        elif plane == "xz":
            slc = self._data[:, idx, :]
        else:
            slc = self._data[:, :, idx]

        normalized = self._normalize_slice(slc)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        # PDF requires RGB (no alpha) and the PDF plugin registered.
        if fmt == "pdf":
            Image.init()
            img = img.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pass format explicitly so a mismatched extension (e.g. format="tiff"
        # with path="out.bin") still writes the requested container.
        pil_format = {"png": "PNG", "pdf": "PDF", "tiff": "TIFF", "tif": "TIFF"}[fmt]
        img.save(str(path), format=pil_format, dpi=(dpi, dpi))
        return path
