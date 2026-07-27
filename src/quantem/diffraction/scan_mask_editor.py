"""Interactive scan-region selection for a 4D-STEM analysis.

An ipywidgets editor for choosing the region of a scan to analyse: circle,
ellipse, square or rectangle, positioned in scan coordinates (X horizontal, Y
vertical) with a live preview over a reference image. The selection is committed
to the analysis object's ``scan_mask`` only on Apply, and can be persisted to a
state file so a later run reuses it.

Constructed through ``BraggPeaksPolymer.edit_scan_mask()`` rather than directly;
it is bound to the analysis whose mask it edits.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, HBox, VBox
from IPython.display import clear_output, display
from matplotlib.patches import Ellipse, Rectangle

__all__ = ["ScanMaskEditor"]


class ScanMaskEditor:
    """Interactive, persistent circular scan-mask editor.

    The horizontal X control maps directly to the scan-column coordinate. The
    vertical Y control is visually inverted relative to the array row index so
    moving the slider upward moves the probe marker upward on an ``origin="upper"``
    scan image.
    """

    SCHEMA_VERSION = 2
    GEOMETRIES = ("circle", "ellipse", "square", "rectangle")

    def __init__(
        self,
        analysis,
        *,
        initial_x=None,
        initial_y=None,
        initial_radius=None,
        initial_geometry="circle",
        initial_size_x=None,
        initial_size_y=None,
        reference_image=None,
        state_path=None,
        overlay_alpha=0.28,
        crosshair_width=2,
        crosshair_size=12,
        autosave=False,
        display_widget=True,
    ):
        self.analysis = analysis
        self.scan_shape = tuple(int(v) for v in analysis.dataset_cartesian.shape[:2])
        self.state_path = None if state_path is None else Path(state_path)
        self.autosave = bool(autosave)
        self.overlay_alpha = float(overlay_alpha)
        self.crosshair_width = float(crosshair_width)
        self.crosshair_size = float(crosshair_size)
        self._syncing = False
        self._dirty = False
        self._saved = False
        self._loaded = False

        rows, columns = self.scan_shape
        center_row = rows // 2 if initial_y is None else int(initial_y)
        center_column = columns // 2 if initial_x is None else int(initial_x)
        radius = (
            max(1, min(rows, columns) // 3)
            if initial_radius is None
            else int(initial_radius)
        )
        geometry = str(initial_geometry).lower()
        size_x = radius if initial_size_x is None else int(initial_size_x)
        size_y = radius if initial_size_y is None else int(initial_size_y)
        loaded_mask = None
        if self.state_path is not None and self.state_path.is_file():
            state = self._read_state(self.state_path)
            center_row = state["center_row"]
            center_column = state["center_column"]
            geometry = state["geometry"]
            size_x = state["size_x"]
            size_y = state["size_y"]
            loaded_mask = state["mask"]
            self._loaded = True
            self._saved = True

        self._validate_geometry(
            center_row, center_column, geometry, size_x, size_y
        )
        self.reference_image = self._resolve_reference_image(reference_image)
        self._preview_mask = (
            loaded_mask.copy()
            if loaded_mask is not None
            else self._geometry_mask(
                center_row, center_column, geometry, size_x, size_y
            )
        )

        maximum_radius = int(np.ceil(np.hypot(rows - 1, columns - 1))) + 1
        self.x_slider = widgets.IntSlider(
            value=center_column,
            min=0,
            max=columns - 1,
            step=1,
            description="X",
            continuous_update=True,
            readout=False,
            style={"description_width": "18px"},
            layout=widgets.Layout(width="375px"),
        )
        # Slider value increases upward, while array row indices increase downward.
        self.y_slider = widgets.IntSlider(
            value=rows - 1 - center_row,
            min=0,
            max=rows - 1,
            step=1,
            description="Y",
            orientation="vertical",
            continuous_update=True,
            readout=False,
            style={"description_width": "18px"},
            layout=widgets.Layout(height="281px", width="52px"),
        )
        self.geometry_selector = widgets.Dropdown(
            options=[
                ("Circular", "circle"),
                ("Elliptical", "ellipse"),
                ("Square", "square"),
                ("Rectangular", "rectangle"),
            ],
            value=geometry,
            description="Shape",
            style={"description_width": "42px"},
            layout=widgets.Layout(width="155px"),
        )
        self.size_x_slider = widgets.IntSlider(
            value=size_x,
            min=1,
            max=maximum_radius,
            step=1,
            description="Radius",
            continuous_update=True,
            readout=False,
            style={"description_width": "55px"},
            layout=widgets.Layout(width="375px"),
        )
        self.size_y_slider = widgets.IntSlider(
            value=size_y,
            min=1,
            max=maximum_radius,
            step=1,
            description="Y radius",
            continuous_update=True,
            readout=False,
            style={"description_width": "68px"},
            layout=widgets.Layout(width="375px"),
        )
        # Historical public attribute retained for callers that customize it.
        self.radius_slider = self.size_x_slider
        self.x_input = widgets.BoundedIntText(
            value=center_column,
            min=0,
            max=columns - 1,
            description="X column",
            style={"description_width": "62px"},
            layout=widgets.Layout(width="150px"),
        )
        self.y_input = widgets.BoundedIntText(
            value=center_row,
            min=0,
            max=rows - 1,
            description="Y row",
            style={"description_width": "52px"},
            layout=widgets.Layout(width="140px"),
        )
        self.size_x_input = widgets.BoundedIntText(
            value=size_x,
            min=1,
            max=maximum_radius,
            description="Radius",
            style={"description_width": "48px"},
            layout=widgets.Layout(width="130px"),
        )
        self.size_y_input = widgets.BoundedIntText(
            value=size_y,
            min=1,
            max=maximum_radius,
            description="Y radius",
            style={"description_width": "58px"},
            layout=widgets.Layout(width="140px"),
        )
        self.radius_input = self.size_x_input

        self.apply_button = widgets.Button(
            description="Apply",
            icon="check",
            button_style="primary",
            tooltip="Commit the preview mask to BraggPeaksPolymer.scan_mask",
            layout=widgets.Layout(width="72px"),
        )
        self.save_button = widgets.Button(
            description="Apply & Save",
            icon="save",
            button_style="success",
            tooltip="Commit and save this mask for the next notebook run",
            disabled=self.state_path is None,
            layout=widgets.Layout(width="105px"),
        )
        self.center_button = widgets.Button(
            description="Center",
            icon="crosshairs",
            tooltip="Center the circle",
            layout=widgets.Layout(width="75px"),
        )
        self.full_button = widgets.Button(
            description="Full",
            icon="expand",
            tooltip="Include the entire scan",
            layout=widgets.Layout(width="70px"),
        )
        self.reset_button = widgets.Button(
            description="Reset",
            icon="undo",
            tooltip="Restore the loaded/default state",
            layout=widgets.Layout(width="72px"),
        )
        self.status = widgets.HTML(layout=widgets.Layout(width="500px"))
        self.output = widgets.Output(
            layout=widgets.Layout(
                width="438px",
                height="356px",
                max_width="438px",
                overflow="hidden",
            )
        )

        self.figure, self.ax = plt.subplots(figsize=(4.0, 3.25))
        finite = self.reference_image[np.isfinite(self.reference_image)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1.0, 99.0])
            if not vmax > vmin:
                vmin, vmax = float(np.min(finite)), float(np.max(finite) + 1.0)
        else:
            vmin, vmax = 0.0, 1.0
        self.image_artist = self.ax.imshow(
            self.reference_image,
            cmap="gray",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        self.mask_artist = self.ax.imshow(
            np.ma.masked_where(~self._preview_mask, self._preview_mask),
            cmap="Reds",
            origin="upper",
            alpha=self.overlay_alpha,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        self.boundary_artist = None
        self.circle_artist = None
        self._replace_boundary_artist()
        (self.center_artist,) = self.ax.plot(
            center_column,
            center_row,
            marker="+",
            color="#ff3030",
            markersize=self.crosshair_size,
            markeredgewidth=self.crosshair_width,
        )
        self.ax.set(
            title="Scan-mask editor",
            xlabel="X — scan column",
            ylabel="Y — scan row",
            xlim=(-0.5, columns - 0.5),
            ylim=(rows - 0.5, -0.5),
        )
        self.ax.title.set_fontsize(10)
        self.ax.xaxis.label.set_fontsize(9)
        self.ax.yaxis.label.set_fontsize(9)
        self.ax.tick_params(labelsize=8)
        self.figure.tight_layout()

        self.x_slider.observe(self._on_x_slider, names="value")
        self.y_slider.observe(self._on_y_slider, names="value")
        self.geometry_selector.observe(self._on_geometry, names="value")
        self.size_x_slider.observe(self._on_size_x_slider, names="value")
        self.size_y_slider.observe(self._on_size_y_slider, names="value")
        self.x_input.observe(self._on_x_input, names="value")
        self.y_input.observe(self._on_y_input, names="value")
        self.size_x_input.observe(self._on_size_x_input, names="value")
        self.size_y_input.observe(self._on_size_y_input, names="value")
        self.apply_button.on_click(lambda _: self.apply())
        self.save_button.on_click(lambda _: self.save())
        self.center_button.on_click(
            lambda _: self.set_mask(x=columns // 2, y=rows // 2)
        )
        self.full_button.on_click(lambda _: self._set_full_scan())
        self.reset_button.on_click(lambda _: self._restore_initial())

        self._initial_geometry = (
            center_column, center_row, geometry, size_x, size_y
        )
        self._initial_mask = self._preview_mask.copy()
        # A loaded/default mask is immediately usable by Run All. Slider edits
        # remain previews until Apply, preventing repeated inference-cache invalidation.
        self.analysis.scan_mask = self._preview_mask.copy()
        self._render()
        self._refresh_status("Loaded saved mask" if self._loaded else "Default mask applied")

        position_row = widgets.HBox(
            [self.geometry_selector, self.x_input, self.y_input],
            layout=widgets.Layout(width="500px"),
        )
        self.size_input_row = widgets.HBox(
            [self.size_x_input, self.size_y_input],
            layout=widgets.Layout(width="500px"),
        )
        toolbar = widgets.HBox(
            [
                self.apply_button,
                self.save_button,
                self.center_button,
                self.full_button,
                self.reset_button,
            ],
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        plot_row = widgets.HBox(
            [self.y_slider, self.output],
            layout=widgets.Layout(
                align_items="center", width="500px", overflow="hidden"
            ),
        )
        self.size_x_row = widgets.HBox(
            [
                widgets.Box(layout=widgets.Layout(width="52px")),
                self.size_x_slider,
            ],
            layout=widgets.Layout(align_items="center", width="500px"),
        )
        self.size_y_row = widgets.HBox(
            [
                widgets.Box(layout=widgets.Layout(width="52px")),
                self.size_y_slider,
            ],
            layout=widgets.Layout(align_items="center", width="500px"),
        )
        x_row = widgets.HBox(
            [
                widgets.Box(layout=widgets.Layout(width="52px")),
                self.x_slider,
            ],
            layout=widgets.Layout(align_items="center", width="500px"),
        )
        self.widget = widgets.VBox(
            [
                toolbar,
                position_row,
                self.size_input_row,
                self.size_x_row,
                self.size_y_row,
                plot_row,
                x_row,
                self.status,
            ],
            layout=widgets.Layout(width="500px", max_width="500px"),
        )
        self._refresh_geometry_controls()
        # Prevent the inline Matplotlib backend from appending a second copy of
        # the figure after the widget cell. The explicitly displayed Output copy
        # remains live and continues to update.
        plt.close(self.figure)
        if display_widget:
            display(self.widget)

    @property
    def x(self):
        """Horizontal scan-column coordinate."""
        return int(self.x_slider.value)

    @property
    def y(self):
        """Vertical scan-row coordinate."""
        return int(self.scan_shape[0] - 1 - self.y_slider.value)

    @property
    def radius(self):
        """Circle radius / square half-width compatibility value."""
        return self.size_x

    @property
    def geometry(self):
        return str(self.geometry_selector.value)

    @property
    def size_x(self):
        """Horizontal radius or half-width in scan pixels."""
        return int(self.size_x_slider.value)

    @property
    def size_y(self):
        """Vertical radius or half-height in scan pixels."""
        if self.geometry in {"circle", "square"}:
            return self.size_x
        return int(self.size_y_slider.value)

    @property
    def mask(self):
        """Current preview mask."""
        return self._preview_mask.copy()

    @property
    def applied_mask(self):
        return None if self.analysis.scan_mask is None else self.analysis.scan_mask.copy()

    def _effective_mask(self):
        """Return the committed mask for ndarray-style compatibility."""
        mask = self.analysis.scan_mask
        return self._preview_mask if mask is None else np.asarray(mask, dtype=bool)

    @property
    def shape(self):
        return self.scan_shape

    @property
    def dtype(self):
        return np.dtype(bool)

    @property
    def size(self):
        return int(np.prod(self.scan_shape))

    @property
    def ndim(self):
        return 2

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self._effective_mask(), dtype=dtype)
        if copy:
            array = array.copy()
        return array

    def __len__(self):
        return self.scan_shape[0]

    def sum(self, *args, **kwargs):
        """NumPy-compatible sum for legacy ``mask_arr = editor`` cells."""
        return self._effective_mask().sum(*args, **kwargs)

    def astype(self, *args, **kwargs):
        return self._effective_mask().astype(*args, **kwargs)

    def copy(self):
        return self._effective_mask().copy()

    def __getitem__(self, key):
        if not isinstance(key, str):
            return self._effective_mask()[key]
        # Compatibility with the historical returned dictionary. Its x0/y0
        # names represented row/column respectively despite their labels.
        values = {
            "mask": self.mask,
            "x0": self.y,
            "y0": self.x,
            "r": self.radius,
            "center_row": self.y,
            "center_column": self.x,
            "geometry": self.geometry,
            "size_x": self.size_x,
            "size_y": self.size_y,
        }
        return values[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _validate_geometry(self, row, column, geometry, size_x, size_y):
        rows, columns = self.scan_shape
        if not 0 <= row < rows or not 0 <= column < columns:
            raise ValueError(
                f"Mask center (row={row}, column={column}) is outside scan shape "
                f"{self.scan_shape}."
            )
        if geometry not in self.GEOMETRIES:
            raise ValueError(
                f"Unknown mask geometry {geometry!r}; choose one of "
                f"{', '.join(self.GEOMETRIES)}."
            )
        if size_x < 1 or size_y < 1:
            raise ValueError("Mask half-sizes must be at least one scan pixel.")
        maximum_radius = int(np.ceil(np.hypot(rows - 1, columns - 1))) + 1
        if size_x > maximum_radius or size_y > maximum_radius:
            raise ValueError(
                f"Mask half-size ({size_x}, {size_y}) exceeds the supported maximum "
                f"{maximum_radius} for scan shape {self.scan_shape}."
            )

    def _geometry_mask(self, row, column, geometry, size_x, size_y):
        yy, xx = np.ogrid[: self.scan_shape[0], : self.scan_shape[1]]
        dy = yy - row
        dx = xx - column
        if geometry == "circle":
            return dy**2 + dx**2 <= size_x**2
        if geometry == "ellipse":
            return (dx / size_x) ** 2 + (dy / size_y) ** 2 <= 1
        if geometry == "square":
            return (np.abs(dx) <= size_x) & (np.abs(dy) <= size_x)
        if geometry == "rectangle":
            return (np.abs(dx) <= size_x) & (np.abs(dy) <= size_y)
        raise ValueError(f"Unknown mask geometry {geometry!r}.")

    def _resolve_reference_image(self, reference_image):
        if reference_image is None:
            virtual_images = getattr(self.analysis.dataset_cartesian, "virtual_images", {})
            if "virtual_image" in virtual_images:
                reference_image = virtual_images["virtual_image"]
                reference_image = getattr(
                    reference_image,
                    "array",
                    getattr(reference_image, "data", reference_image),
                )
            else:
                dataset = self.analysis.dataset_cartesian
                array = getattr(dataset, "array", None)
                if array is not None:
                    reference_image = np.asarray(array).mean(axis=(-2, -1))
                else:
                    reference_image = (
                        dataset.tensor.float().mean(dim=(-2, -1)).detach().cpu().numpy()
                    )
        reference_image = np.asarray(reference_image, dtype=float)
        if reference_image.shape != self.scan_shape:
            raise ValueError(
                f"reference_image shape {reference_image.shape} must match "
                f"scan shape {self.scan_shape}."
            )
        return reference_image

    def _read_state(self, path):
        try:
            with np.load(path, allow_pickle=False) as state:
                version = int(state["schema_version"])
                shape = tuple(int(v) for v in state["scan_shape"])
                if version not in {1, self.SCHEMA_VERSION}:
                    raise ValueError(
                        f"Unsupported scan-mask schema {version}; expected "
                        f"1 or {self.SCHEMA_VERSION}."
                    )
                if shape != self.scan_shape:
                    raise ValueError(
                        f"Saved scan-mask shape {shape} does not match current "
                        f"scan shape {self.scan_shape}."
                    )
                mask = np.asarray(state["mask"], dtype=bool)
                if mask.shape != self.scan_shape:
                    raise ValueError(
                        f"Saved mask array shape {mask.shape} does not match "
                        f"scan shape {self.scan_shape}."
                    )
                if version == 1:
                    geometry = "circle"
                    size_x = size_y = int(state["radius"])
                else:
                    geometry = str(state["geometry"].item())
                    size_x = int(state["size_x"])
                    size_y = int(state["size_y"])
                return {
                    "center_row": int(state["center_row"]),
                    "center_column": int(state["center_column"]),
                    "geometry": geometry,
                    "size_x": size_x,
                    "size_y": size_y,
                    "mask": mask,
                }
        except (OSError, KeyError) as exc:
            raise ValueError(f"Could not load scan-mask state from {path}: {exc}") from exc

    def set_mask(
        self,
        *,
        x=None,
        y=None,
        geometry=None,
        size_x=None,
        size_y=None,
    ):
        self._syncing = True
        try:
            if geometry is not None:
                geometry = str(geometry).lower()
                if geometry not in self.GEOMETRIES:
                    raise ValueError(
                        f"Unknown mask geometry {geometry!r}; choose one of "
                        f"{', '.join(self.GEOMETRIES)}."
                    )
                self.geometry_selector.value = geometry
            if x is not None:
                self.x_slider.value = int(x)
                self.x_input.value = int(x)
            if y is not None:
                self.y_slider.value = self.scan_shape[0] - 1 - int(y)
                self.y_input.value = int(y)
            if size_x is not None:
                self.size_x_slider.value = int(size_x)
                self.size_x_input.value = int(size_x)
            if size_y is not None:
                self.size_y_slider.value = int(size_y)
                self.size_y_input.value = int(size_y)
            if self.geometry in {"circle", "square"}:
                self.size_y_slider.value = self.size_x
                self.size_y_input.value = self.size_x
        finally:
            self._syncing = False
        self._refresh_geometry_controls()
        self._update_preview()
        return self

    def set_circle(self, *, x=None, y=None, radius=None):
        """Compatibility helper that explicitly selects circular geometry."""
        return self.set_mask(
            x=x,
            y=y,
            geometry="circle",
            size_x=radius,
            size_y=radius,
        )

    def apply(self):
        self.analysis.scan_mask = self._preview_mask.copy()
        self._dirty = False
        self._refresh_status("Mask applied")
        if self.autosave and self.state_path is not None:
            self.save(apply_first=False)
        return self.applied_mask

    def save(self, path=None, *, apply_first=True):
        path = self.state_path if path is None else Path(path)
        if path is None:
            raise ValueError("No scan-mask state_path was configured.")
        if apply_first:
            self.apply()
        path.parent.mkdir(parents=True, exist_ok=True)
        sampling = np.asarray(self.analysis.dataset_cartesian.sampling[:2], dtype=float)
        units = np.asarray(
            [str(value) for value in self.analysis.dataset_cartesian.units[:2]],
            dtype="U32",
        )
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                schema_version=np.asarray(self.SCHEMA_VERSION, dtype=np.int64),
                mask=self._preview_mask.astype(bool),
                mask_type=np.asarray(self.geometry),
                geometry=np.asarray(self.geometry),
                center_row=np.asarray(self.y, dtype=np.int64),
                center_column=np.asarray(self.x, dtype=np.int64),
                radius=np.asarray(self.radius, dtype=np.int64),
                size_x=np.asarray(self.size_x, dtype=np.int64),
                size_y=np.asarray(self.size_y, dtype=np.int64),
                scan_shape=np.asarray(self.scan_shape, dtype=np.int64),
                sampling=sampling,
                units=units,
            )
        temporary_path.replace(path)
        self.state_path = path
        self.save_button.disabled = False
        self._saved = True
        self._dirty = False
        self._refresh_status(f"Applied and saved to {path}")
        return path

    def close(self):
        plt.close(self.figure)

    def _restore_initial(self):
        x, y, geometry, size_x, size_y = self._initial_geometry
        self.set_mask(
            x=x,
            y=y,
            geometry=geometry,
            size_x=size_x,
            size_y=size_y,
        )
        self._preview_mask = self._initial_mask.copy()
        self._dirty = True
        self._render()
        self._refresh_status("Initial state restored; click Apply")

    def _set_full_scan(self):
        rows, columns = self.scan_shape
        if self.geometry == "circle":
            size_x = size_y = (
                int(np.ceil(np.hypot(rows - 1, columns - 1))) + 1
            )
        elif self.geometry == "ellipse":
            size_x, size_y = columns, rows
        elif self.geometry == "square":
            size_x = size_y = max(rows, columns)
        else:
            size_x, size_y = columns, rows
        self.set_mask(
            x=columns // 2,
            y=rows // 2,
            size_x=size_x,
            size_y=size_y,
        )

    def _on_x_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.x_input.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_y_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.y_input.value = self.scan_shape[0] - 1 - int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_geometry(self, change):
        if self._syncing:
            return
        self._syncing = True
        try:
            if change["new"] in {"circle", "square"}:
                self.size_y_slider.value = self.size_x
                self.size_y_input.value = self.size_x
        finally:
            self._syncing = False
        self._refresh_geometry_controls()
        self._update_preview()

    def _on_size_x_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.size_x_input.value = int(change["new"])
            if self.geometry in {"circle", "square"}:
                self.size_y_slider.value = int(change["new"])
                self.size_y_input.value = int(change["new"])
        finally:
            self._syncing = False
        self._update_preview()

    def _on_size_y_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.size_y_input.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_x_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.x_slider.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_y_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.y_slider.value = self.scan_shape[0] - 1 - int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_size_x_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.size_x_slider.value = int(change["new"])
            if self.geometry in {"circle", "square"}:
                self.size_y_slider.value = int(change["new"])
                self.size_y_input.value = int(change["new"])
        finally:
            self._syncing = False
        self._update_preview()

    def _on_size_y_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.size_y_slider.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _update_preview(self):
        self._preview_mask = self._geometry_mask(
            self.y, self.x, self.geometry, self.size_x, self.size_y
        )
        self._dirty = True
        self._saved = False
        self._render()
        self._refresh_status("Preview changed; click Apply or Apply & Save")

    def _refresh_geometry_controls(self):
        labels = {
            "circle": ("Radius", None),
            "ellipse": ("X radius", "Y radius"),
            "square": ("Half-size", None),
            "rectangle": ("Half-width", "Half-height"),
        }
        x_label, y_label = labels[self.geometry]
        self.size_x_slider.description = x_label
        self.size_x_input.description = x_label
        if y_label is None:
            self.size_y_slider.layout.display = "none"
            self.size_y_input.layout.display = "none"
            self.size_y_row.layout.display = "none"
        else:
            self.size_y_slider.description = y_label
            self.size_y_input.description = y_label
            self.size_y_slider.layout.display = ""
            self.size_y_input.layout.display = ""
            self.size_y_row.layout.display = ""

    def _replace_boundary_artist(self):
        if self.boundary_artist is not None:
            self.boundary_artist.remove()
        style = {
            "fill": False,
            "edgecolor": "#ff3030",
            "linewidth": 1.05,
            "linestyle": (0, (1.2, 5.5)),
            "alpha": 0.9,
        }
        if self.geometry in {"circle", "ellipse"}:
            size_y = self.size_x if self.geometry == "circle" else self.size_y
            artist = Ellipse(
                (self.x, self.y),
                width=2 * self.size_x,
                height=2 * size_y,
                **style,
            )
        else:
            size_y = self.size_x if self.geometry == "square" else self.size_y
            artist = Rectangle(
                (self.x - self.size_x - 0.5, self.y - size_y - 0.5),
                width=2 * self.size_x + 1,
                height=2 * size_y + 1,
                **style,
            )
        self.ax.add_patch(artist)
        self.boundary_artist = artist
        # Compatibility name retained even when the selected geometry is not circular.
        self.circle_artist = artist

    def _render(self):
        self.mask_artist.set_data(
            np.ma.masked_where(~self._preview_mask, self._preview_mask)
        )
        self._replace_boundary_artist()
        self.center_artist.set_data([self.x], [self.y])
        self.figure.canvas.draw_idle()
        with self.output:
            clear_output(wait=True)
            display(self.figure)

    def _refresh_status(self, message):
        count = int(self._preview_mask.sum())
        total = int(self._preview_mask.size)
        physical = ""
        try:
            row_sampling, column_sampling = (
                float(v) for v in self.analysis.dataset_cartesian.sampling[:2]
            )
            row_unit, column_unit = (
                str(v) for v in self.analysis.dataset_cartesian.units[:2]
            )
            if np.isclose(row_sampling, column_sampling) and row_unit == column_unit:
                if self.geometry == "circle":
                    physical = (
                        f" · radius ≈ {self.size_x * row_sampling:.4g} {row_unit}"
                    )
                else:
                    physical = (
                        f" · half-size ≈ "
                        f"{self.size_x * column_sampling:.4g} × "
                        f"{self.size_y * row_sampling:.4g} {row_unit}"
                    )
        except (TypeError, ValueError):
            pass
        save_state = "saved" if self._saved else "not saved"
        self.status.value = (
            f"<b>{message}</b><br>"
            f"{self.geometry.title()} · X column {self.x} · Y row {self.y} · "
            f"half-size {self.size_x} × {self.size_y} px"
            f"{physical} · {count:,}/{total:,} positions "
            f"({100.0 * count / total:.1f}%) · {save_state}"
        )
