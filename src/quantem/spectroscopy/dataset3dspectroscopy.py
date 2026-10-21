import csv
import json
import os
from typing import Any, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.filter import median_filter
from quantem.spectroscopy.spectroscopy_visualzitions import (
    _plot_background_subtraction as _visualize_background_subtraction,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    _plot_pca_results as _visualize_pca_results,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_attached_spectrum as _visualize_attached_spectrum,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    show_energy_window_map as _visualize_energy_window_map,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    show_mean_spectrum as _visualize_mean_spectrum,
)
from quantem.spectroscopy.utils import (
    _read_csv_without_preamble,
    load_eels_edges_database,
    load_xray_lines_database,
)


class _ModelElementsDict(dict):
    """dict subclass for model_elements with a readable repr."""

    def __repr__(self):
        if not self:
            return "Model Elements:\n  None"
        lines = ["Model Elements:"]
        for element, line_info in self.items():
            if isinstance(line_info, dict) and line_info:
                line_names = ", ".join(sorted(line_info.keys()))
                lines.append(f"  {element}: {line_names}")
            else:
                lines.append(f"  {element}")
        return "\n".join(lines)

    def _repr_html_(self):
        if not self:
            return "<b>Model Elements:</b><br>&nbsp;&nbsp;None"
        rows = "".join(
            f"<tr><td><b>{el}</b></td><td>{', '.join(sorted(info.keys())) if isinstance(info, dict) and info else ''}</td></tr>"
            for el, info in self.items()
        )
        return f"<b>Model Elements:</b><table>{rows}</table>"


class Dataset3dspectroscopy(Dataset3d):
    # stores the element line info so you don't need to reload each time
    element_info = None
    element_info_path = None
    atomic_weights = None

    plot_attached_spectrum = _visualize_attached_spectrum
    _plot_pca_results = _visualize_pca_results
    show_mean_spectrum = _visualize_mean_spectrum
    show_energy_window_map = _visualize_energy_window_map
    _plot_background_subtraction = _visualize_background_subtraction

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=type(self)._token if _token is None else _token,
        )

        self.model_elements = _ModelElementsDict()
        self.attached_spectra = None

    # loads elemental information
    @classmethod
    def load_element_info(cls):
        """Load element database for XEDS X-ray lines or EELS edges."""
        if cls.element_info is not None:
            return cls.element_info

        path = getattr(cls, "element_info_path", None)
        if path is None:
            raise NotImplementedError(
                f"{cls.__name__} must define `element_info_path` to load element metadata."
            )
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

        dataset_type = str(getattr(cls, "dataset_type", "")).lower()
        if path.lower().endswith(".csv"):
            if dataset_type == "eels":
                cls.element_info = load_eels_edges_database(full_path)
            else:
                cls.element_info = load_xray_lines_database(full_path)
        else:
            with open(full_path, "r", encoding="utf-8") as f:
                cls.element_info = json.load(f)["elements"]

        if dataset_type == "xeds":
            cls._normalize_element_info()

        return cls.element_info

    @classmethod
    def _ensure_element_info(cls):
        """Load and return the cached element metadata."""
        return cls.load_element_info() or {}

    @classmethod
    def load_atomic_weights(cls):
        """Load atomic weights table from CSV once per class."""
        if cls.atomic_weights is not None:
            return cls.atomic_weights

        atomic_weights_path = "atomic_weights.csv"
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), atomic_weights_path)
        data = {}
        reader = csv.reader(_read_csv_without_preamble(full_path))
        for row_index, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{atomic_weights_path} row {row_index} must contain element symbol and weight"
                )
            symbol = str(row[0]).strip()
            weight_raw = str(row[1]).strip()
            if not symbol:
                continue
            try:
                weight = float(weight_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{atomic_weights_path} row {row_index} has invalid weight: {weight_raw!r}"
                ) from exc
            data[symbol] = weight

        if not data:
            raise ValueError(f"{atomic_weights_path} did not contain any atomic weights")

        cls.atomic_weights = data
        return cls.atomic_weights

    @staticmethod
    def _normalize_element_specs(specs):
        if isinstance(specs, str):
            return [s.strip() for s in specs.split(",") if s.strip()]
        if isinstance(specs, (list, tuple, set)):
            out = []
            for spec in specs:
                out.extend([s.strip() for s in str(spec).split(",") if s.strip()])
            return out
        raise TypeError("elements must be a string or a sequence of strings")

    @staticmethod
    def _resolve_element_key(all_info, token):
        token_norm = str(token).strip().lower()
        return next((key for key in all_info if str(key).lower() == token_norm), None)

    @staticmethod
    def _line_matches_selectors(line_name, selectors):
        if not selectors:
            return True
        line_norm = str(line_name).strip().lower()
        return any(line_norm == sel or line_norm.startswith(sel) for sel in selectors)

    @staticmethod
    def _line_info_matches_selectors(line_info, selectors):
        if not selectors or not isinstance(line_info, dict):
            return False
        edge_label = str(line_info.get("edge_label", "")).strip().lower()
        return bool(edge_label) and any(
            edge_label == sel or edge_label.startswith(sel) for sel in selectors
        )

    @classmethod
    def _select_lines(cls, line_dict, selectors):
        if not isinstance(line_dict, dict):
            return {}
        if not selectors:
            return dict(line_dict)

        selector_norm = [str(sel).strip().lower() for sel in selectors if str(sel).strip()]
        return {
            line_name: line_info
            for line_name, line_info in line_dict.items()
            if cls._line_matches_selectors(line_name, selector_norm)
            or cls._line_info_matches_selectors(line_info, selector_norm)
        }

    def add_elements_to_model(self, elements):
        """
        Add elements to the model for persistent use in show_mean_spectrum.

        Parameters
        ----------
        elements : list or str
            Element/line spec(s) to add. Examples:
            - 'Al' (all lines for Al)
            - 'Te La' (only Te La line)
            - ['Au Ma', 'Te La', 'Si']
        """
        all_info = type(self)._ensure_element_info()
        if all_info is None:
            return

        specs = type(self)._normalize_element_specs(elements)
        added_this_call = {}

        for spec in specs:
            tokens = str(spec).split()
            if not tokens:
                continue

            element_key = type(self)._resolve_element_key(all_info, tokens[0])
            if element_key is None:
                continue

            selectors = tokens[1:]
            selected_lines = type(self)._select_lines(all_info[element_key], selectors)
            if not selected_lines:
                continue

            existing_before = self.model_elements.get(element_key)
            if not isinstance(existing_before, dict):
                existing_before = {}
            existing_keys_before = set(existing_before.keys())
            if not selectors:
                self.model_elements[element_key] = selected_lines
            else:
                existing = self.model_elements.get(element_key)
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(selected_lines)
                self.model_elements[element_key] = existing

            added_keys = [
                line_name
                for line_name in selected_lines.keys()
                if line_name not in existing_keys_before
            ]
            if added_keys:
                if element_key not in added_this_call:
                    added_this_call[element_key] = []
                added_this_call[element_key].extend(added_keys)
        if not self.model_elements:
            self.model_elements = _ModelElementsDict()

        if added_this_call:
            print("Added to model:")
            for element_key in sorted(added_this_call.keys()):
                unique_lines = sorted(
                    set(str(line_name) for line_name in added_this_call[element_key])
                )
                print(f" - {element_key}: {', '.join(unique_lines)}")
        else:
            print("Added to model: nothing new")

    def median_filter_masked_energies(self, mask: np.ndarray, kernel_width: int = 3):
        """
        Fix bad energy channels by median-filtering along the energy axis in-place.

        This applies a 1D median filter along dimension 2 and writes back only the
        masked energy channels, broadcasting across all scan positions.

        Parameters
        ----------
        mask
            A boolean mask with shape ``(energy,)`` specifying bad energy channels.
        kernel_width
            Width of the median kernel along the energy axis.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.array.shape[2],):
            raise ValueError(
                "Mask shape must match the energy dimension of the dataset. "
                + f"Expected {(self.array.shape[2],)}, got {mask.shape}."
            )

        filtered = median_filter(self.array, size=kernel_width, axes=(2,))
        self.array[:, :, mask] = filtered[:, :, mask]

    def remove_elements_from_model(self, elements):
        """
        Remove element(s) from the persistent model used in show_mean_spectrum.

        Parameters
        ----------
        elements : list or str
            Element/line spec(s) to remove. Examples:
            - 'Al' (remove all Al lines)
            - 'Te La' (remove only Te La line)
            - ['Au Ma', 'Te La']
        """
        if not self.model_elements:
            return

        specs = type(self)._normalize_element_specs(elements)
        for spec in specs:
            tokens = str(spec).split()
            if not tokens:
                continue

            element_key = type(self)._resolve_element_key(self.model_elements, tokens[0])
            if element_key is None:
                continue

            selectors = [str(token).strip().lower() for token in tokens[1:] if str(token).strip()]
            if not selectors:
                self.model_elements.pop(element_key, None)
                continue

            lines_info = self.model_elements.get(element_key)
            if not isinstance(lines_info, dict):
                self.model_elements.pop(element_key, None)
                continue

            self.model_elements[element_key] = {
                line_name: line_info
                for line_name, line_info in lines_info.items()
                if not type(self)._line_matches_selectors(line_name, selectors)
            }
            if not self.model_elements[element_key]:
                self.model_elements.pop(element_key, None)

        if not self.model_elements:
            self.model_elements = _ModelElementsDict()

    def clear_model_elements(self):
        """Clear all elements from the model."""
        self.model_elements = _ModelElementsDict()

    # Storage of spectra alongside dataset

    def add_spectrum_to_data(self, spectrum, energy_axis):
        """
        Store processed spectra in the 3D spectroscopy dataset structure, in a 1D array of 2D arrays. By default, calculate_mean_spectrum will store spectrum at first available index
        """
        from quantem.core.datastructures.dataset1d import Dataset1d

        energy_sampling = (
            energy_axis[1] - energy_axis[0] if len(energy_axis) > 1 else float(self.sampling[2])
        )
        two_d_spectrum = Dataset1d.from_array(
            array=spectrum,
            origin=energy_axis[0],
            sampling=energy_sampling,
            units=self.units[2],
        )

        if self.attached_spectra is not None:
            self.attached_spectra.append(two_d_spectrum)
        else:
            self.attached_spectra = []
            self.attached_spectra.append(two_d_spectrum)

    def clear_attached_spectra(self):
        self.attached_spectra = None

    ## PCA ANALYSIS METHODS
    def perform_pca(
        self,
        n_components: int = 10,
        standardize: bool = True,
        mask: Optional[NDArray] = None,
        plot_results: bool = True,
        return_results=False,
    ) -> dict:
        """
        Perform Principal Component Analysis (PCA) on the spectroscopy dataset.

        Parameters
        ----------
        n_components : int
            Number of principal components to compute
        standardize : bool
            If True, standardize the data before PCA (zero mean, unit variance)
        mask : Optional[NDArray]
            Optional spatial mask to select pixels for analysis. Accepts shape
            (scan_row, scan_col) or a flattened spatial mask.
        plot_results : bool
            If True, plot the explained variance and first few components

        Returns
        -------
        dict
            Dictionary containing:
            - 'pca': PCA result attributes
            - 'components': principal component spectra (n_components x n_energy)
            - 'loadings': spatial loadings (n_components x scan_row x scan_col)
            - 'explained_variance_ratio': explained variance for each component
            - 'reconstructed': reconstructed dataset (dataset3dspectroscopy) using n_components
        """

        from quantem.spectroscopy import Dataset3deels, Dataset3dxeds

        data = np.asarray(self.array, dtype=float)
        scan_row, scan_col, n_energy = data.shape
        n_pixels = scan_row * scan_col

        spectra = data.reshape(n_pixels, n_energy)

        pixel_mask = np.ones(n_pixels, dtype=bool)

        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape == (scan_row, scan_col):
                pixel_mask = mask_array.reshape(-1)
            elif mask_array.shape == (n_pixels,):
                pixel_mask = mask_array
            else:
                raise ValueError(
                    f"mask shape {mask_array.shape} must match spatial shape {(scan_row, scan_col)} "
                    f"or flattened shape {(n_pixels,)}"
                )

        if not np.any(pixel_mask):
            raise ValueError("mask must select at least one spatial pixel")

        selected_spectra = spectra[pixel_mask]

        if standardize:
            mean = np.mean(selected_spectra, axis=0)
            std = np.std(selected_spectra, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            pca_input = (selected_spectra - mean) / std
        else:
            mean = np.zeros(n_energy)
            std = np.ones(n_energy)
            pca_input = selected_spectra

        (
            components,
            loadings,
            explained_variance,
            explained_variance_ratio,
            reconstructed,
        ) = self._run_pca(pca_input, n_components)

        reconstructed = reconstructed * std + mean

        loadings_flat = np.zeros((n_components, n_pixels), dtype=loadings.dtype)
        loadings_flat[:, pixel_mask] = loadings.T
        loadings_spatial = loadings_flat.reshape(n_components, scan_row, scan_col)

        if plot_results:
            self._plot_pca_results(
                components,
                loadings_spatial,
                explained_variance_ratio,
                n_show=min(4, n_components),
            )

        reconstructed_spectra = spectra.copy()
        reconstructed_spectra[pixel_mask] = reconstructed
        reconstructed_array = reconstructed_spectra.reshape(scan_row, scan_col, n_energy)

        dataset_type = str(self.dataset_type).lower()
        if dataset_type == "xeds":
            dataset_class = Dataset3dxeds
        elif dataset_type == "eels":
            dataset_class = Dataset3deels
        else:
            raise ValueError(f"Unsupported spectroscopy dataset_type {self.dataset_type!r}")

        reconstructed_data3d = dataset_class.from_array(
            array=reconstructed_array,
            sampling=self.sampling,
            origin=self.origin,
            units=self.units,
        )

        if return_results:
            return {
                "pca": {
                    "components_": components,
                    "explained_variance_": explained_variance,
                    "explained_variance_ratio_": explained_variance_ratio,
                },
                "components": components,
                "loadings": loadings_spatial,
                "explained_variance_ratio": explained_variance_ratio,
                "explained_variance": explained_variance,
                "reconstructed": reconstructed_data3d,
            }

    def _run_pca(self, data: NDArray | Any, n_components: int):
        array = np.asarray(data, dtype=float)
        n_samples, n_features = array.shape
        max_components = min(n_samples, n_features)
        if not 1 <= n_components <= max_components:
            raise ValueError(f"n_components={n_components} must be between 1 and {max_components}")

        mean = np.mean(array, axis=0)
        centered = torch.as_tensor(array - mean, dtype=torch.float64)
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)

        components = vh[:n_components].cpu().numpy()
        loadings = (centered @ vh[:n_components].T).cpu().numpy()

        denom = max(n_samples - 1, 1)
        explained_variance = ((s[:n_components] ** 2) / denom).cpu().numpy()
        total_variance = torch.sum((s**2) / denom).item()
        explained_variance_ratio = (
            explained_variance / total_variance
            if total_variance > 0
            else np.zeros_like(explained_variance)
        )
        reconstructed = loadings @ components + mean

        return (
            components,
            loadings,
            explained_variance,
            explained_variance_ratio,
            reconstructed,
        )

    def _calibrated_position_to_pixel(self, value, axis):
        if value is None:
            return None

        sampling = float(self.sampling[axis])
        if sampling == 0:
            raise ValueError(f"Cannot convert calibrated ROI on axis {axis}: sampling is zero")

        origin = float(self.origin[axis]) if hasattr(self, "origin") else 0.0
        return int(np.round((float(value) - origin) / sampling))

    def _calibrated_span_to_pixels(self, value, axis):
        if value is None:
            return None

        sampling = abs(float(self.sampling[axis]))
        if sampling == 0:
            raise ValueError(
                f"Cannot convert calibrated ROI span on axis {axis}: sampling is zero"
            )

        pixels = int(np.round(float(value) / sampling))
        if pixels < 1:
            raise ValueError(
                f"Calibrated ROI span on axis {axis} converts to {pixels} pixels; expected >= 1"
            )
        return pixels

    def _validate_roi_bounds(self, y, x, dy, dx):
        errs = []
        ymax = int(self.shape[0])
        xmax = int(self.shape[1])

        for name, val in (("y", y), ("x", x), ("dy", dy), ("dx", dx)):
            if val is None:
                errs.append(f"{name} is None (missing after normalization).")

        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

        if y < 0:
            errs.append(f"y={y} < 0")
        if x < 0:
            errs.append(f"x={x} < 0")
        if dy < 1:
            errs.append(f"dy={dy} < 1")
        if dx < 1:
            errs.append(f"dx={dx} < 1")

        if y >= ymax:
            errs.append(f"y start {y} out of bounds [0, {ymax - 1}]")
        if x >= xmax:
            errs.append(f"x start {x} out of bounds [0, {xmax - 1}]")

        end_y = y + dy
        end_x = x + dx
        if end_y > ymax:
            errs.append(f"y+dy = {end_y} exceeds height {ymax}")
        if end_x > xmax:
            errs.append(f"x+dx = {end_x} exceeds width {xmax}")

        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

    def _resolve_roi(self, roi=None, roi_cal=None):
        selector_count = int(roi is not None) + int(roi_cal is not None)
        if selector_count > 1:
            raise ValueError("Use only one ROI selector: roi or roi_cal")

        if roi is not None:
            roi_spec = roi
        elif roi_cal is not None:
            if len(roi_cal) == 2:
                y_cal, x_cal = roi_cal
                roi_spec = [
                    self._calibrated_position_to_pixel(y_cal, axis=0),
                    self._calibrated_position_to_pixel(x_cal, axis=1),
                ]
            elif len(roi_cal) == 4:
                y_cal, x_cal, dy_cal, dx_cal = roi_cal
                roi_spec = [
                    self._calibrated_position_to_pixel(y_cal, axis=0),
                    self._calibrated_position_to_pixel(x_cal, axis=1),
                    self._calibrated_span_to_pixels(dy_cal, axis=0),
                    self._calibrated_span_to_pixels(dx_cal, axis=1),
                ]
            else:
                raise ValueError("roi_cal must be [y, x] or [y, x, dy, dx]")
        else:
            roi_spec = None

        if roi_spec is None:
            y, x, dy, dx = 0, 0, int(self.shape[0]), int(self.shape[1])
        elif len(roi_spec) == 2:
            y, x = roi_spec
            y, x, dy, dx = int(y), int(x), 1, 1
        elif len(roi_spec) == 4:
            y_val, x_val, dy_val, dx_val = roi_spec
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[0]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[1]) - x if dx_val is None else int(dx_val)
        else:
            raise ValueError(
                "ROI must be None, [y, x], or [y, x, dy, dx]. Use one selector: roi or roi_cal"
            )

        self._validate_roi_bounds(y, x, dy, dx)
        return y, x, dy, dx

    def calculate_mean_spectrum(
        self,
        roi=None,
        energy_range=None,
        mask=None,
        attach_mean_spectrum=True,
        roi_cal=None,
        normalize=False,
    ):
        """Calculate a spectrum from a spatial ROI.

        Parameters
        ----------
        normalize : bool, optional
            If ``True``, scale the mean spectrum to the range [0, 1]. If
            ``False``, return the mean spectrum in original intensity units.
        """
        y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)

        # SPECTRUM CALCULATION --------------------------------------------------------------

        E = self.energy_axis

        # MASK HANDLING ---------------------------------------------------------------------
        if mask is not None:
            # Convert to ndarray and validate
            mask = np.asarray(mask)

            # Check that it's a proper ndarray
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Mask must be a numpy ndarray, got {type(mask)}")

            # Check dimensions - must be 1D
            if mask.ndim != 1:
                raise ValueError(
                    f"Mask must be 1-dimensional, got {mask.ndim}D array with shape {mask.shape}"
                )

            # Convert to bool dtype and validate
            if mask.dtype != bool:
                try:
                    mask = mask.astype(bool)
                except (ValueError, TypeError):
                    raise TypeError(f"Mask cannot be converted to boolean dtype from {mask.dtype}")

            # Check shape matches energy axis
            arr = np.asarray(self.array, dtype=float)
            if mask.shape != (arr.shape[2],):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match energy axis shape ({arr.shape[2]},)"
                )

            arr = arr[y : y + dy, x : x + dx, :][:, :, mask]  # select ROI and masked energies
            arr = np.moveaxis(arr, -1, 0)  # move energy axis to front: (num_masked, dy, dx)
            if arr.shape[0] > 0:
                spec = arr.mean(axis=(1, 2))
            else:
                spec = np.zeros(0)
            E = E[mask]  # Mask the energy axis as well
        else:
            spec = np.empty(self.shape[2], dtype=float)
            for k in range(self.shape[2]):
                img = np.asarray(self.array[:, :, k], dtype=float)
                roi_data = img[y : y + dy, x : x + dx]
                if roi_data.size == 0:
                    raise ValueError("ROI is empty; check y/x/dy/dx.")
                spec[k] = roi_data.mean()

        # APPLY ENERGY RANGE ---------------------------------------------------------------

        if energy_range is not None:
            # Check for errors in energy_range input
            if energy_range[0] >= energy_range[1]:
                raise ValueError("Invalid energy range parameter.")

            # If the entire energy range specified is outside the original energy range of the data, raise an error.
            if energy_range[1] < E[0] or energy_range[0] > E[-1]:
                raise ValueError("Energy range parameter is outside of data bounds.")

            # If either side of input energy_range is beyond the original energy range of the data, default to the limit of the data instead.
            energy_min = np.maximum(energy_range[0], E[0])
            energy_max = np.minimum(energy_range[1], E[-1])

            indices = np.where((E >= energy_min) & (E <= energy_max))[0]
            spec = spec[indices]
            E = E[indices]

        if normalize and spec.size > 0:
            finite = np.isfinite(spec)
            if np.any(finite):
                spec_min = np.min(spec[finite])
                spec_max = np.max(spec[finite])
                if spec_max > spec_min:
                    spec = (spec - spec_min) / (spec_max - spec_min)
                else:
                    spec = np.zeros_like(spec, dtype=float)

        if attach_mean_spectrum:
            self.add_spectrum_to_data(spec, E)

        return spec

    # BACKGROND SUBTRACTION

    def subtract_background(
        self,
        roi=None,
        energy_range=None,
        mask=None,
        target_edge=None,
        window_size=None,
        method="powerlaw",
        polynomial_degree=3,
        return_dataset=True,
        attach_spectrum=True,
        fit_mode="global",
        kernel_width=1,
        show=True,
        show_subtracted=True,
        return_background=False,
    ):
        """
        Subtract fitted background from a 3D spectroscopy dataset.

        Parameters
        ----------
        fit_mode : {"global", "local"}, optional
            ``"global"`` fits one background to the ROI mean spectrum and subtracts
            it from every probe position. ``"local"`` fits a background at each
            probe position from the average spectrum of its nearest spatial
            neighbors.
        kernel_width : int, optional
            Number of nearest spatial neighbors to average for each local
            background fit. The current pixel is included. Used only when
            ``fit_mode="local"``.
        window_size : int, optional
            For XEDS, number of spectral channels in the rolling low-percentile
            envelope used before polynomial fitting. For EELS power-law fitting,
            percent of ``target_edge`` used for the pre-edge fit window. Defaults
            to 50 channels for XEDS and 10 percent for EELS.
        show : bool, optional
            If True, plot the mean raw spectrum, fitted background, and
            background-subtracted spectrum.
        polynomial_degree : int, optional
            Degree of the polynomial power-series background used for XEDS data.
            Ignored for EELS data.
        return_background : bool, optional
            If True, return ``(dataset, background_cube)`` when ``return_dataset``
            is True, otherwise return the background cube.

        Returns
        -------
        Dataset3dspectroscopy or tuple or ndarray or None
            Background-subtracted dataset by default. If ``return_background`` is
            True, also returns the fitted background cube.
        """

        from quantem.spectroscopy import Dataset3deels, Dataset3dxeds

        fit_mode = str(fit_mode).lower()
        if fit_mode not in {"global", "local"}:
            raise ValueError("fit_mode must be 'global' or 'local'")

        E, indices = self._background_energy_axis_and_indices(energy_range, mask)
        array3d = np.asarray(self.array, dtype=float)[:, :, indices]
        y, x, dy, dx = self._resolve_roi(roi=roi)

        if fit_mode == "global":
            input_spectrum = array3d[y : y + dy, x : x + dx, :].mean(axis=(0, 1))
            background = self._fit_background_spectrum(
                input_spectrum,
                E,
                method=method,
                target_edge=target_edge,
                window_size=window_size,
                polynomial_degree=polynomial_degree,
            )
            background_cube = np.broadcast_to(background[None, None, :], array3d.shape)
        else:
            background_cube = self._fit_local_background_cube(
                array3d,
                E,
                method=method,
                target_edge=target_edge,
                window_size=window_size,
                polynomial_degree=polynomial_degree,
                kernel_width=kernel_width,
            )

        spec3D_subtracted = np.maximum(array3d - background_cube, 0)
        input_mean_spectrum = array3d[y : y + dy, x : x + dx, :].mean(axis=(0, 1))
        background_mean_spectrum = background_cube[y : y + dy, x : x + dx, :].mean(axis=(0, 1))
        subtracted_mean_spectrum = spec3D_subtracted[y : y + dy, x : x + dx, :].mean(axis=(0, 1))

        if attach_spectrum:
            self.add_spectrum_to_data(subtracted_mean_spectrum, E)

        if show:
            self._plot_background_subtraction(
                E,
                input_mean_spectrum,
                background_mean_spectrum,
                subtracted_mean_spectrum,
                fit_mode=fit_mode,
                show_subtracted=show_subtracted,
            )

        dataset_type = str(self.dataset_type).lower()
        if dataset_type == "xeds":
            dataset_class = Dataset3dxeds
        elif dataset_type == "eels":
            dataset_class = Dataset3deels
        else:
            raise ValueError(f"Unsupported spectroscopy dataset_type {self.dataset_type!r}")

        output_origin = np.array(self.origin, dtype=float, copy=True)
        output_origin[2] = E[0]

        if return_dataset:
            subtracted_dataset = dataset_class.from_array(
                array=spec3D_subtracted,
                sampling=self.sampling,
                origin=output_origin,
                units=self.units,
            )
            if return_background:
                background_dataset = dataset_class.from_array(
                    array=np.array(background_cube, copy=True),
                    sampling=self.sampling,
                    origin=output_origin,
                    units=self.units,
                )
                return subtracted_dataset, background_dataset
            return subtracted_dataset

        if return_background:
            return background_cube

        print("Notice: no 3D dataset was returned")

    def _background_energy_axis_and_indices(self, energy_range, mask):
        E = np.asarray(self.energy_axis, dtype=float)
        selected = np.ones(E.shape, dtype=bool)

        if energy_range is not None:
            if len(energy_range) != 2:
                raise ValueError("energy_range must be [min_energy, max_energy]")
            e_min = float(energy_range[0])
            e_max = float(energy_range[1])
            if e_min >= e_max:
                raise ValueError("Invalid energy range parameter.")
            if e_max < E[0] or e_min > E[-1]:
                raise ValueError("Energy range parameter is outside of data bounds.")
            e_min = max(e_min, E[0])
            e_max = min(e_max, E[-1])
            selected &= (E >= e_min) & (E <= e_max)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != E.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match energy axis shape {E.shape}"
                )
            selected &= mask

        if not np.any(selected):
            raise ValueError("No energy channels selected. Adjust energy_range or mask")

        indices = np.where(selected)[0]
        return E[indices], indices

    def _fit_background_spectrum(
        self,
        spectrum,
        energy_axis,
        method,
        target_edge,
        window_size,
        polynomial_degree=3,
    ):
        dataset_type = str(self.dataset_type).lower()
        spectrum = np.asarray(spectrum, dtype=float)

        if dataset_type == "xeds":
            return self.calculate_background_polynomial(
                spectrum,
                energy_axis=np.asarray(energy_axis, dtype=float),
                degree=polynomial_degree,
                window_size=50 if window_size is None else window_size,
            )

        if dataset_type != "eels":
            raise ValueError(f"Unsupported spectroscopy dataset_type {self.dataset_type!r}")

        method = str(method).lower()
        if method == "iterative":
            return np.full_like(spectrum, self.calculate_background_iterative(spectrum))
        if method != "powerlaw":
            raise ValueError("EELS background method must be 'powerlaw' or 'iterative'")
        if target_edge is None:
            raise ValueError("target_edge is required for EELS powerlaw background fitting")

        return self._fit_eels_powerlaw_background(
            spectrum,
            np.asarray(energy_axis, dtype=float),
            target_edge=target_edge,
            window_size=10 if window_size is None else window_size,
        )

    def _fit_eels_powerlaw_background(self, spectrum, energy_axis, target_edge, window_size):
        from scipy.optimize import curve_fit

        if window_size < 10 or window_size > 30:
            raise ValueError("Invalid window size. Please input a value of between 10 and 30.")

        target_edge = float(target_edge)
        if target_edge < energy_axis[0] or target_edge > energy_axis[-1]:
            raise ValueError("Target edge is outside of energy range.")

        window_minE = (target_edge - 5) - target_edge * (float(window_size) / 100)
        window_maxE = target_edge - 5
        if window_minE < energy_axis[0]:
            raise ValueError(
                "Insufficient pre-edge background fitting region: "
                f"target_edge={target_edge:g} eV with window_size={window_size}% requires "
                f"{window_minE:.3f}-{window_maxE:.3f} eV, but the available energy range starts at "
                f"{energy_axis[0]:.3f} eV."
            )

        window_indices = np.where((energy_axis >= window_minE) & (energy_axis <= window_maxE))[0]
        if len(window_indices) < 2:
            raise ValueError(
                "Insufficient points in EELS pre-edge background fitting window: "
                f"found {len(window_indices)} points in {window_minE:.3f}-{window_maxE:.3f} eV."
            )

        window_E = energy_axis[window_indices]
        window_I = np.asarray(spectrum, dtype=float)[window_indices]

        def powerlaw_function(E, A, r):
            return A * (E ** (-r))

        popt, _ = curve_fit(powerlaw_function, window_E, window_I, maxfev=2000)
        return powerlaw_function(energy_axis, popt[0], popt[1])

    def _fit_local_background_cube(
        self,
        array3d,
        energy_axis,
        method,
        target_edge,
        window_size,
        polynomial_degree,
        kernel_width,
    ):
        from scipy.spatial import cKDTree

        scan_row, scan_col, n_energy = array3d.shape
        n_pixels = scan_row * scan_col
        try:
            n_neighbors = int(kernel_width)
        except (TypeError, ValueError) as exc:
            raise TypeError("kernel_width must be an integer") from exc
        if n_neighbors < 1:
            raise ValueError("kernel_width must be >= 1")
        n_neighbors = min(n_neighbors, n_pixels)

        row_indices, col_indices = np.indices((scan_row, scan_col))
        coords = np.column_stack((row_indices.reshape(-1), col_indices.reshape(-1)))
        _, neighbor_indices = cKDTree(coords).query(coords, k=n_neighbors)
        if n_neighbors == 1:
            neighbor_indices = neighbor_indices[:, None]

        spectra = array3d.reshape(n_pixels, n_energy)
        background = np.empty_like(spectra)

        for pixel_index, neighbors in enumerate(neighbor_indices):
            local_spectrum = spectra[neighbors].mean(axis=0)
            try:
                background[pixel_index] = self._fit_background_spectrum(
                    local_spectrum,
                    energy_axis,
                    method=method,
                    target_edge=target_edge,
                    window_size=window_size,
                    polynomial_degree=polynomial_degree,
                )
            except Exception as exc:
                i_row, i_col = divmod(pixel_index, scan_col)
                raise RuntimeError(
                    f"Background fit failed at pixel ({i_row}, {i_col}) "
                    f"using method={method!r}, target_edge={target_edge}, "
                    f"window_size={window_size}, kernel_width={kernel_width}: {exc}"
                ) from exc

        return background.reshape(scan_row, scan_col, n_energy)

    @property
    def energy_axis(self):
        energy_axis = np.arange(self.shape[2]) * self.sampling[2] + self.origin[2]
        return energy_axis
