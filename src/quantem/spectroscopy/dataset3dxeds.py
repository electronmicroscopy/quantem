import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.visualization import show_2d
from quantem.spectroscopy import Dataset3dspectroscopy
from quantem.spectroscopy.dataset3dxeds_fitting import (
    peak_autoid as _peak_autoid,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    show_spectrum_images as _visualize_spectrum_images,
)
from quantem.spectroscopy.xeds_fitting import (
    _fit_mean_model_pytorch as _fit_mean_model_pytorch_impl,
)
from quantem.spectroscopy.xeds_fitting import (
    fit_spectrum_mean_pytorch as _fit_spectrum_mean_pytorch,
)
from quantem.spectroscopy.xeds_fitting import (
    fit_spectrum_pytorch as _fit_spectrum_pytorch,
)


class Dataset3dxeds(Dataset3dspectroscopy):
    """An XEDS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (scan_row, scan_col, energy).
    The first two dimensions represent real space sampling, while the last dimension
    represents the energy axis.

    """

    element_info = None
    element_info_path = "x_ray_lines.csv"

    show_spectrum_images = _visualize_spectrum_images
    peak_autoid = _peak_autoid
    _fit_mean_model_pytorch = _fit_mean_model_pytorch_impl
    fit_spectrum_mean_pytorch = _fit_spectrum_mean_pytorch
    fit_spectrum_pytorch = _fit_spectrum_pytorch

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
        """Initialize a 3D XEDS dataset."""
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )
        self.dataset_type = "xeds"

    @staticmethod
    def _normalize_specs(specs, param_name="spec", allow_none=False):
        """Parse specs into a flat list of stripped strings."""
        if specs is None:
            if allow_none:
                return None
            raise TypeError(f"{param_name} must be a string or sequence of strings")
        if isinstance(specs, str):
            return [s.strip() for s in specs.split(",") if s.strip()]
        if isinstance(specs, (list, tuple, set)):
            return [s.strip() for item in specs for s in str(item).split(",") if s.strip()]
        raise TypeError(f"{param_name} must be a string or sequence of strings")

    @staticmethod
    def _normalize_token(text):
        """Return a lowercase alphanumeric-only token for fuzzy matching."""
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    @staticmethod
    def _ordered_element_keys(all_info):
        """Return element keys sorted longest-first for greedy prefix matching."""
        return sorted(map(str, all_info), key=lambda k: (-len(k), k))

    @classmethod
    def _resolve_element_from_label(cls, label, ordered_elements):
        """Extract the element name from a line label like 'FeKa1'."""
        label = str(label)
        for element in ordered_elements:
            if label.startswith(element):
                return element
        m = re.match(r"^[A-Z][a-z]?", label)
        return m.group(0) if m else None

    @classmethod
    def _ensure_element_info(cls):
        """Load element X-ray line data if not already cached."""
        if cls.element_info is None:
            cls.load_element_info()
        return cls.element_info or {}

    @classmethod
    def _normalize_element_info(cls, combine_close_peaks=True, energy_threshold_ev=15):
        """Normalize XEDS X-ray lines and optionally merge unresolved line families."""
        if not isinstance(cls.element_info, dict):
            return cls.element_info

        threshold_kev = float(energy_threshold_ev) / 1000.0

        def line_family(line_name):
            canonical = cls._canonical_line_name(line_name).strip()
            match = re.match(r"^([A-Za-z]+)", canonical)
            return match.group(1) if match else canonical

        def normalized_line_name(line_name):
            canonical = cls._canonical_line_name(line_name).strip()
            match = re.match(r"^([A-Za-z]+)\d+(?:,\d+)+$", canonical)
            return match.group(1) if match else canonical

        def unique_name(lines, name):
            if name not in lines:
                return name
            idx = 2
            while f"{name}__{idx}" in lines:
                idx += 1
            return f"{name}__{idx}"

        def merged_info(entries):
            weights = np.asarray([entry["weight"] for entry in entries], dtype=float)
            energies = np.asarray([entry["energy"] for entry in entries], dtype=float)
            weight_sum = np.sum(weights)
            if weight_sum > 0.0:
                energy = np.sum(energies * weights) / weight_sum
            else:
                energy = np.mean(energies)
            return {"energy (keV)": energy, "weight": weight_sum}

        normalized_info = {}
        for element, lines in cls.element_info.items():
            if not isinstance(lines, dict):
                normalized_info[element] = lines
                continue

            entries_by_family = {}
            normalized_lines = {}
            for line_name, line_info in lines.items():
                if not isinstance(line_info, dict):
                    continue
                try:
                    energy = float(line_info.get("energy (keV)", line_info.get("energy")))
                except (TypeError, ValueError):
                    continue
                try:
                    weight = float(line_info.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight = 0.0

                entry = {
                    "line": normalized_line_name(line_name),
                    "family": line_family(line_name),
                    "energy": energy,
                    "weight": weight,
                }
                entries_by_family.setdefault(entry["family"], []).append(entry)

            for family, entries in entries_by_family.items():
                entries = sorted(entries, key=lambda entry: entry["energy"])
                if not combine_close_peaks:
                    for entry in entries:
                        name = unique_name(normalized_lines, entry["line"])
                        normalized_lines[name] = {
                            "energy (keV)": entry["energy"],
                            "weight": entry["weight"],
                        }
                    continue

                clusters = []
                current = []
                for entry in entries:
                    if not current or entry["energy"] - current[0]["energy"] <= threshold_kev:
                        current.append(entry)
                    else:
                        clusters.append(current)
                        current = [entry]
                if current:
                    clusters.append(current)

                for cluster in clusters:
                    name = family if len(cluster) > 1 else cluster[0]["line"]
                    name = unique_name(normalized_lines, name)
                    normalized_lines[name] = merged_info(cluster)

            normalized_info[element] = dict(
                sorted(
                    normalized_lines.items(),
                    key=lambda item: (item[1]["energy (keV)"], item[0]),
                )
            )

        cls.element_info = normalized_info
        return cls.element_info

    @classmethod
    def _parse_element_selectors(cls, specs, *, allow_none=False, param_name="spec"):
        """Parse element/line specifiers into a dict of {element: set_of_suffixes | None}."""
        tokens = cls._normalize_specs(specs, param_name=param_name, allow_none=allow_none)
        if tokens is None:
            return None

        ordered = cls._ordered_element_keys(cls._ensure_element_info())
        out: dict[str, set[str] | None] = {}
        for raw in tokens:
            compact = re.sub(r"[\s_-]+", "", str(raw).strip())
            if not compact:
                continue
            element = next((k for k in ordered if compact.lower().startswith(k.lower())), None)
            if element is None:
                raise ValueError(f"Could not resolve element from specifier '{raw}'")
            suffix = compact[len(element) :]
            out.setdefault(element, None if not suffix else set())
            if suffix and out[element] is not None:
                out[element].add(suffix)
        return out or None

    @staticmethod
    def _canonical_line_name(line_name: str) -> str:
        """Strip any suffix after '__' from a line name."""
        return str(line_name).split("__", 1)[0]

    @classmethod
    def _iter_selected_lines(cls, element: str, suffix: str, *, raw_spec: str):
        """Yield (line_name, line_info) pairs matching an element and optional suffix."""
        lines = cls._ensure_element_info().get(element) or {}
        if not lines:
            raise ValueError(f"No X-ray lines found for element '{element}'")
        if not suffix:
            yield from lines.items()
            return

        suffix = cls._normalize_token(suffix)
        exact, prefix = [], []
        for line_name, line_info in lines.items():
            token = cls._normalize_token(cls._canonical_line_name(line_name))
            if token == suffix:
                exact.append((line_name, line_info))
            if token.startswith(suffix):
                prefix.append((line_name, line_info))
        matches = exact or prefix
        if not matches:
            raise ValueError(
                f"No X-ray lines matched specifier '{raw_spec}' for element '{element}'"
            )
        yield from matches

    @classmethod
    def _group_labels_by_element(cls, labels: list[str]):
        """Group line labels by their parent element."""
        ordered = cls._ordered_element_keys(cls._ensure_element_info())
        grouped: dict[str, list[str]] = {}
        for lbl in sorted(map(str, labels)):
            element = cls._resolve_element_from_label(lbl, ordered)
            if element:
                grouped.setdefault(element, []).append(lbl)
        return grouped

    @classmethod
    def _select_labels(
        cls, selector: str, *, labels: list[str], labels_by_element: dict[str, list[str]]
    ):
        """Return labels matching a selector string (exact, element, or prefix)."""
        selector = str(selector).strip()
        if not selector:
            return []

        lower_map = {lbl.lower(): lbl for lbl in labels}
        if selector.lower() in lower_map:
            return [lower_map[selector.lower()]]

        elem_map = {elem.lower(): elem for elem in labels_by_element}
        if selector.lower() in elem_map:
            return list(labels_by_element[elem_map[selector.lower()]])

        token = cls._normalize_token(selector)
        return [lbl for lbl in labels if cls._normalize_token(lbl).startswith(token)]

    @staticmethod
    def _line_shell(line_name: str) -> str:
        """Return the shell letter ('K', 'L', 'M', or '?') for a line name."""
        line_name = str(line_name).upper()
        return (
            "K"
            if line_name.startswith("K")
            else "L"
            if line_name.startswith("L")
            else "M"
            if line_name.startswith("M")
            else "?"
        )

    @staticmethod
    def _peak_confidence(
        snr_value: float, line_weight: float, distance_value: float, tolerance: float
    ) -> float:
        """Compute a confidence score for a peak-to-line match."""
        sigma = max(tolerance / 3.0, 1e-9)
        return (
            np.log1p(max(snr_value, 0.0))
            * max(line_weight, 0.0)
            * np.exp(-0.5 * (distance_value / sigma) ** 2)
        )

    @staticmethod
    def _line_matches_selector(line_name: str, selector: str) -> bool:
        """Check whether a line name matches a shell or substring selector."""
        line = str(line_name).strip().lower()
        selector = str(selector).strip().lower()
        return line.startswith(selector) if selector in {"k", "l", "m"} else selector in line

    @classmethod
    def _line_allowed_for_element(
        cls, element_name: str, line_name: str, edge_filters=None
    ) -> bool:
        """Return True if the line passes the edge filter for its element."""
        selectors = None if edge_filters is None else edge_filters.get(str(element_name))
        return selectors is None or any(
            cls._line_matches_selector(line_name, token) for token in selectors
        )

    def _get_spectrum_images(self, method="integration"):
        """Retrieve cached spectrum images for the given method."""
        return {
            "integration": getattr(self, "_spectrum_images", None),
            "fit": getattr(self, "_spectrum_images_pytorch", None),
        }.get(method)

    def _map_to_dataset2d(
        self,
        array,
        name: str | None = None,
        signal_units: str | None = None,
    ) -> Dataset2d:
        """Wrap a real-space map with this dataset's spatial calibration."""
        if isinstance(array, Dataset2d):
            return array
        return Dataset2d.from_array(
            array=np.asarray(array),
            name=name if name is not None else f"{self.name} map",
            origin=np.asarray(self.origin[:2], dtype=float),
            sampling=np.asarray(self.sampling[:2], dtype=float),
            units=list(self.units[:2]),
            signal_units=self.signal_units if signal_units is None else signal_units,
        )

    def _maps_to_dataset2d(
        self,
        maps: dict[str, np.ndarray],
        *,
        name_prefix: str = "",
        signal_units: str | None = None,
    ) -> dict[str, Dataset2d]:
        """Wrap a map dictionary with this dataset's spatial calibration."""
        return {
            key: self._map_to_dataset2d(
                value,
                name=f"{name_prefix}{key}".strip() or str(key),
                signal_units=signal_units,
            )
            for key, value in maps.items()
        }

    def x_ray_lookup(
        self, spec: str | list[str] | tuple[str, ...] | set[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Look up X-ray line energies, weights, and labels.

        Parameters
        ----------
        spec : str | sequence[str]
            One or more element/line specifiers.  Accepted formats include
            element names (``'Fe'``), element + shell (``'Fe K'``), and
            element + line (``'Fe Ka1'``).  Comma-separated strings are
            split automatically.

        Returns
        -------
        energies : ndarray
            1-D array of line energies in keV, sorted by energy.
        weights : ndarray
            Corresponding tabulated line weights (0--1).
        labels : list[str]
            Human-readable labels such as ``'FeKa1'``.

        Raises
        ------
        ValueError
            If no lines match the specifier(s).
        """
        info = type(self)._ensure_element_info()
        ordered = type(self)._ordered_element_keys(info)
        specs = type(self)._normalize_specs(spec, param_name="spec")

        rows: list[tuple[str, float, float]] = []
        for raw in specs:
            compact = re.sub(r"[\s_-]+", "", str(raw).strip())
            if not compact:
                continue
            element = next((k for k in ordered if compact.lower().startswith(k.lower())), None)
            if element is None:
                raise ValueError(f"Could not resolve element from specifier '{raw}'")
            suffix = compact[len(element) :]
            for line_name, line_info in type(self)._iter_selected_lines(
                element, suffix, raw_spec=str(raw)
            ):
                if not isinstance(line_info, dict):
                    continue
                try:
                    energy = float(line_info.get("energy (keV)", line_info.get("energy")))
                except (TypeError, ValueError):
                    continue
                try:
                    weight = float(line_info.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight = 0.0
                rows.append(
                    (f"{element}{type(self)._canonical_line_name(line_name)}", energy, weight)
                )

        if not rows:
            raise ValueError(f"No X-ray lines matched specifier(s): {specs}")

        unique = sorted(
            {(lbl, round(e, 12), round(w, 12)) for lbl, e, w in rows},
            key=lambda t: (t[1], -t[2], t[0]),
        )
        return (
            np.asarray([e for _, e, _ in unique], dtype=float),
            np.asarray([w for _, _, w in unique], dtype=float),
            [lbl for lbl, _, _ in unique],
        )

    def generate_spectrum_images(self, elements=None, width=0.15, return_maps=False):
        """Generate spectrum images by integrating around X-ray line energies.

        For each matched X-ray line, sums the spectral intensity within an
        energy window of ``line_energy +/- width`` at every spatial pixel.
        Results are cached in ``self._spectrum_images`` for later use by
        :meth:`show_spectrum_images` and :meth:`quantify_composition_cliff_lorimer`.

        Parameters
        ----------
        elements : str | sequence[str] | None, optional
            Element/line specifiers (see :meth:`x_ray_lookup`).  If ``None``,
            uses ``self.model_elements``.
        width : float, optional
            Half-width of the integration window in keV.
        return_maps : bool, optional
            If ``True``, return ``(maps, labels)``.

        Returns
        -------
        tuple[list[Dataset2d], list[str]] | None
            Only returned when *return_maps* is ``True``.
        """
        if elements is None:
            if not self.model_elements:
                raise ValueError("elements must be specified")
            elements = list(self.model_elements)

        energies, _, labels = self.x_ray_lookup(elements)
        keep = (energies > self.energy_axis.min()) & (energies < self.energy_axis.max())
        energies = energies[keep]
        labels = [label for label, ok in zip(labels, keep) if ok]

        mask = (self.energy_axis[:, None] > energies[None, :] - width) & (
            self.energy_axis[:, None] < energies[None, :] + width
        )

        scan_row, scan_col, n_energy = self.array.shape
        maps = (
            mask.astype(self.array.dtype).T @ self.array.reshape(-1, n_energy).transpose()
        ).reshape(mask.shape[1], scan_row, scan_col)

        spectrum_images = self._maps_to_dataset2d(dict(zip(labels, maps)))
        self._spectrum_images = {
            **getattr(self, "_spectrum_images", {}),
            **spectrum_images,
        }

        images, titles = self.show_spectrum_images(x_ray_lines=elements, return_maps=True)

        if return_maps:
            return images, titles

    def _integrate(self, spec, width=0.15, return_maps=False, show=True, **kwargs):
        """Integrate the spectrum around specified X-ray lines.

        Sums spectral intensity within ``line_energy +/- width`` for each
        selector.  By default, displays the resulting map(s).

        Parameters
        ----------
        spec : str | sequence[str]
            Element/line specifiers (see :meth:`x_ray_lookup`), e.g.
            ``'Fe Ka'`` or ``['Cu', 'Zn']``.
        width : float, optional
            Half-width of the integration window in keV.
        return_maps : bool, optional
            If ``True``, return the integrated maps.
        show : bool, optional
            If ``True``, display the maps.
        **kwargs
            Forwarded to the plotting function (e.g. ``cmap``, ``roi``).

        Returns
        -------
        Dataset2d | dict[str, Dataset2d]
            Single map when one selector is given, otherwise a dict keyed by
            selector string.
        """
        width = float(width)
        specs = type(self)._normalize_specs(spec, param_name="spec")
        arr = np.asarray(self.array, dtype=float)
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        energy_min, energy_max = energy_axis.min(), energy_axis.max()

        selector_masks, integrated_maps = {}, {}
        for selector in map(str, specs):
            line_energies, _, _ = self.x_ray_lookup(selector.strip())
            line_energies = line_energies[
                (line_energies >= energy_min) & (line_energies <= energy_max)
            ]
            if not len(line_energies):
                raise ValueError(
                    f"No X-ray lines for selector '{selector}' are within the dataset energy range"
                )

            mask = np.any(
                (energy_axis[:, None] >= line_energies[None, :] - width)
                & (energy_axis[:, None] <= line_energies[None, :] + width),
                axis=1,
            )
            selector_masks[selector] = mask
            integrated_maps[selector] = arr[:, :, mask].sum(axis=2)

        if show:
            cmap = kwargs.pop("cmap", "magma")
            if len(integrated_maps) == 1:
                selector = next(iter(integrated_maps))
                self.show_energy_window_map(
                    energy_window=[energy_min, energy_max],
                    roi=kwargs.pop("roi", None),
                    roi_cal=kwargs.pop("roi_cal", None),
                    mask=selector_masks[selector],
                    data_type=kwargs.pop("data_type", "xeds"),
                    cmap=cmap,
                    show=True,
                )
            else:
                show_2d(
                    list(integrated_maps.values()),
                    title=list(integrated_maps),
                    cmap=cmap,
                    scalebar={"sampling": self.sampling[1], "units": self.units[1]},
                    **kwargs,
                )

        integrated_datasets = self._maps_to_dataset2d(
            integrated_maps,
            name_prefix="Integrated XEDS ",
        )
        return (
            integrated_datasets
            if return_maps or len(integrated_datasets) != 1
            else next(iter(integrated_datasets.values()))
        )

    def integrate(self, spec, width=0.15, return_maps=False, show=True, **kwargs):
        """Convenience wrapper for Integrate."""
        return self._integrate(
            spec=spec, width=width, return_maps=return_maps, show=show, **kwargs
        )

    def _build_pytorch_spectrum_images(
        self, abundance_maps: np.ndarray, element_names: list[str] | tuple[str, ...]
    ) -> dict[str, Dataset2d]:
        """Convert per-element abundance maps into per-line spectrum images using weights."""
        maps = np.asarray(abundance_maps)
        if maps.ndim != 3:
            return {}

        line_maps = {}
        for i, element_name in enumerate(element_names):
            if i >= maps.shape[0]:
                break
            try:
                _, line_weights, line_labels = self.x_ray_lookup(str(element_name))
            except ValueError:
                continue
            element_map = np.asarray(maps[i], dtype=float)
            for weight, label in zip(line_weights, line_labels):
                line_maps[str(label)] = self._map_to_dataset2d(
                    element_map * weight,
                    name=str(label),
                )
        return line_maps

    def quantify_composition_cliff_lorimer(
        self, k_factors, method="integration", return_maps=False, verbose=True
    ):
        """Quantify elemental composition using the Cliff-Lorimer thin-film method.

        Parameters
        ----------
        k_factors : dict[str, float]
            Mapping of element/line selectors to their k-factors, e.g.
            ``{'Fe K': 1.0, 'Cu K': 1.45}``.  At least two elements are
            required.
        method : {"integration", "fit"}, optional
            Which cached spectrum images to use for intensity extraction.
        return_maps : bool, optional
            If ``True``, also return per-pixel atomic-percent and weight-percent
            maps.
        verbose : bool, optional
            If ``True``, print the quantification summary table.

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            Atomic-percent and weight-percent compositions keyed by element.
            Intermediate outputs are stored on ``_cliff_lorimer_*`` attributes.
        tuple[tuple[dict[str, float], dict[str, float]], tuple[dict[str, Dataset2d], dict[str, Dataset2d]]]
            When *return_maps* is ``True``, returns ``((atomic_percent,
            weight_percent), (atomic_percent_maps, weight_percent_maps))``.

        Raises
        ------
        ValueError
            If *k_factors* is empty, fewer than two elements are matched, or
            spectrum images are missing.
        """
        if not k_factors:
            raise ValueError("k_factors must be a non-empty dict")
        spectrum_images = self._get_spectrum_images(method)
        if not spectrum_images:
            raise ValueError("No spectrum images available for quantification")

        ordered_elements = type(self)._ordered_element_keys(type(self)._ensure_element_info())
        line_map = {
            str(k): np.asarray(getattr(v, "array", v), dtype=float)
            for k, v in spectrum_images.items()
        }
        labels = list(line_map)
        labels_by_element = type(self)._group_labels_by_element(labels)

        def match(selector: str) -> list[str]:
            return type(self)._select_labels(
                selector, labels=labels, labels_by_element=labels_by_element
            )

        intensities, weighted_intensities = {}, {}
        selector_maps = {} if return_maps else None
        intensity_maps = {} if return_maps else None
        weighted_intensity_maps = {} if return_maps else None

        for selector, k_raw in k_factors.items():
            k_val = float(k_raw)
            sel_labels = match(str(selector).strip())
            if not sel_labels:
                raise ValueError(f"No spectrum images matched selector {selector!r}")

            matched_elements = {
                type(self)._resolve_element_from_label(lbl, ordered_elements) for lbl in sel_labels
            } - {None}
            if len(matched_elements) != 1:
                raise ValueError(
                    f"Selector {selector!r} matched multiple elements: {sorted(matched_elements)}"
                )
            element = next(iter(matched_elements))

            grouped_map = np.sum([line_map[lbl] for lbl in sel_labels], axis=0)
            intensity = float(grouped_map.sum())
            weighted = float(k_val * intensity)
            intensities[element] = intensities.get(element, 0.0) + intensity
            weighted_intensities[element] = weighted_intensities.get(element, 0.0) + weighted

            if return_maps:
                weighted_map = grouped_map * k_val
                selector_maps[str(selector)] = grouped_map
                intensity_maps[element] = intensity_maps.get(element, 0) + grouped_map
                weighted_intensity_maps[element] = (
                    weighted_intensity_maps.get(element, 0) + weighted_map
                )

        if len(weighted_intensities) < 2:
            raise ValueError("At least two elements are required for Cliff-Lorimer quantification")

        weighted_sum = sum(weighted_intensities.values())
        atomic_percent = {
            el: 100.0 * val / weighted_sum if weighted_sum > 0 else 0.0
            for el, val in weighted_intensities.items()
        }

        if type(self).atomic_weights is None:
            type(self).load_atomic_weights()
        atomic_weights = type(self).atomic_weights or {}
        missing = [el for el in atomic_percent if el not in atomic_weights]
        if missing:
            raise ValueError(f"Atomic weights not found for elements: {missing}")

        weight_sum = sum(
            (atomic_percent[el] / 100.0) * float(atomic_weights[el]) for el in atomic_percent
        )
        weight_percent = {
            el: (atomic_percent[el] / 100.0) * float(atomic_weights[el]) / weight_sum * 100.0
            if weight_sum > 0
            else 0.0
            for el in atomic_percent
        }

        ordered = sorted(weighted_intensities, key=weighted_intensities.get, reverse=True)
        table_text = "\n".join(
            [
                "Element  Intensity      Weighted Intensity    Atomic %    Weight %",
                "-------  -------------  --------------------  ----------  ----------",
                *[
                    f"{el:<7}  {intensities[el]:>13.3f}  {weighted_intensities[el]:>20.3f}  {atomic_percent[el]:>10.3f}  {weight_percent[el]:>10.3f}"
                    for el in ordered
                ],
            ]
        )
        self._cliff_lorimer_intensities = intensities
        self._cliff_lorimer_weighted_intensities = weighted_intensities
        self._cliff_lorimer_atomic_percent = atomic_percent
        self._cliff_lorimer_weight_percent = weight_percent
        self._cliff_lorimer_summary_table = table_text
        self._cliff_lorimer_selector_maps = None
        self._cliff_lorimer_intensity_maps = None
        self._cliff_lorimer_weighted_intensity_maps = None
        self._cliff_lorimer_atomic_percent_maps = None
        self._cliff_lorimer_weight_percent_maps = None

        if verbose:
            print(table_text)

        if return_maps:
            weighted_stack = np.stack(list(weighted_intensity_maps.values()), axis=0)
            weighted_sum_map = weighted_stack.sum(axis=0)
            atomic_percent_maps = {
                el: np.divide(
                    wmap * 100.0,
                    weighted_sum_map,
                    out=np.zeros_like(weighted_sum_map, dtype=float),
                    where=weighted_sum_map > 0,
                )
                for el, wmap in weighted_intensity_maps.items()
            }
            mass_maps = {
                el: atomic_percent_maps[el] / 100.0 * float(atomic_weights[el])
                for el in atomic_percent_maps
            }
            mass_sum_map = np.sum(np.stack(list(mass_maps.values()), axis=0), axis=0)
            weight_percent_maps = {
                el: np.divide(
                    mmap * 100.0,
                    mass_sum_map,
                    out=np.zeros_like(mass_sum_map, dtype=float),
                    where=mass_sum_map > 0,
                )
                for el, mmap in mass_maps.items()
            }
            atomic_percent_maps = self._maps_to_dataset2d(
                atomic_percent_maps,
                name_prefix="Atomic percent ",
                signal_units="%",
            )
            weight_percent_maps = self._maps_to_dataset2d(
                weight_percent_maps,
                name_prefix="Weight percent ",
                signal_units="%",
            )
            self._cliff_lorimer_selector_maps = self._maps_to_dataset2d(
                selector_maps,
                name_prefix="Cliff-Lorimer selector ",
            )
            self._cliff_lorimer_intensity_maps = self._maps_to_dataset2d(
                intensity_maps,
                name_prefix="Cliff-Lorimer intensity ",
            )
            self._cliff_lorimer_weighted_intensity_maps = self._maps_to_dataset2d(
                weighted_intensity_maps,
                name_prefix="Cliff-Lorimer weighted intensity ",
            )
            self._cliff_lorimer_atomic_percent_maps = atomic_percent_maps
            self._cliff_lorimer_weight_percent_maps = weight_percent_maps
            return (atomic_percent, weight_percent), (atomic_percent_maps, weight_percent_maps)

        return atomic_percent, weight_percent

    def clear_spectrum_images(self):
        """Clear cached integration-based spectrum images."""
        self._spectrum_images = {}

    def clear_spectrum_images_pytorch(self):
        """Clear cached PyTorch fit-based spectrum images."""
        self._spectrum_images_pytorch = {}

    def calculate_background_polynomial(
        self,
        spectrum,
        energy_axis=None,
        degree=3,
        percentile=10,
        window_size=50,
    ):
        """
        Fit an XEDS continuum background with a polynomial power series in energy.

        A rolling low-percentile envelope is used as the fit target so sharp
        characteristic X-ray peaks do not dominate the continuum fit.
        """

        spectrum = np.asarray(spectrum, dtype=float)
        if spectrum.ndim != 1:
            raise ValueError("spectrum must be a 1D array")
        if spectrum.size == 0:
            raise ValueError("spectrum must contain at least one channel")

        if energy_axis is None:
            energy_axis = np.asarray(self.energy_axis, dtype=float)
            if energy_axis.shape != spectrum.shape:
                energy_axis = float(self.origin[2]) + float(self.sampling[2]) * np.arange(
                    spectrum.size, dtype=float
                )
        else:
            energy_axis = np.asarray(energy_axis, dtype=float)
        if energy_axis.shape != spectrum.shape:
            raise ValueError("energy_axis must have the same shape as spectrum")

        if isinstance(degree, bool):
            raise TypeError("degree must be a non-negative integer")
        try:
            degree = int(degree)
        except (TypeError, ValueError) as exc:
            raise TypeError("degree must be a non-negative integer") from exc
        if degree < 0:
            raise ValueError("degree must be >= 0")

        try:
            percentile = float(percentile)
        except (TypeError, ValueError) as exc:
            raise TypeError("percentile must be a number between 0 and 100") from exc
        if percentile < 0 or percentile > 100:
            raise ValueError("percentile must be between 0 and 100")

        if isinstance(window_size, bool):
            raise TypeError("window_size must be a positive integer")
        try:
            window_size = int(window_size)
        except (TypeError, ValueError) as exc:
            raise TypeError("window_size must be a positive integer") from exc
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        window_size = min(window_size, spectrum.size)

        finite = np.isfinite(spectrum) & np.isfinite(energy_axis)
        if np.count_nonzero(finite) < degree + 1:
            raise ValueError("not enough finite spectrum points for the requested degree")

        half_window = window_size // 2
        envelope = np.full_like(spectrum, np.nan, dtype=float)
        for channel in range(spectrum.size):
            start = max(0, channel - half_window)
            end = min(spectrum.size, channel + half_window + 1)
            values = spectrum[start:end]
            values = values[np.isfinite(values)]
            if values.size:
                envelope[channel] = np.percentile(values, percentile)

        fit_mask = finite & np.isfinite(envelope)
        if np.count_nonzero(fit_mask) < degree + 1:
            raise ValueError("not enough background fit points for the requested degree")

        fit_energy = energy_axis[fit_mask]
        fit_counts = envelope[fit_mask]
        energy_min = float(np.min(fit_energy))
        energy_span = float(np.max(fit_energy) - energy_min)
        if energy_span <= 0:
            if degree != 0:
                raise ValueError("energy_axis must span more than one value for degree > 0")
            return np.full_like(spectrum, max(float(np.median(fit_counts)), 0.0), dtype=float)

        # Scaling improves conditioning; this remains a polynomial in energy.
        def scaled_energy(energy):
            return 2.0 * (np.asarray(energy, dtype=float) - energy_min) / energy_span - 1.0

        def polynomial_background(energy, *coefficients):
            energy_scaled = scaled_energy(energy)
            background = np.zeros_like(energy_scaled, dtype=float)
            for power, coefficient in enumerate(coefficients):
                background += coefficient * (energy_scaled**power)
            return background

        scaled_fit_energy = scaled_energy(fit_energy)
        initial_coefficients = np.polynomial.polynomial.polyfit(
            scaled_fit_energy,
            fit_counts,
            deg=degree,
        )
        try:
            coefficients, _ = curve_fit(
                polynomial_background,
                fit_energy,
                fit_counts,
                p0=initial_coefficients,
                maxfev=10000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            coefficients = initial_coefficients

        background = polynomial_background(energy_axis, *coefficients)
        finite_counts = spectrum[finite]
        max_count = max(float(np.max(finite_counts)), float(np.max(fit_counts)), 0.0)
        background = np.nan_to_num(background, nan=0.0, posinf=max_count, neginf=0.0)
        return np.maximum(background, 0.0)

    def calculate_background_powerlaw(self, spectrum, *args, **kwargs):
        """Compatibility wrapper for the XEDS polynomial background fit."""
        return self.calculate_background_polynomial(spectrum, *args, **kwargs)
