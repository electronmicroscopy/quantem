from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.imaging_utils import dft_upsample, rotate_image
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import ScalebarConfig, show_2d


class StrainMapAutocorrelation(AutoSerialize):
    _token = object()

    def __init__(
        self,
        dataset: Dataset4dstem,
        input_data: Any | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use StrainMapAutocorrelation.from_dataset() or StrainMapAutocorrelation.from_array() to instantiate this class."
            )
        super().__init__()
        self.dataset = dataset
        self.input_data = input_data
        self.strain = None
        self.metadata: dict[str, Any] = {}
        self.transform: Dataset2d | None = None
        self.transform_rotated: Dataset2d | None = None

        self.mean_img_peaks: NDArray | None = None
        self.mean_img_weights: NDArray | None = None

        self.mask_diffraction = np.ones(self.dataset.array.shape[2:])
        self.mask_diffraction_inv = np.zeros(self.dataset.array.shape[2:])

        self.u_ref: np.ndarray | None = None
        self.v_ref: np.ndarray | None = None
        self.u_array: np.ndarray | None = None
        self.v_array: np.ndarray | None = None

        self.real_space = True

    @classmethod
    def from_dataset(cls, dataset: Dataset4dstem, *, name: str | None = None) -> "StrainMapAutocorrelation":
        if not isinstance(dataset, Dataset4dstem):
            raise TypeError("StrainMapAutocorrelation.from_dataset expects a Dataset4dstem instance.")
        if name is not None:
            dataset.name = name
        return cls(dataset=dataset, input_data=dataset, _token=cls._token)

    @classmethod
    def from_array(cls, array: NDArray, *, name: str = "strain_map_autocorrelation") -> "StrainMapAutocorrelation":
        arr = ensure_valid_array(array)
        if arr.ndim != 4:
            raise ValueError(
                "StrainMapAutocorrelation.from_array expects a 4D array with shape (scan_r, scan_c, dp_r, dp_c)."
            )
        ds4 = Dataset4dstem.from_array(arr, name=name)
        return cls(dataset=ds4, input_data=array, _token=cls._token)

    def diffraction_mask(
        self,
        threshold=None,
        edge_blend=64.0,
        plot_mask=True,
        figsize=(8, 4),
    ):
        dp_mean = np.mean(self.dataset.array, axis=(0, 1))
        mask_init = dp_mean < threshold
        mask_init[:, 0] = True
        mask_init[0, :] = True
        mask_init[:, -1] = True
        mask_init[-1, :] = True

        self.mask_diffraction = np.sin(
            np.clip(
                distance_transform_edt(np.logical_not(mask_init)) / edge_blend,
                0.0,
                1.0,
            )
            * np.pi
            / 2,
        ) ** 2
        int_edge = np.min(dp_mean[self.mask_diffraction > 0.99])
        self.mask_diffraction_inv = (1 - self.mask_diffraction) * int_edge

        if plot_mask:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            ax[0].imshow(
                np.log(np.maximum(dp_mean, np.min(dp_mean[dp_mean > 0]))),
                cmap="gray",
            )
            ax[1].imshow(
                np.log(
                    dp_mean * self.mask_diffraction + self.mask_diffraction_inv,
                ),
                cmap="gray",
            )

        return self

    def preprocess(
        self,
        mode: str = "linear",
        q_to_r_rotation_ccw_deg: float | None = None,
        q_transpose: bool | None = None,
        skip=None,
        plot_transform: bool = True,
        cropping_factor: float = 0.25,
        gamma: float = 0.5,
        **plot_kwargs: Any,
    ) -> "StrainMapAutocorrelation":
        mode_in = mode.strip().lower()
        if mode_in in {"linear", "patterson", "paterson", "acf", "autocorrelation"}:
            mode_norm = "linear"
        elif mode_in in {"log", "cepstrum", "cepstral"}:
            mode_norm = "log"
        elif mode_in in {"gamma", "power", "sqrt"}:
            mode_norm = "gamma"
        else:
            raise ValueError(
                "mode must be 'linear', 'log', or 'gamma' (aliases: 'patterson'->'linear', 'cepstrum'/'cepstral'->'log')."
            )

        self.metadata["mode"] = mode_norm
        if mode_norm == "gamma":
            self.metadata["gamma"] = gamma

        qrow_unit = self.dataset.units[2]
        qcol_unit = self.dataset.units[3]

        if qrow_unit in {"A", "Å"}:
            qrow_sampling_ang = self.dataset.sampling[2]
        elif qrow_unit == "mrad":
            wavelength = electron_wavelength_angstrom(self.dataset.metadata["energy"])
            qrow_sampling_ang = self.dataset.sampling[2] / 1000.0 / wavelength
        else:
            qrow_sampling_ang = 1.0
            qrow_unit = "pixels"

        if qcol_unit in {"A", "Å"}:
            qcol_sampling_ang = self.dataset.sampling[3]
        elif qcol_unit == "mrad":
            wavelength = electron_wavelength_angstrom(self.dataset.metadata["energy"])
            qcol_sampling_ang = self.dataset.sampling[3] / 1000.0 / wavelength
        else:
            qcol_sampling_ang = 1.0
            qcol_unit = "pixels"

        self.metadata["sampling_real"] = np.array(
            (
                1.0 / (qrow_sampling_ang * self.dataset.shape[2]),
                1.0 / (qcol_sampling_ang * self.dataset.shape[3]),
            ),
            dtype=float,
        )

        if qrow_unit == "pixels" and qcol_unit == "pixels":
            self.metadata["real_units"] = "1/pixels"
        else:
            self.metadata["real_units"] = r"$\mathrm{\AA}$"

        parent_rot = self.dataset.metadata.get("q_to_r_rotation_ccw_deg", None)
        parent_tr = self.dataset.metadata.get("q_transpose", None)

        used_parent = False
        if q_to_r_rotation_ccw_deg is None and parent_rot is not None:
            q_to_r_rotation_ccw_deg = parent_rot
            used_parent = True
        if q_transpose is None and parent_tr is not None:
            q_transpose = parent_tr
            used_parent = True

        if used_parent:
            import warnings

            warnings.warn(
                "StrainMapAutocorrelation.preprocess: using parent Dataset4dstem metadata "
                f"(q_to_r_rotation_ccw_deg={q_to_r_rotation_ccw_deg or 0.0}, "
                f"q_transpose={q_transpose or False}).",
                UserWarning,
            )

        if q_to_r_rotation_ccw_deg is None or q_transpose is None:
            import warnings

            q_to_r_rotation_ccw_deg = 0.0 if q_to_r_rotation_ccw_deg is None else q_to_r_rotation_ccw_deg
            q_transpose = False if q_transpose is None else q_transpose
            warnings.warn(
                "StrainMapPatterson.preprocess: setting q_to_r_rotation_ccw_deg=0.0 and q_transpose=False.",
                UserWarning,
            )

        self.metadata["q_to_r_rotation_ccw_deg"] = q_to_r_rotation_ccw_deg
        self.metadata["q_transpose"] = q_transpose

        arr = self.dataset.array if skip is None else self.dataset.array[::skip, ::skip]
        dp = arr * self.mask_diffraction[None, None, :, :] + self.mask_diffraction_inv[None, None, :, :]

        if mode_norm == "linear":
            dp_proc = dp
        elif mode_norm == "log":
            dp_proc = np.log1p(dp)
        elif mode_norm == "gamma":
            dp_proc = np.power(np.clip(dp, 0.0, None), self.metadata["gamma"])
        else:
            raise RuntimeError("Unreachable: normalized mode mapping failed.")

        im = np.mean(np.abs(np.fft.fft2(dp_proc)), axis=(0, 1))
        im = np.fft.fftshift(im)

        self.transform = Dataset2d.from_array(
            im,
            origin=(im.shape[0] // 2, im.shape[1] // 2),
            sampling=(1.0, 1.0),
            units=(qrow_unit, qcol_unit),
            signal_units="intensity",
        )

        im_plot = self.transform.array
        if self.metadata["q_transpose"]:
            im_plot = im_plot.T

        self.transform_rotated = Dataset2d.from_array(
            rotate_image(
                im_plot,
                self.metadata["q_to_r_rotation_ccw_deg"],
                clockwise=False,
            ),
            origin=(im.shape[0] // 2, im.shape[1] // 2),
            sampling=(1.0, 1.0),
            units=(self.metadata["real_units"], self.metadata["real_units"]),
            signal_units="intensity",
        )

        if plot_transform:
            self.plot_transform(cropping_factor=cropping_factor, **plot_kwargs)

        return self

    def plot_transform(
        self,
        cropping_factor: float = 0.25,
        scalebar_fraction: float = 0.25,
        **plot_kwargs: Any,
    ):
        if self.transform is None or self.transform_rotated is None:
            raise ValueError("Run preprocess() first to compute transform images.")

        sampling = np.mean(self.metadata["sampling_real"])
        units = self.metadata.get("real_units", r"$\mathrm{\AA}$")

        W = self.transform.shape[1]
        view_w_px = W * cropping_factor
        target_units = scalebar_fraction * view_w_px * sampling
        sb_len = _nice_length_units(target_units)

        kr = (np.arange(self.transform.shape[0], dtype=float) - self.transform.shape[0] // 2)[:, None]
        kc = (np.arange(self.transform.shape[1], dtype=float) - self.transform.shape[1] // 2)[None, :]
        qmag = np.sqrt(kr * kr + kc * kc)
        im0 = self.transform.array
        tmp = im0 * qmag
        i0 = np.unravel_index(np.nanargmax(tmp), tmp.shape)
        vmin = 0.0
        vmax = im0[i0]

        defaults = dict(
            vmin=vmin,
            vmax=vmax,
            title=("Original Transform", "Rotated Transform"),
            scalebar=ScalebarConfig(
                sampling=sampling,
                units=units,
                length=sb_len if sb_len > 0 else None,
            ),
        )
        defaults.update(plot_kwargs)

        fig, ax = show_2d([self.transform, self.transform_rotated], **defaults)

        for a in _flatten_axes(ax):
            _apply_center_crop_limits(a, self.transform.shape, cropping_factor)

        return fig, ax
    
    def plot_single_transform(
        self,
        row: int = 0,
        col: int = 0,
        cropping_factor: float = 0.25,
        scalebar_fraction: float = 0.25,
        **plot_kwargs: Any,
    ):
        if self.transform is None or self.transform_rotated is None:
            raise ValueError("Run preprocess() first to compute transform images.")

        sampling = np.mean(self.metadata["sampling_real"])
        units = self.metadata.get("real_units", r"$\mathrm{\AA}$")

        W = self.transform.shape[1]
        view_w_px = W * cropping_factor
        target_units = scalebar_fraction * view_w_px * sampling
        sb_len = _nice_length_units(target_units)

        kr = (np.arange(self.transform.shape[0], dtype=float) - self.transform.shape[0] // 2)[:, None]
        kc = (np.arange(self.transform.shape[1], dtype=float) - self.transform.shape[1] // 2)[None, :]
        qmag = np.sqrt(kr * kr + kc * kc)
        im0 = self.transform.array
        tmp = im0 * qmag
        i0 = np.unravel_index(np.nanargmax(tmp), tmp.shape)
        vmin = 0.0
        vmax = im0[i0]

        defaults = dict(
            vmin=vmin,
            vmax=vmax,
            title=(f"Row={row} Col={col} Transform"),
            scalebar=ScalebarConfig(
                sampling=sampling,
                units=units,
                length=sb_len if sb_len > 0 else None,
            ),
        )
        defaults.update(plot_kwargs)
        if row >= 0 and row < self.dataset.array.shape[0] and col >= 0 and col < self.dataset.array.shape[1]:
            dp = self.dataset.array[row, col] * self.mask_diffraction + self.mask_diffraction_inv
        else:
            raise ValueError("row or column value out of bounds")
        mode = self.metadata.get("mode", "linear").lower()
        if mode == "gamma":
            g = self.metadata["gamma"]

        
        if mode == "linear":
            im = np.fft.fftshift(np.abs(np.fft.fft2(dp)))
        elif mode == "log":
            im = np.fft.fftshift(np.abs(np.fft.fft2(np.log1p(dp))))
        elif mode == "gamma":
            im = np.fft.fftshift(np.abs(np.fft.fft2(np.power(np.clip(dp, 0.0, None), g))))
        else:
            raise ValueError("metadata['mode'] must be 'linear', 'log', or 'gamma'")

    
        fig, ax = show_2d(im, **defaults)
        rot_ccw = self.metadata["q_to_r_rotation_ccw_deg"]
        q_transpose = self.metadata["q_transpose"]

        _overlay_lattice_vectors(
            ax=ax,
            shape=self.transform.shape,
            u_rc= self.u_fit.array[row, col, :],
            v_rc=self.v_fit.array[row, col, :],
            rot_ccw_deg=rot_ccw,
            q_transpose=q_transpose,
            peaks_plot=self.mean_img_peaks,
        )
        
        for a in _flatten_axes(ax):
            _apply_center_crop_limits(a, self.transform.shape, cropping_factor)

    def choose_lattice_vector(
        self,
        *,
        u: tuple[float, float] | NDArray,
        v: tuple[float, float] | NDArray,
        define_in_rotated: bool = False,
        refine_gaussian: bool = True,
        refine_dft: bool = False,
        refine_all_peaks: bool = False,
        refine_radius_px: float = 2.0,
        upsample: int = 16,
        gaussian_maxfev: int = 100,
        threshold_percentile: float = 0.9975,
        min_peak_spacing: float = 0,
        plot: bool = True,
        cropping_factor: float = 0.25,
        **plot_kwargs: Any,
    ) -> "StrainMapAutocorrelation":
        if self.transform is None or self.transform_rotated is None:
            raise ValueError("Run preprocess() first to compute transform images.")

        u_rc = np.asarray(u, dtype=float).reshape(2)
        v_rc = np.asarray(v, dtype=float).reshape(2)

        rot_ccw = self.metadata["q_to_r_rotation_ccw_deg"]
        q_transpose = self.metadata["q_transpose"]

        if define_in_rotated:
            u_rc = _display_vec_to_raw(u_rc, rotation_ccw_deg=rot_ccw, transpose=q_transpose)
            v_rc = _display_vec_to_raw(v_rc, rotation_ccw_deg=rot_ccw, transpose=q_transpose)

        u_fit_abs, v_fit_abs, peaks, weights = _refine_lattice_vectors(
            self.transform.array,
            u_rc=u_rc,
            v_rc=v_rc,
            radius_px=refine_radius_px,
            refine_gaussian=refine_gaussian,
            refine_dft=refine_dft,
            refine_all_peaks=refine_all_peaks,
            peaks=None,
            weights=None,
            upsample=upsample,
            maxfev=gaussian_maxfev,
            threshold_percentile=threshold_percentile,
            min_peak_spacing = min_peak_spacing,
        )

        self.u = u_fit_abs[:2]
        self.v = v_fit_abs[:2]
        if refine_all_peaks:
            self.mean_img_peaks = peaks
            self.mean_img_weights = weights

        self.metadata["choose_define_in_rotated"] = define_in_rotated
        self.metadata["choose_refine_gaussian"] = refine_gaussian
        self.metadata["choose_refine_dft"] = refine_dft
        self.metadata["choose_refine_all_peaks"] = refine_all_peaks
        self.metadata["choose_refine_radius_px"] = refine_radius_px
        self.metadata["choose_upsample"] = upsample
        self.metadata["choose_gaussian_maxfev"] = gaussian_maxfev
        self.metadata["choose_threshold_percentile"] = threshold_percentile

        if plot:
            fig, ax = self.plot_transform(cropping_factor=cropping_factor, **plot_kwargs)
            _overlay_lattice_vectors(
                ax=ax,
                shape=self.transform.shape,
                u_rc=self.u,
                v_rc=self.v,
                rot_ccw_deg=rot_ccw,
                q_transpose=q_transpose,
                peaks_plot=self.mean_img_peaks,
            )
            return self

        return self

    def fit_lattice_vectors(
        self,
        refine_gaussian: bool = True,
        refine_dft: bool = False,
        refine_all_peaks: bool = False,
        refine_radius_px: float = 2.0,
        upsample: int = 16,
        gaussian_maxfev: int = 100,
        progressbar: bool = True,
    ) -> "StrainMapAutocorrelation":
        if self.u is None or self.v is None:
            raise ValueError("Run choose_lattice_vector() first to set initial lattice vectors (self.u, self.v).")
        if refine_all_peaks:
            if self.mean_img_peaks is None or self.mean_img_weights is None:
                raise ValueError("Run choose_lattice_vector() with refine_all_peaks=True to determine which peaks to fit")
    
        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]

        self.u_peak_fit = Dataset3d.from_shape(
            (scan_r, scan_c, 5),
            name="u_peak_fit",
            signal_units="mixed",
        )
        self.v_peak_fit = Dataset3d.from_shape(
            (scan_r, scan_c, 5),
            name="v_peak_fit",
            signal_units="mixed",
        )

        self.u_array = np.zeros((scan_r, scan_c, 2))
        self.v_array = np.zeros((scan_r, scan_c, 2))

        mode = self.metadata.get("mode", "linear").lower()
        if mode == "gamma":
            g = self.metadata["gamma"]

        it = np.ndindex(scan_r, scan_c)
        if progressbar:
            try:
                from tqdm.auto import tqdm  # type: ignore

                it = tqdm(it, total=scan_r * scan_c, desc="fit_lattice_vectors", leave=True)
            except Exception:
                pass

        u0 = np.asarray(self.u, dtype=float).reshape(2)
        v0 = np.asarray(self.v, dtype=float).reshape(2)


        for r, c in it:
            dp = self.dataset.array[r, c] * self.mask_diffraction + self.mask_diffraction_inv

            if mode == "linear":
                im = np.fft.fftshift(np.abs(np.fft.fft2(dp)))
            elif mode == "log":
                im = np.fft.fftshift(np.abs(np.fft.fft2(np.log1p(dp))))
            elif mode == "gamma":
                im = np.fft.fftshift(np.abs(np.fft.fft2(np.power(np.clip(dp, 0.0, None), g))))
            else:
                raise ValueError("metadata['mode'] must be 'linear', 'log', or 'gamma'")

            u_fit_abs, v_fit_abs, _, _ = _refine_lattice_vectors(
                im,
                u_rc=u0,
                v_rc=v0,
                radius_px=refine_radius_px,
                refine_gaussian=refine_gaussian,
                refine_dft=refine_dft,
                refine_all_peaks=refine_all_peaks,
                peaks=self.mean_img_peaks,
                weights=self.mean_img_weights,
                upsample=upsample,
                maxfev=gaussian_maxfev,
            )

            self.u_peak_fit.array[r, c, :] = u_fit_abs
            self.v_peak_fit.array[r, c, :] = v_fit_abs

            self.u_array[r, c, 0] = u_fit_abs[0]
            self.u_array[r, c, 1] = u_fit_abs[1]
            self.v_array[r, c, 0] = v_fit_abs[0]
            self.v_array[r, c, 1] = v_fit_abs[1]

        self.metadata["fit_refine_gaussian"] = refine_gaussian
        self.metadata["fit_refine_dft"] = refine_dft
        self.metadata["fit_refine_all_peaks"] = refine_all_peaks
        self.metadata["fit_refine_radius_px"] = refine_radius_px
        self.metadata["fit_upsample"] = upsample
        self.metadata["fit_gaussian_maxfev"] = gaussian_maxfev

        return self

    def plot_lattice_vectors(
        self,
        subtract_mean: bool = True,
        max_shift: float = 1.0,
        cmap: str = "PiYG_r",
        axsize: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        **imshow_kwargs: Any,
    ):
        if self.u_array is None or self.v_array is None:
            raise ValueError("Run fit_lattice_vectors() first to compute u_array and v_array.")
        if self.u is None or self.v is None:
            raise ValueError("Run choose_lattice_vector() first to set self.u and self.v.")

        im0 = self.u_array[:, :, 0]
        im1 = self.u_array[:, :, 1]
        im2 = self.v_array[:, :, 0]
        im3 = self.v_array[:, :, 1]

        du0 = im0 - self.u[0]
        du1 = im1 - self.u[1]
        dv0 = im2 - self.v[0]
        dv1 = im3 - self.v[1]

        max_shift2 = max_shift * max_shift
        mu = (du0 * du0 + du1 * du1) <= max_shift2
        mv = (dv0 * dv0 + dv1 * dv1) <= max_shift2

        if subtract_mean:
            if np.any(mu):
                im0 = im0 - np.mean(im0[mu])
                im1 = im1 - np.mean(im1[mu])
            else:
                im0 = im0 - np.mean(im0)
                im1 = im1 - np.mean(im1)

            if np.any(mv):
                im2 = im2 - np.mean(im2[mv])
                im3 = im3 - np.mean(im3[mv])
            else:
                im2 = im2 - np.mean(im2)
                im3 = im3 - np.mean(im3)

        vals = []
        if np.any(mu):
            vals.append(np.abs(im0[mu]))
            vals.append(np.abs(im1[mu]))
        if np.any(mv):
            vals.append(np.abs(im2[mv]))
            vals.append(np.abs(im3[mv]))

        if vals:
            vlim = np.max(np.concatenate(vals))
        else:
            vlim = np.max(np.abs(np.stack([im0, im1, im2, im3], axis=0)))

        vmin = -vlim
        vmax = vlim

        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color="black")

        m0 = np.ma.array(im0, mask=~mu)
        m1 = np.ma.array(im1, mask=~mu)
        m2 = np.ma.array(im2, mask=~mv)
        m3 = np.ma.array(im3, mask=~mv)

        if axsize is None and figsize is None:
            axsize = (4.0, 4.0)
        if figsize is None:
            figsize = (axsize[0] * 4.0, axsize[1])

        fig, ax = plt.subplots(1, 4, figsize=figsize)

        ax[0].imshow(m0, cmap=cm, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax[1].imshow(m1, cmap=cm, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax[2].imshow(m2, cmap=cm, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax[3].imshow(m3, cmap=cm, vmin=vmin, vmax=vmax, **imshow_kwargs)

        ax[0].set_title("u_r")
        ax[1].set_title("u_c")
        ax[2].set_title("v_r")
        ax[3].set_title("v_c")

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        return fig, ax


def _nice_length_units(target: float) -> float:
    if not np.isfinite(target) or target <= 0:
        return 0.0
    exp = np.floor(np.log10(target))
    base = target / (10.0**exp)
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10.0**exp)


def _apply_center_crop_limits(ax: Any, shape: tuple[int, int], cropping_factor: float) -> None:
    if cropping_factor >= 1.0:
        return
    if not (0.0 < cropping_factor <= 1.0):
        raise ValueError("cropping_factor must be in (0, 1].")

    H, W = shape
    r0 = H // 2
    c0 = W // 2
    half_h = 0.5 * cropping_factor * H
    half_w = 0.5 * cropping_factor * W

    ax.set_xlim(c0 - half_w, c0 + half_w)

    y0, y1 = ax.get_ylim()
    if y0 > y1:
        ax.set_ylim(r0 + half_h, r0 - half_h)
    else:
        ax.set_ylim(r0 - half_h, r0 + half_h)


def _flatten_axes(ax: Any) -> list[Any]:
    if isinstance(ax, np.ndarray):
        return list(ax.ravel())
    if isinstance(ax, (list, tuple)):
        out: list[Any] = []
        for a in ax:
            out.extend(_flatten_axes(a))
        return out
    return [ax]


def _raw_vec_to_display(vec_rc: NDArray, *, rotation_ccw_deg: float, transpose: bool) -> NDArray:
    v = np.asarray(vec_rc, dtype=float).reshape(2)
    dr, dc = v[0], v[1]

    if transpose:
        dr, dc = dc, dr

    theta = np.deg2rad(rotation_ccw_deg)
    ct = np.cos(theta)
    st = np.sin(theta)

    dr2 = ct * dr - st * dc
    dc2 = st * dr + ct * dc
    return np.array((dr2, dc2), dtype=float)


def _display_vec_to_raw(vec_rc: NDArray, *, rotation_ccw_deg: float, transpose: bool) -> NDArray:
    v = np.asarray(vec_rc, dtype=float).reshape(2)
    dr, dc = v[0], v[1]

    theta = np.deg2rad(rotation_ccw_deg)
    ct = np.cos(theta)
    st = np.sin(theta)

    dr2 = ct * dr + st * dc
    dc2 = -st * dr + ct * dc

    if transpose:
        dr2, dc2 = dc2, dr2

    return np.array((dr2, dc2), dtype=float)


def _plot_lattice_vectors(ax: Any, center_rc: tuple[float, float], u_rc: NDArray, v_rc: NDArray) -> None:
    r0, c0 = center_rc

    def _draw(vec: NDArray, label: str, color: tuple[float, float, float]) -> None:
        dr, dc = vec[0], vec[1]
        ax.plot([c0, c0 + dc], [r0, r0 + dr], linewidth=2.75, color=color)
        ax.plot([c0 + dc], [r0 + dr], marker="o", markersize=6.0, color=color)
        ax.text(c0 + dc, r0 + dr, f" {label}", color=color, fontsize=18, va="center")

    _draw(np.asarray(u_rc, dtype=float).reshape(2), "u", (1.0, 0.0, 0.0))
    _draw(np.asarray(v_rc, dtype=float).reshape(2), "v", (0.0, 0.7, 1.0))

def _plot_peaks(ax: Any, center_rc: tuple[float, float], peaks_plot: NDArray) -> None:
    r0, c0 = center_rc

    def _draw(vec: NDArray, color: tuple[float, float, float]) -> None:
        dr, dc = vec[0], vec[1]
        ax.plot([c0 + dc], [r0 + dr], marker="o", markersize=6.0, color=color)

    for p in peaks_plot:
        _draw(np.asarray(p, dtype=float).reshape(2), (0.0, 1.0, 0.0))

def _overlay_lattice_vectors(
    *,
    ax: Any,
    shape: tuple[int, int],
    u_rc: NDArray,
    v_rc: NDArray,
    rot_ccw_deg: float,
    q_transpose: bool,
    peaks_plot: NDArray | None = None,
) -> None:
    axs = _flatten_axes(ax)
    if not axs:
        return

    H, W = shape
    center_rc = (H // 2, W // 2)

    _plot_lattice_vectors(axs[0], center_rc, u_rc, v_rc)
    if peaks_plot is not None:
        _plot_peaks(axs[0], center_rc, peaks_plot)

    if len(axs) >= 2:
        u_disp = _raw_vec_to_display(u_rc, rotation_ccw_deg=rot_ccw_deg, transpose=q_transpose)
        v_disp = _raw_vec_to_display(v_rc, rotation_ccw_deg=rot_ccw_deg, transpose=q_transpose)
        _plot_lattice_vectors(axs[1], center_rc, u_disp, v_disp)


def _parabolic_vertex_delta(v_m1: float, v_0: float, v_p1: float) -> float:
    denom = v_m1 - 2.0 * v_0 + v_p1
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    delta = 0.5 * (v_m1 - v_p1) / denom
    if not np.isfinite(delta):
        return 0.0
    return np.clip(delta, -1.0, 1.0)


def _refine_peak_subpixel(
    im: NDArray,
    *,
    r_guess: float,
    c_guess: float,
    radius_px: float = 2.0,
) -> tuple[float, float]:
    im = np.asarray(im, dtype=float)
    H, W = im.shape

    r0 = int(np.clip(int(np.round(r_guess)), 0, H - 1))
    c0 = int(np.clip(int(np.round(c_guess)), 0, W - 1))
    rad = int(max(0, int(np.ceil(radius_px))))

    r1 = max(0, r0 - rad)
    r2 = min(H, r0 + rad + 1)
    c1 = max(0, c0 - rad)
    c2 = min(W, c0 + rad + 1)

    win = im[r1:r2, c1:c2]
    if win.size == 0:
        return r_guess, c_guess

    ir, ic = np.unravel_index(np.argmax(win), win.shape)
    r_peak = r1 + ir
    c_peak = c1 + ic

    if 0 < r_peak < H - 1:
        col = im[r_peak - 1 : r_peak + 2, c_peak]
        dr = _parabolic_vertex_delta(col[0], col[1], col[2])
    else:
        dr = 0.0

    if 0 < c_peak < W - 1:
        row = im[r_peak, c_peak - 1 : c_peak + 2]
        dc = _parabolic_vertex_delta(row[0], row[1], row[2])
    else:
        dc = 0.0

    return r_peak + dr, c_peak + dc


def _refine_peak_subpixel_dft(
    im: NDArray,
    *,
    r0: float,
    c0: float,
    upsample: int,
) -> tuple[float, float]:
    if upsample <= 1:
        return r0, c0

    im = np.asarray(im, dtype=float)
    if torch.is_tensor(r0):
        r0 = float(r0.item())
    if torch.is_tensor(c0):
        c0 = float(c0.item())
    F = np.fft.fft2(np.fft.fftshift(im))

    up = upsample
    du = int(np.fix(np.ceil(1.5 * up)))
    patch = np.abs(dft_upsample(F, up=up, shift=(r0, c0), device="cpu"))
    patch = np.asarray(patch, dtype=float)

    i0, j0 = np.unravel_index(np.argmax(patch), patch.shape)

    if 0 < i0 < patch.shape[0] - 1:
        col = patch[i0 - 1 : i0 + 2, j0]
        di = _parabolic_vertex_delta(col[0], col[1], col[2])
    else:
        di = 0.0

    if 0 < j0 < patch.shape[1] - 1:
        row = patch[i0, j0 - 1 : j0 + 2]
        dj = _parabolic_vertex_delta(row[0], row[1], row[2])
    else:
        dj = 0.0
    M, N = im.shape
    dr = ((float(i0) - du + di)) / up
    dc = ((float(j0) - du + dj)) / up

    return r0 + dr, c0 + dc


def _refine_lattice_vectors(
    im: NDArray,
    *,
    u_rc: NDArray,
    v_rc: NDArray,
    radius_px: float = 2.0,
    refine_gaussian: bool = True,
    refine_dft: bool = False,
    refine_all_peaks: bool = False,
    peaks: NDArray | None = None,
    weights: NDArray | None = None,
    upsample: int = 16,
    maxfev: int = 100,
    threshold_percentile: float = 0.9975,
    min_peak_spacing: float = 0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    from scipy.optimize import curve_fit

    im = np.asarray(im, dtype=float)
    if im.ndim != 2:
        raise ValueError("im must be 2D.")

    H, W = im.shape
    r_center = H // 2
    c_center = W // 2

    def _parabolic_peak_rc_amp(*, r_guess: float, c_guess: float) -> tuple[float, float, float]:
        r0 = int(np.clip(int(np.round(r_guess)), 0, H - 1))
        c0 = int(np.clip(int(np.round(c_guess)), 0, W - 1))
        win = im[
            max(0, r0 - 1) : min(H, r0 + 2),
            max(0, c0 - 1) : min(W, c0 + 2),
        ]
        if win.size == 0:
            return r_guess, c_guess, 0.0

        ir, ic = np.unravel_index(np.argmax(win), win.shape)
        r_peak = max(0, r0 - 1) + ir
        c_peak = max(0, c0 - 1) + ic

        r_ref = r_peak
        c_ref = c_peak

        if 0 < r_peak < H - 1:
            col = im[r_peak - 1 : r_peak + 2, c_peak]
            dr = _parabolic_vertex_delta(col[0], col[1], col[2])
        else:
            dr = 0.0

        if 0 < c_peak < W - 1:
            row = im[r_peak, c_peak - 1 : c_peak + 2]
            dc = _parabolic_vertex_delta(row[0], row[1], row[2])
        else:
            dc = 0.0

        r_sub = r_ref + dr
        c_sub = c_ref + dc
        r_int = int(np.clip(int(np.round(r_sub)), 0, H - 1))
        c_int = int(np.clip(int(np.round(c_sub)), 0, W - 1))
        amp = im[r_int, c_int]

        return r_sub, c_sub, amp

    def _fit_gaussian_isotropic(
        *,
        r0: float,
        c0: float,
        radius_px: float,
        maxfev: int,
    ) -> tuple[float, float, float, float, float]:
        rad = int(max(1, int(np.ceil(radius_px))))
        r0i = int(np.clip(int(np.round(r0)), 0, H - 1))
        c0i = int(np.clip(int(np.round(c0)), 0, W - 1))

        r1 = max(0, r0i - rad)
        r2 = min(H, r0i + rad + 1)
        c1 = max(0, c0i - rad)
        c2 = min(W, c0i + rad + 1)

        win = im[r1:r2, c1:c2]
        if win.size == 0:
            return r0, c0, 0.0, 0.0, 0.0

        ir, ic = np.unravel_index(np.argmax(win), win.shape)
        r_peak = r1 + ir
        c_peak = c1 + ic

        bg0 = np.median(win)
        amp0 = win[ir, ic] - bg0
        sig0 = max(0.75, radius_px / 2.0)

        rr = np.arange(r1, r2, dtype=float)[:, None]
        cc = np.arange(c1, c2, dtype=float)[None, :]
        RR = np.broadcast_to(rr, win.shape)
        CC = np.broadcast_to(cc, win.shape)

        def _g2(
            coords: tuple[NDArray, NDArray],
            row: float,
            col: float,
            amp: float,
            sigma: float,
            background: float,
        ) -> NDArray:
            r, c = coords
            sig = np.maximum(sigma, 1e-12)
            return background + amp * np.exp(-((r - row) ** 2 + (c - col) ** 2) / (2.0 * sig * sig))

        p0 = (r_peak, c_peak, max(0.0, amp0), sig0, bg0)

        rlo = r1 - 0.5
        rhi = (r2 - 1) + 0.5
        clo = c1 - 0.5
        chi = (c2 - 1) + 0.5

        bounds_lo = (rlo, clo, 0.0, 0.25, -np.inf)
        bounds_hi = (rhi, chi, np.inf, radius_px * 4.0, np.inf)

        try:
            popt, _ = curve_fit(
                _g2,
                (RR.ravel(), CC.ravel()),
                win.ravel(),
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=maxfev,
            )
            row, col, amp, sig, bg = popt
            if not (np.isfinite(row) and np.isfinite(col) and np.isfinite(amp) and np.isfinite(sig) and np.isfinite(bg)):
                return r0, c0, p0[2], 0.0, 0.0
            return row, col, amp, sig, bg
        except Exception:
            return r0, c0, p0[2], 0.0, 0.0

    def _refine_one(vec: NDArray) -> NDArray:
        vec = np.asarray(vec, dtype=float).reshape(2)
        r_guess = r_center + vec[0]
        c_guess = c_center + vec[1]

        r_par, c_par, amp_par = _parabolic_peak_rc_amp(r_guess=r_guess, c_guess=c_guess)

        if refine_gaussian:
            r_fit, c_fit, amp, sig, bg = _fit_gaussian_isotropic(
                r0=r_par,
                c0=c_par,
                radius_px=radius_px,
                maxfev=maxfev,
            )
        else:
            r_fit, c_fit, amp, sig, bg = r_par, c_par, amp_par, 0.0, 0.0

        if refine_dft and upsample > 1:
            r_dft, c_dft = _refine_peak_subpixel_dft(
                im,
                r0=r_fit,
                c0=c_fit,
                upsample=upsample,
            )
            r_fit, c_fit = r_dft, c_dft

        return np.array((r_fit - r_center, c_fit - c_center, amp, sig, bg), dtype=float)

    def _find_initial_peaks_weights(initial_peaks: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        if initial_peaks is None:
            threshold = np.percentile(im, threshold_percentile*100)
            p = np.logical_and.reduce((
                im>np.roll(im,(-1,-1),axis = (0,1)),
                im>np.roll(im,(-1, 0),axis = (0,1)),
                im>np.roll(im,(-1, 1),axis = (0,1)),
                im>np.roll(im,( 0,-1),axis = (0,1)),
                im>np.roll(im,( 0, 1),axis = (0,1)),
                im>np.roll(im,( 1,-1),axis = (0,1)),
                im>np.roll(im,( 1, 0),axis = (0,1)),
                im>np.roll(im,( 1, 1),axis = (0,1)),
                im>threshold,
                ) )
            initial_peaks = np.argwhere(p)
            r0 = (r_center, c_center)
            initial_peaks = initial_peaks - r0
            if min_peak_spacing > 0:
                intensities = im[initial_peaks[:, 0] + r_center, initial_peaks[:, 1] + c_center]
                sorted_indices = np.argsort(intensities)[::-1]
                initial_peaks_sorted = initial_peaks[sorted_indices]
                accepted_peaks = []
                
                for peak in initial_peaks_sorted:
                    if len(accepted_peaks) == 0:
                        accepted_peaks.append(peak)
                    else:
                        distances = np.sqrt(np.sum((np.array(accepted_peaks) - peak)**2, axis=1))
                        if np.all(distances >= min_peak_spacing):
                            accepted_peaks.append(peak)
                
                initial_peaks = np.array(accepted_peaks)
        
        peak_sp = np.zeros(initial_peaks.shape)
        weights = np.zeros((initial_peaks.shape[0], 1))
        it = 0
        for peak in initial_peaks:
            refined_peak = _refine_one(peak)
            peak_sp[it, :] = refined_peak[:2]
            weights[it] = refined_peak[2]
            it += 1
        return peak_sp, weights
    

    if refine_all_peaks:
        if peaks is None or weights is None:
            pts, weights = _find_initial_peaks_weights(None)
        else:
            pts,_ = _find_initial_peaks_weights(peaks)
        
        A = np.column_stack((u_rc, v_rc))
        ab0_float = np.linalg.lstsq(A, pts.T, rcond=None)[0]
        ab0 = (np.round(ab0_float)).T

        weights /= weights.sum()
        A = np.ones((pts.shape[0], 3))
        A[:,:2] = ab0
        pts_weighted = pts * np.sqrt(weights)
        A_weighted = A * np.sqrt(weights)
        uvr0 = np.linalg.lstsq(A_weighted, pts_weighted, rcond=None)[0]
        u_refined = uvr0[0,:]
        v_refined = uvr0[1,:]
        _,_, amp_par_u = _parabolic_peak_rc_amp(r_guess=u_refined[0], c_guess=u_refined[1])
        _,_, amp_par_v = _parabolic_peak_rc_amp(r_guess=v_refined[0], c_guess=v_refined[1])

        return np.array((u_refined[0], u_refined[1], amp_par_u, 0, 0), dtype=float), np.array((v_refined[0], v_refined[1], amp_par_v, 0, 0), dtype=float), pts, weights

    return _refine_one(u_rc), _refine_one(v_rc), None, None

