from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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


class StrainMap(AutoSerialize):
    _token = object()

    def __init__(
        self,
        dataset: Dataset4dstem,
        input_data: Any | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use StrainMap.from_data() to instantiate this class.")
        super().__init__()
        self.dataset = dataset
        self.input_data = input_data
        self.strain = None
        self.metadata: dict[str, Any] = {}
        self.transform: Dataset2d | None = None
        self.transform_rotated: Dataset2d | None = None
        self.u: NDArray | None = None
        self.v: NDArray | None = None
        self.mask_diffraction = np.ones(self.dataset.array.shape[2:])
        self.mask_diffraction_inv = np.zeros(self.dataset.array.shape[2:])

    @classmethod
    def from_data(cls, data: NDArray | Dataset4dstem, *, name: str = "strain_map") -> "StrainMap":
        if isinstance(data, Dataset4dstem):
            return cls(dataset=data, input_data=data, _token=cls._token)

        arr = ensure_valid_array(data)
        if arr.ndim != 4:
            raise ValueError(
                "StrainMap.from_data expects a 4D array with shape (scan_r, scan_c, dp_r, dp_c)."
            )

        ds4 = Dataset4dstem.from_array(arr, name=name)
        return cls(dataset=ds4, input_data=data, _token=cls._token)


    def diffraction_mask(
        self,
        threshold = None,
        edge_blend = 64.0,
        plot_mask = True,
        figsize = (8,4),
    ):
        dp_mean = np.mean(self.dataset.array,axis=(0,1))
        mask_init = dp_mean < threshold
        mask_init[:,0] = True
        mask_init[0,:] = True
        mask_init[:,-1] = True
        mask_init[-1,:] = True

        self.mask_diffraction = np.sin(
            np.clip(
                distance_transform_edt(np.logical_not(mask_init)) / edge_blend,
                0.0,
                1.0,
            )*np.pi/2,
        )**2
        # int_edge = np.sum(dp_mean*self.mask_diffraction) / np.sum(self.mask_diffraction)
        int_edge = np.min(dp_mean[self.mask_diffraction>0.99])
        self.mask_diffraction_inv = (1 - self.mask_diffraction) * int_edge

        if plot_mask:
            fig,ax = plt.subplots(1,2,figsize=figsize)
            ax[0].imshow(
                np.log(np.maximum(dp_mean,np.min(dp_mean[dp_mean>0]))),
                cmap = 'gray',
            )
            ax[1].imshow(
                np.log(
                    dp_mean*self.mask_diffraction + \
                    self.mask_diffraction_inv,
                ),
                cmap = 'gray',
            )
        
        return self


    def preprocess(
        self,
        mode: str = "linear",
        q_to_r_rotation_ccw_deg: float | None = None,
        q_transpose: bool | None = None,
        skip = None,
        plot_transform: bool = True,
        cropping_factor: float = 0.25,
        **plot_kwargs: Any,
    ) -> "StrainMap":

        self.metadata["mode"] = mode

        qrow_unit = str(self.dataset.units[2])
        qcol_unit = str(self.dataset.units[3])

        if qrow_unit in {"A", "Å"}:
            qrow_sampling_ang = float(self.dataset.sampling[2])
        elif qrow_unit == "mrad":
            wavelength = float(electron_wavelength_angstrom(float(self.dataset.metadata["energy"])))
            qrow_sampling_ang = float(self.dataset.sampling[2]) / 1000.0 / wavelength
        else:
            qrow_sampling_ang = 1.0
            qrow_unit = "pixels"

        if qcol_unit in {"A", "Å"}:
            qcol_sampling_ang = float(self.dataset.sampling[3])
        elif qcol_unit == "mrad":
            wavelength = float(electron_wavelength_angstrom(float(self.dataset.metadata["energy"])))
            qcol_sampling_ang = float(self.dataset.sampling[3]) / 1000.0 / wavelength
        else:
            qcol_sampling_ang = 1.0
            qcol_unit = "pixels"

        self.metadata["sampling_real"] = np.array(
            (
                1.0 / (qrow_sampling_ang * float(self.dataset.shape[2])),
                1.0 / (qcol_sampling_ang * float(self.dataset.shape[3])),
            ),
            dtype=float,
        )

        if qrow_unit == "pixels" and qcol_unit == "pixels":
            self.metadata["real_units"] = "1/pixels"
        else:
            self.metadata["real_units"] = r"$\mathrm{\AA}$"

        if q_to_r_rotation_ccw_deg is None or q_transpose is None:
            parent_rot = self.dataset.metadata.get("q_to_r_rotation_ccw_deg", None)
            parent_tr = self.dataset.metadata.get("q_transpose", None)
            if q_to_r_rotation_ccw_deg is None and parent_rot is not None:
                q_to_r_rotation_ccw_deg = float(parent_rot)
            if q_transpose is None and parent_tr is not None:
                q_transpose = bool(parent_tr)
            if (parent_rot is not None or parent_tr is not None) and (
                q_to_r_rotation_ccw_deg is not None or q_transpose is not None
            ):
                import warnings

                warnings.warn(
                    f"StrainMap.preprocess: using parent Dataset4dstem metadata "
                    f"(q_to_r_rotation_ccw_deg={float(q_to_r_rotation_ccw_deg or 0.0)}, "
                    f"q_transpose={bool(q_transpose or False)}).",
                    UserWarning,
                )

        if q_to_r_rotation_ccw_deg is None or q_transpose is None:
            import warnings

            q_to_r_rotation_ccw_deg = (
                0.0 if q_to_r_rotation_ccw_deg is None else float(q_to_r_rotation_ccw_deg)
            )
            q_transpose = False if q_transpose is None else bool(q_transpose)
            warnings.warn(
                "StrainMap.preprocess: setting q_to_r_rotation_ccw_deg=0.0 and q_transpose=False.",
                UserWarning,
            )

        self.metadata["q_to_r_rotation_ccw_deg"] = float(q_to_r_rotation_ccw_deg)
        self.metadata["q_transpose"] = bool(q_transpose)

        if skip is None:
            if self.metadata["mode"] == "linear":
                im = np.mean(np.abs(np.fft.fft2(
                    self.dataset.array * self.mask_diffraction[None,None,:,:] + \
                    self.mask_diffraction_inv[None,None,:,:] 
                )), axis=(0, 1))
            elif self.metadata["mode"]  == "log":
                im = np.mean(np.abs(np.fft.fft2(np.log1p(
                    self.dataset.array * self.mask_diffraction[None,None,:,:] + \
                    self.mask_diffraction_inv[None,None,:,:] 
                ))), axis=(0, 1))
            else:
                raise ValueError("mode must be 'linear' or 'log'")
        else:
            if self.metadata["mode"] == "linear":
                im = np.mean(np.abs(np.fft.fft2(
                    self.dataset.array[::skip,::skip] * self.mask_diffraction[None,None,:,:] + \
                    self.mask_diffraction_inv[None,None,:,:] 
                )), axis=(0, 1))
            elif self.metadata["mode"]  == "log":
                im = np.mean(np.abs(np.fft.fft2(np.log1p(
                    self.dataset.array[::skip,::skip] * self.mask_diffraction[None,None,:,:] + \
                    self.mask_diffraction_inv[None,None,:,:] 
                ))), axis=(0, 1))
            else:
                raise ValueError("mode must be 'linear' or 'log'")

        im = np.fft.fftshift(im)

        self.transform = Dataset2d.from_array(
            im,
            origin=(im.shape[0] // 2, im.shape[1] // 2),
            sampling=(1.0, 1.0),
            units=(qrow_unit, qcol_unit),
            signal_units="intensity",
        )

        im_plot = self.transform.array
        if bool(self.metadata["q_transpose"]):
            im_plot = im_plot.T

        self.transform_rotated = Dataset2d.from_array(
            rotate_image(
                im_plot,
                float(self.metadata["q_to_r_rotation_ccw_deg"]),
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

        sampling = float(np.mean(self.metadata["sampling_real"]))
        units = str(self.metadata.get("real_units", r"$\mathrm{\AA}$"))

        W = int(self.transform.shape[1])
        view_w_px = float(W) * float(cropping_factor)
        target_units = float(scalebar_fraction) * view_w_px * sampling
        sb_len = _nice_length_units(target_units)

        # intensity scaling: compute from transform, apply same scaling to both panels
        kr = (np.arange(self.transform.shape[0], dtype=float) - self.transform.shape[0] // 2)[:, None]
        kc = (np.arange(self.transform.shape[1], dtype=float) - self.transform.shape[1] // 2)[None, :]
        qmag = np.sqrt(kr * kr + kc * kc)
        im0 = self.transform.array
        tmp = im0 * qmag
        i0 = np.unravel_index(int(np.nanargmax(tmp)), tmp.shape)
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


    def choose_lattice_vector(
        self,
        *,
        u: tuple[float, float] | NDArray,
        v: tuple[float, float] | NDArray,
        define_in_rotated: bool = False,
        refine_subpixel: bool = True,
        refine_subpixel_dft: bool = False,
        refine_radius_px: float = 2.0,
        refine_log: bool = False,
        upsample: int = 16,
        plot: bool = True,
        cropping_factor: float = 0.25,
        **plot_kwargs: Any,
    ) -> "StrainMap":
        if self.transform is None or self.transform_rotated is None:
            raise ValueError("Run preprocess() first to compute transform images.")

        u_rc = np.asarray(u, dtype=float).reshape(2)
        v_rc = np.asarray(v, dtype=float).reshape(2)

        rot_ccw = float(self.metadata.get("q_to_r_rotation_ccw_deg", 0.0))
        q_transpose = bool(self.metadata.get("q_transpose", False))

        if define_in_rotated:
            u_rc = _display_vec_to_raw(u_rc, rotation_ccw_deg=rot_ccw, transpose=q_transpose)
            v_rc = _display_vec_to_raw(v_rc, rotation_ccw_deg=rot_ccw, transpose=q_transpose)

        if refine_subpixel_dft:
            refine_subpixel = True

        if refine_subpixel:
            u_rc, v_rc = _refine_lattice_vectors(
                self.transform.array,
                u_rc=u_rc,
                v_rc=v_rc,
                radius_px=float(refine_radius_px),
                log_fit=bool(refine_log),
                refine_dft=bool(refine_subpixel_dft),
                upsample=int(upsample),
            )

        self.u = u_rc
        self.v = v_rc
        self.metadata["lattice_u_rc"] = self.u.copy()
        self.metadata["lattice_v_rc"] = self.v.copy()

        if plot:
            fig, ax = self.plot_transform(cropping_factor=cropping_factor, **plot_kwargs)
            _overlay_lattice_vectors(
                ax=ax,
                shape=self.transform.shape,
                u_rc=self.u,
                v_rc=self.v,
                rot_ccw_deg=rot_ccw,
                q_transpose=q_transpose,
            )
            return self

        return self


    def fit_lattice_vectors(
        self,
        refine_subpixel: bool = True,
        refine_subpixel_dft: bool = False,
        refine_radius_px: float = 2.0,
        upsample: int = 16,
        refine_log: bool = False,
        progressbar: bool = True,
    ) -> "StrainMap":
        from quantem.core.datastructures.dataset3d import Dataset3d

        if self.u is None or self.v is None:
            raise ValueError("Run choose_lattice_vector() first to set initial lattice vectors (self.u, self.v).")

        if refine_subpixel_dft:
            refine_subpixel = True

        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]
        self.u_fit = Dataset3d.from_shape(
            (scan_r, scan_c, 2),
            name="u_fits",
            signal_units="pixels",
        )
        self.v_fit = Dataset3d.from_shape(
            (scan_r, scan_c, 2),
            name="v_fits",
            signal_units="pixels",
        )

        mode = str(self.metadata.get("mode", "linear")).lower()

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
            dp = self.dataset.array[r, c]*self.mask_diffraction + \
                self.mask_diffraction_inv

            if mode == "linear":
                im = np.fft.fftshift(np.abs(np.fft.fft2(dp)))
            elif mode == "log":
                im = np.fft.fftshift(np.abs(np.fft.fft2(np.log1p(dp))))
            else:
                raise ValueError("metadata['mode'] must be 'linear' or 'log'")

            if refine_subpixel:
                u_rc, v_rc = _refine_lattice_vectors(
                    im,
                    u_rc=u0,
                    v_rc=v0,
                    radius_px=float(refine_radius_px),
                    log_fit=bool(refine_log),
                    refine_dft=bool(refine_subpixel_dft),
                    upsample=int(upsample),
                )
            else:
                u_rc = u0
                v_rc = v0

            self.u_fit.array[r, c, :] = u_rc
            self.v_fit.array[r, c, :] = v_rc

        self.metadata["fit_refine_subpixel"] = bool(refine_subpixel)
        self.metadata["fit_refine_subpixel_dft"] = bool(refine_subpixel_dft)
        self.metadata["fit_refine_radius_px"] = float(refine_radius_px)
        self.metadata["fit_refine_log"] = bool(refine_log)
        self.metadata["fit_upsample"] = int(upsample)

        return self


    def plot_lattice_vectors(
        self,
        subtract_mean: bool = True,
        scalebar: bool = False,
        **plot_kwargs: Any,
    ):
        if getattr(self, "u_fit", None) is None or getattr(self, "v_fit", None) is None:
            raise ValueError("Run fit_lattice_vectors() first to compute u_fit and v_fit.")

        im0 = self.u_fit.array[:,:,0]
        im1 = self.u_fit.array[:,:,1]
        im2 = self.v_fit.array[:,:,0]
        im3 = self.v_fit.array[:,:,1]

        if subtract_mean:
            im0 = im0 - float(np.nanmean(im0))
            im1 = im1 - float(np.nanmean(im1))
            im2 = im2 - float(np.nanmean(im2))
            im3 = im3 - float(np.nanmean(im3))

        vlim = float(np.nanmax(np.abs(np.stack([im0, im1, im2, im3], axis=0))))
        vmin = -vlim
        vmax = vlim

        defaults: dict[str, Any] = dict(
            title=("u_r", "u_c", "v_r", "v_c"),
            vmin=vmin,
            vmax=vmax,
        )

        if scalebar:
            s0 = float(self.dataset.sampling[0]) if len(self.dataset.sampling) > 0 else 1.0
            s1 = float(self.dataset.sampling[1]) if len(self.dataset.sampling) > 1 else s0
            sampling_scan = float(np.mean([s0, s1]))
            units_scan = str(self.dataset.units[0]) if len(self.dataset.units) > 0 else "pixels"
            defaults["scalebar"] = ScalebarConfig(sampling=sampling_scan, units=units_scan)

        defaults.update(plot_kwargs)

        fig, ax = show_2d([im0, im1, im2, im3], **defaults)
        return fig, ax


    def fit_strain(
        self,
        mask_reference = None,
        plot_strain = True,
    ):
        if self.u_fit is None or self.v_fit is None:
            raise ValueError("Run fit_lattice_vectors() first to compute u_fit and v_fit.")

        u_fit = self.u_fit.array
        v_fit = self.v_fit.array
        scan_r, scan_c = int(u_fit.shape[0]), int(u_fit.shape[1])

        if mask_reference is None:
            self.u_ref = np.median(u_fit.reshape(-1, 2), axis=0)
            self.v_ref = np.median(v_fit.reshape(-1, 2), axis=0)
        else:
            m = np.asarray(mask_reference, dtype=bool)
            self.u_ref = np.array(
                (
                    np.median(u_fit[m, 0]),
                    np.median(u_fit[m, 1]),
                ),
                dtype=float,
            )
            self.v_ref = np.array(
                (
                    np.median(v_fit[m, 0]),
                    np.median(v_fit[m, 1]),
                ),
                dtype=float,
            )

        Uref = np.stack((self.u_ref, self.v_ref), axis=1).astype(float)
        det = float(np.linalg.det(Uref))
        if not np.isfinite(det) or abs(det) < 1e-12:
            Uref_inv = np.linalg.pinv(Uref)
        else:
            Uref_inv = np.linalg.inv(Uref)

        # init
        self.strain_trans = Dataset4d.from_shape(
            (scan_r, scan_c, 2, 2),
            name="transformation matrix",
            signal_units="fractional",
        )

        # Loop over probe positions
        for r in range(scan_r):
            for c in range(scan_c):
                U = np.stack((u_fit[r, c, :], v_fit[r, c, :]), axis=1)
                self.strain_trans.array[r, c, :, :] = U @ Uref_inv

        # get strains in orthogonal directions
        self.strain_raw_err = Dataset2d.from_array(
            self.strain_trans.array[:, :, 0, 0] - 1,
            name="strain err",
            signal_units="fractional",
        )
        self.strain_raw_ecc = Dataset2d.from_array(
            self.strain_trans.array[:, :, 1, 1] - 1,
            name="strain ecc",
            signal_units="fractional",
        )
        self.strain_raw_erc = Dataset2d.from_array(
            self.strain_trans.array[:, :, 1, 0]*0.5 + self.strain_trans.array[:, :, 0, 1]*0.5,
            name="strain erc",
            signal_units="fractional",
        )
        self.strain_rotation = Dataset2d.from_array(
            self.strain_trans.array[:, :, 1, 0]*-0.5 + self.strain_trans.array[:, :, 0, 1]*0.5,
            name="strain rotation",
            signal_units="fractional",
        )

        return self


    def plot_strain(
        self,
        ref_u_v=(1.0, 0.0),
        ref_angle_degrees=None,
        strain_range_percent=(-3.0, 3.0),
        rotation_range_degrees=(-2.0, 2.0),
        plot_rotation=True,
        cmap_strain="PiYG_r",
        cmap_rotation="PiYG_r",
        layout="horizontal",
        figsize=(6, 6),
    ):
        import matplotlib.pyplot as plt

        if ref_angle_degrees is None:
            ref_vec = self.u_ref * float(ref_u_v[0]) + self.v_ref * float(ref_u_v[1])
            ref_angle = float(np.arctan2(ref_vec[1], ref_vec[0]))
        else:
            ref_angle = float(np.deg2rad(ref_angle_degrees))

        angle = ref_angle + np.deg2rad(self.metadata["q_to_r_rotation_ccw_deg"])
        print(np.round(np.rad2deg(angle),2))

        c = float(np.cos(angle))
        s = float(np.sin(angle))

        err = self.strain_raw_err.array
        ecc = self.strain_raw_ecc.array
        erc = self.strain_raw_erc.array

        euu = err * (c * c) + 2.0 * erc * (c * s) + ecc * (s * s)
        evv = err * (s * s) - 2.0 * erc * (c * s) + ecc * (c * c)
        euv = (ecc - err) * (c * s) + erc * (c * c - s * s)

        self.strain_euu = self.strain_raw_err.copy()
        self.strain_evv = self.strain_raw_ecc.copy()
        self.strain_euv = self.strain_raw_erc.copy()
        self.strain_euu.array[...] = euu
        self.strain_evv.array[...] = evv
        self.strain_euv.array[...] = euv

        if layout == "horizontal":
            if plot_rotation:
                fig, ax = plt.subplots(1, 4, figsize=figsize)

                ax[0].imshow(
                    self.strain_euu.array * 100,
                    vmin=strain_range_percent[0],
                    vmax=strain_range_percent[1],
                    cmap=cmap_strain,
                )
                ax[1].imshow(
                    self.strain_evv.array * 100,
                    vmin=strain_range_percent[0],
                    vmax=strain_range_percent[1],
                    cmap=cmap_strain,
                )
                ax[2].imshow(
                    self.strain_euv.array * 100,
                    vmin=strain_range_percent[0],
                    vmax=strain_range_percent[1],
                    cmap=cmap_strain,
                )
                ax[3].imshow(
                    np.rad2deg(self.strain_rotation.array),
                    vmin=rotation_range_degrees[0],
                    vmax=rotation_range_degrees[1],
                    cmap=cmap_rotation,
                )
                return fig, ax

            fig, ax = plt.subplots(1, 3, figsize=figsize)
            ax[0].imshow(
                self.strain_euu.array * 100,
                vmin=strain_range_percent[0],
                vmax=strain_range_percent[1],
                cmap=cmap_strain,
            )
            ax[1].imshow(
                self.strain_evv.array * 100,
                vmin=strain_range_percent[0],
                vmax=strain_range_percent[1],
                cmap=cmap_strain,
            )
            ax[2].imshow(
                self.strain_euv.array * 100,
                vmin=strain_range_percent[0],
                vmax=strain_range_percent[1],
                cmap=cmap_strain,
            )
            return fig, ax

        raise ValueError("layout must be 'horizontal'")


def _nice_length_units(target: float) -> float:
    target = float(target)
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
    return float(nice * (10.0**exp))


def _apply_center_crop_limits(ax: Any, shape: tuple[int, int], cropping_factor: float) -> None:
    cf = float(cropping_factor)
    if cf >= 1.0:
        return
    if not (0.0 < cf <= 1.0):
        raise ValueError("cropping_factor must be in (0, 1].")

    H, W = int(shape[0]), int(shape[1])
    r0 = float(H // 2)
    c0 = float(W // 2)
    half_h = 0.5 * cf * H
    half_w = 0.5 * cf * W

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
    dr, dc = float(v[0]), float(v[1])

    if transpose:
        dr, dc = dc, dr

    theta = float(np.deg2rad(rotation_ccw_deg))
    ct = float(np.cos(theta))
    st = float(np.sin(theta))

    dr2 = ct * dr - st * dc
    dc2 = st * dr + ct * dc
    return np.array((dr2, dc2), dtype=float)


def _display_vec_to_raw(vec_rc: NDArray, *, rotation_ccw_deg: float, transpose: bool) -> NDArray:
    v = np.asarray(vec_rc, dtype=float).reshape(2)
    dr, dc = float(v[0]), float(v[1])

    theta = float(np.deg2rad(rotation_ccw_deg))
    ct = float(np.cos(theta))
    st = float(np.sin(theta))

    dr2 = ct * dr + st * dc
    dc2 = -st * dr + ct * dc

    if transpose:
        dr2, dc2 = dc2, dr2

    return np.array((dr2, dc2), dtype=float)


def _plot_lattice_vectors(ax: Any, center_rc: tuple[float, float], u_rc: NDArray, v_rc: NDArray) -> None:
    r0, c0 = float(center_rc[0]), float(center_rc[1])

    def _draw(vec: NDArray, label: str, color: tuple[float, float, float]) -> None:
        dr, dc = float(vec[0]), float(vec[1])
        ax.plot([c0, c0 + dc], [r0, r0 + dr], linewidth=2.75, color=color)
        ax.plot([c0 + dc], [r0 + dr], marker="o", markersize=6.0, color=color)
        ax.text(c0 + dc, r0 + dr, f" {label}", color=color, fontsize=18, va="center")

    _draw(np.asarray(u_rc, dtype=float).reshape(2), "u", (1.0, 0.0, 0.0))
    _draw(np.asarray(v_rc, dtype=float).reshape(2), "v", (0.0, 0.7, 1.0))


def _overlay_lattice_vectors(
    *,
    ax: Any,
    shape: tuple[int, int],
    u_rc: NDArray,
    v_rc: NDArray,
    rot_ccw_deg: float,
    q_transpose: bool,
) -> None:
    axs = _flatten_axes(ax)
    if not axs:
        return

    H, W = int(shape[0]), int(shape[1])
    center_rc = (float(H // 2), float(W // 2))

    _plot_lattice_vectors(axs[0], center_rc, u_rc, v_rc)

    if len(axs) >= 2:
        u_disp = _raw_vec_to_display(u_rc, rotation_ccw_deg=float(rot_ccw_deg), transpose=bool(q_transpose))
        v_disp = _raw_vec_to_display(v_rc, rotation_ccw_deg=float(rot_ccw_deg), transpose=bool(q_transpose))
        _plot_lattice_vectors(axs[1], center_rc, u_disp, v_disp)


def _parabolic_vertex_delta(v_m1: float, v_0: float, v_p1: float) -> float:
    denom = (v_m1 - 2.0 * v_0 + v_p1)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    delta = 0.5 * (v_m1 - v_p1) / denom
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -1.0, 1.0))


def _refine_peak_subpixel(
    im: NDArray,
    *,
    r_guess: float,
    c_guess: float,
    radius_px: float = 2.0,
    log_fit: bool = False,
) -> tuple[float, float]:
    im = np.asarray(im, dtype=float)
    H, W = im.shape

    r0 = int(np.clip(int(np.round(r_guess)), 0, H - 1))
    c0 = int(np.clip(int(np.round(c_guess)), 0, W - 1))
    rad = int(max(0, int(np.ceil(float(radius_px)))))

    r1 = max(0, r0 - rad)
    r2 = min(H, r0 + rad + 1)
    c1 = max(0, c0 - rad)
    c2 = min(W, c0 + rad + 1)

    win = im[r1:r2, c1:c2]
    if win.size == 0:
        return float(r_guess), float(c_guess)

    ir, ic = np.unravel_index(int(np.argmax(win)), win.shape)
    r_peak = r1 + int(ir)
    c_peak = c1 + int(ic)

    if 0 < r_peak < H - 1:
        col = im[r_peak - 1 : r_peak + 2, c_peak]
        if log_fit:
            col = np.log(np.clip(col, 1e-12, None))
        dr = _parabolic_vertex_delta(float(col[0]), float(col[1]), float(col[2]))
    else:
        dr = 0.0

    if 0 < c_peak < W - 1:
        row = im[r_peak, c_peak - 1 : c_peak + 2]
        if log_fit:
            row = np.log(np.clip(row, 1e-12, None))
        dc = _parabolic_vertex_delta(float(row[0]), float(row[1]), float(row[2]))
    else:
        dc = 0.0

    return float(r_peak) + dr, float(c_peak) + dc


def _refine_peak_subpixel_dft(
    im: NDArray,
    *,
    r0: float,
    c0: float,
    upsample: int,
    log_fit: bool = False,
) -> tuple[float, float]:
    if int(upsample) <= 1:
        return float(r0), float(c0)

    im = np.asarray(im, dtype=float)
    F = np.fft.fft2(im)

    up = int(upsample)
    du = int(np.ceil(1.5 * up))

    patch = dft_upsample(F, up=up, shift=(float(r0), float(c0)), device="cpu")
    patch = np.asarray(patch, dtype=float)

    i0, j0 = np.unravel_index(int(np.argmax(patch)), patch.shape)
    i0 = int(i0)
    j0 = int(j0)

    if 0 < i0 < patch.shape[0] - 1:
        col = patch[i0 - 1 : i0 + 2, j0]
        if log_fit:
            col = np.log(np.clip(col, 1e-12, None))
        di = _parabolic_vertex_delta(float(col[0]), float(col[1]), float(col[2]))
    else:
        di = 0.0

    if 0 < j0 < patch.shape[1] - 1:
        row = patch[i0, j0 - 1 : j0 + 2]
        if log_fit:
            row = np.log(np.clip(row, 1e-12, None))
        dj = _parabolic_vertex_delta(float(row[0]), float(row[1]), float(row[2]))
    else:
        dj = 0.0

    dr = (float(i0) - float(du) + float(di)) / float(up)
    dc = (float(j0) - float(du) + float(dj)) / float(up)

    return float(r0) + dr, float(c0) + dc


def _refine_lattice_vectors(
    im: NDArray,
    *,
    u_rc: NDArray,
    v_rc: NDArray,
    radius_px: float = 2.0,
    log_fit: bool = False,
    refine_dft: bool = True,
    upsample: int = 16,
) -> tuple[NDArray, NDArray]:
    im = np.asarray(im, dtype=float)
    if im.ndim != 2:
        raise ValueError("im must be 2D.")

    H, W = im.shape
    r_center = float(H // 2)
    c_center = float(W // 2)

    def _refine(vec: NDArray) -> NDArray:
        vec = np.asarray(vec, dtype=float).reshape(2)
        r_guess = r_center + float(vec[0])
        c_guess = c_center + float(vec[1])

        r1, c1 = _refine_peak_subpixel(
            im,
            r_guess=float(r_guess),
            c_guess=float(c_guess),
            radius_px=float(radius_px),
            log_fit=bool(log_fit),
        )

        if refine_dft and int(upsample) > 1:
            r2, c2 = _refine_peak_subpixel_dft(
                im,
                r0=float(r1),
                c0=float(c1),
                upsample=int(upsample),
                log_fit=bool(log_fit),
            )
        else:
            r2, c2 = float(r1), float(c1)

        return np.array((r2 - r_center, c2 - c_center), dtype=float)

    return _refine(u_rc), _refine(v_rc)
