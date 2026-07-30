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
from quantem.diffraction.strain import StrainMap


class StrainMapAutocorrelation(AutoSerialize):
    """Cepstral / autocorrelation lattice fitting and strain mapping for 4D-STEM.

    An alternative to correlation-based :class:`~quantem.diffraction.bragg_vectors.BraggVectors`
    that needs no disk template: every diffraction pattern is transformed into the
    autocorrelation (real-space) domain, where the crystal periodicity shows up as a
    lattice of sharp peaks, and those peaks are fit per scan position to recover the
    local lattice vectors. Because the transform lives in real space, the peaks track
    real-space lattice spacings (they *expand* under tension, opposite to Bragg disks),
    so ``real_space=True`` -- :func:`~quantem.diffraction.strain._strain_tensor` then
    reduces both this and the reciprocal-space Bragg path to the same deformation
    gradient, and the two strain maps can be compared directly.

    Workflow (each step writes state consumed by the next):

    1. :meth:`diffraction_mask` -- build a soft mask over the detector that suppresses
       the bright central beam / vacuum so the transform is dominated by the lattice.
    2. :meth:`preprocess` -- transform every pattern and average the magnitudes into a
       mean transform image. Three ``mode`` choices set the intensity scaling applied
       before the FFT: ``"linear"`` (Patterson / autocorrelation), ``"log"``
       (cepstrum), or ``"gamma"`` (power law). The detector->scan rotation
       (``q_to_r_rotation_ccw_deg`` + ``q_transpose``) is read from the parent dataset
       metadata, the single source of truth shared with the DPC/CoM and Bragg
       workflows.
    3. :meth:`choose_lattice_vector` -- refine a hand-picked initial ``(u, v)`` basis
       against the mean transform, optionally auto-detecting and re-fitting all peaks.
    4. :meth:`fit_lattice_vectors` -- the heavy step: transform and fit the lattice
       vectors at every scan position into ``u_array``/``v_array`` of shape
       ``(scan_row, scan_col, 2)``.
    5. :meth:`create_mask` -- compute the per-position weight :attr:`mask_weight`
       (lattice signal strength) used to weight the reference lattice.
    6. :meth:`calculate_strain_map` -- hand the lattice vectors (and
       :attr:`mask_weight`) to a :class:`~quantem.diffraction.strain.StrainMap`.

    Use :meth:`from_dataset` or :meth:`from_array` to construct an instance.

    Parameters
    ----------
    dataset : Dataset4dstem
        The 4D-STEM dataset to analyze.
    input_data : Any, optional
        The original object passed to the constructor (a dataset or array), retained
        for provenance.
    """

    _token = object()

    # Cepstral / Patterson lattice vectors are measured in the autocorrelation
    # (real-space) domain -- the peaks track real-space lattice spacings and expand
    # under tension -- so real_space=True. _strain_tensor reduces both this and the
    # reciprocal-space Bragg path to F.T, so the two strain maps agree.
    real_space: bool = True

    def __init__(
        self,
        dataset: Dataset4dstem,
        input_data: Any | None = None,
        _token: object | None = None,
    ):
        """Private constructor; use :meth:`from_dataset` or :meth:`from_array`.

        Direct instantiation is blocked by the ``_token`` guard. The factory
        classmethods are the supported entry points and document their arguments.

        Parameters
        ----------
        dataset : Dataset4dstem
            The 4D-STEM dataset to analyze.
        input_data : Any, optional
            The original object passed to a factory (kept for provenance).
        _token : object, optional
            Internal sentinel; must match the class token or a ``RuntimeError`` is
            raised.
        """
        if _token is not self._token:
            raise RuntimeError(
                "Use StrainMapAutocorrelation.from_dataset() or StrainMapAutocorrelation.from_array() to instantiate this class."
            )
        # Explicit (two-arg) super() rather than the bare super(): the zero-arg form
        # needs a compiler-created __class__ closure cell that is absent when this
        # method's source is re-exec'd from a string (Jupyter autoreload), which would
        # raise "super(): __class__ cell not found".
        super(StrainMapAutocorrelation, self).__init__()
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

        # initial basis from choose_lattice_vector(); per-position fits from
        # fit_lattice_vectors(); per-position weight from create_mask().
        self.u: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.u_peak_fit: Dataset3d | None = None
        self.v_peak_fit: Dataset3d | None = None
        self.u_ref: np.ndarray | None = None
        self.v_ref: np.ndarray | None = None
        self.u_array: np.ndarray | None = None
        self.v_array: np.ndarray | None = None
        self.mask_weight: np.ndarray | None = None
        self._gpu_cache: torch.Tensor | None = None

    @classmethod
    def from_dataset(cls, dataset: Dataset4dstem, *, name: str | None = None) -> "StrainMapAutocorrelation":
        """Create a cepstral strain workflow bound to a 4D-STEM dataset.

        Parameters
        ----------
        dataset : Dataset4dstem
            The 4D-STEM dataset to analyze.
        name : str, optional
            If given, sets ``dataset.name``.

        Returns
        -------
        StrainMapAutocorrelation
            A new workflow instance bound to ``dataset``.
        """
        if not isinstance(dataset, Dataset4dstem):
            raise TypeError("StrainMapAutocorrelation.from_dataset expects a Dataset4dstem instance.")
        if name is not None:
            dataset.name = name
        return cls(dataset=dataset, input_data=dataset, _token=cls._token)

    @classmethod
    def from_array(cls, array: NDArray, *, name: str = "strain_map_autocorrelation") -> "StrainMapAutocorrelation":
        """Create a cepstral strain workflow from a raw 4D array.

        Parameters
        ----------
        array : np.ndarray
            4D-STEM data with shape ``(scan_row, scan_col, dp_row, dp_col)``.
        name : str, default="strain_map_autocorrelation"
            Name for the wrapped :class:`Dataset4dstem`.

        Returns
        -------
        StrainMapAutocorrelation
            A new workflow instance wrapping the data in a :class:`Dataset4dstem`.
        """
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
        threshold_percentile=50.0,
        edge_blend=64.0,
        plot_mask=True,
        figsize=(8, 4),
    ):
        """Build a soft detector mask suppressing the central beam and vacuum.

        Pixels of the mean diffraction pattern below ``threshold`` (plus the detector
        border) are treated as "outside" the useful signal. The kept region is feathered
        with a raised-cosine taper of width ``edge_blend`` (via a Euclidean distance
        transform) into :attr:`mask_diffraction` (multiplicative, in ``[0, 1]``), and a
        complementary fill :attr:`mask_diffraction_inv` replaces the masked region with a
        flat edge intensity. Both are applied to every pattern in :meth:`preprocess` and
        :meth:`fit_lattice_vectors` so the transform is dominated by the crystalline
        signal rather than the bright unscattered beam.

        Parameters
        ----------
        threshold : float, optional
            Absolute mean-intensity level below which detector pixels are masked out. If
            ``None`` (default), the level is taken from ``threshold_percentile`` of the
            mean diffraction pattern, so no manual intensity value is needed.
        threshold_percentile : float, default=50.0
            Percentile (``0``-``100``) of the mean diffraction pattern used to set the
            masking level when ``threshold`` is ``None``. The default keeps the brighter
            half of the detector (disks and central beam) and masks the dimmer vacuum;
            raise it to mask more aggressively. Ignored when ``threshold`` is given.
        edge_blend : float, default=64.0
            Feather width in pixels of the raised-cosine taper between kept and masked
            regions; larger values give a softer transition.
        plot_mask : bool, default=True
            If ``True``, show the raw mean pattern beside the masked/filled pattern (log
            scaled) so the mask can be checked.
        figsize : tuple of float, default=(8, 4)
            Figure size in inches for the diagnostic plot.

        Returns
        -------
        StrainMapAutocorrelation
            ``self``, with :attr:`mask_diffraction` and :attr:`mask_diffraction_inv` set.
        """
        dp_mean = np.mean(self.dataset.array, axis=(0, 1))
        if threshold is None:
            threshold = np.percentile(dp_mean, threshold_percentile)
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
        """Transform every pattern into the autocorrelation domain and average it.

        For each diffraction pattern the masked/filled pattern (from
        :meth:`diffraction_mask`) is intensity-scaled per ``mode``, Fourier transformed,
        and its magnitude accumulated; the mean over all scan positions is stored
        (fft-shifted, origin centered) as :attr:`transform`. A display copy rotated by
        the detector->scan rotation is stored as :attr:`transform_rotated`. The crystal
        periodicity appears as a lattice of peaks about the center, which later steps fit.

        The detector->scan rotation (``q_to_r_rotation_ccw_deg``) and transpose
        (``q_transpose``) default to the parent dataset metadata -- the same source used
        by the DPC/CoM and :class:`BraggVectors` workflows -- so the strain frame is
        consistent across methods; pass them explicitly to override.

        Parameters
        ----------
        mode : {"linear", "log", "gamma"}, default="linear"
            Intensity scaling applied before the FFT. ``"linear"`` is the Patterson /
            autocorrelation (aliases ``"patterson"``, ``"acf"``, ``"autocorrelation"``);
            ``"log"`` is the cepstrum, ``log1p(I)`` (aliases ``"cepstrum"``,
            ``"cepstral"``); ``"gamma"`` raises intensity to the power ``gamma``
            (aliases ``"power"``, ``"sqrt"``).
        q_to_r_rotation_ccw_deg : float, optional
            Counter-clockwise detector->scan rotation in degrees for the rotated display
            transform. Defaults to ``dataset.metadata["q_to_r_rotation_ccw_deg"]`` if
            present, else ``0`` (with a warning).
        q_transpose : bool, optional
            Whether to transpose the detector axes before rotating the display transform.
            Defaults to ``dataset.metadata["q_transpose"]`` if present, else ``False``.
        skip : int, optional
            If given, subsample the scan by this stride (``array[::skip, ::skip]``) when
            building the mean transform -- a fast preview over fewer patterns.
        plot_transform : bool, default=True
            If ``True``, show the original and rotated mean transforms via
            :meth:`plot_transform`.
        cropping_factor : float, default=0.25
            Fraction of the transform width/height shown when ``plot_transform`` is
            ``True`` (the lattice peaks sit near the center).
        gamma : float, default=0.5
            Exponent for ``mode="gamma"`` (ignored otherwise).
        **plot_kwargs
            Forwarded to :meth:`plot_transform`.

        Returns
        -------
        StrainMapAutocorrelation
            ``self``, with :attr:`transform` and :attr:`transform_rotated` set.
        """
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
                "StrainMapAutocorrelation.preprocess: setting q_to_r_rotation_ccw_deg=0.0 and q_transpose=False.",
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
        """Show the original and rotated mean transform images side by side.

        The color range is set from the brightest lattice peak (the global max of the
        radially weighted transform) so the central DC peak does not wash out the panel,
        and both panels are cropped to the central ``cropping_factor`` window where the
        lattice peaks lie. A scale bar is drawn in real-space units.

        Parameters
        ----------
        cropping_factor : float, default=0.25
            Fraction of the full transform width/height shown about the center.
        scalebar_fraction : float, default=0.25
            Target scale-bar length as a fraction of the cropped view width (snapped to
            a "nice" round value).
        **plot_kwargs
            Forwarded to :func:`~quantem.core.visualization.show_2d` (overriding the
            defaults computed here).

        Returns
        -------
        tuple
            ``(fig, ax)`` from :func:`~quantem.core.visualization.show_2d`.
        """
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
        """Show the transform of a single scan position with its fitted lattice vectors.

        Recomputes the transform for the pattern at ``(row, col)`` using the current
        ``mode``, and overlays the per-position fitted ``u``/``v`` vectors (from
        :meth:`fit_lattice_vectors`) plus any detected peaks. Useful for inspecting the
        fit quality at a specific position.

        Parameters
        ----------
        row : int, default=0
            Scan row index of the pattern to transform.
        col : int, default=0
            Scan column index of the pattern to transform.
        cropping_factor : float, default=0.25
            Fraction of the full transform width/height shown about the center.
        scalebar_fraction : float, default=0.25
            Target scale-bar length as a fraction of the cropped view width.
        **plot_kwargs
            Forwarded to :func:`~quantem.core.visualization.show_2d`.

        Returns
        -------
        None
            Draws the figure; nothing is returned.
        """
        if self.transform is None or self.transform_rotated is None:
            raise ValueError("Run preprocess() first to compute transform images.")
        if self.u_peak_fit is None or self.v_peak_fit is None:
            raise ValueError("Run fit_lattice_vectors() first.")

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
            u_rc= self.u_peak_fit.array[row, col, :2],
            v_rc=self.v_peak_fit.array[row, col, :2],
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
        """Refine a hand-picked initial lattice basis against the mean transform.

        Takes an approximate basis ``(u, v)`` -- read off the transform plot by eye --
        and refines each vector to the nearest transform peak, storing the result in
        :attr:`u` and :attr:`v`. These seed the per-position fit in
        :meth:`fit_lattice_vectors`. Vectors are ``(row, col)`` offsets from the
        transform center, in transform pixels.

        Parameters
        ----------
        u : tuple of float or np.ndarray
            Initial first lattice vector ``(d_row, d_col)`` relative to the center.
        v : tuple of float or np.ndarray
            Initial second lattice vector ``(d_row, d_col)`` relative to the center.
        define_in_rotated : bool, default=False
            If ``True``, ``u``/``v`` are given in the rotated (display) frame and are
            converted back to the raw detector frame before fitting.
        refine_gaussian : bool, default=True
            If ``True``, refine each peak by a 2D isotropic Gaussian fit; otherwise use
            the parabolic sub-pixel estimate only.
        refine_dft : bool, default=False
            If ``True``, additionally refine by DFT upsampling (uses ``upsample``).
        refine_all_peaks : bool, default=False
            If ``True``, auto-detect all peaks above ``threshold_percentile`` and fit
            the basis to the full set by weighted least squares (storing the detected
            peaks/weights for reuse), rather than refining only ``u`` and ``v``.
        refine_radius_px : float, default=2.0
            Half-width in pixels of the window used for sub-pixel/Gaussian refinement.
        upsample : int, default=16
            DFT upsampling factor used when ``refine_dft=True``.
        gaussian_maxfev : int, default=100
            Maximum function evaluations for the Gaussian fit.
        threshold_percentile : float, default=0.9975
            Intensity percentile (0--1) above which local maxima are kept as peaks when
            ``refine_all_peaks=True``.
        min_peak_spacing : float, default=0
            Minimum spacing in pixels between accepted peaks when
            ``refine_all_peaks=True`` (0 disables the spacing filter).
        plot : bool, default=True
            If ``True``, show the transform with the refined ``u``/``v`` overlaid.
        cropping_factor : float, default=0.25
            Fraction of the transform shown about the center when ``plot=True``.
        **plot_kwargs
            Forwarded to :meth:`plot_transform`.

        Returns
        -------
        StrainMapAutocorrelation
            ``self``, with :attr:`u` and :attr:`v` set (and the detected peaks/weights
            when ``refine_all_peaks=True``).
        """
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
        device: str = "cpu",
        batch_size: int | None = None,
        save_to_gpu: bool = True,
    ) -> "StrainMapAutocorrelation":
        """Fit the lattice vectors at every scan position (the heavy step).

        For each scan position the masked/filled pattern is transformed (same ``mode``
        as :meth:`preprocess`) and the lattice basis is refined from the
        :meth:`choose_lattice_vector` seed ``(self.u, self.v)``. The fitted vectors are
        written to :attr:`u_array`/:attr:`v_array` (shape ``(scan_row, scan_col, 2)``,
        row/col components) for :meth:`calculate_strain_map`; the full fit records
        (position, amplitude, width, background) are kept in :attr:`u_peak_fit`/
        :attr:`v_peak_fit` (shape ``(scan_row, scan_col, 5)``).

        Whenever ``refine_dft=False`` (including ``refine_all_peaks=True``) the transforms
        and the isotropic-Gaussian peak fits are batched across scan positions on
        ``device`` (the analogue of the correlation pipeline's
        :func:`~quantem.diffraction.disk_detection.detect_disks_batch`): the Gaussian fit
        is a vectorized Levenberg-Marquardt solve rather than a per-position
        ``scipy.optimize.curve_fit``. For ``refine_all_peaks=True`` every detected peak is
        batch-refined across the stack and the basis is solved per position by weighted
        least squares -- avoiding the ``n_positions x n_peaks`` ``curve_fit`` calls of the
        old loop. This removes the dominant per-call overhead and runs far faster (and
        faster still on a GPU). The batched Levenberg-Marquardt fit reproduces the
        per-position ``scipy.optimize.curve_fit`` to ~1e-6 px on clean, well-isolated
        peaks; on noisy, non-Gaussian cepstral peaks the bounded-trf and LM optima can
        land a few hundredths of a pixel apart (amplitude-preserving and well below
        strain-relevant precision). Only ``refine_dft=True`` falls back to the
        per-position path.

        Parameters
        ----------
        refine_gaussian : bool, default=True
            If ``True``, refine each peak with a 2D isotropic Gaussian fit; otherwise
            use the parabolic sub-pixel estimate only.
        refine_dft : bool, default=False
            If ``True``, additionally refine by DFT upsampling (uses ``upsample``).
        refine_all_peaks : bool, default=False
            If ``True``, fit the basis to all peaks detected in
            :meth:`choose_lattice_vector` (which must have been called with
            ``refine_all_peaks=True``) rather than just ``u`` and ``v``.
        refine_radius_px : float, default=2.0
            Half-width in pixels of the window used for sub-pixel/Gaussian refinement.
        upsample : int, default=16
            DFT upsampling factor used when ``refine_dft=True``.
        gaussian_maxfev : int, default=100
            Maximum function evaluations for each Gaussian fit.
        progressbar : bool, default=True
            If ``True``, show a tqdm progress bar over the scan positions.

        Returns
        -------
        StrainMapAutocorrelation
            ``self``, with :attr:`u_array`, :attr:`v_array`, :attr:`u_peak_fit`, and
            :attr:`v_peak_fit` set.
        """
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

        u0 = np.asarray(self.u, dtype=float).reshape(2)
        v0 = np.asarray(self.v, dtype=float).reshape(2)

        if not refine_dft:
            # Fast path: batch the transforms and isotropic-Gaussian fits across scan
            # positions on `device` (per-position scipy.optimize.curve_fit -> vectorized
            # LM). Covers both the 2-vector fit and the all-peaks basis fit; matches the
            # per-position result to ~1e-6 px on clean peaks (a few 0.01 px on noisy,
            # non-Gaussian peaks -- optimizer variance). Only DFT upsampling still falls back.
            self._fit_lattice_vectors_batched(
                u0=u0,
                v0=v0,
                refine_gaussian=refine_gaussian,
                refine_radius_px=refine_radius_px,
                device=device,
                progressbar=progressbar,
                refine_all_peaks=refine_all_peaks,
                peaks=self.mean_img_peaks if refine_all_peaks else None,
                weights=self.mean_img_weights if refine_all_peaks else None,
                batch_size = batch_size,
                save_to_gpu = save_to_gpu
            )
        else:
            # Per-position fallback for DFT upsampling (refine_dft=True).
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

        # Populate the per-position weight directly from the fitted peak amplitudes,
        # mirroring the correlation pipeline where BraggVectors.fit_lattice emits
        # mask_weight. create_mask() is therefore optional -- call it only to plot the
        # weight map, recompute, or switch to the radial estimator.
        self.mask_weight = self._amplitude_mask_weight()

        return self

    def _fit_lattice_vectors_batched(
        self,
        *,
        u0: NDArray,
        v0: NDArray,
        refine_gaussian: bool,
        refine_radius_px: float,
        device: str,
        progressbar: bool,
        refine_all_peaks: bool = False,
        peaks: NDArray | None = None,
        weights: NDArray | None = None,
        batch_size: int | None = None,
        save_to_gpu: bool = True,
    ) -> None:
        """Batched torch implementation of the non-DFT :meth:`fit_lattice_vectors` path.

        The transform (mask/fill -> ``mode`` -> ``|FFT|`` -> fftshift) and the
        parabolic/isotropic-Gaussian peak refinement are evaluated for a stack of scan
        positions at once on ``device``, the analogue of the correlation pipeline's
        :func:`~quantem.diffraction.disk_detection.detect_disks_batch`. This reproduces
        :func:`_refine_lattice_vectors` (with ``refine_dft=False``) -- to ~1e-6 px on
        clean peaks, within a few 0.01 px on noisy non-Gaussian peaks (bounded-trf vs LM
        optimizer variance) -- while removing the per-position
        ``scipy.optimize.curve_fit`` overhead. Results are
        written into :attr:`u_array`/:attr:`v_array` and :attr:`u_peak_fit`/
        :attr:`v_peak_fit`.

        With ``refine_all_peaks=False`` only the two seed vectors ``u0``/``v0`` are
        refined per position. With ``refine_all_peaks=True`` every peak in ``peaks`` (the
        basis detected by :meth:`choose_lattice_vector`, weighted by ``weights``) is
        batch-refined across the stack and the basis is recovered per position by the
        same intensity-weighted least-squares fit as the per-position all-peaks branch --
        so that path is batched too rather than looping ``scipy.optimize.curve_fit`` over
        ``n_positions x n_peaks``. In both cases the amplitude/width/background stored in
        :attr:`u_peak_fit`/:attr:`v_peak_fit` (columns 2-4, the source of the mask weight)
        come from the single-peak refinement at ``u0``/``v0`` -- the background-subtracted
        Gaussian height. The all-peaks least squares only improves the u/v *positions*
        (columns 0-1); the raw cepstral value at those positions rides the
        central-autocorrelation pedestal and would invert the mask (bright in vacuum), so
        it is not used for the weight.
        """
        scan_r, scan_c, H, W = self.dataset.array.shape
        n_pos = scan_r * scan_c
        rcent, ccent = H // 2, W // 2

        mode = self.metadata.get("mode", "linear").lower()
        if mode not in ("linear", "log", "gamma"):
            raise ValueError("metadata['mode'] must be 'linear', 'log', or 'gamma'")
        gamma = float(self.metadata["gamma"]) if mode == "gamma" else None

        dev = torch.device(device)
        mask = torch.as_tensor(self.mask_diffraction, dtype=torch.float64, device=dev)
        mask_inv = torch.as_tensor(self.mask_diffraction_inv, dtype=torch.float64, device=dev)
        use_cache = False
        if save_to_gpu and str(dev) != "cpu":
            if not hasattr(self, '_gpu_cache') or self._gpu_cache is None:
                try:
                    print(f"Loading dataset to {dev}", end = " ", flush = True)
                    self._gpu_cache = torch.as_tensor(
                        np.asarray(self.dataset.array),
                        dtype = torch.float32,
                        device=dev,
                    )
                    use_cache = True
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"Out of memory, will read per-batch")
                    self._gpu_cache = None
                    use_cache = False
            else:
                # self._gpu_cache = None
                use_cache = True
            
        if refine_all_peaks:
            # Precompute the fixed pieces of the per-position all-peaks fit (these do not
            # change across scan positions): the detected peak seeds, their normalized
            # weights, and the seed basis used to assign integer (h, k) indices.
            peaks_arr = np.asarray(peaks, dtype=float).reshape(-1, 2)
            w = np.asarray(weights, dtype=float).reshape(-1)
            wsum = float(w.sum())
            w_norm = w / wsum if wsum != 0 else np.full(w.shape, 1.0 / max(1, w.size))
            sqrt_w = np.sqrt(w_norm)[:, None]
            A_seed = np.column_stack((u0, v0))  # (2, 2)
            n_pk = peaks_arr.shape[0]

        # Match the correlation chunking heuristic (see BraggVectors._detect_positions).
        if batch_size is None:
            batch_size = int(min(1024, max(1, 16_000_000 // (H * W))))

        starts = range(0, n_pos, batch_size)
        if progressbar:
            try:
                from tqdm.auto import tqdm  # type: ignore

                bar = tqdm(total=n_pos, desc="fit_lattice_vectors", leave=True)
            except Exception:
                bar = None
        else:
            bar = None

        for start in starts:
            stop = min(start + batch_size, n_pos)
            idxs = [(idx // scan_c, idx % scan_c) for idx in range(start, stop)]

            if use_cache:
                rows = [r for r, c in idxs]
                cols = [c for r, c in idxs]
                dps = self._gpu_cache[rows, cols]
            else:
                dps = torch.stack(
                    [
                        torch.as_tensor(
                            np.asarray(self.dataset.array[r, c]), dtype=torch.float64, device=dev
                        )
                        for r, c in idxs
                    ],
                    dim=0,
                )

            dpm = dps * mask + mask_inv
            if mode == "linear":
                tr = dpm
            elif mode == "log":
                tr = torch.log1p(dpm)
            else:  # gamma
                tr = dpm.clamp(min=0.0).pow(gamma)
            ims = torch.fft.fftshift(torch.fft.fft2(tr).abs(), dim=(-2, -1))

            if refine_all_peaks:
                # Batch-refine each detected peak across the stack (one vectorized LM
                # solve per peak), then recover the basis per position with the same
                # weighted least squares as _refine_lattice_vectors' all-peaks branch.
                pts_all = np.empty((len(idxs), n_pk, 2), dtype=float)
                for j in range(n_pk):
                    rj = _refine_peaks_batched(
                        ims, peaks_arr[j], radius_px=refine_radius_px,
                        refine_gaussian=refine_gaussian,
                    ).cpu().numpy()
                    pts_all[:, j, :] = rj[:, :2]
                # Amplitude/width/background for the mask weight come from the SAME
                # single-peak refinement at the u/v seeds as the single-peak branch
                # below -- i.e. the background-subtracted Gaussian height. (The all-peaks
                # lstsq only improves the u/v *positions*; the raw cepstral value at those
                # positions rides the central-autocorrelation pedestal and would invert
                # the mask in vacuum.)
                u_fit = _refine_peaks_batched(
                    ims, u0, radius_px=refine_radius_px, refine_gaussian=refine_gaussian
                ).cpu().numpy()
                v_fit = _refine_peaks_batched(
                    ims, v0, radius_px=refine_radius_px, refine_gaussian=refine_gaussian
                ).cpu().numpy()
                for k, (r, c) in enumerate(idxs):
                    pts = pts_all[k]  # (n_pk, 2) row/col offsets from center
                    ab = np.round(np.linalg.lstsq(A_seed, pts.T, rcond=None)[0]).T
                    M = np.ones((n_pk, 3))
                    M[:, :2] = ab  # integer (h, k) indices + constant (center) column
                    uvc = np.linalg.lstsq(M * sqrt_w, pts * sqrt_w, rcond=None)[0]  # (3, 2)
                    u_ref, v_ref = uvc[0], uvc[1]
                    self.u_peak_fit.array[r, c, :] = (
                        u_ref[0], u_ref[1], u_fit[k, 2], u_fit[k, 3], u_fit[k, 4]
                    )
                    self.v_peak_fit.array[r, c, :] = (
                        v_ref[0], v_ref[1], v_fit[k, 2], v_fit[k, 3], v_fit[k, 4]
                    )
                    self.u_array[r, c, 0] = u_ref[0]
                    self.u_array[r, c, 1] = u_ref[1]
                    self.v_array[r, c, 0] = v_ref[0]
                    self.v_array[r, c, 1] = v_ref[1]
            else:
                u_np = _refine_peaks_batched(
                    ims, u0, radius_px=refine_radius_px, refine_gaussian=refine_gaussian
                ).cpu().numpy()
                v_np = _refine_peaks_batched(
                    ims, v0, radius_px=refine_radius_px, refine_gaussian=refine_gaussian
                ).cpu().numpy()
                for k, (r, c) in enumerate(idxs):
                    self.u_peak_fit.array[r, c, :] = u_np[k]
                    self.v_peak_fit.array[r, c, :] = v_np[k]
                    self.u_array[r, c, 0] = u_np[k, 0]
                    self.u_array[r, c, 1] = u_np[k, 1]
                    self.v_array[r, c, 0] = v_np[k, 0]
                    self.v_array[r, c, 1] = v_np[k, 1]

            if bar is not None:
                bar.update(len(idxs))

        if bar is not None:
            bar.close()

    def _amplitude_mask_weight(self) -> np.ndarray:
        """Per-position weight from the fitted u/v peak amplitudes, min-max to ``[0, 1]``.

        The mean of the two fitted lattice-peak amplitudes measures the lattice signal at
        each position. It is min-max normalized to ``[0, 1]`` with no contrast windowing;
        display contrast is applied later via :meth:`StrainMap.plot_strain`'s
        ``mask_range``. Degenerate input (constant / non-finite / amplitudes unavailable)
        falls back to uniform full weight.
        """
        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]
        if self.u_peak_fit is None or self.v_peak_fit is None:
            return np.ones((scan_r, scan_c))
        signal = (self.u_peak_fit.array[:, :, 2] + self.v_peak_fit.array[:, :, 2]) / 2.0
        lo = np.nanmin(signal)
        hi = np.nanmax(signal)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return (signal - lo) / (hi - lo)
        return np.ones((scan_r, scan_c))

    def create_mask(
        self,
        use_radial_method: bool = False,
        exclusion_radius_fraction: float = 0.1,
        plot: bool = True,
        figsize: tuple[float, float] = (5, 4),
    ):
        """(Re)compute the per-position weight :attr:`mask_weight` from lattice signal.

        Usually **optional**: :meth:`fit_lattice_vectors` already populates
        :attr:`mask_weight` from the fitted peak amplitudes (mirroring the correlation
        pipeline, where ``BraggVectors.fit_lattice`` emits ``mask_weight``). Call this
        only to plot the weight map, to recompute it, or to switch to the radial
        estimator.

        Builds a ``(scan_row, scan_col)`` weight in ``[0, 1]`` measuring the lattice
        signal at each position, the analogue of :attr:`BraggVectors.mask_weight`. It is
        the default reference weighting handed to :meth:`calculate_strain_map`, so strong,
        well-fit positions dominate the reference lattice and weak/vacuum positions are
        down-weighted.

        The raw signal is min-max normalized to ``[0, 1]`` with **no** contrast windowing
        or smoothing. Display contrast is applied via :meth:`StrainMap.plot_strain`'s
        ``mask_range`` argument (e.g. ``mask_range=(0.6, 0.8)``) -- weights at/below
        ``low`` render black, at/above ``high`` render full color. (Min-max is sensitive
        to a few very bright positions, which compress the bulk toward 0; pick
        ``mask_range`` to match where the bulk actually sits -- the weight-map plot here
        shows it.)

        Parameters
        ----------
        use_radial_method : bool, default=False
            If ``True``, weight by the total diffracted intensity *outside* a central
            disk (radius ``exclusion_radius_fraction`` of the detector width), excluding
            the bright unscattered beam. Use a **small** exclusion (~0.1) so the Bragg
            disks are kept: a large exclusion keeps only the far-corner diffuse scatter,
            which is *anti*-correlated with crystallinity and inverts the contrast. If
            ``False`` (default, recommended), weight by the mean fitted peak amplitude
            from :meth:`fit_lattice_vectors` (requires it to have been run), identical to
            the weight ``fit_lattice_vectors`` stores automatically.
        exclusion_radius_fraction : float, default=0.1
            Central-disk radius (fraction of detector width) excluded by the radial
            method.
        plot : bool, default=True
            If ``True``, show the resulting per-position weight map (the analogue of the
            correlation pipeline's :meth:`BraggVectors.fit_lattice` plot).
        figsize : tuple of float, default=(5, 4)
            Figure size in inches for the weight-map plot.

        Returns
        -------
        StrainMapAutocorrelation
            ``self``, with :attr:`mask_weight` set.
        """
        if not isinstance(self.dataset, (Dataset4d, Dataset4dstem)):
            raise ValueError("Dataset must be Dataset4d or Dataset4dstem.")

        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]

        if use_radial_method:
            # center / radius are in DETECTOR coordinates (the last two axes)
            center_y = self.dataset.shape[-2] / 2.0
            center_x = self.dataset.shape[-1] / 2.0
            y, x = np.ogrid[:self.dataset.shape[-2], :self.dataset.shape[-1]]
            radius_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            exclusion_radius = exclusion_radius_fraction * self.dataset.shape[-1]
            outside_mask = radius_map > exclusion_radius
            signal = np.empty(shape=(scan_r, scan_c))
            for r in range(scan_r):
                for c in range(scan_c):
                    dp = self.dataset.array[r, c]
                    signal[r, c] = np.sum(dp[outside_mask])
            # min-max normalization to [0, 1]; constant/degenerate -> ones
            lo = np.nanmin(signal)
            hi = np.nanmax(signal)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                self.mask_weight = (signal - lo) / (hi - lo)
            else:
                self.mask_weight = np.ones((scan_r, scan_c))
        else:
            if self.u_peak_fit is None or self.v_peak_fit is None:
                raise RuntimeError("For intensity-based masking, run fit_lattice_vectors() first.")
            self.mask_weight = self._amplitude_mask_weight()

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            handle = ax.imshow(self.mask_weight, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title("Per-position lattice weight (mask_weight)")
            ax.set_xlabel("scan column")
            ax.set_ylabel("scan row")
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

        return self


    def calculate_strain_map(
        self,
        u_ref: np.ndarray | None = None,
        v_ref: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> StrainMap:
        """Build a :class:`StrainMap` from the fitted per-position lattice vectors.

        Mirrors :meth:`BraggVectors.calculate_strain_map` so the downstream strain cells
        (``plot_strain``, ``update_reference``, ``estimate_strain_precision``) are
        identical for the correlation and cepstral workflows. Because the cepstral
        vectors are real-space, ``real_space=True`` is passed and
        :func:`~quantem.diffraction.strain._strain_tensor` yields strain matching the
        correlation result on the same data.

        Parameters
        ----------
        u_ref : np.ndarray, optional
            ``(2,)`` reference for the first lattice vector. Defaults to the median over
            the scan inside :class:`StrainMap`.
        v_ref : np.ndarray, optional
            ``(2,)`` reference for the second lattice vector. Defaults to the median over
            the scan inside :class:`StrainMap`.
        mask : np.ndarray, optional
            ``(scan_row, scan_col)`` per-position weighting used when computing the
            reference lattice. Defaults to :attr:`mask_weight` from :meth:`create_mask`
            (the lattice signal strength), so strong, well-fit positions dominate the
            reference.

        Returns
        -------
        StrainMap
            A strain map initialized from the fitted lattice vectors.
        """
        if self.u_array is None or self.v_array is None:
            raise ValueError("Run fit_lattice_vectors() before calculate_strain_map().")
        if not isinstance(self.dataset, (Dataset4d, Dataset4dstem)):
            raise ValueError("Dataset must be Dataset4d or Dataset4dstem.")

        if mask is None:
            mask = self.mask_weight

        ds_sampling = float(self.dataset.sampling[0])
        ds_units = str(self.dataset.units[0])

        return StrainMap(
            u_array=self.u_array,
            v_array=self.v_array,
            ds_shape=tuple(self.dataset.shape),
            real_space=self.real_space,
            u_ref=u_ref,
            v_ref=v_ref,
            mask=mask,
            ds_sampling=ds_sampling,
            ds_units=ds_units,
            q_to_r_rotation_ccw_deg = self.metadata['q_to_r_rotation_ccw_deg'],
            q_transpose = self.metadata['q_transpose'],
        )

    def plot_lattice_vectors(
        self,
        subtract_mean: bool = True,
        max_shift: float = 1.0,
        cmap: str = "PiYG_r",
        axsize: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        **imshow_kwargs: Any,
    ):
        """Plot the four per-position lattice-vector component maps.

        Shows ``u_row``, ``u_col``, ``v_row``, ``v_col`` from :attr:`u_array`/
        :attr:`v_array` as four panels (optionally with the mean subtracted), on a
        shared symmetric color range. Positions whose vectors deviate from the
        :meth:`choose_lattice_vector` seed by more than ``max_shift`` are masked out, so
        bad fits do not blow up the color scale. A quick diagnostic of fit smoothness
        before computing strain.

        Parameters
        ----------
        subtract_mean : bool, default=True
            If ``True``, subtract the (in-range) mean of each component so deviations
            from the average lattice are shown.
        max_shift : float, default=1.0
            Maximum allowed deviation (pixels) of a vector from the seed before its
            position is masked out of the plot and the color-range statistics.
        cmap : str, default="PiYG_r"
            Diverging colormap; masked positions are drawn black.
        axsize : tuple of float, optional
            Per-panel size in inches; defaults to ``(4, 4)`` when ``figsize`` is unset.
        figsize : tuple of float, optional
            Overall figure size in inches; defaults to ``(4 * axsize[0], axsize[1])``.
        **imshow_kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        tuple
            ``(fig, ax)`` with the four-panel figure.
        """
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
    """Round ``target`` to the nearest "nice" scale-bar length (1/2/5 x 10^n)."""
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
    """Zoom ``ax`` to the central ``cropping_factor`` fraction of a ``shape`` image.

    Preserves the existing y-axis direction (inverted for image coordinates).
    """
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
    """Flatten a matplotlib axes container (array/list/tuple) to a flat list of axes."""
    if isinstance(ax, np.ndarray):
        return list(ax.ravel())
    if isinstance(ax, (list, tuple)):
        out: list[Any] = []
        for a in ax:
            out.extend(_flatten_axes(a))
        return out
    return [ax]


def _raw_vec_to_display(vec_rc: NDArray, *, rotation_ccw_deg: float, transpose: bool) -> NDArray:
    """Map a raw-detector ``(row, col)`` vector into the rotated display frame.

    Applies the optional axis transpose, then a counter-clockwise rotation of
    ``rotation_ccw_deg``. Inverse of :func:`_display_vec_to_raw`.
    """
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
    """Map a rotated-display ``(row, col)`` vector back to the raw detector frame.

    Inverse of :func:`_raw_vec_to_display`: undo the rotation, then the transpose.
    """
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
    """Draw the ``u`` (red) and ``v`` (cyan) lattice vectors from ``center_rc`` on ``ax``."""
    r0, c0 = center_rc

    def _draw(vec: NDArray, label: str, color: tuple[float, float, float]) -> None:
        dr, dc = vec[0], vec[1]
        ax.plot([c0, c0 + dc], [r0, r0 + dr], linewidth=2.75, color=color)
        ax.plot([c0 + dc], [r0 + dr], marker="o", markersize=6.0, color=color)
        ax.text(c0 + dc, r0 + dr, f" {label}", color=color, fontsize=18, va="center")

    _draw(np.asarray(u_rc, dtype=float).reshape(2), "u", (1.0, 0.0, 0.0))
    _draw(np.asarray(v_rc, dtype=float).reshape(2), "v", (0.0, 0.7, 1.0))

def _plot_peaks(ax: Any, center_rc: tuple[float, float], peaks_plot: NDArray) -> None:
    """Mark each detected peak (green dot), offset from ``center_rc``, on ``ax``."""
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
    """Overlay lattice vectors on the original (and, if present, rotated) transform axes.

    Draws ``u``/``v`` (and any ``peaks_plot``) on the first axis in raw coordinates, and
    on the second axis (if any) in the rotated display frame.
    """
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
    """Sub-pixel vertex offset (in ``[-1, 1]``) of a parabola through three samples.

    Given values at offsets ``-1, 0, +1``, returns the offset of the parabola's extremum
    from the center sample; ``0`` for a degenerate (flat/non-finite) fit.
    """
    denom = v_m1 - 2.0 * v_0 + v_p1
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    delta = 0.5 * (v_m1 - v_p1) / denom
    if not np.isfinite(delta):
        return 0.0
    return np.clip(delta, -1.0, 1.0)


def _parabolic_peak_rc_amp(im: NDArray, r_guess: float, c_guess: float) -> tuple[float, float, float]:
    """3-point parabolic sub-pixel peak near ``(r_guess, c_guess)`` and its amplitude.

    Snaps to the brightest pixel in the 3x3 window around the (absolute-coordinate)
    guess, refines row and column independently by a parabolic vertex, and returns
    ``(r_sub, c_sub, amp)`` -- the sub-pixel peak position and the image value at the
    rounded vertex. Shared by the per-position :func:`_refine_lattice_vectors` and the
    batched :meth:`StrainMapAutocorrelation._fit_lattice_vectors_batched` so they agree
    exactly.
    """
    H, W = im.shape
    r0 = int(np.clip(int(np.round(r_guess)), 0, H - 1))
    c0 = int(np.clip(int(np.round(c_guess)), 0, W - 1))
    win = im[max(0, r0 - 1) : min(H, r0 + 2), max(0, c0 - 1) : min(W, c0 + 2)]
    if win.size == 0:
        return r_guess, c_guess, 0.0

    ir, ic = np.unravel_index(np.argmax(win), win.shape)
    r_peak = max(0, r0 - 1) + ir
    c_peak = max(0, c0 - 1) + ic

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

    r_sub = r_peak + dr
    c_sub = c_peak + dc
    r_int = int(np.clip(int(np.round(r_sub)), 0, H - 1))
    c_int = int(np.clip(int(np.round(c_sub)), 0, W - 1))
    return r_sub, c_sub, float(im[r_int, c_int])


def _refine_peak_subpixel(
    im: NDArray,
    *,
    r_guess: float,
    c_guess: float,
    radius_px: float = 2.0,
) -> tuple[float, float]:
    """Refine a peak near ``(r_guess, c_guess)`` to sub-pixel ``(row, col)``.

    Finds the brightest pixel in a ``radius_px`` window, then applies independent
    parabolic vertex offsets along each axis.
    """
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
    """Refine a peak location to subpixel precision via local DFT upsampling.

    Uses a matrix-multiply DFT (the Guizar-Sicairos upsampled cross-correlation
    trick) to evaluate the image's Fourier interpolant on a fine grid in a small
    neighborhood around the initial estimate, then locates the maximum of that
    upsampled patch with a 3-point parabolic vertex refinement. This avoids
    interpolating the whole image and is accurate to roughly ``1 / upsample`` of a
    pixel.

    Parameters
    ----------
    im : NDArray
        2D real image (a transform panel) whose peak is being refined.
    r0, c0 : float
        Initial peak estimate in pixel coordinates (row, column). If ``r0`` or
        ``c0`` is a torch tensor it is converted to a Python float.
    upsample : int
        DFT upsampling factor. Values ``<= 1`` skip refinement and return the
        input estimate unchanged; larger values give finer subpixel resolution.

    Returns
    -------
    tuple[float, float]
        The refined ``(row, column)`` peak location in pixel coordinates.
    """
    if upsample <= 1:
        return r0, c0

    im = np.asarray(im, dtype=float)
    if torch.is_tensor(r0):
        r0 = float(r0.item())
    if torch.is_tensor(c0):
        c0 = float(c0.item())
    F = np.fft.fft2(np.fft.fftshift(im))

    up = upsample
    H, W = im.shape
    du = int(np.floor(np.ceil(1.5 * up) / 2.0))
    off_r = -(-H // 2)          # ceil(H/2)
    off_c = -(-W // 2)          # ceil(W/2)

    shift = (du + up * (r0 - off_r), du + up * (c0 - off_c))
    patch = np.abs(dft_upsample(F, up=up, shift=shift))
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

    dr = ((du - float(i0) - di)) / up
    dc = ((du - float(j0) - dj)) / up

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
    """Refine the two lattice vectors of a transform panel to subpixel precision.

    Starting from integer-pixel guesses for the ``u`` and ``v`` lattice vectors
    (expressed as row/column offsets from the panel center), this locates the
    corresponding autocorrelation/cepstral peaks and refines them with up to
    three successively finer stages: a 3-point parabolic vertex estimate, an
    isotropic 2D Gaussian least-squares fit, and DFT upsampling. When
    ``refine_all_peaks`` is set, all bright peaks are detected and the lattice
    basis is recovered by an intensity-weighted least-squares fit to the full
    peak lattice instead of refining only the two seed vectors.

    Parameters
    ----------
    im : NDArray
        2D real transform panel (Patterson/cepstral image) to fit.
    u_rc, v_rc : NDArray
        Length-2 initial lattice vectors as ``(row, column)`` offsets relative to
        the panel center.
    radius_px : float, optional
        Half-width (in pixels) of the fitting window used for the Gaussian fit.
        Default 2.0.
    refine_gaussian : bool, optional
        If True (default), refine each peak with an isotropic 2D Gaussian fit.
    refine_dft : bool, optional
        If True, follow the Gaussian fit with DFT-upsampled refinement
        (requires ``upsample > 1``). Default False.
    refine_all_peaks : bool, optional
        If True, detect every bright peak and solve a weighted least-squares fit
        for the lattice basis rather than refining only ``u_rc`` and ``v_rc``.
        Default False.
    peaks : NDArray or None, optional
        Precomputed peak positions (row/col offsets from center) to use when
        ``refine_all_peaks`` is set; if None they are detected automatically.
    weights : NDArray or None, optional
        Precomputed peak weights paired with ``peaks``; if None they are derived
        from peak amplitudes.
    upsample : int, optional
        DFT upsampling factor for ``refine_dft``. Default 16.
    maxfev : int, optional
        Maximum function evaluations for the Gaussian ``curve_fit``. Default 100.
    threshold_percentile : float, optional
        Fractional intensity percentile (0-1) used as the peak-detection
        threshold when auto-detecting peaks. Default 0.9975.
    min_peak_spacing : float, optional
        Minimum allowed spacing (in pixels) between detected peaks; ``0`` (default)
        disables the spacing filter.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        ``(u_result, v_result, pts, weights)``. ``u_result`` and ``v_result`` are
        length-5 arrays ``(row_offset, col_offset, amplitude, sigma, background)``
        giving the refined lattice vectors relative to the panel center. ``pts``
        and ``weights`` are the detected peak positions and their normalized
        weights when ``refine_all_peaks`` is True, otherwise both are ``None``.
        When ``refine_all_peaks`` is True the position comes from the
        intensity-weighted all-peaks least squares, but the amplitude/sigma/background
        come from the single-peak refinement at the u/v seed -- the
        background-subtracted Gaussian height, not the raw cepstral value, which rides
        the central pedestal and inverts the mask in vacuum.
    """
    from scipy.optimize import curve_fit

    im = np.asarray(im, dtype=float)
    if im.ndim != 2:
        raise ValueError("im must be 2D.")

    H, W = im.shape
    r_center = H // 2
    c_center = W // 2

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

        r_par, c_par, amp_par = _parabolic_peak_rc_amp(im, r_guess, c_guess)

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
        # Position from the weighted all-peaks lstsq; amplitude/width/background from
        # the SAME single-peak refinement at the u/v seeds as the single-peak return
        # below -- the background-subtracted Gaussian height (the crystalline order
        # parameter), not the raw cepstral value, which rides the central pedestal and
        # would invert the mask in vacuum.
        u_amp_fit = _refine_one(u_rc)
        v_amp_fit = _refine_one(v_rc)

        return np.array((u_refined[0], u_refined[1], u_amp_fit[2], u_amp_fit[3], u_amp_fit[4]), dtype=float), np.array((v_refined[0], v_refined[1], v_amp_fit[2], v_amp_fit[3], v_amp_fit[4]), dtype=float), pts, weights

    return _refine_one(u_rc), _refine_one(v_rc), None, None


def _refine_peaks_batched(
    ims: torch.Tensor,
    vec: NDArray,
    *,
    radius_px: float,
    refine_gaussian: bool,
    lm_iters: int = 50,
) -> torch.Tensor:
    """Batched, vectorized version of the single-image ``_refine_one`` peak refinement.

    Refines one lattice peak (located near ``center + vec``) for every transform in the
    stack ``ims`` of shape ``(B, H, W)`` at once. The parabolic sub-pixel step matches
    :func:`_refine_lattice_vectors`' ``_parabolic_peak_rc_amp`` exactly; when
    ``refine_gaussian`` is set, the per-position ``scipy.optimize.curve_fit`` of an
    isotropic 2D Gaussian is replaced by a batched Levenberg-Marquardt solve of the
    identical objective and bounds. On clean, well-isolated peaks the two agree to
    ~1e-6 px; on noisy, non-Gaussian peaks (e.g. a sinc-like cepstral peak on the sloped
    tail of a bright neighbor) the bounded-trf and LM optima can differ by a few 0.01 px
    -- amplitude-preserving and well below strain-relevant precision. The batched solve
    removes the dominant per-call overhead, so it runs far faster (and scales onto a GPU
    via the tensor ``device``).

    Parameters
    ----------
    ims : torch.Tensor
        Stack of transformed (fftshifted ``|FFT|``) images, shape ``(B, H, W)``, float.
    vec : NDArray
        Lattice vector ``(row, col)`` relative to the panel center; the peak is sought
        near ``(H // 2 + row, W // 2 + col)``.
    radius_px : float
        Half-width in pixels of the Gaussian-fit window.
    refine_gaussian : bool
        If ``True``, refine with the batched isotropic-Gaussian LM fit; otherwise return
        the parabolic estimate only (``sigma`` and ``background`` set to 0).
    lm_iters : int, default=50
        Number of Levenberg-Marquardt iterations.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 5)``: ``(row_offset, col_offset, amplitude, sigma, background)`` with
        ``row_offset``/``col_offset`` measured relative to the panel center (matching
        :func:`_refine_lattice_vectors`).
    """
    Bn, Hn, Wn = ims.shape
    dev, dt = ims.device, ims.dtype
    rcent, ccent = Hn // 2, Wn // 2
    bidx = torch.arange(Bn, device=dev)

    # Clamp the seed into the interior so the 3x3 window below is always in-bounds
    # (mirrors the clamping in _parabolic_peak_rc_amp; matters only for edge peaks,
    # which are interior in normal use but must not crash the batched solve).
    r0 = int(np.clip(round(rcent + float(vec[0])), 1, Hn - 2))
    c0 = int(np.clip(round(ccent + float(vec[1])), 1, Wn - 2))

    # --- 3x3 argmax around the seed (matches _parabolic_peak_rc_amp) ---
    win3 = ims[:, r0 - 1 : r0 + 2, c0 - 1 : c0 + 2].reshape(Bn, -1)
    am = win3.argmax(1)
    r_peak = r0 - 1 + torch.div(am, 3, rounding_mode="floor")
    c_peak = c0 - 1 + (am % 3)

    def _parab(vm1: torch.Tensor, v0_: torch.Tensor, vp1: torch.Tensor) -> torch.Tensor:
        denom = vm1 - 2.0 * v0_ + vp1
        delta = 0.5 * (vm1 - vp1) / denom
        delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        delta = torch.where(denom == 0, torch.zeros_like(delta), delta)
        return delta.clamp(-1.0, 1.0)

    v0_ = ims[bidx, r_peak, c_peak]
    dr = _parab(ims[bidx, (r_peak - 1).clamp(0, Hn - 1), c_peak], v0_, ims[bidx, (r_peak + 1).clamp(0, Hn - 1), c_peak])
    dr = torch.where((r_peak > 0) & (r_peak < Hn - 1), dr, torch.zeros_like(dr))
    dc = _parab(ims[bidx, r_peak, (c_peak - 1).clamp(0, Wn - 1)], v0_, ims[bidx, r_peak, (c_peak + 1).clamp(0, Wn - 1)])
    dc = torch.where((c_peak > 0) & (c_peak < Wn - 1), dc, torch.zeros_like(dc))
    r_sub = r_peak.to(dt) + dr
    c_sub = c_peak.to(dt) + dc

    if not refine_gaussian:
        ri = r_sub.round().long().clamp(0, Hn - 1)
        ci = c_sub.round().long().clamp(0, Wn - 1)
        amp = ims[bidx, ri, ci]
        zeros = torch.zeros(Bn, device=dev, dtype=dt)
        return torch.stack([r_sub - rcent, c_sub - ccent, amp, zeros, zeros], 1)

    # --- isotropic Gaussian fit over a (2*rad+1)^2 window, batched LM ---
    rad = int(max(1, np.ceil(radius_px)))
    P = 2 * rad + 1
    r0i = r_sub.round().long()
    c0i = c_sub.round().long()
    off = torch.arange(-rad, rad + 1, device=dev)
    rows_idx = (r0i[:, None, None] + off[None, :, None]).expand(Bn, P, P)
    cols_idx = (c0i[:, None, None] + off[None, None, :]).expand(Bn, P, P)
    bexp = bidx[:, None, None].expand(Bn, P, P)
    win = ims[bexp, rows_idx.clamp(0, Hn - 1), cols_idx.clamp(0, Wn - 1)]
    RR = rows_idx.to(dt).reshape(Bn, -1)
    CC = cols_idx.to(dt).reshape(Bn, -1)
    y = win.reshape(Bn, -1)

    am2 = y.argmax(1)
    r_pk2 = (r0i - rad + torch.div(am2, P, rounding_mode="floor")).to(dt)
    c_pk2 = (c0i - rad + (am2 % P)).to(dt)
    bg0 = y.median(1).values
    amp0 = (y.max(1).values - bg0).clamp(min=0.0)
    sig0 = torch.full((Bn,), max(0.75, radius_px / 2.0), device=dev, dtype=dt)
    p = torch.stack([r_pk2, c_pk2, amp0, sig0, bg0], 1)

    rlo = (r0i - rad).to(dt) - 0.5
    rhi = (r0i + rad).to(dt) + 0.5
    clo = (c0i - rad).to(dt) - 0.5
    chi = (c0i + rad).to(dt) + 0.5
    sig_hi = radius_px * 4.0

    lam = torch.full((Bn,), 1e-2, device=dev, dtype=dt)
    eye5 = torch.eye(5, device=dev, dtype=dt)[None]

    def _sse(p_: torch.Tensor) -> torch.Tensor:
        row, col, amp, sig, bg = [p_[:, i : i + 1] for i in range(5)]
        sig = sig.clamp(min=1e-9)
        E = torch.exp(-((RR - row) ** 2 + (CC - col) ** 2) / (2.0 * sig * sig))
        return ((bg + amp * E - y) ** 2).sum(1)

    for _ in range(lm_iters):
        row, col, amp, sig, bg = [p[:, i : i + 1] for i in range(5)]
        sig = sig.clamp(min=1e-9)
        d_row = RR - row
        d_col = CC - col
        E = torch.exp(-(d_row * d_row + d_col * d_col) / (2.0 * sig * sig))
        res = (bg + amp * E) - y
        g_row = amp * E * d_row / (sig * sig)
        g_col = amp * E * d_col / (sig * sig)
        g_amp = E
        g_sig = amp * E * ((d_row * d_row + d_col * d_col) / (sig**3))
        g_bg = torch.ones_like(E)
        J = torch.stack([g_row, g_col, g_amp, g_sig, g_bg], 2)  # (B, N, 5)
        JT = J.transpose(1, 2)
        JTJ = JT @ J
        JTr = JT @ res[..., None]
        diag = torch.diagonal(JTJ, dim1=1, dim2=2).clamp(min=1e-12)
        A = JTJ + lam[:, None, None] * eye5 * diag[:, None, :]
        delta = torch.linalg.solve(A, -JTr)[..., 0]
        pn = p + delta
        pn[:, 0] = pn[:, 0].clamp(rlo, rhi)
        pn[:, 1] = pn[:, 1].clamp(clo, chi)
        pn[:, 2] = pn[:, 2].clamp(min=0.0)
        pn[:, 3] = pn[:, 3].clamp(0.25, sig_hi)
        s_new = _sse(pn)
        s_old = res.pow(2).sum(1)
        better = s_new < s_old
        p = torch.where(better[:, None], pn, p)
        lam = torch.where(better, (lam * 0.5).clamp(min=1e-9), (lam * 3.0).clamp(max=1e6))

    row, col, amp, sig, bg = [p[:, i] for i in range(5)]
    ok = torch.isfinite(p).all(1)
    row = torch.where(ok, row, r_sub)
    col = torch.where(ok, col, c_sub)
    return torch.stack([row - rcent, col - ccent, amp, sig, bg], 1)