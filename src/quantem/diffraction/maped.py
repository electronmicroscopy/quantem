from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndi_shift
from scipy.signal import convolve2d
from scipy.signal.windows import tukey
from tqdm import tqdm

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.imaging_utils import (
    cross_correlation_shift_torch,
    weighted_cross_correlation_shift,
)
from quantem.core.visualization import show_2d


class MAPED(AutoSerialize):
    """
    Merge-Averaged Precession Electron Diffraction (MAPED) helper.

    This class manages a set of 4D-STEM datasets and provides utilities to:
    - compute mean BF and mean DP summaries,
    - choose/find diffraction origins,
    - align diffraction space and real space,
    - merge datasets into a single composite Dataset4dstem.
    """

    _token = object()

    def __init__(self, datasets: list[Dataset4dstem], _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError("Use MAPED.from_datasets() to instantiate this class.")
        super().__init__()
        self.datasets = datasets
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_datasets(cls, datasets: Sequence[Dataset4dstem]) -> MAPED:
        """
        Construct a MAPED instance from a non-empty sequence of Dataset4dstem.

        Parameters
        ----------
        datasets
            Sequence of Dataset4dstem instances.

        Returns
        -------
        MAPED
            New MAPED instance.
        """
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
            raise TypeError("MAPED.from_datasets expects a sequence of Dataset4dstem instances.")
        ds_list: list[Dataset4dstem] = []
        for d in datasets:
            if not isinstance(d, Dataset4dstem):
                raise TypeError(
                    "MAPED.from_datasets expects a sequence of Dataset4dstem instances."
                )
            ds_list.append(d)
        if not ds_list:
            raise ValueError(
                "MAPED.from_datasets expects a non-empty sequence of Dataset4dstem instances."
            )
        return cls(datasets=ds_list, _token=cls._token)

    def preprocess(
        self,
        plot_summary: bool = True,
        scale: float | Sequence[float] | None = None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Compute dataset summary images.

        Parameters
        ----------
        plot_summary : bool, optional
            If True, display summary plots (default True).
        scale : float or sequence of float or None, optional
            Per-dataset scaling factor(s) (default None).

        Attributes
        ----------
        scales : np.ndarray
            Per-dataset scaling factors (n,).
        dp_mean : list[np.ndarray]
            Mean diffraction patterns (H, W), one per dataset.
        im_bf : list[np.ndarray]
            Mean bright-field images (R, C), one per dataset.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)
        if scale is None:
            self.scales = np.ones(n, dtype=float)
        elif isinstance(scale, (int, float, np.floating)):
            self.scales = np.full(n, float(scale), dtype=float)
        else:
            self.scales = np.asarray(list(scale), dtype=float)
            if self.scales.shape != (n,):
                raise ValueError(
                    "scale must be a scalar or a sequence with the same length as datasets."
                )
        if np.any(self.scales == 0):
            raise ValueError("scale entries must be nonzero.")

        self.dp_mean: list[np.ndarray] = []
        self.im_bf: list[np.ndarray] = []

        for d in self.datasets:
            if hasattr(d, "get_dp_mean"):
                try:
                    d.get_dp_mean()
                except TypeError:
                    try:
                        d.get_dp_mean(returnval=False)
                    except Exception:
                        pass

            dp = getattr(d, "dp_mean", None)
            if dp is None:
                arr = np.asarray(d.array)
                dp_arr = np.mean(arr, axis=(0, 1))
            else:
                dp_arr = np.asarray(dp.array if hasattr(dp, "array") else dp)

            arr = np.asarray(d.array)
            im_bf_arr = np.mean(arr, axis=(2, 3))

            self.dp_mean.append(np.asarray(dp_arr))
            self.im_bf.append(np.asarray(im_bf_arr))

        if plot_summary:
            tiles = [[(self.im_bf[i] / self.scales[i]), self.dp_mean[i]] for i in range(n)]
            titles = [
                [f"{i} - Mean Bright Field", f"{i} - Mean Diffraction Pattern"] for i in range(n)
            ]
            show_2d(tiles, title=titles, **plot_kwargs)

        return self

    def diffraction_origin(
        self,
        origins=None,
        sigma=None,
        plot_origins: bool = True,
        plot_indices=None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Choose or automatically find the origin in diffraction space.

        Parameters
        ----------
        origins : tuple or sequence, optional
            Optional manual origins. Can be:
            - a single (row, col) tuple, applied to all datasets
            - a list of (row, col) tuples of length n (one per dataset)
        sigma : float, optional
            Optional low-pass smoothing sigma (pixels) applied to each mean DP prior to peak finding.
        plot_origins : bool, optional
            If True, plot mean diffraction patterns with overlaid origin markers.
        plot_indices : sequence of int, optional
            Optional indices to plot. If None, plots all datasets.
        **plot_kwargs
            Passed to show_2d.

        Attributes
        ----------
        diffraction_origins : np.ndarray
            Array of shape (n, 2) with integer (row, col) origins.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")

        if plot_indices is None:
            plot_indices_list = list(range(n))
        else:
            plot_indices_list = list(plot_indices)
            for i in plot_indices_list:
                if i < 0 or i >= n:
                    raise IndexError("plot_indices contains an out-of-range index.")

        if origins is None:
            origins_arr = np.zeros((n, 2), dtype=int)
            for i in range(n):
                dp = np.asarray(self.dp_mean[i])
                if sigma is not None and float(sigma) > 0:
                    dp_use = gaussian_filter(
                        dp.astype(float, copy=False), float(sigma), mode="nearest"
                    )
                else:
                    dp_use = dp
                r, c = np.unravel_index(int(np.argmax(dp_use)), dp_use.shape)
                origins_arr[i, 0] = int(r)
                origins_arr[i, 1] = int(c)
        else:
            if isinstance(origins, tuple) and len(origins) == 2:
                origins_arr = np.tile(np.asarray(origins, dtype=int)[None, :], (n, 1))
            else:
                origins_list = list(origins)
                if len(origins_list) != n:
                    raise ValueError(
                        "origins must be a single (row,col) tuple or a list of length n."
                    )
                origins_arr = np.asarray(origins_list, dtype=int)
                if origins_arr.shape != (n, 2):
                    raise ValueError("origins must have shape (n, 2) after conversion.")

        self.diffraction_origins = origins_arr

        if plot_origins:
            arrays = [np.asarray(self.dp_mean[i]) for i in plot_indices_list]
            titles = [f"{i} - Mean Diffraction Pattern" for i in plot_indices_list]
            fig, ax = show_2d(arrays, title=titles, returnfig=True, **plot_kwargs)
            axs = np.ravel(np.asarray(ax, dtype=object))
            for j, i in enumerate(plot_indices_list):
                r, c = self.diffraction_origins[i]
                axs[j].plot([c], [r], marker="+", color="red", markersize=16, markeredgewidth=2)

        return self

    def diffraction_align(
        self,
        edge_blend: float = 16.0,
        padding=None,
        pad_val: str | float = "min",
        upsample_factor: int = 100,
        weight_scale: float = 1 / 8,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align mean diffraction patterns using weighted cross-correlation in Fourier space.

        Parameters
        ----------
        edge_blend : float
            Tukey window edge taper (pixels).
        padding : int or None
            Passed to shift_images for plotting.
        pad_val : str or float
            Passed to shift_images for plotting.
        upsample_factor : int
            Subpixel upsampling factor for correlation peak estimation.
        weight_scale : float
            Radial weight falloff scale (fraction of mean DP size).
        plot_aligned : bool
            If True, plot aligned mean diffraction patterns.
        **plot_kwargs
            Passed to show_2d when plotting.

        Attributes
        ----------
        diffraction_shifts : np.ndarray
            Array of shape (n, 2) with (row, col) shifts to align diffraction patterns.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")
        if not hasattr(self, "diffraction_origins"):
            raise RuntimeError(
                "Run diffraction_origin() first so self.diffraction_origins exists."
            )

        H, W = np.asarray(self.dp_mean[0]).shape

        w = (
            tukey(H, alpha=2.0 * float(edge_blend) / float(H))[:, None]
            * tukey(W, alpha=2.0 * float(edge_blend) / float(W))[None, :]
        )

        r = np.fft.fftfreq(H, 1.0 / float(H))[:, None]
        c = np.fft.fftfreq(W, 1.0 / float(W))[None, :]

        n = len(self.dp_mean)
        self.diffraction_shifts = np.zeros((n, 2), dtype=float)

        G_ref = np.fft.fft2(w * np.asarray(self.dp_mean[0]))
        xy0 = np.asarray(self.diffraction_origins[0], dtype=float)

        for ind in range(1, n):
            G = np.fft.fft2(w * np.asarray(self.dp_mean[ind]))
            xy = np.asarray(self.diffraction_origins[ind], dtype=float)

            dr2 = (r - xy0[0] + xy[0]) ** 2 + (c - xy0[1] + xy[1]) ** 2
            im_weight = np.clip(
                1.0 - np.sqrt(dr2) / float(np.mean((H, W))) / float(weight_scale),
                0.0,
                1.0,
            )
            im_weight = np.sin(im_weight * np.pi / 2.0) ** 2

            shift_rc, G_shift = weighted_cross_correlation_shift(
                im_ref=G_ref,
                im=G,
                weight_real=im_weight * 0.0 + 1.0,
                upsample_factor=int(upsample_factor),
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )
            self.diffraction_shifts[ind, :] = np.asarray(shift_rc, dtype=float)

            G_ref = G_ref * (ind / (ind + 1)) + G_shift / (ind + 1)

        self.diffraction_shifts -= np.mean(self.diffraction_shifts, axis=0)[None, :]

        if plot_aligned:
            im_aligned = shift_images(
                images=self.dp_mean,
                shifts_rc=self.diffraction_shifts,
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
            )
            show_2d(im_aligned, **plot_kwargs)

        return self

    def real_space_align(  # torch.grid_sample
        self,
        num_images=None,
        num_iter: int = 3,
        edge_blend: float = 1.0,
        padding=None,
        pad_val: str | float = "median",
        upsample_factor: int = 100,
        max_shift=None,
        shift_method: str = "bilinear",
        edge_filter: bool = True,
        edge_sigma: float = 2.0,
        hanning_filter: bool = False,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align real-space mean BF images using iterative average-reference correlation.

        Parameters
        ----------
        num_images : int, optional
            If provided, align only the first num_images images.
        num_iter : int
            Number of refinement iterations.
        edge_blend : float
            Used to set default correlation padding when max_shift is None.
        padding : int or None
            Passed to shift_images for plotting.
        pad_val : str or float
            Passed to shift_images for plotting.
        upsample_factor : int
            Subpixel upsampling factor for correlation peak estimation.
        max_shift : float, optional
            Optional maximum shift constraint passed to weighted_cross_correlation_shift.
        shift_method : str
            Passed to shift_images for plotting ('bilinear' or 'fourier').
        edge_filter : bool
            If True, correlate on gradient magnitude instead of raw intensity.
        edge_sigma : float
            Gaussian sigma applied to gradients when edge_filter is True.
        hanning_filter : bool
            If True, apply a Hanning window prior to FFT.
        plot_aligned : bool
            If True, plot aligned mean BF images.
        **plot_kwargs
            Passed to show_2d when plotting.

        Attributes
        ----------
        real_space_shifts : np.ndarray
            Array of shape (n_total, 2) with (row, col) shifts for aligned datasets.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        if not hasattr(self, "im_bf"):
            raise RuntimeError("Run preprocess() first so self.im_bf exists.")
        if len(self.im_bf) == 0:
            raise RuntimeError("No images found in self.im_bf.")

        H, W = self.im_bf[0].shape
        for im in self.im_bf:
            if im.shape != (H, W):
                raise ValueError("all self.im_bf images must have the same shape")

        n_total = len(self.im_bf)
        if num_images is None:
            n = n_total
        else:
            n = int(num_images)
            if n <= 0:
                raise ValueError("num_images must be positive")
            n = min(n, n_total)

        if int(num_iter) < 1:
            raise ValueError("num_iter must be >= 1")

        if max_shift is not None:
            pad_cc = int(np.ceil(float(max_shift))) + 4
        else:
            pad_cc = int(np.ceil(float(edge_blend))) + 4

        Hp = H + 2 * pad_cc
        Wp = W + 2 * pad_cc
        r0 = pad_cc
        c0 = pad_cc

        w_h = np.ones((H, W), dtype=float)
        if hanning_filter:
            w_h = np.hanning(H)[:, None] * np.hanning(W)[None, :]
        w_h_pad = np.zeros((Hp, Wp), dtype=float)
        w_h_pad[r0 : r0 + H, c0 : c0 + W] = w_h
        w_h_sum = float(np.sum(w_h_pad))
        if w_h_sum <= 0:
            raise RuntimeError("hanning window sum is zero")

        if edge_filter:
            wx = np.array(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=float,
            )
        else:
            wx = None

        base_pad = np.zeros((n, Hp, Wp), dtype=float)
        for i in range(n):
            im0 = np.asarray(self.im_bf[i], dtype=float)

            if edge_filter:
                gx = convolve2d(im0, wx, mode="same", boundary="symm")
                gy = convolve2d(im0, wx.T, mode="same", boundary="symm")
                gx = gaussian_filter(gx, float(edge_sigma), mode="nearest")
                gy = gaussian_filter(gy, float(edge_sigma), mode="nearest")
                im_use = np.sqrt(gx * gx + gy * gy)
            else:
                im_use = im0

            base_pad[i, r0 : r0 + H, c0 : c0 + W] = im_use

        shifts = np.zeros((n, 2), dtype=float)

        for _ in range(int(num_iter)):
            G_list = np.empty((n, Hp, Wp), dtype=np.complex128)

            for i in range(n):
                im_a = ndi_shift(
                    base_pad[i],
                    shift=(shifts[i, 0], shifts[i, 1]),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                im_mean = float(np.sum(im_a * w_h_pad) / w_h_sum)
                im_win = (im_a - im_mean) * w_h_pad
                G_list[i] = np.fft.fft2(im_win)

            G_ref = np.mean(G_list, axis=0)

            for i in range(1, n):
                drc = weighted_cross_correlation_shift(
                    im_ref=G_ref,
                    im=G_list[i],
                    weight_real=None,
                    upsample_factor=int(upsample_factor),
                    max_shift=max_shift,
                    fft_input=True,
                    fft_output=False,
                    return_shifted_image=False,
                )
                shifts[i, 0] += float(drc[0])
                shifts[i, 1] += float(drc[1])

            shifts -= shifts[0][None, :]

        shifts -= np.mean(shifts, axis=0)[None, :]

        self.real_space_shifts = np.zeros((n_total, 2), dtype=float)
        self.real_space_shifts[:n, :] = shifts

        if plot_aligned:
            im_aligned = shift_images(
                images=self.im_bf[:n],
                shifts_rc=self.real_space_shifts[:n, :],
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
                shift_method=shift_method,
            )
            show_2d(im_aligned, **plot_kwargs)

        return self

    def merge_datasets(
        self,
        real_space_padding=0,
        real_space_edge_blend=1.0,
        diffraction_padding=0,
        diffraction_edge_blend=0.0,
        diffraction_pad_val="min",
        shift_method: str = "bilinear",
        dtype=None,
        scale_output: bool = False,
        plot_result: bool = True,
        **plot_kwargs: Any,
    ) -> Dataset4dstem:
        """
        Merge aligned datasets into a single Dataset4dstem.

        Notes
        -----
        Requires the following attributes to be present on ``self``:

        self.real_space_shifts
            From ``real_space_align()``.
        self.diffraction_shifts
            From ``diffraction_align()``.

        Parameters
        ----------
        real_space_padding
            Output scan padding in pixels (adds border to scan grid).
        real_space_edge_blend
            Tukey taper width for scan-space interpolation weights.
        diffraction_padding
            Output diffraction padding in pixels (adds border around DPs).
        diffraction_edge_blend
            Tukey taper width for diffraction-space weights.
        diffraction_pad_val
            Pad value for diffraction padding ('min','max','mean','median' or float).
        shift_method
            Diffraction shift method: 'bilinear' or 'fourier'.
        dtype
            Output dtype. If None, uses parent dtype.
        scale_output
            If True and dtype is integer, scale to full dynamic range using global max.
        plot_result
            If True, plot merged BF and merged mean DP.
        **plot_kwargs
            Passed to show_2d.

        Returns
        -------
        Dataset4dstem
            Merged dataset.
        """
        if not hasattr(self, "real_space_shifts"):
            raise RuntimeError("Run real_space_align() first so self.real_space_shifts exists.")
        if not hasattr(self, "diffraction_shifts"):
            raise RuntimeError("Run diffraction_align() first so self.diffraction_shifts exists.")

        arrays = [np.asarray(d.array) for d in self.datasets]
        n = len(arrays)
        if n == 0:
            raise RuntimeError("No datasets found in self.datasets.")

        Rs, Cs, H, W = arrays[0].shape
        for a in arrays:
            if a.shape != (Rs, Cs, H, W):
                raise ValueError("All dataset arrays must have the same shape (Rs, Cs, H, W).")

        rs_shifts = np.asarray(self.real_space_shifts, dtype=float)
        dp_shifts = np.asarray(self.diffraction_shifts, dtype=float)
        if rs_shifts.shape != (n, 2):
            raise ValueError("self.real_space_shifts must have shape (n, 2).")
        if dp_shifts.shape != (n, 2):
            raise ValueError("self.diffraction_shifts must have shape (n, 2).")

        if dtype is None:
            dtype_out = np.asarray(arrays[0]).dtype
            warnings.warn(f"dtype=None; using parent dtype {dtype_out}.", RuntimeWarning)
        else:
            dtype_out = np.dtype(dtype)

        real_space_padding = int(real_space_padding)
        diffraction_padding = int(diffraction_padding)

        Rout = Rs + 2 * real_space_padding
        Cout = Cs + 2 * real_space_padding

        Hp = H + 2 * diffraction_padding
        Wp = W + 2 * diffraction_padding
        rp0 = diffraction_padding
        cp0 = diffraction_padding

        method = str(shift_method).strip().lower()
        if method not in {"bilinear", "fourier"}:
            raise ValueError("shift_method must be 'bilinear' or 'fourier'.")

        if real_space_edge_blend and float(real_space_edge_blend) > 0:
            alpha_r = min(1.0, 2.0 * float(real_space_edge_blend) / float(Rs))
            alpha_c = min(1.0, 2.0 * float(real_space_edge_blend) / float(Cs))
            w_rs = tukey(Rs, alpha=alpha_r)[:, None] * tukey(Cs, alpha=alpha_c)[None, :]
        else:
            w_rs = np.ones((Rs, Cs), dtype=float)
        w_rs = w_rs.astype(float, copy=False)

        if diffraction_edge_blend and float(diffraction_edge_blend) > 0:
            alpha_dr = min(1.0, 2.0 * float(diffraction_edge_blend) / float(H))
            alpha_dc = min(1.0, 2.0 * float(diffraction_edge_blend) / float(W))
            w_dp = tukey(H, alpha=alpha_dr)[:, None] * tukey(W, alpha=alpha_dc)[None, :]
        else:
            w_dp = np.ones((H, W), dtype=float)
        w_dp = w_dp.astype(float, copy=False)

        dp_means = [np.mean(a, axis=(0, 1), dtype=np.float64) for a in arrays]
        v = np.stack(dp_means, axis=0).reshape(-1)

        if isinstance(diffraction_pad_val, str):
            s = diffraction_pad_val.strip().lower()
            if s == "min":
                pad_val_dp = float(np.min(v))
            elif s == "max":
                pad_val_dp = float(np.max(v))
            elif s == "mean":
                pad_val_dp = float(np.mean(v))
            elif s == "median":
                pad_val_dp = float(np.median(v))
            else:
                raise ValueError(
                    "diffraction_pad_val must be a float or one of {'min','max','mean','median'}."
                )
        else:
            pad_val_dp = float(diffraction_pad_val)

        wdp_pad = np.zeros((Hp, Wp), dtype=float)
        wdp_pad[rp0 : rp0 + H, cp0 : cp0 + W] = w_dp

        wdp_shifted = np.zeros((n, Hp, Wp), dtype=float)
        if method == "fourier":
            kr = np.fft.fftfreq(Hp)[:, None]
            kc = np.fft.fftfreq(Wp)[None, :]
            Fw = np.fft.fft2(wdp_pad)
            ramps: list[np.ndarray] = []
            for i in range(n):
                dr, dc = dp_shifts[i, 0], dp_shifts[i, 1]
                ramp = np.exp(-2j * np.pi * (kr * dr + kc * dc))
                ramps.append(ramp)
                w_i = np.fft.ifft2(Fw * ramp).real
                wdp_shifted[i] = np.clip(w_i, 0.0, 1.0)
        else:
            for i in range(n):
                w_i = ndi_shift(
                    wdp_pad,
                    shift=(dp_shifts[i, 0], dp_shifts[i, 1]),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                wdp_shifted[i] = np.clip(w_i, 0.0, 1.0)
            ramps = []

        coverage = np.clip(np.sum(wdp_shifted, axis=0), 0.0, 1.0)
        edge_w_dp = 1.0 - coverage

        merged = np.zeros((Rout, Cout, Hp, Wp), dtype=np.float64)

        dp_local = np.zeros((H, W), dtype=np.float64)
        dp_pad = np.zeros((Hp, Wp), dtype=np.float64)
        dp_shifted_tmp = np.zeros((Hp, Wp), dtype=np.float64)
        num_tmp = np.zeros((Hp, Wp), dtype=np.float64)
        den_tmp = np.zeros((Hp, Wp), dtype=np.float64)

        for ro in tqdm(range(Rout), desc="Merging (rows)"):
            r_base = ro - real_space_padding
            for co in range(Cout):
                c_base = co - real_space_padding

                num_tmp.fill(0.0)
                den_tmp.fill(0.0)
                max_wi = 0.0

                for i in range(n):
                    r_in = r_base - rs_shifts[i, 0]
                    c_in = c_base - rs_shifts[i, 1]

                    r0 = int(np.floor(r_in))
                    c0 = int(np.floor(c_in))
                    if r0 < 0 or r0 >= Rs - 1 or c0 < 0 or c0 >= Cs - 1:
                        continue

                    dr = r_in - r0
                    dc = c_in - c0

                    w00 = (1.0 - dr) * (1.0 - dc)
                    w10 = dr * (1.0 - dc)
                    w01 = (1.0 - dr) * dc
                    w11 = dr * dc

                    wi = (
                        w00 * w_rs[r0, c0]
                        + w10 * w_rs[r0 + 1, c0]
                        + w01 * w_rs[r0, c0 + 1]
                        + w11 * w_rs[r0 + 1, c0 + 1]
                    )
                    if wi <= 0.0:
                        continue
                    if wi > max_wi:
                        max_wi = wi

                    a = arrays[i]
                    dp_local[:] = (
                        w00 * a[r0, c0]
                        + w10 * a[r0 + 1, c0]
                        + w01 * a[r0, c0 + 1]
                        + w11 * a[r0 + 1, c0 + 1]
                    )

                    dp_pad.fill(0.0)
                    dp_pad[rp0 : rp0 + H, cp0 : cp0 + W] = dp_local * w_dp

                    if method == "fourier":
                        ramp = ramps[i]
                        dp_shifted_tmp[:] = np.fft.ifft2(np.fft.fft2(dp_pad) * ramp).real
                    else:
                        dp_shifted_tmp[:] = ndi_shift(
                            dp_pad,
                            shift=(dp_shifts[i, 0], dp_shifts[i, 1]),
                            order=1,
                            mode="constant",
                            cval=0.0,
                            prefilter=False,
                        )

                    num_tmp += wi * dp_shifted_tmp
                    den_tmp += wi * wdp_shifted[i]

                if max_wi <= 0.0:
                    merged[ro, co] = 0.0
                    continue

                num = num_tmp + edge_w_dp * pad_val_dp
                den = den_tmp + edge_w_dp

                out = np.empty_like(num)
                np.divide(num, den, out=out, where=den != 0.0)
                out[den == 0.0] = 0.0
                merged[ro, co] = out

        self.im_bf_merged = np.mean(merged, axis=(2, 3), dtype=np.float64)
        self.dp_mean_merged = np.mean(merged, axis=(0, 1), dtype=np.float64)

        if np.issubdtype(dtype_out, np.integer):
            info = np.iinfo(dtype_out)
            dmin = float(info.min)
            dmax = float(info.max)

            merged_f = merged

            if scale_output:
                peak = float(np.max(merged_f))
                if peak <= 0.0:
                    merged_scaled = merged_f
                else:
                    merged_scaled = merged_f * (dmax / peak)

                if np.issubdtype(dtype_out, np.unsignedinteger):
                    lo, hi = 0.0, dmax
                else:
                    lo, hi = dmin, dmax

                merged_out = np.rint(np.clip(merged_scaled, lo, hi)).astype(dtype_out)
            else:
                below = float(np.min(merged_f))
                above = float(np.max(merged_f))
                if below < dmin or above > dmax:
                    warnings.warn(
                        f"Output overflow for dtype {dtype_out}: data range [{below}, {above}] exceeds "
                        f"[{dmin}, {dmax}]. Values will be clipped.",
                        RuntimeWarning,
                    )
                merged_out = np.rint(np.clip(merged_f, dmin, dmax)).astype(dtype_out)
        else:
            merged_out = merged.astype(dtype_out, copy=False)

        dataset_merged = Dataset4dstem.from_array(array=merged_out)
        dataset_merged.im_bf_merged = self.im_bf_merged
        dataset_merged.dp_mean_merged = self.dp_mean_merged

        if plot_result:
            show_2d(
                [[self.im_bf_merged, self.dp_mean_merged]],
                title=[["Merged Bright Field", "Merged Mean Diffraction Pattern"]],
                **plot_kwargs,
            )

        return dataset_merged


class MAPEDTorch(AutoSerialize):
    """
    Merge-Averaged Precession Electron Diffraction (MAPED) helper coded in PyTorch.

    This class manages a set of 4D-STEM datasets and provides utilities to:
    - compute mean BF and mean DP summaries,
    - choose/find diffraction origins,
    - align diffraction space and real space,
    - merge datasets into a single composite Dataset4dstem.
    """

    _token = object()

    def __init__(
        self,
        datasets: list[torch.Tensor],
        device: str | Any,
        dtype: str | Any,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use MAPED.from_datasets() to instantiate this class.")
        super().__init__()
        self.datasets = datasets
        self.metadata: dict[str, Any] = {}
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_datasets(cls, datasets: Sequence[torch.Tensor]) -> MAPED:
        """
        Construct a MAPED instance from a non-empty sequence of Dataset4dstem.

        Parameters
        ----------
        datasets
            Sequence of Dataset4dstem instances.

        Returns
        -------
        MAPED
            New MAPED instance.
        """
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
            raise TypeError("MAPED.from_datasets expects a sequence of Torch tensor instances.")
        ds_list: list[torch.Tensor] = []
        for d in datasets:
            if not isinstance(d, torch.Tensor):
                raise TypeError(
                    "MAPED.from_datasets expects a sequence of Torch tensor instances."
                )
            ds_list.append(d)

        dtypes = np.array([dataset.dtype for dataset in datasets])
        devices = np.array([dataset.device for dataset in datasets])

        # check that all datasets have the same dtype and device
        if not np.all(dtypes == dtypes[0]):
            raise TypeError("All datasets need to have the same type")
        if not np.all(devices == devices[0]):
            raise TypeError("All datasets need to have the same device")

        if not ds_list:
            raise ValueError(
                "MAPED.from_datasets expects a non-empty sequence of Torch tensor instances."
            )
        return cls(datasets=ds_list, _token=cls._token, device=devices[0], dtype=dtypes[0])

    def preprocess(
        self,
        plot_summary: bool = True,
        scale: float | Sequence[float] | None = None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Compute dataset summary images.

        Parameters
        ----------
        plot_summary : bool, optional
            If True, display summary plots (default True).
        scale : float or sequence of float or None, optional
            Per-dataset scaling factor(s) (default None).

        Attributes
        ----------
        scales : torch.tensor
            Per-dataset scaling factors (n,).
        dp_mean : list[torch.tensor]
            Mean diffraction patterns (H, W), one per dataset.
        im_bf : list[torch.tensor]
            Mean bright-field images (R, C), one per dataset.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)

        if scale is None:
            self.scales = torch.ones(n, dtype=self.dtype, device=self.device)
        elif isinstance(scale, (int, float, np.floating)):
            self.scales = torch.full(n, float(scale), dtype=float)
        else:
            self.scales = torch.tensor(scale, dtype=self.dtype, device=self.device)
            if self.scales.dim != (n,):
                raise ValueError(
                    "scale must be a scalar or a sequence with the same length as datasets."
                )
        if torch.any(self.scales == 0):
            raise ValueError("scale entries must be nonzero.")

        self.dp_mean: list[torch.Tensor] = []
        self.im_bf: list[torch.Tensor] = []

        for d in self.datasets:
            dp_arr = torch.mean(d, dim=(0, 1))
            im_bf_arr = torch.mean(d, dim=(2, 3))

            self.dp_mean.append(dp_arr)
            self.im_bf.append(im_bf_arr)

        if plot_summary:
            tiles = [[(self.im_bf[i] / self.scales[i]), self.dp_mean[i]] for i in range(n)]
            titles = [
                [f"{i} - Mean Bright Field", f"{i} - Mean Diffraction Pattern"] for i in range(n)
            ]
            show_2d(tiles, title=titles, **plot_kwargs)

        return self

    def diffraction_origin(
        self,
        origins: tuple | list | None = None,
        sigma: float | None = None,
        plot_origins: bool = True,
        plot_indices: list | None = None,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Choose or automatically find the origin in diffraction space.

        Parameters
        ----------
        origins : tuple or list, optional
            Optional manual origins. Can be:
            - a single (row, col) tuple, applied to all datasets
            - a list of (row, col) tuples of length n (one per dataset)
        sigma : float, optional
            Optional low-pass smoothing sigma (pixels) applied to each mean DP prior to peak finding.
        plot_origins : bool, optional
            If True, plot mean diffraction patterns with overlaid origin markers.
        plot_indices : list, optional
            Optional indices to plot. If None, plots all datasets.
        **plot_kwargs
            Passed to show_2d.

        Attributes
        ----------
        diffraction_origins : np.ndarray
            Array of shape (n, 2) with integer (row, col) origins.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        n = len(self.datasets)
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")

        if plot_indices is None:
            plot_indices_list = list(range(n))
        else:
            plot_indices_list = list(plot_indices)
            for i in plot_indices_list:
                if i < 0 or i >= n:
                    raise IndexError("plot_indices contains an out-of-range index.")

        if sigma is not None and float(sigma) > 0:
            gaussian_filter_torch = torchvision.transforms.GaussianBlur(
                kernel_size=[2 * int(2 * float(sigma)) + 1, 2 * int(2 * float(sigma)) + 1],
                sigma=[sigma, sigma],
            )

            dp_means_use = gaussian_filter_torch(torch.stack(self.dp_mean))
        else:
            dp_means_use = torch.stack(self.dp_mean)

        if origins is None:
            origins_arr = torch.zeros((n, 2), dtype=torch.int)
            for i in range(n):
                dp_use = dp_means_use[i]

                r, c = torch.unravel_index(torch.argmax(dp_use), dp_use.shape)
                origins_arr[i, 0] = int(r)
                origins_arr[i, 1] = int(c)
        else:
            if isinstance(origins, tuple) and len(origins) == 2:
                origins_arr = torch.tile(
                    torch.tensor(origins, dtype=torch.int, device=self.device)[None, :], (n, 1)
                )
            else:
                origins_list = list(origins)
                if len(origins_list) != n:
                    raise ValueError(
                        "origins must be a single (row,col) tuple or a list of length n."
                    )
                origins_arr = torch.tensor(origins_list, dtype=torch.int, device=self.device)
                if origins_arr.shape != (n, 2):
                    raise ValueError("origins must have shape (n, 2) after conversion.")

        self.diffraction_origins = origins_arr

        if plot_origins:
            arrays = [np.asarray(self.dp_mean[i].cpu()) for i in plot_indices_list]
            titles = [f"{i} - Mean Diffraction Pattern" for i in plot_indices_list]
            fig, ax = show_2d(arrays, title=titles, returnfig=True, **plot_kwargs)
            axs = np.ravel(np.asarray(ax, dtype=object))
            for j, i in enumerate(plot_indices_list):
                r, c = self.diffraction_origins[i].cpu().numpy()
                axs[j].plot([c], [r], marker="+", color="red", markersize=16, markeredgewidth=2)

        return self

    def dscan_align(
        self,
        iterations: int,
        upsample_factor: int = 100,
        plot_aligned: bool = True,
        edge_blend: float = 2.0,
        fit_shifts: bool = True,
        mode: str = "linear",
    ):
        for i, dataset in enumerate(self.datasets):
            _, aligned_dataset, _ = dscan_correct(
                dataset,
                iterations,
                upsample_factor=upsample_factor,
                plot_aligned=plot_aligned,
                edge_blend=edge_blend,
                device=self.device,
                fit_shifts=fit_shifts,
                mode=mode,
            )
            self.datasets[i] = aligned_dataset

        return self

    def diffraction_align(
        self,
        edge_blend: float = 16.0,
        padding=None,
        pad_val: str | float = "min",
        upsample_factor: int = 100,
        weight_scale: float = 1 / 8,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align mean diffraction patterns using weighted cross-correlation in Fourier space.

        Parameters
        ----------
        edge_blend : float
            Tukey window edge taper (pixels).
        padding : int or tuple, optional
            Passed to shift_images for plotting.
        pad_val : str or float
            Passed to shift_images for plotting.
        upsample_factor : int
            Subpixel upsampling factor for correlation peak estimation.
        weight_scale : float
            Radial weight falloff scale (fraction of mean DP size).
        plot_aligned : bool
            If True, plot aligned mean diffraction patterns.
        **plot_kwargs
            Passed to show_2d when plotting.

        Attributes
        ----------
        diffraction_shifts : np.ndarray
            Array of shape (n, 2) with (row, col) shifts to align diffraction patterns.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        if not hasattr(self, "dp_mean"):
            raise RuntimeError("Run preprocess() first so self.dp_mean exists.")
        if not hasattr(self, "diffraction_origins"):
            raise RuntimeError(
                "Run diffraction_origin() first so self.diffraction_origins exists."
            )

        H, W = self.dp_mean[0].shape

        w = (
            tukey_torch(
                H,
                alpha=2.0 * float(edge_blend) / float(H),
                device=self.device,
                dtype=torch.float32,
            )[:, None]
            * tukey_torch(
                W,
                alpha=2.0 * float(edge_blend) / float(W),
                device=self.device,
                dtype=torch.float32,
            )[None, :]
        )

        r = torch.fft.fftfreq(H, 1.0 / float(H))[:, None]
        c = torch.fft.fftfreq(W, 1.0 / float(W))[None, :]

        n = len(self.dp_mean)
        self.diffraction_shifts = torch.zeros((n, 2), device=self.device, dtype=torch.float32)

        G_ref = torch.fft.fft2(w * self.dp_mean[0])
        xy0 = self.diffraction_origins[0]

        kr = torch.fft.fftfreq(H, device=self.device)[:, None]
        kc = torch.fft.fftfreq(W, device=self.device)[None, :]

        for ind in range(1, n):
            G = torch.fft.fft2(w * self.dp_mean[ind])
            xy = self.diffraction_origins[ind]

            dr2 = (r - xy0[0] + xy[0]) ** 2 + (c - xy0[1] + xy[1]) ** 2
            im_weight = torch.clip(
                1.0
                - torch.sqrt(dr2)
                / float(torch.mean(torch.tensor([H, W], device=self.device, dtype=torch.float32)))
                / float(weight_scale),
                0.0,
                1.0,
            )
            im_weight = torch.sin(im_weight * torch.pi / 2.0) ** 2
            shift_rc = cross_correlation_shift_torch(  # not torchified yet
                im_ref=G_ref,
                im=G,
                # weight_real=im_weight * 0.0 + 1.0,
                upsample_factor=int(upsample_factor),
                fft_input=True,
            )

            phase_ramp = torch.exp(-2j * torch.pi * (kr * shift_rc[0] + kc * shift_rc[1]))

            G_shift = G * phase_ramp
            self.diffraction_shifts[ind, :] = torch.tensor(
                shift_rc, device=self.device, dtype=torch.float32
            )

            G_ref = G_ref * (ind / (ind + 1)) + G_shift / (ind + 1)

        self.diffraction_shifts -= torch.mean(self.diffraction_shifts, axis=0)[None, :]
        if plot_aligned:
            im_aligned = shift_images_torch(
                images=torch.stack(self.dp_mean),
                shifts_rc=self.diffraction_shifts,
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
            )
            show_2d(im_aligned.unbind(0), **plot_kwargs)

        return self

    def real_space_align(
        self,
        num_images=None,
        num_iter: int = 3,
        edge_blend: float = 1.0,
        padding=None,
        pad_val: str | float = "median",
        upsample_factor: int = 100,
        max_shift=None,
        shift_method: str = "bilinear",
        edge_filter: bool = True,
        edge_sigma: float = 2.0,
        hanning_filter: bool = False,
        plot_aligned: bool = True,
        **plot_kwargs: Any,
    ) -> MAPED:
        """
        Align real-space mean BF images using iterative average-reference correlation.

        Parameters
        ----------
        num_images : int, optional
            If provided, align only the first num_images images.
        num_iter : int
            Number of refinement iterations.
        edge_blend : float
            Used to set default correlation padding when max_shift is None.
        padding : int or tuple, optional
            Passed to shift_images for plotting.
        pad_val : float
            Passed to shift_images for plotting.
        upsample_factor  : int
            Subpixel upsampling factor for correlation peak estimation.
        max_shift : float
            Optional maximum shift constraint passed to weighted_cross_correlation_shift.
        shift_method : 'bilinear' or 'fourier'
            Passed to shift_images for plotting ('bilinear' or 'fourier').
        edge_filter : bool
            If True, correlate on gradient magnitude instead of raw intensity.
        edge_sigma : float
            Gaussian sigma applied to gradients when edge_filter is True.
        hanning_filter : bool
            If True, apply a Hanning window prior to FFT.
        plot_aligned : bool
            If True, plot aligned mean BF images.
        **plot_kwargs
            Passed to show_2d when plotting.

        Attributes
        ----------
        real_space_shifts : np.ndarray
            Array of shape (n_total, 2) with (row, col) shifts for aligned datasets.

        Returns
        -------
        MAPED
            self (updated instance)
        """
        if not hasattr(self, "im_bf"):
            raise RuntimeError("Run preprocess() first so self.im_bf exists.")
        if len(self.im_bf) == 0:
            raise RuntimeError("No images found in self.im_bf.")

        H, W = self.im_bf[0].shape
        for im in self.im_bf:
            if im.shape != (H, W):
                raise ValueError("all self.im_bf images must have the same shape")

        n_total = len(self.im_bf)
        if num_images is None:
            n = n_total
        else:
            n = int(num_images)
            if n <= 0:
                raise ValueError("num_images must be positive")
            n = min(n, n_total)

        if int(num_iter) < 1:
            raise ValueError("num_iter must be >= 1")

        if max_shift is not None:
            pad_cc = int(np.ceil(float(max_shift))) + 4
        else:
            pad_cc = int(np.ceil(float(edge_blend))) + 4

        Hp = H + 2 * pad_cc
        Wp = W + 2 * pad_cc
        r0 = pad_cc
        c0 = pad_cc

        w_h = torch.ones((H, W), dtype=torch.float32, device=self.device)
        if hanning_filter:
            w_h = (
                torch.hann_window(H, dtype=torch.float32, device=self.device)[:, None]
                * torch.hann_window(W, dtype=torch.float32, device=self.device)[None, :]
            )
        w_h_pad = torch.zeros((Hp, Wp), dtype=torch.float32, device=self.device)
        w_h_pad[r0 : r0 + H, c0 : c0 + W] = w_h
        w_h_sum = torch.sum(w_h_pad)
        if w_h_sum <= 0:
            raise RuntimeError("hanning window sum is zero")

        if edge_filter:
            wx = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            wx = None

        base_pad = torch.zeros((n, Hp, Wp), dtype=torch.float32, device=self.device)
        for i in range(n):
            im0 = self.im_bf[i]

            if edge_filter:
                pad_symmetric = wx.shape[-1] // 2
                im0_pad = F.pad(
                    im0.unsqueeze(0).unsqueeze(0),
                    pad=(pad_symmetric, pad_symmetric, pad_symmetric, pad_symmetric),
                    mode="reflect",
                )

                gx = F.conv2d(im0_pad, wx.unsqueeze(0).unsqueeze(0))[0, 0]
                gy = F.conv2d(im0_pad, wx.T.unsqueeze(0).unsqueeze(0))[0, 0]

                gaussian_filt = torchvision.transforms.GaussianBlur(
                    kernel_size=[
                        2 * int(2 * float(edge_sigma)) + 1,
                        2 * int(2 * float(edge_sigma)) + 1,
                    ],
                    sigma=[edge_sigma, edge_sigma],
                )
                gx = gaussian_filt(gx.unsqueeze(0))
                gy = gaussian_filt(gy.unsqueeze(0))
                im_use = torch.sqrt(gx * gx + gy * gy)
            else:
                im_use = im0

            base_pad[i, r0 : r0 + H, c0 : c0 + W] = im_use

        shifts = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        for _ in range(int(num_iter)):
            G_list = torch.empty((n, Hp, Wp), dtype=torch.complex128)

            # shift images to current guess
            ims_a = shift_images_torch(base_pad, shifts)
            ims_mean = torch.sum(ims_a * w_h_pad, dim=(1, 2)) / w_h_sum

            ims_win = (ims_a - ims_mean[:, None, None]) * w_h_pad[None]
            G_list = torch.fft.fft2(ims_win)

            G_ref = torch.mean(G_list, axis=0)

            # perform cross correlation again
            for i in range(1, n):
                drc = cross_correlation_shift_torch(
                    im_ref=G_ref,
                    im=G_list[i],
                    # weight_real=None,
                    upsample_factor=int(upsample_factor),
                    # max_shift=max_shift,
                    fft_input=True,
                    # fft_output=False,
                    # return_shifted_image=False,
                )

                shifts[i, 0] += float(drc[0])
                shifts[i, 1] += float(drc[1])

            shifts -= shifts[0][None, :].clone()

        shifts -= torch.mean(shifts, dim=0)[None, :]

        self.real_space_shifts = torch.zeros((n_total, 2), dtype=torch.float32, device=self.device)
        self.real_space_shifts[:n, :] = shifts

        if plot_aligned:
            im_aligned = shift_images_torch(
                images=torch.stack(self.im_bf[:n]),
                shifts_rc=self.real_space_shifts[:n, :],
                edge_blend=float(edge_blend),
                padding=padding,
                pad_val=pad_val,
                mode=shift_method,
                blend=False,
            )
            show_2d(im_aligned.sum(0), **plot_kwargs)

        return self

    def merge_datasets(
        self,
        real_space_padding: int = 0,
        real_space_edge_blend: float = 1.0,
        diffraction_padding: int = 0,
        diffraction_edge_blend: float = 0.0,
        diffraction_pad_val: str | float = "min",
        shift_method: str = "bilinear",
        dtype=None,
        scale_output: bool = False,
        plot_result: bool = True,
        batch_size: int = None,
        **plot_kwargs: Any,
    ) -> Dataset4dstem:
        """
        Merge aligned datasets into a single Dataset4dstem.

        Notes
        -----
        Requires the following attributes to be present on ``self``:

        self.real_space_shifts
            From ``real_space_align()``.
        self.diffraction_shifts
            From ``diffraction_align()``.

        Parameters
        ----------
        real_space_padding : int
            Output scan padding in pixels (adds border to scan grid).
        real_space_edge_blend : float
            Tukey taper width for scan-space interpolation weights.
        diffraction_padding : int
            Output diffraction padding in pixels (adds border around DPs).
        diffraction_edge_blend : float
            Tukey taper width for diffraction-space weights.
        diffraction_pad_val : str | float
            Pad value for diffraction padding ('min','max','mean','median' or float).
        shift_method : str
            Diffraction shift method: 'bilinear' or 'fourier'.
        dtype : str or torch.dtype, optional
            Output dtype. If None, uses parent dtype.
        scale_output : bool
            If True and dtype is integer, scale to full dynamic range using global max.
        plot_result : bool
            If True, plot merged BF and merged mean DP.
        batch_size : int, optional
            Number of rows to process per batch. If None, uses adaptive sizing (1-32 rows).
        **plot_kwargs
            Passed to show_2d.

        Returns
        -------
        Dataset4dstem
            Merged dataset.
        """

        if not hasattr(self, "real_space_shifts"):
            raise RuntimeError("Run real_space_align() first so self.real_space_shifts exists.")
        if not hasattr(self, "diffraction_shifts"):
            raise RuntimeError("Run diffraction_align() first so self.diffraction_shifts exists.")

        arrays = self.datasets
        n = len(arrays)
        if n == 0:
            raise RuntimeError("No datasets found in self.datasets.")

        Rs, Cs, H, W = arrays[0].shape
        for a in arrays:
            if a.shape != (Rs, Cs, H, W):
                raise ValueError("All dataset arrays must have the same shape (Rs, Cs, H, W).")

        rs_shifts = self.real_space_shifts
        dp_shifts = self.diffraction_shifts
        if rs_shifts.shape != (n, 2):
            raise ValueError("self.real_space_shifts must have shape (n, 2).")
        if dp_shifts.shape != (n, 2):
            raise ValueError("self.diffraction_shifts must have shape (n, 2).")

        if dtype is None:
            dtype_out = arrays[0].dtype
            warnings.warn(f"dtype=None; using parent dtype {dtype_out}.", RuntimeWarning)
        else:
            dtype_out = torch.dtype(dtype)

        real_space_padding = int(real_space_padding)
        diffraction_padding = int(diffraction_padding)

        Rout = Rs + 2 * real_space_padding
        Cout = Cs + 2 * real_space_padding

        Hp = H + 2 * diffraction_padding
        Wp = W + 2 * diffraction_padding
        rp0 = diffraction_padding
        cp0 = diffraction_padding

        method = str(shift_method).strip().lower()
        if method not in {"bilinear", "fourier"}:
            raise ValueError("shift_method must be 'bilinear' or 'fourier'.")

        # set up real space edge blending weights
        if real_space_edge_blend and float(real_space_edge_blend) > 0:
            alpha_r = min(1.0, 2.0 * float(real_space_edge_blend) / float(Rs))
            alpha_c = min(1.0, 2.0 * float(real_space_edge_blend) / float(Cs))
            w_rs = (
                tukey_torch(Rs, alpha=alpha_r, device=self.device, dtype=torch.float32)[:, None]
                * tukey_torch(Cs, alpha=alpha_c, device=self.device, dtype=torch.float32)[None, :]
            )
        else:
            w_rs = torch.ones((Rs, Cs), dtype=torch.float32, device=self.device)

        # set up diffraction space edge blending weights
        if diffraction_edge_blend and float(diffraction_edge_blend) > 0:
            alpha_dr = min(1.0, 2.0 * float(diffraction_edge_blend) / float(H))
            alpha_dc = min(1.0, 2.0 * float(diffraction_edge_blend) / float(W))
            w_dp = (
                tukey_torch(H, alpha=alpha_dr, device=self.device, dtype=torch.float32)[:, None]
                * tukey_torch(W, alpha=alpha_dc, device=self.device, dtype=torch.float32)[None, :]
            )
        else:
            w_dp = torch.ones((H, W), dtype=torch.float32, device=self.device)

        v = torch.stack(self.dp_mean, axis=0).reshape(-1)

        if isinstance(diffraction_pad_val, str):
            s = diffraction_pad_val.strip().lower()
            if s == "min":
                pad_val_dp = float(torch.min(v))
            elif s == "max":
                pad_val_dp = float(torch.max(v))
            elif s == "mean":
                pad_val_dp = float(torch.mean(v))
            elif s == "median":
                pad_val_dp = float(torch.median(v))
            else:
                raise ValueError(
                    "diffraction_pad_val must be a float or one of {'min','max','mean','median'}."
                )
        else:
            pad_val_dp = float(diffraction_pad_val)

        wdp_pad = torch.zeros((Hp, Wp), dtype=torch.float32, device=self.device)
        wdp_pad[rp0 : rp0 + H, cp0 : cp0 + W] = w_dp

        wdp_shifted = torch.zeros((n, Hp, Wp), dtype=torch.float32, device=self.device)
        if method == "fourier":
            kr = torch.fft.fftfreq(Hp, device=self.device)[:, None]
            kc = torch.fft.fftfreq(Wp, device=self.device)[None, :]
            Fw = torch.fft.fft2(wdp_pad)
            ramps: list[torch.Tensor] = []
            for i in range(n):
                dr, dc = dp_shifts[i, 0], dp_shifts[i, 1]

                ramp = torch.exp(-2j * torch.pi * (kr * dr + kc * dc))
                ramps.append(ramp)
                w_i = torch.fft.ifft2(Fw * ramp).real
                wdp_shifted[i] = torch.clip(w_i, 0.0, 1.0)
        else:
            for i in range(n):
                w_i = shift_images_torch(
                    wdp_pad,
                    shifts_rc=dp_shifts[i, :],
                    mode="bilinear",
                )
                wdp_shifted[i] = w_i
            wdp_shifted = torch.clip(w_i, 0.0, 1.0)

        coverage = torch.clip(torch.sum(wdp_shifted, dim=0), 0.0, 1.0)
        edge_w_dp = 1.0 - coverage

        # Determine batch size (somewhat arbitrary)
        if batch_size is None:
            batch_size = max(1, min(32, Rout // 2))

        c_out = torch.arange(Cout, dtype=torch.float32, device=self.device)
        c_base = c_out - real_space_padding

        merged = torch.zeros((Rout, Cout, Hp, Wp), dtype=torch.float64, device=self.device)

        # start batching

        for batch_start in tqdm(
            range(0, Rout, batch_size),
            desc="Merging (batches)",
            total=(Rout + batch_size - 1) // batch_size,
        ):
            batch_end = min(batch_start + batch_size, Rout)
            batch_rows = torch.arange(
                batch_start, batch_end, dtype=torch.float32, device=self.device
            )

            num_batch = torch.zeros(
                (batch_end - batch_start, Cout, Hp, Wp), dtype=torch.float32, device=self.device
            )
            den_batch = torch.zeros(
                (batch_end - batch_start, Cout, Hp, Wp), dtype=torch.float32, device=self.device
            )

            r_base_batch = batch_rows.unsqueeze(1) - real_space_padding  # (batch_size, 1)
            c_base_batch = c_base.unsqueeze(0)  # (1, Cout)

            for i in range(n):
                a = arrays[i]
                if isinstance(a, torch.Tensor):
                    a = a.float()
                else:
                    a = torch.tensor(a, dtype=torch.float32, device=self.device)

                r_in = r_base_batch.expand(-1, Cout) - rs_shifts[i, 0]  # (batch_size, Cout)
                c_in = (
                    c_base_batch.expand(batch_end - batch_start, -1) - rs_shifts[i, 1]
                )  # (batch_size, Cout)

                c_norm = 2.0 * c_in / (Cs - 1) - 1.0  # (batch_size, Cout)
                r_norm = 2.0 * r_in / (Rs - 1) - 1.0  # (batch_size, Cout)

                a_reshaped = (
                    a.view(Rs, Cs, H * W).permute(2, 0, 1).unsqueeze(0)
                )  # (1, H*W, Rs, Cs)

                # Reshape w_rs from (Rs, Cs) to (1, 1, Rs, Cs)
                w_rs_reshaped = w_rs.unsqueeze(0).unsqueeze(0)  # (1, 1, Rs, Cs)

                dp_interp_list = []
                wi_list = []

                # Loop through batches, vectorize columns per batch
                for b in range(batch_end - batch_start):
                    grid_batch = torch.stack(
                        [c_norm[b : b + 1, :], r_norm[b : b + 1, :]], dim=-1
                    ).unsqueeze(2)  # (1, Cout, 1, 2)

                    dp_sample = torch.nn.functional.grid_sample(
                        a_reshaped,
                        grid_batch,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )

                    wi_sample = torch.nn.functional.grid_sample(
                        w_rs_reshaped,
                        grid_batch,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )

                    dp_b = dp_sample.squeeze(0).squeeze(-1).view(H, W, Cout).permute(2, 0, 1)
                    wi_b = wi_sample.squeeze(0).squeeze(-1).squeeze(0)

                    dp_interp_list.append(dp_b)
                    wi_list.append(wi_b)

                dp_interp = torch.stack(dp_interp_list)
                wi = torch.stack(wi_list)

                dp_padded = torch.zeros(
                    (batch_end - batch_start, Cout, Hp, Wp),
                    dtype=torch.float32,
                    device=self.device,
                )
                dp_padded[:, :, rp0 : rp0 + H, cp0 : cp0 + W] = (
                    dp_interp * w_dp.unsqueeze(0).unsqueeze(0)
                ).float()

                if method == "fourier":
                    ramp = ramps[i]
                    fft_result = torch.fft.fft2(dp_padded)
                    ramp_exp = ramp.unsqueeze(0).unsqueeze(0)
                    dp_shifted = torch.fft.ifft2(fft_result * ramp_exp).real
                else:
                    dp_shifted = torch.zeros_like(dp_padded)
                    for batch_idx in range(batch_end - batch_start):
                        for co in range(Cout):
                            dp_shifted[batch_idx, co] = shift_images_torch(
                                dp_padded[batch_idx, co].unsqueeze(0),
                                shifts_rc=dp_shifts[i, :].unsqueeze(0),
                                mode="bilinear",
                            ).squeeze(0)

                wi_exp = wi.unsqueeze(-1).unsqueeze(-1)
                wdp_i = wdp_shifted[i].unsqueeze(0).unsqueeze(0)

                num_batch += wi_exp * dp_shifted
                den_batch += wi_exp * wdp_i

                # clear memory
                del a, a_reshaped, w_rs_reshaped, dp_padded, dp_shifted
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Final division for this batch
            num_final = num_batch + edge_w_dp.unsqueeze(0).unsqueeze(0) * pad_val_dp
            den_final = den_batch + edge_w_dp.unsqueeze(0).unsqueeze(0)

            merged[batch_start:batch_end] = torch.where(
                den_final != 0.0,
                (num_final / den_final).to(torch.float64),
                torch.zeros_like(num_final).to(torch.float64),
            )

            del num_batch, den_batch, num_final, den_final  # clear memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.im_bf_merged = torch.mean(merged, dim=(2, 3))
        self.dp_mean_merged = torch.mean(merged, dim=(0, 1))

        self.im_bf_merged = torch.mean(merged, dim=(2, 3))
        self.dp_mean_merged = torch.mean(merged, dim=(0, 1))

        # dtype scaling and clipping
        try:
            info = torch.iinfo(dtype_out)
            is_int_dtype = True
        except TypeError:
            is_int_dtype = False

        if is_int_dtype:
            dmin = float(info.min)
            dmax = float(info.max)

            merged_f = merged

            if scale_output:
                peak = torch.max(merged_f).item()
                if peak <= 0.0:
                    merged_scaled = merged_f
                else:
                    merged_scaled = merged_f * (dmax / peak)

                lo, hi = (0.0, dmax) if dtype_out == torch.uint8 else (dmin, dmax)
                merged_out = torch.rint(torch.clamp(merged_scaled, lo, hi)).to(dtype=dtype_out)
            else:
                below = torch.min(merged_f).item()
                above = torch.max(merged_f).item()
                if below < dmin or above > dmax:
                    warnings.warn(
                        f"Output overflow for dtype {dtype_out}: data range [{below}, {above}] exceeds "
                        f"[{dmin}, {dmax}]. Values will be clipped.",
                        RuntimeWarning,
                    )
                merged_out = torch.rint(torch.clamp(merged_f, dmin, dmax)).to(dtype=dtype_out)
        else:
            merged_out = merged.to(dtype=dtype_out)

        dataset_merged = Dataset4dstem.from_array(array=merged_out.cpu().numpy())
        dataset_merged.im_bf_merged = self.im_bf_merged
        dataset_merged.dp_mean_merged = self.dp_mean_merged

        if plot_result:
            show_2d(
                [[self.im_bf_merged, self.dp_mean_merged]],
                title=[["Merged Bright Field", "Merged Mean Diffraction Pattern"]],
                **plot_kwargs,
            )

        return dataset_merged


def shift_images(
    images: list[np.ndarray],
    shifts_rc: np.ndarray,
    edge_blend: float = 8.0,
    padding: int | None = None,
    pad_val: str | float = 0.0,
    shift_method: str = "bilinear",
):
    """
    Shift and blend a stack of 2D images into a common padded canvas.

    Parameters
    ----------
    images : list of np.ndarray
        Sequence of (H, W) arrays.
    shifts_rc : np.ndarray
        Array-like of shape (n, 2) with (row, col) shifts for each image.
    edge_blend : float, optional
        Tukey taper width in pixels for image blending.
    padding : int
        Output padding. If None, set from max shift and edge_blend.
    pad_val : str | float optional
        Fill value outside support ('min','max','mean','median' or float).
    shift_method : str
        'bilinear' or 'fourier'.

    Returns
    -------
    np.ndarray
        Blended image of shape (H + 2*padding, W + 2*padding).
    """
    images = [np.asarray(im, dtype=float) for im in images]
    if len(images) == 0:
        raise ValueError("images must be non-empty")

    H, W = images[0].shape
    for im in images:
        if im.shape != (H, W):
            raise ValueError("all images must have the same shape")

    shifts_rc = np.asarray(shifts_rc, dtype=float)
    if shifts_rc.shape != (len(images), 2):
        raise ValueError("shifts_rc must have shape (len(images), 2)")

    if isinstance(pad_val, str):
        s = pad_val.strip().lower()
        v = np.stack(images, axis=0).reshape(-1)
        if s == "min":
            pad_val_f = float(np.min(v))
        elif s == "max":
            pad_val_f = float(np.max(v))
        elif s == "mean":
            pad_val_f = float(np.mean(v))
        elif s == "median":
            pad_val_f = float(np.median(v))
        else:
            raise ValueError("pad_val must be a float or one of {'min','max','mean','median'}")
    else:
        pad_val_f = float(pad_val)

    if padding is None:
        max_shift = float(np.max(np.abs(shifts_rc))) if shifts_rc.size else 0.0
        padding = int(np.ceil(max_shift + float(edge_blend))) + 2
    padding = int(padding)

    alpha_r = min(1.0, 2.0 * float(edge_blend) / float(H)) if edge_blend > 0 else 0.0
    alpha_c = min(1.0, 2.0 * float(edge_blend) / float(W)) if edge_blend > 0 else 0.0
    w = tukey(H, alpha=alpha_r)[:, None] * tukey(W, alpha=alpha_c)[None, :]
    w = w.astype(float, copy=False)

    Hp = H + 2 * padding
    Wp = W + 2 * padding

    stack_w = np.zeros((len(images), Hp, Wp), dtype=float)
    stack = np.zeros_like(stack_w)

    r0 = padding
    c0 = padding
    stack_w[:, r0 : r0 + H, c0 : c0 + W] = w[None, :, :]
    for ind, im in enumerate(images):
        stack[ind, r0 : r0 + H, c0 : c0 + W] = im * w

    method = str(shift_method).strip().lower()
    if method not in {"bilinear", "fourier"}:
        raise ValueError("shift_method must be 'bilinear' or 'fourier'")

    if method == "fourier":
        kr = np.fft.fftfreq(Hp)[:, None]
        kc = np.fft.fftfreq(Wp)[None, :]
        for ind in range(len(images)):
            dr, dc = shifts_rc[ind, 0], shifts_rc[ind, 1]
            ramp = np.exp(-2j * np.pi * (kr * dr + kc * dc))

            F = np.fft.fft2(stack[ind])
            stack[ind] = np.fft.ifft2(F * ramp).real

            Fw = np.fft.fft2(stack_w[ind])
            stack_w[ind] = np.fft.ifft2(Fw * ramp).real
            stack_w[ind] = np.clip(stack_w[ind], 0.0, 1.0)
    else:
        for ind in range(len(images)):
            stack[ind] = ndi_shift(
                stack[ind],
                shift=(shifts_rc[ind, 0], shifts_rc[ind, 1]),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            stack_w[ind] = ndi_shift(
                stack_w[ind],
                shift=(shifts_rc[ind, 0], shifts_rc[ind, 1]),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            stack_w[ind] = np.clip(stack_w[ind], 0.0, 1.0)

    edge_w = np.clip(1.0 - np.sum(stack_w, axis=0), 0.0, 1.0)

    num = np.sum(stack, axis=0) + edge_w * pad_val_f
    den = np.sum(stack_w, axis=0) + edge_w

    out = np.empty_like(num)
    np.divide(num, den, out=out, where=den != 0.0)
    out[den == 0.0] = 0.0

    return out


def tukey_torch(N, alpha=0.5, device=None, dtype=torch.float32):
    """
    Creates a 1D Tukey window of length N and shape parameter alpha.

    Parameters
    ----------
    N : int
        Length of the window.
    alpha : float
        Shape parameter for the Tukey window.
    device : torch.device | str
        Device on which to create the window.
    dtype : torch.dtype
        torch.dtype, Data type of the window.

    Returns
    -------
    window : torch.Tensor
        1D Tukey window of length N.
    """
    n = torch.arange(N, device=device, dtype=dtype)
    w = torch.ones(N, device=device, dtype=dtype)

    if alpha <= 0:
        return w
    if alpha >= 1:
        return torch.hann_window(N, device=device, dtype=dtype)

    edge = alpha * (N - 1) / 2

    left = n < edge
    right = n >= (N - 1 - edge)

    w[left] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[left] / (alpha * (N - 1)) - 1)))

    w[right] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[right] / (alpha * (N - 1)) - 2 / alpha + 1)))

    return w


def shift_images_torch(
    images,
    shifts_rc,
    mode="bilinear",
    blend: bool = False,
    edge_blend: float = 8.0,
    padding=None,
    pad_val: str | float = 0.0,
):
    """
    Shift (and optionally blend) a stack of 2D images by per-image (dr, dc) pixel shifts using grid_sample.

    Parameters
    ----------
    images : torch.Tensor, shape (n, H, W) or (H, W)
        Stack of images (or a single image).
    shifts_rc : torch.Tensor, shape (n, 2) or (2,)
        Per-image shifts as (row_shift, col_shift) in pixels.
    mode : 'bilinear' or 'nearest'
    blend : bool, whether to blend the shifted images using a Tukey window
    edge_blend : float, Tukey edge width in pixels used when blending
    padding : int or None, canvas padding. If None, computed from max shift + edge_blend
    pad_val : float or one of 'min','max','mean','median', fill value outside support

    Returns
    -------
    torch.Tensor
        Shifted (and blended) images. If the input was a single image, returns an array
        of shape (Hp, Wp). Otherwise returns (n, Hp, Wp) for blended result or (n, H, W)
        for the non-blended case.
    """
    single = images.dim() == 2
    if single:
        images = images.unsqueeze(0)
        shifts_rc = shifts_rc.unsqueeze(0)

    n, H, W = images.shape

    shifts_rc = shifts_rc.to(dtype=torch.float32, device=images.device)

    if not blend:
        # simple shift per-image without padding/blending — keep original behavior
        imgs = images.unsqueeze(1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=images.device),
            torch.linspace(-1, 1, W, device=images.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = base_grid.unsqueeze(0).expand(n, -1, -1, -1).clone()  # (n, H, W, 2)
        grid[..., 0] -= 2.0 * shifts_rc[:, 1].view(n, 1, 1) / W  # col shift → x
        grid[..., 1] -= 2.0 * shifts_rc[:, 0].view(n, 1, 1) / H  # row shift → y

        shifted = F.grid_sample(imgs, grid, mode=mode, padding_mode="zeros", align_corners=True)
        result = shifted[:, 0]  # (n, H, W)
        return result[0] if single else result

    # --- blending path ---
    # determine pad_val numeric
    if isinstance(pad_val, str):
        s = pad_val.strip().lower()
        v = images.reshape(-1)
        if s == "min":
            pad_val_f = float(torch.min(v).item())
        elif s == "max":
            pad_val_f = float(torch.max(v).item())
        elif s == "mean":
            pad_val_f = float(torch.mean(v).item())
        elif s == "median":
            pad_val_f = float(torch.median(v).item())
        else:
            raise ValueError("pad_val must be a float or one of {'min','max','mean','median'}")
    else:
        pad_val_f = float(pad_val)

    # padding (compute from max shift if not provided)
    max_shift = float(torch.max(torch.abs(shifts_rc)).item()) if shifts_rc.numel() else 0.0
    if padding is None:
        padding = int(np.ceil(max_shift + float(edge_blend))) + 2
    padding = int(padding)

    alpha_r = min(1.0, 2.0 * float(edge_blend) / float(H)) if edge_blend > 0 else 0.0
    alpha_c = min(1.0, 2.0 * float(edge_blend) / float(W)) if edge_blend > 0 else 0.0

    w = (
        tukey_torch(H, alpha=alpha_r, device=images.device, dtype=torch.float32)[:, None]
        * tukey_torch(W, alpha=alpha_c, device=images.device, dtype=torch.float32)[None, :]
    )

    Hp = H + 2 * padding
    Wp = W + 2 * padding
    r0 = padding
    c0 = padding

    # build padded stacks
    stack = torch.zeros((n, Hp, Wp), dtype=torch.float32, device=images.device)
    stack_w = torch.zeros_like(stack)
    for ind in range(n):
        stack[ind, r0 : r0 + H, c0 : c0 + W] = images[ind].to(dtype=torch.float32) * w
        stack_w[ind, r0 : r0 + H, c0 : c0 + W] = w

    # shift both stack and stack_w using grid_sample on (n,1,Hp,Wp)
    imgs = stack.unsqueeze(1)
    imgs_w = stack_w.unsqueeze(1)

    # Build base normalized grid for Hp, Wp
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, Hp, device=images.device),
        torch.linspace(-1, 1, Wp, device=images.device),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (Hp, Wp, 2)
    grid = base_grid.unsqueeze(0).expand(n, -1, -1, -1).clone()  # (n, Hp, Wp, 2)
    grid[..., 0] -= 2.0 * shifts_rc[:, 1].view(n, 1, 1) / Wp  # col shift → x
    grid[..., 1] -= 2.0 * shifts_rc[:, 0].view(n, 1, 1) / Hp  # row shift → y

    shifted = F.grid_sample(imgs, grid, mode=mode, padding_mode="zeros", align_corners=True)
    shifted_w = F.grid_sample(imgs_w, grid, mode=mode, padding_mode="zeros", align_corners=True)

    shifted = shifted[:, 0]
    shifted_w = shifted_w[:, 0]

    shifted_w = torch.clamp(shifted_w, 0.0, 1.0)

    edge_w = torch.clamp(1.0 - torch.sum(shifted_w, dim=0), 0.0, 1.0)

    num = torch.sum(shifted, dim=0) + edge_w * pad_val_f
    den = torch.sum(shifted_w, dim=0) + edge_w

    out = torch.empty_like(num)
    mask = den != 0.0
    out[mask] = num[mask] / den[mask]
    out[~mask] = 0.0

    return out


def fit_surface_lstsq(img, mode="linear"):
    """
    Fits an image with a linear or quadratic function

    Parameters
    ----------
    img : torch.Tensor
        Image to fit, of shape (H, W)
    mode : str
        Fitting mode, either "linear" or "quadratic"

    Returns
    ------
    fitted : torch.Tensor
        Array of shape (H, W) of the fit function over the image
    coeffs : torch.Tensor
        fitting coefficients
    """
    H, W = img.shape
    x_1d = torch.arange(img.shape[1], device=img.device, dtype=torch.float32)
    y_1d = torch.arange(img.shape[0], device=img.device, dtype=torch.float32)

    xx, yy = torch.meshgrid(x_1d, y_1d)

    x = xx.flatten()
    y = yy.flatten()
    z = img.flatten()

    if mode == "linear":
        A = torch.stack([x, y, torch.ones_like(x)], dim=1)
    elif mode == "quadratic":
        A = torch.stack([x**2, y**2, x * y, x, y, torch.ones_like(x)], dim=1)

    coeffs, _, _, _ = torch.linalg.lstsq(A, z.unsqueeze(1))

    fitted = (A @ coeffs).reshape(H, W)
    return fitted, coeffs


def dscan_correct(
    dataset,
    iterations,
    upsample_factor: int = 100,
    plot_aligned: bool = True,
    edge_blend: float = 2.0,
    device="cpu",
    fit_shifts=True,
    mode="linear",
):
    """
    Align diffraction patterns using cross-correlation.

    Parameters
    ----------
    dataset : torch.Tensor
        Input 4D dataset
    iterations : int
        Number of refinement iterations
    upsample_factor : int
        Upsampling factor for sub-pixel accuracy
    plot_aligned : bool
        Whether to plot results after each iteration
    edge_blend : float
        Edge blending parameter for Tukey window
    device : torch.device
        Device to use
    fit_shifts : bool
        Whether to fit shifts to a smooth surface
    mode : str
        "linear" or "quadratic" for surface fitting

    Returns
    -------
    tuple
        A tuple ``(diffraction_shifts, shifted_dps, G_ref_final)`` where
        ``diffraction_shifts`` is a ``torch.Tensor`` of shape (H_rs, W_rs, 2) with
        per-scan-position shifts, ``shifted_dps`` is the aligned dataset (same shape
        as ``dataset``), and ``G_ref_final`` is the final complex Fourier-domain
        reference (torch.Tensor).
    """
    H_rs, W_rs, H_dp, W_dp = dataset.shape

    w = (
        tukey_torch(
            H_dp,
            alpha=2.0 * float(edge_blend) / float(H_dp),
            device=device,
            dtype=torch.float32,
        )[:, None]
        * tukey_torch(
            W_dp,
            alpha=2.0 * float(edge_blend) / float(W_dp),
            device=device,
            dtype=torch.float32,
        )[None, :]
    )

    diffraction_shifts = torch.zeros((H_rs, W_rs, 2), device=device, dtype=torch.float32)
    shifted_dps = dataset.clone()

    kr = torch.fft.fftfreq(H_dp, device=device)[:, None]
    kc = torch.fft.fftfreq(W_dp, device=device)[None, :]

    for iteration in range(iterations):
        G_ref = torch.fft.fft2(shifted_dps.mean(dim=(0, 1)) * w)

        for h_rs in tqdm(range(H_rs), desc=f"Iteration {iteration + 1}/{iterations}"):
            for w_rs in range(W_rs):
                ind = w_rs + h_rs * H_rs
                dp = shifted_dps[h_rs, w_rs]  # <-- Read from current shifted_dps, not original
                G = torch.fft.fft2(w * dp)
                shift = cross_correlation_shift_torch(
                    G_ref, G, upsample_factor=upsample_factor, fft_input=True
                )
                diffraction_shifts[h_rs, w_rs] = shift

                phase_ramp = torch.exp(-1j * torch.pi * (kr * shift[0] + kc * shift[1]))
                G_shift = G * phase_ramp

                shifted_dps[h_rs, w_rs, :, :] = torch.fft.ifft2(G_shift).real
                G_ref = G_ref * (ind / (ind + 1)) + G_shift / (ind + 1)

        G_ref_final = G_ref.clone()

        if fit_shifts:
            diffraction_shifts_1, _ = fit_surface_lstsq(diffraction_shifts[:, :, 0], mode=mode)
            diffraction_shifts_2, _ = fit_surface_lstsq(diffraction_shifts[:, :, 1], mode=mode)
            diffraction_shifts_old = diffraction_shifts.clone()
            diffraction_shifts = torch.stack((diffraction_shifts_1, diffraction_shifts_2), dim=2)

            # Recompute fitted shifts
            for h_rs in tqdm(range(H_rs), desc="Applying fitted shifts"):
                for w_rs in range(W_rs):
                    dp = shifted_dps[h_rs, w_rs]  # <-- Also read from shifted_dps here
                    G = torch.fft.fft2(w * dp)
                    shift = diffraction_shifts[h_rs, w_rs]

                    phase_ramp = torch.exp(-1j * torch.pi * (kr * shift[0] + kc * shift[1]))
                    G_shift = G * phase_ramp

                    shifted_dps[h_rs, w_rs, :, :] = torch.fft.ifft2(G_shift).real

        if plot_aligned:
            if fit_shifts:
                show_2d(
                    [
                        [
                            diffraction_shifts_old[:, :, 0],
                            diffraction_shifts[:, :, 0],
                            diffraction_shifts[:, :, 0] - diffraction_shifts_old[:, :, 0],
                        ],
                        [
                            diffraction_shifts_old[:, :, 1],
                            diffraction_shifts[:, :, 1],
                            diffraction_shifts[:, :, 1] - diffraction_shifts_old[:, :, 1],
                        ],
                    ],
                    title=[
                        ["Shifts x", "Fit x", "Residual x"],
                        ["Shifts y", "Fit y", "Residual y"],
                    ],
                    cmap="RdBu_r",
                    vmax=3,
                    vmin=-3,
                )

            dp_mean_before = dataset.mean(dim=(0, 1))
            dp_mean = shifted_dps.mean(dim=(0, 1))
            dp_max = torch.max(
                torch.max(shifted_dps, dim=0, keepdim=False).values, dim=0, keepdim=False
            ).values
            show_2d(
                [dp_mean_before, dp_mean, dp_max],
                vmax=0.75,
            )

    return diffraction_shifts, shifted_dps, G_ref_final
