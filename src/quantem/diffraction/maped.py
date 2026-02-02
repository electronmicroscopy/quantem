from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.signal.windows import tukey

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d
from quantem.core.utils.imaging_utils import weighted_cross_correlation_shift


class MAPED(AutoSerialize):
    _token = object()

    def __init__(self, datasets: list[Dataset4dstem], _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError("Use MAPED.from_datasets() to instantiate this class.")
        AutoSerialize.__init__(self)
        self.datasets = datasets
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_datasets(cls, datasets: Sequence[Dataset4dstem]) -> "MAPED":
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
            raise TypeError("MAPED.from_datasets expects a sequence of Dataset4dstem instances.")
        ds_list: list[Dataset4dstem] = []
        for d in datasets:
            if not isinstance(d, Dataset4dstem):
                raise TypeError("MAPED.from_datasets expects a sequence of Dataset4dstem instances.")
            ds_list.append(d)
        if len(ds_list) == 0:
            raise ValueError("MAPED.from_datasets expects a non-empty sequence of Dataset4dstem instances.")
        return cls(datasets=ds_list, _token=cls._token)

    def preprocess(
        self,
        plot_summary: bool = True,
        scale: float | Sequence[float] | None = None,
        **plot_kwargs: Any,
    ) -> "MAPED":
        n = len(self.datasets)
        if scale is None:
            self.scales = np.ones(n, dtype=float)
        elif isinstance(scale, (int, float, np.floating)):
            self.scales = np.full(n, float(scale), dtype=float)
        else:
            self.scales = np.asarray(list(scale), dtype=float)
            if self.scales.shape != (n,):
                raise ValueError("scale must be a scalar or a sequence with the same length as datasets.")
        if np.any(self.scales == 0):
            raise ValueError("scale entries must be nonzero.")

        self.dp_mean = []
        self.im_bf = []

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
                dp_arr = np.mean(np.asarray(d.array), axis=(0, 1))
            else:
                dp_arr = np.asarray(dp.array if hasattr(dp, "array") else dp)

            im_bf_arr = np.mean(np.asarray(d.array), axis=(2, 3))

            self.dp_mean.append(np.asarray(dp_arr))
            self.im_bf.append(np.asarray(im_bf_arr))

        if plot_summary:
            tiles = [[(self.im_bf[i] / self.scales[i]), self.dp_mean[i]] for i in range(n)]
            titles = [[f"{i} - Mean Bright Field", f"{i} - Mean Diffraction Pattern"] for i in range(n)]
            show_2d(
                tiles,
                titles=titles,
                **plot_kwargs,
            )

        return self


    def diffraction_find_origin(
        self,
        origins=None,
        sigma=None,
        plot_origins: bool = True,
        plot_indices=None,
    ):
        """
        Choose or automatically find the origin in diffraction space.

        Parameters
        ----------
        origins
            Optional manual origins. Can be:
            - a single (row, col) tuple, applied to all datasets
            - a list of (row, col) tuples of length n (one per dataset)
            - a list of (row, col) tuples shorter than n, used for plot/inspection only (will error if not broadcastable)
        sigma
            Optional low-pass smoothing sigma (pixels) applied to each mean DP prior to peak finding.
        plot_origins
            If True, plot mean diffraction patterns with overlaid origin markers.
        plot_indices
            Optional indices to plot. If None, plots all datasets.

        Stores
        ------
        self.diffraction_origins : np.ndarray
            Array of shape (n, 2) with integer (row, col) origins.
        """
        import numpy as _np

        try:
            from scipy.ndimage import gaussian_filter as _gaussian_filter
        except Exception:  # pragma: no cover
            _gaussian_filter = None

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
            origins_arr = _np.zeros((n, 2), dtype=int)
            for i in range(n):
                dp = _np.asarray(self.dp_mean[i])
                if sigma is not None and float(sigma) > 0:
                    if _gaussian_filter is None:
                        raise ImportError("scipy is required for sigma smoothing (gaussian_filter).")
                    dp_use = _gaussian_filter(dp.astype(float, copy=False), float(sigma))
                else:
                    dp_use = dp
                ind = int(_np.argmax(dp_use))
                r, c = _np.unravel_index(ind, dp_use.shape)
                origins_arr[i, 0] = int(r)
                origins_arr[i, 1] = int(c)
        else:
            if isinstance(origins, tuple) and len(origins) == 2:
                origins_arr = _np.tile(_np.asarray(origins, dtype=int)[None, :], (n, 1))
            else:
                origins_list = list(origins)
                if len(origins_list) != n:
                    raise ValueError("origins must be a single (row,col) tuple or a list of length n.")
                origins_arr = _np.asarray(origins_list, dtype=int)
                if origins_arr.shape != (n, 2):
                    raise ValueError("origins must have shape (n, 2) after conversion.")

        self.diffraction_origins = origins_arr

        if plot_origins:
            dp_tiles = [[_np.asarray(self.dp_mean[i]) for i in plot_indices_list]]
            titles = [[f"{i} - Mean Diffraction Pattern" for i in plot_indices_list]]
            fig, axs = show_2d(dp_tiles, titles=titles, returnfig=True, **{})
            if not isinstance(axs, (list, _np.ndarray)):
                axs = [axs]
            axs_flat = _np.ravel(axs)
            for j, i in enumerate(plot_indices_list):
                ax = axs_flat[j]
                r, c = self.diffraction_origins[i]
                ax.plot([c], [r], marker="+", color="red", markersize=16, markeredgewidth=2)
            return fig, axs

        return self


    def diffraction_align(
        self,
        edge_blend = 16.0,
        padding = None,
        pad_val = 'min',
        upsample_factor = 100,
        weight_scale = 1/8,
        plot_aligned = True,
        linewidth = 2,
        **kwargs,
    ):
        """
        Refine the diffraction space origins, set padding, align images

        """

        # window function
        from scipy.signal.windows import tukey
        w = tukey(self.dp_mean[0].shape[0], alpha=2.0*edge_blend/self.dp_mean[0].shape[0])[:,None] * \
            tukey(self.dp_mean[0].shape[1], alpha=2.0*edge_blend/self.dp_mean[0].shape[1])[None,:]

        # coordinates
        r = np.fft.fftfreq(self.dp_mean[0].shape[0],1/self.dp_mean[0].shape[0])[:,None]
        c = np.fft.fftfreq(self.dp_mean[0].shape[1],1/self.dp_mean[0].shape[1])[None,:]

        # init
        self.diffraction_shifts = np.zeros((len(self.dp_mean),2))

        # correlation alignment
        G_ref = np.fft.fft2(w * self.dp_mean[0])
        xy0 = self.diffraction_origins[0]
        for ind in range(1,len(self.dp_mean)):
            G = np.fft.fft2(w * self.dp_mean[ind])
            xy = self.diffraction_origins[ind]

            dr2 = (r - xy0[0] + xy[0])**2 \
                + (c - xy0[1] + xy[1])**2
            im_weight = np.clip(1 - np.sqrt(dr2)/np.mean(self.dp_mean[0].shape)/weight_scale, 0.0, 1.0)
            im_weight = np.sin(im_weight*np.pi/2)**2

            shift, G_shift = weighted_cross_correlation_shift(
                im_ref=G_ref,
                im=G,
                weight_real=im_weight*0+1.0,
                upsample_factor = upsample_factor,
                fft_input = True,
                fft_output = True,
                return_shifted_image = True,
            )
            self.diffraction_shifts[ind,:] = shift

            # update reference
            G_ref = G_ref*(ind/(ind+1)) + G_shift/(ind+1)

        # Center shifts
        self.diffraction_shifts -= np.mean(self.diffraction_shifts,axis=0)[None,:]

        # Generate output image 

        if plot_aligned:
            im_aligned = shift_images(
                images = self.dp_mean,
                shifts_rc = self.diffraction_shifts,
                edge_blend = edge_blend,
                padding = padding,
                pad_val = pad_val,
            )
            show_2d(
                im_aligned,
                **kwargs,
            )


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
        **kwargs,
    ):
        import numpy as np
        from scipy.ndimage import gaussian_filter, shift as ndi_shift
        from scipy.signal import convolve2d
        from scipy.signal.windows import tukey

        from quantem.core.utils.imaging_utils import weighted_cross_correlation_shift
        from quantem.core.visualization import show_2d

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

        wx = None
        if edge_filter:
            wx = np.array(
                [
                    [-1.0, -2.0, -1.0],
                    [ 0.0,  0.0,  0.0],
                    [ 1.0,  2.0,  1.0],
                ],
                dtype=float,
            )

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
                    shift=(float(shifts[i, 0]), float(shifts[i, 1])),
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
                shift_method=str(shift_method),
            )
            show_2d(im_aligned, **kwargs)

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
        **kwargs,
    ):
        import warnings

        import numpy as np
        from scipy.ndimage import shift as ndi_shift
        from scipy.signal.windows import tukey
        from tqdm import tqdm

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
                raise ValueError("diffraction_pad_val must be a float or one of {'min','max','mean','median'}.")
        else:
            pad_val_dp = float(diffraction_pad_val)

        wdp_pad = np.zeros((Hp, Wp), dtype=float)
        wdp_pad[rp0 : rp0 + H, cp0 : cp0 + W] = w_dp

        wdp_shifted = np.zeros((n, Hp, Wp), dtype=float)
        if method == "fourier":
            kr = np.fft.fftfreq(Hp)[:, None]
            kc = np.fft.fftfreq(Wp)[None, :]
            ramps = []
            Fw = np.fft.fft2(wdp_pad)
            for i in range(n):
                dr, dc = float(dp_shifts[i, 0]), float(dp_shifts[i, 1])
                ramp = np.exp(-2j * np.pi * (kr * dr + kc * dc))
                ramps.append(ramp)
                w_i = np.fft.ifft2(Fw * ramp).real
                wdp_shifted[i] = np.clip(w_i, 0.0, 1.0)
        else:
            for i in range(n):
                w_i = ndi_shift(
                    wdp_pad,
                    shift=(float(dp_shifts[i, 0]), float(dp_shifts[i, 1])),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                wdp_shifted[i] = np.clip(w_i, 0.0, 1.0)
            ramps = None

        edge_w_dp = 1.0 - np.max(wdp_shifted, axis=0)
        edge_w_dp = np.clip(edge_w_dp, 0.0, 1.0)

        merged = np.zeros((Rout, Cout, Hp, Wp), dtype=np.float64)

        dp_local = np.zeros((H, W), dtype=np.float64)
        dp_pad = np.zeros((Hp, Wp), dtype=np.float64)
        dp_shifted_tmp = np.zeros((Hp, Wp), dtype=np.float64)
        num_tmp = np.zeros((Hp, Wp), dtype=np.float64)
        den_tmp = np.zeros((Hp, Wp), dtype=np.float64)

        for ro in tqdm(range(Rout), desc="Merging (rows)"):
            r_base = float(ro - real_space_padding)
            for co in range(Cout):
                c_base = float(co - real_space_padding)

                num_tmp.fill(0.0)
                den_tmp.fill(0.0)
                max_wi = 0.0

                for i in range(n):
                    r_in = r_base - float(rs_shifts[i, 0])
                    c_in = c_base - float(rs_shifts[i, 1])

                    r0 = int(np.floor(r_in))
                    c0 = int(np.floor(c_in))
                    if r0 < 0 or r0 >= Rs - 1 or c0 < 0 or c0 >= Cs - 1:
                        continue

                    dr = float(r_in - r0)
                    dc = float(c_in - c0)

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
                        dp_shifted_tmp[:] = np.fft.ifft2(np.fft.fft2(dp_pad) * ramps[i]).real
                    else:
                        dp_shifted_tmp[:] = ndi_shift(
                            dp_pad,
                            shift=(float(dp_shifts[i, 0]), float(dp_shifts[i, 1])),
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

            merged_f = merged  # float64

            if scale_output:
                peak = float(np.max(merged_f))
                if peak <= 0.0:
                    scale = 1.0
                    merged_scaled = merged_f
                else:
                    scale = dmax / peak
                    merged_scaled = merged_f * scale

                if np.issubdtype(dtype_out, np.unsignedinteger):
                    if float(np.min(merged_scaled)) < 0.0:
                        warnings.warn(
                            f"scale_output=True with unsigned dtype {dtype_out}: "
                            "negative values present; they will be clipped to 0.",
                            RuntimeWarning,
                        )
                    lo, hi = 0.0, dmax
                else:
                    lo, hi = dmin, dmax

                if float(np.min(merged_scaled)) < lo or float(np.max(merged_scaled)) > hi:
                    warnings.warn(
                        f"Output overflow for dtype {dtype_out} after scaling: "
                        f"data range [{float(np.min(merged_scaled))}, {float(np.max(merged_scaled))}] exceeds "
                        f"[{lo}, {hi}]. Values will be clipped.",
                        RuntimeWarning,
                    )

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
                titles=[["Merged Bright Field", "Merged Mean Diffraction Pattern"]],
                **kwargs,
            )

        return dataset_merged


def shift_images(
    images,
    shifts_rc,
    edge_blend: float = 8.0,
    padding=None,
    pad_val=0.0,
    shift_method: str = "bilinear",
):
    import numpy as np
    from scipy.ndimage import shift as ndi_shift
    from scipy.signal.windows import tukey

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
            pad_val = float(np.min(v))
        elif s == "max":
            pad_val = float(np.max(v))
        elif s == "mean":
            pad_val = float(np.mean(v))
        elif s == "median":
            pad_val = float(np.median(v))
        else:
            raise ValueError("pad_val must be a float or one of {'min','max','mean','median'}")
    else:
        pad_val = float(pad_val)

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
            dr, dc = float(shifts_rc[ind, 0]), float(shifts_rc[ind, 1])
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
                shift=(float(shifts_rc[ind, 0]), float(shifts_rc[ind, 1])),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            stack_w[ind] = ndi_shift(
                stack_w[ind],
                shift=(float(shifts_rc[ind, 0]), float(shifts_rc[ind, 1])),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            stack_w[ind] = np.clip(stack_w[ind], 0.0, 1.0)

    # edge_w = 1.0 - np.clip(np.max(stack_w, axis=0), 0.0, 1.0)
    edge_w = len(images) - np.sum(stack_w, axis=0)

    num = np.sum(stack, axis=0) + edge_w * pad_val
    den = np.sum(stack_w, axis=0) + edge_w
    out = num / den

    return out
