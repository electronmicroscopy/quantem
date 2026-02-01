from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.signal.windows import tukey

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d


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
        edge_blend = 8.0,
        padding = None,
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
        shifts = np.zeros((len(self.dp_mean),2))

        # correlation alignment
        G_ref = np.fft.fft2(w * self.dp_mean[0])
        xy0 = self.diffraction_origins[0]
        for ind in range(1,2):
            G = np.conj(np.fft.fft2(w * self.dp_mean[ind]))
            xy = self.diffraction_origins[ind]

            dr2 = (r - xy0[0] + xy[0])**2 \
                + (c - xy0[1] + xy[1])**2
            im_weight = np.clip(1 - np.sqrt(dr2)/np.mean(self.dp_mean[0].shape)/weight_scale, 0.0, 1.0)
            im_weight = np.sin(im_weight*np.pi/2)**2

            im_corr = np.real(np.fft.ifft2(G_ref * G)) * im_weight



        if plot_aligned:
            show_2d(
                np.fft.fftshift(im_corr),
                norm = {
                    'upper_quantile':1.0,
                },
                **kwargs,
            )


    def real_space_align(
        self
    ):
        pass


        
    def merge_datasets(
        self
    ):
        pass


        