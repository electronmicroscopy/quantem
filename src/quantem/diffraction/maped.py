from __future__ import annotations

from typing import Any, Sequence

import numpy as np

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
        *,
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
            show_2d(
                [
                    [self.im_bf[i] / self.scales[i] for i in range(n)],
                    [self.dp_mean[i] for i in range(n)],
                ],
                **plot_kwargs,
            )

        return self
