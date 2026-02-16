from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.ndimage import shift as ndi_shift
from scipy.signal.windows import tukey

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.models.base import Model, ModelContext, OriginND, Overlay, PreparedModel
from quantem.core.utils.imaging_utils import cross_correlation_shift
from quantem.core.visualization import show_2d

__all__ = ["ModelDiffraction"]


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ModelDiffraction(AutoSerialize):
    _token = object()

    def __init__(self, dataset: Any, _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError("Use ModelDiffraction.from_dataset() or .from_file().")
        super().__init__()
        self.dataset = dataset
        self.metadata: dict[str, Any] = {}
        self.index_shape: tuple[int, ...] | None = None
        self.image_ref: np.ndarray | None = None
        self.preprocess_shifts: np.ndarray | None = None
        self.model: Model | None = None
        self.prepared: PreparedModel | None = None
        self.fit_mask: np.ndarray | None = None

    @classmethod
    def from_dataset(self, dataset: Dataset2d | Dataset3d | Dataset4d | Dataset4dstem | Any) -> "ModelDiffraction":
        if isinstance(dataset, (Dataset2d, Dataset3d, Dataset4d, Dataset4dstem)):
            return ModelDiffraction(dataset=dataset, _token=ModelDiffraction._token)
        raise TypeError("from_dataset expects a Dataset2d, Dataset3d, Dataset4d, or Dataset4dstem instance.")

    def preprocess(
        self,
        *,
        align: bool = False,
        edge_blend: float = 8.0,
        upsample_factor: int = 32,
        max_shift: float | None = None,
        shift_order: int = 1,
    ) -> "ModelDiffraction":
        arr = np.asarray(self.dataset.array)
        if arr.ndim < 2:
            raise ValueError("dataset.array must have at least 2 dimensions.")
        H, W = int(arr.shape[-2]), int(arr.shape[-1])
        self.index_shape = tuple(arr.shape[:-2])

        stack = arr.reshape((-1, H, W)).astype(np.float32, copy=False)
        n = int(stack.shape[0])

        if (not align) or n <= 1:
            self.image_ref = np.mean(stack, axis=0)
            self.preprocess_shifts = None
            self.metadata["preprocess"] = {
                "align": bool(align),
                "edge_blend": float(edge_blend),
                "upsample_factor": int(upsample_factor),
                "max_shift": None if max_shift is None else float(max_shift),
                "shift_order": int(shift_order),
            }
            return self

        alpha_r = 0.0 if edge_blend <= 0 else min(1.0, 2.0 * float(edge_blend) / float(H))
        alpha_c = 0.0 if edge_blend <= 0 else min(1.0, 2.0 * float(edge_blend) / float(W))
        w = tukey(H, alpha=alpha_r)[:, None] * tukey(W, alpha=alpha_c)[None, :]
        w = w.astype(np.float32, copy=False)

        shifts = np.zeros((n, 2), dtype=np.float32)
        F_ref = np.fft.fft2(w * stack[0])

        for i in range(1, n):
            F_i = np.fft.fft2(w * stack[i])
            drc, F_shift = cross_correlation_shift(
                F_ref,
                F_i,
                upsample_factor=int(upsample_factor),
                max_shift=max_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )
            shifts[i, 0] = float(drc[0])
            shifts[i, 1] = float(drc[1])
            F_ref = F_ref * (i / (i + 1)) + F_shift / (i + 1)

        shifts -= np.mean(shifts, axis=0, keepdims=True)

        aligned = np.empty_like(stack, dtype=np.float32)
        for i in range(n):
            aligned[i] = ndi_shift(
                stack[i],
                shift=(float(shifts[i, 0]), float(shifts[i, 1])),
                order=int(shift_order),
                mode="nearest",
                prefilter=False,
            )

        self.image_ref = np.mean(aligned, axis=0)
        self.preprocess_shifts = shifts.reshape(self.index_shape + (2,))

        self.metadata["preprocess"] = {
            "align": bool(align),
            "edge_blend": float(edge_blend),
            "upsample_factor": int(upsample_factor),
            "max_shift": None if max_shift is None else float(max_shift),
            "shift_order": int(shift_order),
        }
        return self

    def _ensure_image_ref(self) -> None:
        if self.image_ref is not None:
            return
        arr = np.asarray(self.dataset.array)
        if arr.ndim < 2:
            raise ValueError("dataset.array must have at least 2 dimensions.")
        H, W = int(arr.shape[-2]), int(arr.shape[-1])
        self.index_shape = tuple(arr.shape[:-2])
        stack = arr.reshape((-1, H, W)).astype(np.float32, copy=False)
        self.image_ref = np.mean(stack, axis=0)
        self.preprocess_shifts = None

    def define_model(
        self,
        *,
        origin_row: float | tuple[float, float] | tuple[float, float, float | None],
        origin_col: float | tuple[float, float] | tuple[float, float, float | None],
        components: list[Any],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        origin_key: str = "origin",
    ) -> "ModelDiffraction":
        self._ensure_image_ref()
        if self.image_ref is None:
            raise RuntimeError("image_ref not available.")
        H, W = int(self.image_ref.shape[-2]), int(self.image_ref.shape[-1])

        dev = torch.device(device) if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32

        mask_t = None
        if mask is not None:
            if torch.is_tensor(mask):
                mask_t = mask.to(device=dev, dtype=torch.float32)
            else:
                m = np.asarray(mask)
                if m.shape != (H, W):
                    raise ValueError("mask must have shape (H, W).")
                mask_t = torch.as_tensor(m.astype(np.float32, copy=False), device=dev)

        ctx = ModelContext(H=H, W=W, device=dev, dtype=dt, mask=mask_t)

        m = Model()
        origin = OriginND.from_row_col(key=str(origin_key), origin_row=origin_row, origin_col=origin_col)
        comps = [origin] + list(components)
        m.add(comps)

        self.model = m
        self.prepared = m.compile(ctx)
        self.metadata["define_model"] = {"origin_key": str(origin_key), "device": str(dev), "dtype": str(dt)}
        return self

    def _apply_overlays(self, ax: Any, overlays: list[Overlay]) -> None:
        for ov in overlays:
            d = dict(ov.data)
            if ov.kind in {"points", "points_rc"} and ("r" in d) and ("c" in d):
                r = np.asarray(d["r"]).ravel()
                c = np.asarray(d["c"]).ravel()
                s = float(d.get("s", 60.0))
                marker = d.get("marker", "x")
                color = d.get("color", "orange")
                ax.scatter(c, r, s=s, marker=marker, c=color)
                continue

    def plot_mean_model(
        self,
        *,
        power: float = 0.25,
        returnfig: bool = False,
        show_overlays: bool = True,
        axsize: tuple[int, int] = (6, 6),
    ) -> tuple[Any, Any] | None:
        self._ensure_image_ref()
        if self.image_ref is None:
            raise RuntimeError("image_ref not available.")
        if self.prepared is None:
            raise RuntimeError("No model defined. Call .define_model(...) first.")

        ref = np.asarray(self.image_ref, dtype=np.float32)
        init = _to_numpy(self.prepared.render_initial()).astype(np.float32, copy=False)

        refp = ref if power == 1.0 else np.maximum(ref, 0.0) ** float(power)
        initp = init if power == 1.0 else np.maximum(init, 0.0) ** float(power)

        vmin = float(np.min([refp.min(), initp.min()]))
        vmax = float(np.max([refp.max(), initp.max()]))

        fig, ax = show_2d(
            [refp, initp],
            title=["image_ref", "model"],
            cmap="gray",
            cbar=False,
            returnfig=True,
            axsize=axsize,
            norm="linear_minmax",
            vmin=vmin,
            vmax=vmax,
        )

        H, W = int(ref.shape[-2]), int(ref.shape[-1])

        boundaries: list[float] = []
        for comp in getattr(self.prepared, "components", []):
            b = getattr(comp, "boundary_px", None)
            if b is not None:
                boundaries.append(float(b))

        inset = 0
        pad = 0
        if boundaries:
            pos = [b for b in boundaries if b > 0.0]
            neg = [b for b in boundaries if b < 0.0]
            if pos:
                inset = int(np.ceil(max(pos)))
            if neg:
                pad = int(np.ceil(-min(neg)))

        if isinstance(ax, np.ndarray):
            axes = list(ax.ravel())
        elif isinstance(ax, (list, tuple)):
            axes = list(ax)
        else:
            axes = [ax]

        x0 = (-pad + inset)
        x1 = (W - 1) + pad - inset
        y0 = (-pad + inset)
        y1 = (H - 1) + pad - inset

        for a in axes[:2]:
            a.set_xlim(x0, x1)
            a.set_ylim(y1, y0)

        if show_overlays:
            overlays = self.prepared.overlays()
            if len(axes) >= 1:
                self._apply_overlays(axes[0], overlays)
            if len(axes) >= 2:
                self._apply_overlays(axes[1], overlays)

        if returnfig:
            return fig, ax
        return None
