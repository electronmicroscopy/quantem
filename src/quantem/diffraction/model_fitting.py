from __future__ import annotations

"""High-level diffraction model fitting workflow utilities."""

import warnings
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import shift as ndi_shift
from scipy.signal.windows import tukey

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.models.base import Model, ModelContext, Overlay, PreparedModel
from quantem.core.models.diffraction import Origin2D
from quantem.core.utils.imaging_utils import cross_correlation_shift
from quantem.core.visualization import show_2d


def _to_numpy(x: Any) -> np.ndarray:
    """Convert arrays/tensors to NumPy arrays."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ModelDiffraction(AutoSerialize):
    """
    End-to-end helper for defining and fitting additive diffraction forward models.

    This class wraps a diffraction dataset, builds an average reference image
    (`image_ref`), compiles a composable component model, and provides optimization
    routines for:
    - fitting to the mean reference image,
    - fitting selected individual diffraction patterns.

    Features
    --------
    - Build a mean reference image with optional stack alignment.
    - Define a composable model from origin/background/template/lattice components.
    - Refine a mean model with Adam or L-BFGS.
    - Fit all or selected patterns with optional progress bars.
    - Plot reference/model comparisons and component overlays.

    Typical workflow
    ----------------
    >>> md = ModelDiffraction.from_dataset(ds).preprocess().define_model(...)
    >>> md.refine_mean_model(...)
    >>> md.fit_all_patterns(...)
    >>> md.plot_mean_model(...)
    """

    _token = object()

    def __init__(self, dataset: Any, _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError("Use ModelDiffraction.from_dataset() or .from_file().")
        super().__init__()
        self.dataset = dataset
        self.metadata: dict[str, Any] = {}
        self.image_ref: np.ndarray | None = None
        self.preprocess_shifts: np.ndarray | None = None
        self.index_shape: tuple[int, ...] | None = None
        self.model: Model | None = None
        self.prepared: PreparedModel | None = None
        self.x_mean: torch.Tensor | None = None
        self.x_defined: torch.Tensor | None = None
        self.x_initial: torch.Tensor | None = None
        self.mean_refined: bool = False
        self.x_patterns: torch.Tensor | None = None
        self.pattern_fit_losses: np.ndarray | None = None
        self.pattern_fit_linear_indices: np.ndarray | None = None
        self.pattern_fit_indices: list[tuple[int, ...]] | None = None

    @staticmethod
    def _weak_softplus(x: torch.Tensor, *, scale: float) -> torch.Tensor:
        s = torch.as_tensor(float(scale), device=x.device, dtype=x.dtype)
        return torch.nn.functional.softplus(x / s) * s

    @classmethod
    def _apply_intensity_transform(
        cls, x: torch.Tensor, *, mode: str, weak_softplus_scale: float
    ) -> torch.Tensor:
        m = str(mode).lower()
        if m == "none":
            return x
        if m == "weak_softplus":
            return cls._weak_softplus(x, scale=weak_softplus_scale)
        raise ValueError("intensity_transform must be one of: 'none', 'weak_softplus'.")

    @classmethod
    def from_dataset(cls, dataset: Dataset2d | Dataset3d | Dataset4d | Dataset4dstem | Any) -> "ModelDiffraction":
        """
        Construct a ModelDiffraction object from a QuantEM dataset container.

        Parameters
        ----------
        dataset
            Dataset2d, Dataset3d, Dataset4d, or Dataset4dstem instance.

        Returns
        -------
        ModelDiffraction
            New model-fitting helper bound to the provided dataset.
        """
        if isinstance(dataset, (Dataset2d, Dataset3d, Dataset4d, Dataset4dstem)):
            return cls(dataset=dataset, _token=cls._token)
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
        """
        Precompute the mean reference image used for model fitting.

        Parameters
        ----------
        align
            If True, align the flattened pattern stack before averaging.
        edge_blend
            Tukey edge taper width (pixels) used for robust FFT alignment.
        upsample_factor
            Sub-pixel alignment upsampling factor for cross-correlation shift.
        max_shift
            Optional maximum shift magnitude during alignment.
        shift_order
            Interpolation order used when applying shifts to patterns.

        Returns
        -------
        ModelDiffraction
            Returns self.

        Notes
        -----
        - `dataset.array` is interpreted as `(..., H, W)`, where leading dimensions
          are flattened into a pattern stack.
        - The computed stack-average is stored in `self.image_ref`.
        - If `align=False`, preprocessing is a direct mean over stack elements.
        """
        arr = np.asarray(self.dataset.array)
        if arr.ndim < 2:
            raise ValueError("dataset.array must have at least 2 dimensions.")
        H, W = arr.shape[-2], arr.shape[-1]
        self.index_shape = tuple(arr.shape[:-2])

        stack = arr.reshape((-1, H, W)).astype(np.float32, copy=False)
        n = stack.shape[0]

        if not align or n <= 1:
            self.image_ref = np.mean(stack, axis=0)
            self.preprocess_shifts = None
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
        return self

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
        """
        Define and compile a diffraction model against `image_ref`.

        Parameters
        ----------
        origin_row, origin_col
            Initial origin parameter specification. Supported forms are:
            - scalar: fixed initial value with no explicit bounds
            - `(value, deviation)`: symmetric bounds `(value - deviation, value + deviation)`
            - `(value, lower_bound, upper_bound)`: explicit bounds
        components
            Sequence of model components (e.g. `DiskTemplate`, backgrounds, lattice).
            Components are rendered additively in the provided order.
        device
            Torch device used for compiled parameters and rendering.
        dtype
            Torch dtype used for compiled parameters and rendering.
        mask
            Optional `(H, W)` mask for weighted loss during optimization.
        origin_key
            Name used to register/retrieve the origin component in context fields.

        Returns
        -------
        ModelDiffraction
            Returns self with compiled model state.

        Notes
        -----
        - If `image_ref` is missing, `preprocess()` is run automatically.
        - `Origin2D` is inserted automatically before user components.
        - Component dependency ordering still matters for shared context fields:
          for example, `DiskTemplate` should appear before `SyntheticDiskLattice`
          when the lattice references that template.
        - This method resets fit state (`x_defined`, `x_initial`, `x_mean`,
          `x_patterns`, and pattern-fit metadata).
        """
        if self.image_ref is None:
            self.preprocess()

        if self.image_ref is None:
            raise RuntimeError("image_ref not available.")
        H, W = int(self.image_ref.shape[0]), int(self.image_ref.shape[1])

        dev = torch.device(device) if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32

        mask_t = None
        if mask is not None:
            if torch.is_tensor(mask):
                mask_t = mask.to(device=dev, dtype=dt)
            else:
                m = np.asarray(mask)
                if m.shape != (H, W):
                    raise ValueError("mask must have shape (H, W).")
                mask_t = torch.as_tensor(m.astype(np.float32, copy=False), device=dev, dtype=dt)

        ctx = ModelContext(H=H, W=W, device=dev, dtype=dt, mask=mask_t, fields={})
        m = Model()
        m.add([Origin2D(origin_key=str(origin_key), row=origin_row, col=origin_col)])
        m.add(list(components))

        self.model = m
        self.prepared = m.compile(ctx)
        self.x_defined = self.prepared.x0.detach().clone()
        self.x_initial = self.x_defined.detach().clone()
        self.x_mean = self.x_initial.detach().clone()
        self.mean_refined = False
        self.x_patterns = None
        self.pattern_fit_losses = None
        self.pattern_fit_linear_indices = None
        self.pattern_fit_indices = None
        return self

    def _fit_target_image(
        self,
        *,
        target: torch.Tensor,
        x_start: torch.Tensor,
        n_steps: int,
        lr: float,
        method: str,
        power: float | None,
        fit_disk_pixels: bool | None,
        fit_only_disk_pixels: bool,
        enforce_disk_max_one: bool,
        enforce_disk_center_of_mass: bool,
        intensity_transform: str,
        weak_softplus_scale: float,
        progress: bool = False,
        progress_desc: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        if self.prepared is None:
            raise RuntimeError("Call .define_model(...) first.")

        ctx = self.prepared.ctx
        lb = self.prepared.lb
        ub = self.prepared.ub

        x = x_start.detach().clone().to(device=ctx.device, dtype=ctx.dtype)
        x.requires_grad_(True)

        if fit_disk_pixels is None:
            fit_disk_pixels = any(p.tags.get("role") == "disk_pixel" for p in self.prepared.params)

        freeze = torch.zeros_like(x, dtype=torch.bool)
        disk_mask = torch.zeros_like(x, dtype=torch.bool)
        for p in self.prepared.params:
            if p.tags.get("role") == "disk_pixel":
                disk_mask[p.index] = True

        # Cache template groups for optional per-step projection constraints.
        disk_templates = self.prepared.ctx.fields.get("disk_templates", {})
        disk_param_groups: list[dict[str, Any]] = []
        if enforce_disk_max_one or enforce_disk_center_of_mass:
            grouped: dict[str, list[tuple[int, int]]] = {}
            for p in self.prepared.params:
                if p.tags.get("role") != "disk_pixel":
                    continue
                name = str(p.tags.get("disk"))
                i_flat = int(p.tags.get("i"))
                grouped.setdefault(name, []).append((i_flat, int(p.index)))

            for name, pairs in grouped.items():
                if name not in disk_templates:
                    continue
                dmeta = disk_templates[name]
                shape = dmeta.get("shape", None)
                if shape is None:
                    continue
                Ht, Wt = int(shape[0]), int(shape[1])
                order = sorted(pairs, key=lambda t: t[0])
                flat_i = torch.as_tensor([t[0] for t in order], device=ctx.device, dtype=torch.long)
                p_idx = torch.as_tensor([t[1] for t in order], device=ctx.device, dtype=torch.long)
                dr = dmeta["dr"][flat_i]
                dc = dmeta["dc"][flat_i]
                disk_param_groups.append(
                    {
                        "param_idx": p_idx,
                        "shape": (Ht, Wt),
                        "dr": dr,
                        "dc": dc,
                    }
                )

        if fit_only_disk_pixels:
            if not fit_disk_pixels:
                raise ValueError("fit_only_disk_pixels=True requires fit_disk_pixels=True.")
            freeze[:] = True
            freeze[disk_mask] = False
        elif not fit_disk_pixels:
            freeze[disk_mask] = True
        x_frozen = x.detach().clone()

        target_t = target.to(device=ctx.device, dtype=ctx.dtype)
        target_t = self._apply_intensity_transform(
            target_t, mode=intensity_transform, weak_softplus_scale=weak_softplus_scale
        )
        if power is not None:
            target_t = torch.clamp(target_t, min=0.0) ** float(power)

        def clamp_inplace() -> None:
            with torch.no_grad():
                x.data = torch.max(torch.min(x.data, ub), lb)
                if torch.any(freeze):
                    x.data[freeze] = x_frozen[freeze]
                if (enforce_disk_max_one or enforce_disk_center_of_mass) and disk_param_groups:
                    eps = torch.as_tensor(1e-12, device=ctx.device, dtype=ctx.dtype)
                    for g in disk_param_groups:
                        p_idx = g["param_idx"]
                        if torch.all(freeze[p_idx]):
                            continue
                        vals = x.data[p_idx]
                        vals = torch.clamp(vals, min=0.0)

                        if enforce_disk_center_of_mass:
                            mass = torch.sum(vals)
                            if mass > eps:
                                r_com = torch.sum(vals * g["dr"]) / mass
                                c_com = torch.sum(vals * g["dc"]) / mass
                                Ht, Wt = g["shape"]
                                img = vals.reshape(Ht, Wt)[None, None, :, :]
                                yy = torch.linspace(-1.0, 1.0, Ht, device=ctx.device, dtype=ctx.dtype)
                                xx = torch.linspace(-1.0, 1.0, Wt, device=ctx.device, dtype=ctx.dtype)
                                gy, gx = torch.meshgrid(yy, xx, indexing="ij")
                                sx = gx + (2.0 * c_com / max(Wt - 1, 1))
                                sy = gy + (2.0 * r_com / max(Ht - 1, 1))
                                grid = torch.stack((sx, sy), dim=-1)[None, :, :, :]
                                img_shift = F.grid_sample(
                                    img,
                                    grid,
                                    mode="bilinear",
                                    padding_mode="zeros",
                                    align_corners=True,
                                )
                                vals = torch.clamp(img_shift[0, 0].reshape(-1), min=0.0)

                        if enforce_disk_max_one:
                            vmax = torch.max(vals)
                            if vmax > eps:
                                vals = vals / vmax

                        x.data[p_idx] = vals

        def loss_fn() -> torch.Tensor:
            clamp_inplace()
            pred = self.prepared.render(x)
            pred = self._apply_intensity_transform(
                pred, mode=intensity_transform, weak_softplus_scale=weak_softplus_scale
            )
            if power is not None:
                pred = torch.clamp(pred, min=0.0) ** float(power)
            if ctx.mask is not None:
                m = ctx.mask
                diff = (pred - target_t) * m
                denom = torch.clamp(torch.sum(m), min=1.0)
                return torch.sum(diff * diff) / denom
            diff = pred - target_t
            return torch.mean(diff * diff)

        if method == "adam":
            opt = torch.optim.Adam([x], lr=lr)
            step_iter: Any = range(int(n_steps))
            if progress:
                try:
                    from tqdm.auto import trange

                    step_iter = trange(int(n_steps), desc=progress_desc or "Refining", leave=False)
                except Exception:
                    warnings.warn("progress=True requested but tqdm is unavailable.", stacklevel=2)
            for _ in step_iter:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn()
                loss.backward()
                opt.step()
                clamp_inplace()
        elif method == "lbfgs":
            opt = torch.optim.LBFGS([x], lr=lr, max_iter=int(n_steps), line_search_fn="strong_wolfe")

            def closure() -> torch.Tensor:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn()
                loss.backward()
                return loss

            opt.step(closure)
            clamp_inplace()
        else:
            raise ValueError("method must be one of: 'lbfgs', 'adam'.")

        with torch.no_grad():
            final_loss = float(loss_fn().detach().cpu())
        return x.detach().clone(), final_loss

    def _resolve_pattern_indices(self, indices: Any, n: int, index_shape: tuple[int, ...]) -> np.ndarray:
        if indices is None:
            out = np.arange(n, dtype=np.int64)
        elif isinstance(indices, (int, np.integer)):
            out = np.asarray([int(indices)], dtype=np.int64)
        elif isinstance(indices, slice):
            out = np.arange(n, dtype=np.int64)[indices]
        elif isinstance(indices, tuple) and all(isinstance(i, (int, np.integer)) for i in indices):
            if len(index_shape) == 0:
                if len(indices) != 0:
                    raise ValueError("indices tuple must be empty for single-pattern datasets.")
                out = np.asarray([0], dtype=np.int64)
            else:
                out = np.asarray([np.ravel_multi_index(tuple(int(i) for i in indices), index_shape)], dtype=np.int64)
        elif isinstance(indices, tuple):
            if len(index_shape) == 0:
                raise ValueError("slice tuple indices are not valid for single-pattern datasets.")
            grid = np.arange(n, dtype=np.int64).reshape(index_shape)
            out = np.asarray(grid[indices], dtype=np.int64).ravel()
        elif isinstance(indices, np.ndarray) and indices.dtype == np.bool_:
            if indices.shape != index_shape:
                raise ValueError(f"Boolean mask must have shape {index_shape}.")
            out = np.flatnonzero(indices.ravel()).astype(np.int64, copy=False)
        elif isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
            seq = list(indices)
            if len(seq) == 0:
                out = np.asarray([], dtype=np.int64)
            elif isinstance(seq[0], (tuple, list, np.ndarray)):
                if len(index_shape) == 0:
                    raise ValueError("multi-index selection is not valid for single-pattern datasets.")
                out = np.asarray(
                    [np.ravel_multi_index(tuple(int(j) for j in i), index_shape) for i in seq],
                    dtype=np.int64,
                )
            else:
                out = np.asarray([int(i) for i in seq], dtype=np.int64)
        else:
            raise TypeError("Unsupported indices type for fit_all_patterns.")

        if out.ndim != 1:
            out = out.ravel()
        if np.any(out < 0) or np.any(out >= n):
            raise IndexError(f"indices must be in [0, {n - 1}].")
        return out

    def refine_mean_model(
        self,
        *,
        n_steps: int = 50,
        lr: float = 1e-3,
        method: str = "adam",
        power: float | None = 1.0,
        fit_disk_pixels: bool | None = None,
        fit_only_disk_pixels: bool = False,
        enforce_disk_max_one: bool = True,
        enforce_disk_center_of_mass: bool = True,
        warmup_disk_steps: int = 0,
        overwrite_initial: bool = True,
        intensity_transform: str = "none",
        weak_softplus_scale: float = 1e-3,
        progress: bool = False,
    ) -> "ModelDiffraction":
        """
        Refine model parameters against the mean reference image.

        Parameters
        ----------
        n_steps
            Number of optimization steps/iterations for the main phase.
        lr
            Optimizer learning rate.
        method
            Optimizer name: `"adam"` or `"lbfgs"`.
        power
            Optional power-law transform applied to both target and prediction
            before computing MSE loss.
        fit_disk_pixels
            Controls whether `disk_pixel` parameters are trainable. If None,
            inferred from model parameters.
        fit_only_disk_pixels
            If True, freeze all non-disk parameters.
        warmup_disk_steps
            Optional number of disk-only warmup steps run before the main phase.
        overwrite_initial
            If True, store refined parameters as new initial parameters for
            subsequent per-pattern fitting.
        intensity_transform
            Optional intensity transform (`"none"` or `"weak_softplus"`) applied
            before the power transform and loss evaluation.
        weak_softplus_scale
            Scale parameter used when `intensity_transform="weak_softplus"`.
        progress
            If True, show a tqdm progress bar (when tqdm is available).

        Returns
        -------
        ModelDiffraction
            Returns self with updated `x_mean` (and optionally `x_initial`).
        """
        method = str(method).lower()

        if self.image_ref is None:
            self.preprocess()
        if self.image_ref is None or self.prepared is None or self.x_mean is None or self.x_initial is None:
            raise RuntimeError("Call .define_model(...) first.")

        ctx = self.prepared.ctx
        target = torch.as_tensor(self.image_ref, device=ctx.device, dtype=ctx.dtype)
        x_start = self.x_initial if overwrite_initial else self.x_mean
        if int(warmup_disk_steps) > 0:
            x_start, _ = self._fit_target_image(
                target=target,
                x_start=x_start,
                n_steps=int(warmup_disk_steps),
                lr=float(lr),
                method=method,
                power=power,
                fit_disk_pixels=True,
                fit_only_disk_pixels=True,
                enforce_disk_max_one=bool(enforce_disk_max_one),
                enforce_disk_center_of_mass=bool(enforce_disk_center_of_mass),
                intensity_transform=intensity_transform,
                weak_softplus_scale=float(weak_softplus_scale),
                progress=bool(progress),
                progress_desc="Refine mean model (disk warmup)",
            )
        x_fit, _ = self._fit_target_image(
            target=target,
            x_start=x_start,
            n_steps=int(n_steps),
            lr=float(lr),
            method=method,
            power=power,
            fit_disk_pixels=fit_disk_pixels,
            fit_only_disk_pixels=bool(fit_only_disk_pixels),
            enforce_disk_max_one=bool(enforce_disk_max_one),
            enforce_disk_center_of_mass=bool(enforce_disk_center_of_mass),
            intensity_transform=intensity_transform,
            weak_softplus_scale=float(weak_softplus_scale),
            progress=bool(progress),
            progress_desc="Refine mean model",
        )

        self.x_mean = x_fit
        self.mean_refined = True
        if overwrite_initial:
            self.x_initial = x_fit.detach().clone()
        return self

    def fit_all_patterns(
        self,
        *,
        indices: Any = None,
        use_refined_init: bool = True,
        strict_refined_init: bool = False,
        n_steps: int = 50,
        lr: float = 1e-3,
        method: str = "adam",
        power: float | None = 1.0,
        fit_disk_pixels: bool | None = None,
        fit_only_disk_pixels: bool = False,
        enforce_disk_max_one: bool = True,
        enforce_disk_center_of_mass: bool = True,
        intensity_transform: str = "none",
        weak_softplus_scale: float = 1e-3,
        progress: bool = False,
    ) -> "ModelDiffraction":
        """
        Fit selected diffraction patterns using the compiled model.

        Parameters
        ----------
        indices
            Pattern selector. Supported forms include None (all patterns),
            integer, slice, tuple indexing, list of linear indices, list of
            multi-indices, or boolean mask shaped like scan dimensions.
        use_refined_init
            If True, initialize per-pattern fitting from `x_initial`.
        strict_refined_init
            If True, raise when refined init is requested before mean refinement.
            If False, emit warning and fall back to defined initialization.
        n_steps, lr, method, power, fit_disk_pixels, fit_only_disk_pixels
            Optimization settings analogous to `refine_mean_model`.
        intensity_transform, weak_softplus_scale
            Prediction/target intensity transform options.
        progress
            If True, show a tqdm progress bar over selected patterns.

        Returns
        -------
        ModelDiffraction
            Returns self with:
            - `x_patterns`: fitted parameter vectors `(n_selected, n_params)`
            - `pattern_fit_losses`: per-pattern final losses
            - index bookkeeping for selected patterns.
        """
        method = str(method).lower()

        if self.prepared is None or self.x_defined is None or self.x_initial is None:
            raise RuntimeError("Call .define_model(...) first.")

        arr = np.asarray(self.dataset.array)
        if arr.ndim < 2:
            raise ValueError("dataset.array must have at least 2 dimensions.")
        H, W = arr.shape[-2], arr.shape[-1]
        index_shape = tuple(arr.shape[:-2])
        stack = arr.reshape((-1, H, W)).astype(np.float32, copy=False)
        n = int(stack.shape[0])

        linear = self._resolve_pattern_indices(indices=indices, n=n, index_shape=index_shape)
        if linear.size == 0:
            raise ValueError("No patterns selected for fitting.")

        if use_refined_init:
            if not self.mean_refined:
                msg = "fit_all_patterns(use_refined_init=True) was requested before refine_mean_model()."
                if strict_refined_init:
                    raise RuntimeError(msg)
                warnings.warn(f"{msg} Falling back to defined initial parameters.", stacklevel=2)
                x_seed = self.x_defined
            else:
                x_seed = self.x_initial
        else:
            x_seed = self.x_defined

        n_sel = int(linear.size)
        x_fit_all = torch.empty(
            (n_sel, self.x_defined.numel()),
            device=self.prepared.ctx.device,
            dtype=self.prepared.ctx.dtype,
        )
        losses = np.empty((n_sel,), dtype=np.float32)

        pat_iter: Any = enumerate(linear)
        if progress:
            try:
                from tqdm.auto import tqdm

                pat_iter = enumerate(tqdm(linear, desc="Fit patterns", leave=False))
            except Exception:
                warnings.warn("progress=True requested but tqdm is unavailable.", stacklevel=2)

        for j, i_lin in pat_iter:
            target = torch.as_tensor(stack[int(i_lin)], device=self.prepared.ctx.device, dtype=self.prepared.ctx.dtype)
            x_fit, loss = self._fit_target_image(
                target=target,
                x_start=x_seed,
                n_steps=int(n_steps),
                lr=float(lr),
                method=method,
                power=power,
                fit_disk_pixels=fit_disk_pixels,
                fit_only_disk_pixels=bool(fit_only_disk_pixels),
                enforce_disk_max_one=bool(enforce_disk_max_one),
                enforce_disk_center_of_mass=bool(enforce_disk_center_of_mass),
                intensity_transform=intensity_transform,
                weak_softplus_scale=float(weak_softplus_scale),
                progress=False,
            )
            x_fit_all[j] = x_fit
            losses[j] = float(loss)

        self.x_patterns = x_fit_all
        self.pattern_fit_losses = losses
        self.pattern_fit_linear_indices = linear
        if len(index_shape) == 0:
            self.pattern_fit_indices = [tuple() for _ in linear]
        else:
            self.pattern_fit_indices = [tuple(int(k) for k in np.unravel_index(int(i), index_shape)) for i in linear]
        return self

    def _apply_overlays(self, ax: Any, overlays: list[Overlay]) -> None:
        for ov in overlays:
            if ov.kind != "points_rc":
                continue
            d = dict(ov.data)
            r = _to_numpy(d["r"]).ravel()
            c = _to_numpy(d["c"]).ravel()
            ax.scatter(
                c,
                r,
                s=float(d.get("s", 60.0)),
                marker=d.get("marker", "x"),
                color=d.get("color", "orange"),
            )

    def plot_mean_model(
        self,
        *,
        power: float = 0.25,
        returnfig: bool = False,
        show_overlays: bool = True,
        axsize: tuple[int, int] = (6, 6),
    ) -> tuple[Any, Any] | None:
        """
        Plot `image_ref` and the current mean-model prediction side-by-side.

        Parameters
        ----------
        power
            Display transform exponent applied to both reference and model images.
        returnfig
            If True, return `(fig, ax)` from `show_2d`.
        show_overlays
            If True, draw component overlays (e.g., origin and lattice markers).
        axsize
            Base axis size passed to the plotting helper.

        Returns
        -------
        tuple[Any, Any] | None
            `(fig, ax)` if `returnfig=True`, else None.
        """
        if self.image_ref is None:
            self.preprocess()
        if self.image_ref is None or self.prepared is None:
            raise RuntimeError("Call .define_model(...) first.")
        if self.x_mean is None:
            self.x_mean = self.prepared.x0.detach().clone()

        ref = np.asarray(self.image_ref, dtype=np.float32)
        mod = _to_numpy(self.prepared.render(self.x_mean)).astype(np.float32, copy=False)

        refp = ref if power == 1.0 else np.maximum(ref, 0.0) ** float(power)
        modp = mod if power == 1.0 else np.maximum(mod, 0.0) ** float(power)

        vmin = float(min(refp.min(), modp.min()))
        vmax = float(max(refp.max(), modp.max()))

        fig, ax = show_2d(
            [refp, modp],
            title=["image_ref", "model"],
            cmap="gray",
            cbar=False,
            returnfig=True,
            axsize=axsize,
            vmin=vmin,
            vmax=vmax,
        )

        H, W = ref.shape[-2], ref.shape[-1]
        pad = 0
        boundaries = []
        for c in getattr(self.prepared, "components", []):
            b = getattr(c, "boundary_px", None)
            if b is not None:
                boundaries.append(float(b))
        if boundaries:
            min_b = float(np.min(boundaries))
            if min_b < 0.0:
                pad = int(np.ceil(-min_b))

        axes: list[Any]
        if isinstance(ax, np.ndarray):
            axes = list(ax.ravel())
        elif isinstance(ax, (list, tuple)):
            axes = list(ax)
        else:
            axes = [ax]

        for a in axes[:2]:
            a.set_xlim(-pad, (W - 1) + pad)
            a.set_ylim((H - 1) + pad, -pad)

        if show_overlays:
            ovs = self.prepared.overlays(self.x_mean)
            if len(axes) >= 1:
                self._apply_overlays(axes[0], ovs)
            if len(axes) >= 2:
                self._apply_overlays(axes[1], ovs)

        if returnfig:
            return fig, ax
        return None
