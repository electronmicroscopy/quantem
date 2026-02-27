from __future__ import annotations

import warnings
from typing import Any, Literal, Sequence, cast

import numpy as np
import torch
from scipy.ndimage import shift as ndi_shift
from scipy.signal.windows import tukey
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d, Dataset4dstem
from quantem.core.fitting.base import AdditiveRenderModel, RenderComponent, RenderContext
from quantem.core.fitting.diffraction import OriginND
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.utils.imaging_utils import cross_correlation_shift
from quantem.core.visualization import show_2d


def _parse_init(value: float | int | Sequence[float | int | None], *, name: str) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"{name} cannot be empty.")
        if value[0] is None:
            raise ValueError(f"{name} initial value cannot be None.")
        return float(value[0])
    return float(cast(float | int, value))


class ModelDiffraction(OptimizerMixin, AutoSerialize):
    _token = object()
    DEFAULT_LR = 1e-3
    DEFAULT_OPTIMIZER_TYPE = "adamw"

    def __init__(self, dataset: Any, _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError("Use ModelDiffraction.from_dataset() or .from_file().")
        AutoSerialize.__init__(self)
        OptimizerMixin.__init__(self)
        self.dataset = dataset
        self.metadata: dict[str, Any] = {}
        self.image_ref: np.ndarray | None = None
        self.preprocess_shifts: np.ndarray | None = None
        self.index_shape: tuple[int, ...] | None = None

        self.ctx: RenderContext | None = None
        self.model: AdditiveRenderModel | None = None
        self.target_mean: torch.Tensor | None = None

        self.state_initialized: torch.Tensor | None = None
        self.state_mean_refined: torch.Tensor | None = None
        self.state_current: torch.Tensor | None = None
        self.mean_refined: bool = False
        self.mean_fit_history: list[float] = []
        self.mean_fit_lrs: list[float] = []

    @classmethod
    def from_dataset(
        cls, dataset: Dataset2d | Dataset3d | Dataset4d | Dataset4dstem | Any
    ) -> "ModelDiffraction":
        if isinstance(dataset, (Dataset2d, Dataset3d, Dataset4d, Dataset4dstem)):
            return cls(dataset=dataset, _token=cls._token)
        raise TypeError(
            "from_dataset expects a Dataset2d, Dataset3d, Dataset4d, or Dataset4dstem instance."
        )

    def get_optimization_parameters(self) -> Any:
        if self.model is None:
            return []
        return self.model.parameters()

    def _get_model_state_vector(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return parameters_to_vector(self.model.parameters()).detach().clone()

    def _load_model_state_vector(self, state: torch.Tensor) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        dst = next(self.model.parameters(), None)
        if dst is None:
            raise RuntimeError("Model has no parameters.")
        vec = state.detach().clone().to(device=dst.device, dtype=dst.dtype)
        vector_to_parameters(vec, self.model.parameters())

    def _clear_histories(self) -> None:
        for name in ("mean_fit_history", "mean_fit_lrs", "fit_history", "fit_lrs"):
            if hasattr(self, name):
                val = getattr(self, name)
                if isinstance(val, list):
                    val.clear()

    def _render_state_array(self, state: torch.Tensor) -> np.ndarray:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        live = self._get_model_state_vector()
        try:
            self._load_model_state_vector(state)
            arr = self.model(self.ctx).cpu().detach().numpy()
            # arr = to_numpy(self.model(self.ctx)).astype(np.float32, copy=False)
        finally:
            self._load_model_state_vector(live)
        return arr

    @property
    def render_initialized(self) -> np.ndarray:
        if self.state_initialized is None:
            raise RuntimeError("initialized state is unavailable. Call .define_model(...) first.")
        return self._render_state_array(self.state_initialized)

    @property
    def render_mean_refined(self) -> np.ndarray:
        if self.state_mean_refined is None:
            raise RuntimeError(
                "mean_refined state is unavailable. Run .fit_mean_diffraction_pattern(...) first."
            )
        return self._render_state_array(self.state_mean_refined)

    @property
    def render_current(self) -> np.ndarray:
        if self.state_current is None:
            raise RuntimeError("current state is unavailable. Call .define_model(...) first.")
        return self._render_state_array(self.state_current)

    def reset(
        self, reset_to: Literal["initialized", "mean_refined"] = "mean_refined"
    ) -> "ModelDiffraction":
        if reset_to == "initialized":
            state = self.state_initialized
            if state is None:
                raise RuntimeError(
                    "initialized state is unavailable. Call .define_model(...) first."
                )
        elif reset_to == "mean_refined":
            state = self.state_mean_refined
            if state is None:
                raise RuntimeError(
                    "mean_refined state is unavailable. Run .fit_mean_diffraction_pattern(...) first."
                )
        else:
            raise ValueError("reset_to must be 'initialized' or 'mean_refined'.")

        self._load_model_state_vector(state)
        self.state_current = self._get_model_state_vector()
        self._clear_histories()
        return self

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
        h, w = arr.shape[-2], arr.shape[-1]
        self.index_shape = tuple(arr.shape[:-2])

        stack = arr.reshape((-1, h, w)).astype(np.float32, copy=False)
        n = stack.shape[0]
        if not align or n <= 1:
            self.image_ref = np.mean(stack, axis=0)
            self.preprocess_shifts = None
            return self

        alpha_r = 0.0 if edge_blend <= 0 else min(1.0, 2.0 * float(edge_blend) / float(h))
        alpha_c = 0.0 if edge_blend <= 0 else min(1.0, 2.0 * float(edge_blend) / float(w))
        window = tukey(h, alpha=alpha_r)[:, None] * tukey(w, alpha=alpha_c)[None, :]
        window = window.astype(np.float32, copy=False)

        shifts = np.zeros((n, 2), dtype=np.float32)
        fft_ref = np.fft.fft2(window * stack[0])
        for i in range(1, n):
            fft_i = np.fft.fft2(window * stack[i])
            drc, fft_shift = cross_correlation_shift(
                fft_ref,
                fft_i,
                upsample_factor=int(upsample_factor),
                max_shift=max_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )
            if not isinstance(drc, (list, tuple, np.ndarray)) or len(drc) < 2:
                raise RuntimeError("cross_correlation_shift returned an invalid shift vector.")
            shifts[i, 0] = float(drc[0])
            shifts[i, 1] = float(drc[1])
            fft_ref = fft_ref * (i / (i + 1)) + fft_shift / (i + 1)

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
        origin_row: float | Sequence[float],
        origin_col: float | Sequence[float],
        components: list[RenderComponent],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        origin_key: str = "origin",
    ) -> "ModelDiffraction":
        if self.image_ref is None:
            self.preprocess()
        if self.image_ref is None:
            raise RuntimeError("image_ref not available.")

        h, w = int(self.image_ref.shape[0]), int(self.image_ref.shape[1])
        dev = torch.device(device) if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32

        mask_t = None
        if mask is not None:
            mask_t = (
                mask.to(device=dev, dtype=dt)
                if torch.is_tensor(mask)
                else torch.as_tensor(mask, device=dev, dtype=dt)
            )
            if tuple(mask_t.shape) != (h, w):
                raise ValueError("mask must have shape (H, W).")

        origin = OriginND(
            ndim=2,
            init=[
                _parse_init(origin_row, name="origin_row"),
                _parse_init(origin_col, name="origin_col"),
            ],
        )
        origin._quantem_origin_key = str(origin_key)  # type: ignore[attr-defined]

        for component in components:
            if hasattr(component, "set_origin"):
                component.set_origin(origin)  # type: ignore[misc]
            elif hasattr(component, "origin") and getattr(component, "origin") is None:
                component.origin = origin  # type: ignore[attr-defined]

        self.model = AdditiveRenderModel(origin=origin, components=list(components)).to(
            device=dev, dtype=dt
        )
        self.ctx = RenderContext(shape=(h, w), device=dev, dtype=dt, mask=mask_t, fields={})
        self.target_mean = torch.as_tensor(self.image_ref, device=dev, dtype=dt)

        x0 = parameters_to_vector(self.model.parameters()).detach().clone()
        self.state_initialized = x0.detach().clone()
        self.state_current = x0.detach().clone()
        self.state_mean_refined = None
        self.mean_refined = False
        self._clear_histories()
        self.remove_optimizer()
        return self

    def fit_mean_diffraction_pattern(
        self,
        *,
        n_steps: int = 200,
        reset: bool | Literal["initialized", "mean_refined"] = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraint_weight: float = 1.0,
        progress: bool = False,
        **kwargs: Any,
    ) -> "ModelDiffraction":
        if self.model is None or self.ctx is None or self.target_mean is None:
            raise RuntimeError("Call .define_model(...) first.")
        if reset is True:
            self.reset("initialized")
        elif isinstance(reset, str):
            if reset not in ("initialized", "mean_refined"):
                raise ValueError("reset must be False, True, 'initialized', or 'mean_refined'.")
            self.reset(reset_to=cast(Literal["initialized", "mean_refined"], reset))
        elif reset not in (False,):
            raise ValueError("reset must be False, True, 'initialized', or 'mean_refined'.")

        optimizer_rebuilt = False
        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)
            optimizer_rebuilt = True
        elif self.optimizer is None:
            if self.optimizer_params:
                self.set_optimizer(self.optimizer_params)
            else:
                self.set_optimizer({"type": self.DEFAULT_OPTIMIZER_TYPE, "lr": self.DEFAULT_LR})
            optimizer_rebuilt = True

        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iter=int(n_steps))
        elif self.scheduler is None and self.scheduler_params:
            self.set_scheduler(self.scheduler_params, num_iter=int(n_steps))
        elif optimizer_rebuilt and self.scheduler is not None and self.optimizer is not None:
            self.scheduler.optimizer = self.optimizer

        iterator: Any = range(int(n_steps))
        if progress:
            try:
                from tqdm.auto import trange

                iterator = trange(int(n_steps), desc="Fit mean diffraction", leave=False)
            except Exception:
                warnings.warn("progress=True requested but tqdm is unavailable.", stacklevel=2)

        for _ in iterator:
            self.zero_optimizer_grad()
            pred = self.model(self.ctx)
            if self.ctx.mask is not None:
                diff = (pred - self.target_mean) * self.ctx.mask
                denom = torch.clamp(torch.sum(self.ctx.mask), min=1.0)
                loss_data = torch.sum(diff * diff) / denom
            else:
                loss_data = torch.mean((pred - self.target_mean) ** 2)
            loss = loss_data + float(constraint_weight) * self.model.total_constraint_loss(
                self.ctx
            )
            loss.backward()
            self.step_optimizer()
            loss_value = float(loss.detach().cpu())
            self.step_scheduler(loss_value)
            self.mean_fit_history.append(loss_value)
            self.mean_fit_lrs.append(float(self.get_current_lr()))

        x_fit = self._get_model_state_vector()
        self.state_current = x_fit.detach().clone()
        self.state_mean_refined = x_fit.detach().clone()
        self.mean_refined = True
        return self

    def plot_losses(
        self, figax: tuple[Any, Any] | None = None, plot_lrs: bool = True
    ) -> tuple[Any, Any]:
        import matplotlib.pyplot as plt

        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        losses = np.asarray(self.mean_fit_history, dtype=np.float64)
        if losses.size == 0:
            ax.text(
                0.5,
                0.5,
                "No fit history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            if figax is None:
                plt.tight_layout()
                plt.show()
            return fig, ax

        iters = np.arange(losses.size)
        lines: list[Any] = []
        lines.extend(ax.semilogy(iters, losses, c="k", lw=2, label="loss"))
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss", color="k")
        ax.tick_params(axis="y", which="both", colors="k")
        ax.spines["left"].set_color("k")
        ax.set_xbound(-2, max(1, int(iters.max())) + 2)

        lrs = np.asarray(self.mean_fit_lrs, dtype=np.float64)
        if plot_lrs and lrs.size > 0:
            if lrs.size == losses.size and not np.allclose(lrs, lrs[0]):
                ax_lr = ax.twinx()
                ax.set_zorder(2)
                ax_lr.set_zorder(1)
                ax.patch.set_visible(False)
                ax_lr.spines["left"].set_visible(False)
                lines.extend(
                    ax_lr.semilogy(
                        np.arange(lrs.size), lrs, c="tab:blue", lw=2, ls="--", label="LR"
                    )
                )
                ax_lr.set_ylabel("LR", color="tab:blue")
                ax_lr.tick_params(axis="y", which="both", colors="tab:blue")
                ax_lr.spines["right"].set_color("tab:blue")
            else:
                ax.set_title(f"LR: {float(lrs[-1]):.2e}", fontsize=10)

        labels = [line.get_label() for line in lines]
        if len(labels) > 1:
            ax.legend(lines, labels, loc="upper right")

        if figax is None:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def visualize(
        self, *, power: float = 0.25, cbar: bool = False, axsize: tuple[int, int] = (6, 6)
    ) -> tuple[Any, Any]:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        if self.image_ref is None:
            self.preprocess()
        if self.image_ref is None or self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")

        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        self.plot_losses(figax=(fig, ax_top), plot_lrs=True)

        ref = np.asarray(self.image_ref, dtype=np.float32)
        pred = self.render_current
        refp = ref if power == 1.0 else np.maximum(ref, 0.0) ** float(power)
        predp = pred if power == 1.0 else np.maximum(pred, 0.0) ** float(power)
        vmin = float(min(refp.min(), predp.min()))
        vmax = float(max(refp.max(), predp.max()))

        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.15)
        axs = np.array(
            [fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])], dtype=object
        )
        show_2d(
            [refp, predp],
            figax=(fig, axs),
            title=["image_ref", "model"],
            cmap="gray",
            cbar=bool(cbar),
            returnfig=False,
            axsize=axsize,
            vmin=vmin,
            vmax=vmax,
        )

        if len(self.mean_fit_history) > 0:
            fig.suptitle(
                f"Final loss: {self.mean_fit_history[-1]:.3e} | Iters: {len(self.mean_fit_history)}",
                fontsize=13,
                y=0.98,
            )
        plt.show()
        return fig, axs

    def plot_mean_model(
        self,
        *,
        power: float = 0.25,
        returnfig: bool = False,
        axsize: tuple[int, int] = (6, 6),
        **_: Any,
    ) -> tuple[Any, Any] | None:
        if self.image_ref is None:
            self.preprocess()
        if self.image_ref is None or self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")

        ref = np.asarray(self.image_ref, dtype=np.float32)
        pred = self.render_current

        refp = ref if power == 1.0 else np.maximum(ref, 0.0) ** float(power)
        predp = pred if power == 1.0 else np.maximum(pred, 0.0) ** float(power)
        vmin = float(min(refp.min(), predp.min()))
        vmax = float(max(refp.max(), predp.max()))

        fig, ax = show_2d(
            [refp, predp],
            title=["image_ref", "model"],
            cmap="gray",
            cbar=False,
            returnfig=True,
            axsize=axsize,
            vmin=vmin,
            vmax=vmax,
        )
        if returnfig:
            return fig, ax
        return None
