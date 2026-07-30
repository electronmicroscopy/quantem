from __future__ import annotations

from typing import Any, Literal, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import shift as ndi_shift
from scipy.signal.windows import tukey
from tqdm import tqdm

from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d, Dataset4dstem
from quantem.core.fitting.background import DCBackground, GaussianBackground
from quantem.core.fitting.base import (
    AdditiveRenderModel,
    FitBase,
    OriginND,
    RenderComponent,
    RenderContext,
)
from quantem.core.fitting.diffraction import DiskTemplate, SyntheticDiskLattice
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.optimizer_mixin import OptimizerParams
from quantem.core.utils.imaging_utils import cross_correlation_shift
from quantem.diffraction.model_fitting_visualizations import ModelDiffractionVisualizations
from quantem.diffraction.strain import StrainMap


def _parse_init(value: float | int | Sequence[float | int | None], *, name: str) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"{name} cannot be empty.")
        if value[0] is None:
            raise ValueError(f"{name} initial value cannot be None.")
        return float(value[0])
    return float(cast(float | int, value))


class ModelDiffraction(ModelDiffractionVisualizations, FitBase, AutoSerialize):
    _token = object()
    DEFAULT_LR = 5e-2
    DEFAULT_OPTIMIZER_TYPE = "adam"

    def __init__(self, dataset: Any, _token: object | None = None):
        if _token is not self._token:
            raise RuntimeError("Use ModelDiffraction.from_dataset() or .from_file().")
        AutoSerialize.__init__(self)
        FitBase.__init__(self)

        # Dataset/input references
        self.dataset = dataset
        self.image_ref: np.ndarray | None = None
        self.preprocess_shifts: np.ndarray | None = None
        self.index_shape: tuple[int, ...] | None = None
        self.target_mean: torch.Tensor | None = None

        # Diffraction-specific state/checkpoints
        self.state_mean_refined: dict[str, torch.Tensor] | None = None
        self.mean_refined: bool = False

        self.state_individual_refined: np.ndarray | None = None
        self.individual_refined: bool = False

        # Misc metadata
        self.metadata: dict[str, Any] = {}

        self.u_ref: np.ndarray | None = None
        self.v_ref: np.ndarray | None = None
        self.u_array: np.ndarray | None = None
        self.v_array: np.ndarray | None = None

        self.real_space = False

    @classmethod
    def from_dataset(
        cls, dataset: Dataset2d | Dataset3d | Dataset4d | Dataset4dstem | Any
    ) -> "ModelDiffraction":
        if isinstance(dataset, (Dataset2d, Dataset3d, Dataset4d, Dataset4dstem)):
            return cls(dataset=dataset, _token=cls._token)
        raise TypeError(
            "from_dataset expects a Dataset2d, Dataset3d, Dataset4d, or Dataset4dstem instance."
        )

    @property
    def components(self) -> torch.nn.ModuleList:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self.model.components

    def get_component(self, name: str) -> RenderComponent:
        """
        Return a live model component by resolved name.

        Parameters
        ----------
        name : str
            Resolved component name.

        Returns
        -------
        RenderComponent
            The live component object.

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If no component matches ``name``.
        """
        return self._resolve_component_by_name(name)

    def get_rendered_component(self, name: str) -> np.ndarray:
        """
        Render a component and return a NumPy array.

        Parameters
        ----------
        name : str
            Resolved component name.

        Returns
        -------
        np.ndarray
            Rendered component image.

        Raises
        ------
        RuntimeError
            If model/context are not defined.
        KeyError
            If no component matches ``name``.
        """
        if self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        ctx = self.ctx
        component = self._resolve_component_by_name(name)
        rendered = component(ctx)
        return rendered.detach().cpu().numpy()

    def get_rendered_disk_template(self, name: str | None = None) -> np.ndarray:
        """
        Return a DiskTemplate patch as a numpy array--not rendered onto the full frame.

        Parameters
        ----------
        name : str | None, optional
            DiskTemplate component name. If omitted, requires exactly one DiskTemplate.

        Returns
        -------
        np.ndarray
            Template-sized array from ``template_raw``.

        Raises
        ------
        RuntimeError
            If model/context are not defined, no DiskTemplate exists, or multiple
            DiskTemplates exist when ``name`` is omitted.
        TypeError
            If a named component exists but is not a DiskTemplate.
        """
        if self.ctx is None or self.model is None:
            raise RuntimeError("Call .define_model(...) first.")

        if name is not None:
            component = self._resolve_component_by_name(name)
            if not isinstance(component, DiskTemplate):
                raise TypeError(f"Component '{name}' is not a DiskTemplate.")
            return component.template_raw.detach().cpu().numpy()
        matches = [m for m in self.model.components if isinstance(m, DiskTemplate)]
        if len(matches) == 0:
            raise RuntimeError("No DiskTemplate components found.")
        if len(matches) > 1:
            raise RuntimeError("Multiple DiskTemplate components found; pass name explicitly.")
        disk = cast(DiskTemplate, matches[0])
        return disk.template_raw.detach().cpu().numpy()

    def set_disk_template_trainable(
        self, enabled: bool, name: str | None = None, rebuild_optimizer: bool = True
    ) -> None:
        """
        Toggle DiskTemplate ``template_raw`` trainability.

        Parameters
        ----------
        enabled : bool
            If ``True``, enable optimization of ``template_raw``.
        name : str | None, optional
            DiskTemplate component name. If ``None``, applies to all DiskTemplate
            components in the current model.
        rebuild_optimizer : bool, optional
            If ``True``, rebuild optimizer param groups after toggling.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If ``name`` does not match any component.
        RuntimeError
            If model is not defined or no DiskTemplate components are found.
        TypeError
            If ``name`` resolves to a non-DiskTemplate component.

        Notes
        -----
        This toggles only ``template_raw.requires_grad``. Other DiskTemplate
        parameters (for example ``intensity_raw``) are unchanged. When
        ``rebuild_optimizer=True``, optimizer param groups are rebuilt to match
        current ``requires_grad`` flags.
        """
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")

        if name is not None:
            component = self._resolve_component_by_name(name)
            if not isinstance(component, DiskTemplate):
                raise TypeError(f"Component '{name}' is not a DiskTemplate.")
            self.set_parameter_trainable(
                name,
                "template_raw",
                enabled=enabled,
                rebuild_optimizer=rebuild_optimizer,
            )
            return

        disk_names = [
            component_name
            for component_name, component in self._iter_named_components()
            if isinstance(component, DiskTemplate)
        ]
        if len(disk_names) == 0:
            raise RuntimeError("No DiskTemplate components found.")

        for disk_name in disk_names:
            self.set_parameter_trainable(
                disk_name,
                "template_raw",
                enabled=enabled,
                rebuild_optimizer=False,
            )
        if rebuild_optimizer:
            self._rebuild_optimizer_after_trainability_change()

    def get_component_constraints(self, name: str) -> dict[str, dict[str, Any]]:
        component = self._resolve_component_by_name(name)
        return {
            "hard": dict(component.hard_constraints),
            "soft": dict(component.soft_constraints),
        }

    def get_overlay_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return origin and lattice disk-center coordinates for overlay plotting.

        Parameters
        ----------
        None

        Returns
        -------
        origin_rc : np.ndarray
            Origin coordinate array with shape ``(2,)`` as ``(row, col)``.
        disk_centers_rc : np.ndarray
            Disk-center array with shape ``(N, 2)`` as ``(row, col)``.

        Raises
        ------
        RuntimeError
            If model/context are not defined.

        Notes
        -----
        Coordinates are computed from current model parameters without mutating state.
        Boundary filtering matches ``SyntheticDiskLattice.forward`` behavior.
        """
        if self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")

        with torch.no_grad():
            origin = cast(OriginND, self.model.origin)
            origin_rc = origin.coords[:2].detach().cpu().numpy().astype(np.float32, copy=False)

            centers: list[np.ndarray] = []
            for module in self.model.components:
                component = cast(RenderComponent, module)
                if not isinstance(component, SyntheticDiskLattice):
                    continue
                if component.origin is None:
                    continue
                uv_indices = cast(torch.Tensor, component.uv_indices)
                if torch.numel(uv_indices) == 0:
                    continue

                uv = torch.as_tensor(uv_indices, device=self.ctx.device)
                u = uv[:, 0].to(dtype=self.ctx.dtype)
                v = uv[:, 1].to(dtype=self.ctx.dtype)
                r0, c0 = component.origin.coords[0], component.origin.coords[1]
                centers_r = r0 + u * component.u_row + v * component.v_row
                centers_c = c0 + u * component.u_col + v * component.v_col

                b = torch.as_tensor(
                    component.boundary_px, device=self.ctx.device, dtype=self.ctx.dtype
                )
                keep = (centers_r >= b) & (centers_r <= (self.ctx.shape[0] - 1) - b)
                keep = keep & (centers_c >= b) & (centers_c <= (self.ctx.shape[1] - 1) - b)
                mk = component._mask_keep(self.ctx, centers_r, centers_c)
                if mk is not None:
                    keep = keep & mk
                if torch.any(keep):
                    rc = torch.stack((centers_r[keep], centers_c[keep]), dim=1)
                    centers.append(rc.detach().cpu().numpy().astype(np.float32, copy=False))

            if centers:
                disk_centers_rc = np.concatenate(centers, axis=0)
            else:
                disk_centers_rc = np.zeros((0, 2), dtype=np.float32)

        return origin_rc, disk_centers_rc

    def preprocess(
        self,
        *,
        align: bool = False,
        edge_blend: float = 8.0,
        upsample_factor: int = 32,
        max_shift: float | None = None,
        shift_order: int = 1,
        gamma: float = 0.5,
        mode: str = "linear",
        rows=None,
        cols = None,
    ) -> "ModelDiffraction":
        arr = np.asarray(self.dataset.array)
        self.mask = np.ones(arr.shape[:2])

        if arr.ndim < 2:
            raise ValueError("dataset.array must have at least 2 dimensions.")
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
        
        if mode_norm == "linear":
            arr = arr
        elif mode_norm == "log":
            arr = np.log1p(arr)
        elif mode_norm == "gamma":
            arr = np.power(np.clip(arr, 0.0, None), self.metadata["gamma"])
        else:
            raise RuntimeError("Unreachable: normalized mode mapping failed.")
        self.dataset.array = np.asarray(arr)

        if rows is None and cols is None:
            rows = range(self.dataset.shape[0])
            cols = range(self.dataset.shape[1])
        elif rows is not None and cols is None:
            cols = range(self.dataset.shape[1])
        elif rows is None and cols is not None:
            rows = range(self.dataset.shape[0])
        else:
            rows = rows
            cols = cols
        
        if isinstance(rows, int):
            rows = np.array([rows]).astype(int)
        else:        
            rows = np.asarray(rows).astype(int)
        
        if isinstance(cols, int):
            cols = np.array([cols]).astype(int)
        else:        
            cols = np.asarray(cols).astype(int)
        
        arr = arr[np.ix_(rows, cols)]

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

        s0 = self._get_model_state_dict_copy()
        self.state_initialized = s0
        self.state_mean_refined = None
        self.mean_refined = False
        self._clear_fit_history_all()
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
        constraint_params: dict[str, Any] | None = None,
        constraint_config_params: dict[str, Any] | None = None,
        progress: bool = True,
    ) -> "ModelDiffraction":
        """
        Fit the mean diffraction pattern.

        Parameters
        ----------
        n_steps : int, optional
            Number of optimization steps.
        reset : bool | Literal["initialized", "mean_refined"], optional
            Reset behavior before fitting.
        optimizer_params : dict | None, optional
            Optimizer override for this fit call.
        scheduler_params : dict | None, optional
            Scheduler override for this fit call.
        constraint_weight : float, optional
            Global multiplier for soft-constraint loss.
        constraint_params : dict[str, Any] | None, optional
            Optional constraint updates applied once to components before fitting.
            If ``None``, previously assigned constraints are reused.
        progress : bool, optional
            If ``True``, show progress bar.

        Returns
        -------
        ModelDiffraction
            Self, with updated fit state and history.

        Raises
        ------
        RuntimeError
            If model/context/target are not defined.
        ValueError
            If ``reset`` has an unsupported value.

        Notes
        -----
        Constraint assignments persist on components across fit calls.
        """
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

        self.fit_render(
            # target=torch.tensor(self.dataset[30,30].array.astype("float32")),
            target=self.target_mean,
            n_steps=int(n_steps),
            constraint_weight=float(constraint_weight),
            constraint_params=constraint_params,
            constraint_config_params=constraint_config_params,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            progress=bool(progress),
            run_key="mean",
        )

        s_fit = self._get_model_state_dict_copy()
        self.state_mean_refined = self._clone_state_dict(s_fit)
        self.mean_refined = True
        return self

    def reset(
        self,
        reset_to: Literal["initialized", "mean_refined", "individual"] = "mean_refined",
        reset_history: bool = True,
        individual_row: int = 0,
        individual_col: int = 0,
    ) -> "ModelDiffraction":
        if reset_to == "initialized":
            state = self.state_initialized
            if state is None:
                raise RuntimeError(
                    "initialized state is unavailable. Call .define_model(...) first."
                )
            if reset_history:
                self._clear_fit_history_all()
        elif reset_to == "mean_refined":
            state = self.state_mean_refined
            if state is None:
                raise RuntimeError(
                    "mean_refined state is unavailable. Run .fit_mean_diffraction_pattern(...) first."
                )
            if reset_history:
                mean_hist = self.fit_history.get("mean")
                self._clear_fit_history_all()
                if mean_hist is not None:
                    self.fit_history["mean"] = mean_hist
        elif reset_to == "individual":
            if self.state_individual_refined is None:
                raise ValueError("individual states is unavalible. Run fit_individual_diffraction_pattern(....) first")
            if (individual_row >= self.state_individual_refined.shape[0]) or (individual_col >= self.state_individual_refined.shape[1]):
                raise ValueError("row and column values not in range")
            state = self.state_individual_refined[individual_row, individual_col]
            if reset_history:
                self._clear_fit_history_all()
        else:
            raise ValueError("reset_to must be 'initialized' or 'mean_refined' or 'individual'.")

        self._load_model_state_dict_copy(state)
        return self
    
    def fit_individual_diffraction_pattern(
        self,
        *,
        rows=None,
        cols = None,
        n_steps: int = 200,
        reset: bool | Literal["initialized", "mean_refined"],
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraint_weight: float = 1.0,
        constraint_params: dict[str, Any] | None = None,
        constraint_config_params: dict[str, Any] | None = None,
        progress: bool = True,
        batch_size: int | None = None,
        frozen_components: list[str] | str | None = None,
        sample_trainability: dict[str, Any] | None = None,
        save_to_gpu: bool = True,
        **_compat_kwargs: Any,
    ) -> "ModelDiffraction":
        if batch_size is not None:
            return self.fit_individual_diffraction_pattern_batched(
                rows=rows,
                cols=cols,
                batch_size=int(batch_size),
                n_steps=int(n_steps),
                reset=cast(Literal["initialized", "mean_refined"], reset),
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params,
                constraint_weight=float(constraint_weight),
                constraint_params=constraint_params,
                constraint_config_params=constraint_config_params,
                frozen_components=frozen_components,
                sample_trainability=sample_trainability,
                progress=progress,
                save_to_gpu = save_to_gpu
            )

        if self.model is None or self.ctx is None or self.target_mean is None:
            raise RuntimeError("Call .define_model(...) first.")
        if reset not in ("initialized", "mean_refined"):
            raise ValueError("reset must be initialized', or 'mean_refined'.")
        self.reset(reset_to=cast(Literal["initialized", "mean_refined"], reset))
        if not isinstance(self.dataset, Dataset4d):
            raise ValueError("Dataset must be Dataset4d or Dataset4dstem for fit_individual_diffraction_pattern")
        
        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]

        if rows is None and cols is None:
            rows = range(self.dataset.shape[0])
            cols = range(self.dataset.shape[1])
        elif rows is not None and cols is None:
            cols = range(self.dataset.shape[1])
        elif rows is None and cols is not None:
            rows = range(self.dataset.shape[0])
        else:
            rows = rows
            cols = cols
        
        if isinstance(rows, int):
            rows = np.array([rows]).astype(int)
        else:        
            rows = np.asarray(rows).astype(int)
        
        if isinstance(cols, int):
            cols = np.array([cols]).astype(int)
        else:        
            cols = np.asarray(cols).astype(int)

        self.state_individual_refined = np.full(shape=(scan_r, scan_c), fill_value=None, dtype=object)
        
        if progress:
            pbar = tqdm(total=rows.shape[0] * cols.shape[0], desc="Fit individual")

        for r in rows:
            for c in cols:
                # print(self.dataset.array[r,c].shape)
                self.reset(reset_to=cast(Literal["initialized", "mean_refined"], reset), reset_history=False)
                self.fit_render(
                    target=torch.as_tensor(self.dataset.array[r,c],device=self.ctx.device,dtype=self.ctx.dtype),
                    n_steps=int(n_steps),
                    constraint_weight=float(constraint_weight),
                    constraint_params=constraint_params,
                    optimizer_params=optimizer_params,
                    scheduler_params=scheduler_params,
                    progress=False,
                    run_key=f"individual_{r}_{c}",
                )

                s_fit = self._get_model_state_dict_copy()
                self.state_individual_refined[r,c] = self._clone_state_dict(s_fit)
                # self.reset(reset_to=cast(Literal["initialized", "mean_refined"], reset), reset_history=False)
                if progress:
                    pbar.update(1)
        if progress:
            pbar.close()

        self.individual_refined=True
        return self

    def fit_individual_diffraction_pattern_batched(
        self,
        *,
        rows: Any = None,
        cols: Any = None,
        batch_size: int = 16,
        n_steps: int = 200,
        reset: Literal["initialized", "mean_refined"] = "mean_refined",
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraint_weight: float = 1.0,
        constraint_params: dict[str, Any] | None = None,
        constraint_config_params: dict[str, Any] | None = None,
        frozen_components: list[str] | str | None = None,
        sample_trainability: dict[str, Any] | None = None,
        progress: bool = True,
        save_to_gpu: bool = True,
    ) -> "ModelDiffraction":
        """
        Per-pattern fit, vectorized across a batch dimension on a single GPU.

        See ``fit_individual_diffraction_pattern`` for argument semantics. The
        batched version runs ``batch_size`` patterns in parallel per optimizer
        step, with per-sample stacked parameters and per-sample Adam moments.

        Soft constraint losses are added per-sample via
        ``model.total_constraint_loss``-style accumulation (currently only
        ``DiskTemplate`` contributes). Hard constraints and parameter bounds
        are still enforced between steps.

        Parameters
        ----------
        scheduler_params : dict | None, optional
            Either a single spec dict (``{"type": "cosine", "t_max": N}``)
            applied to every parameter group, or a dict keyed by component
            name/class. Supported types: ``none``, ``cosine`` (a.k.a.
            ``cosine_annealing``), ``linear``, ``exponential``. ``plateau``
            and ``cyclic`` fall back to constant LR with a warning.
        constraint_weight : float
            Global multiplier on the per-sample soft-constraint loss.
        constraint_config_params : dict | None
            Applied to the model once before the fit (``apply_constraint_configs``).
        frozen_components : list[str] | str | None
            Component name(s) whose parameters are frozen across all samples.
            Resolved via ``AdditiveRenderModel._component_constraint_name``;
            both instance names (``"disk0"``) and class names (``"DiskTemplate"``)
            work.
        sample_trainability : dict[str, ArrayLike] | None
            Per-canonical-key (e.g. ``"disk.template_raw"``) boolean array of
            length ``len(positions)``; ``False`` entries freeze that parameter
            for the matching scan position.
        """
        if self.model is None or self.ctx is None or self.target_mean is None:
            raise RuntimeError("Call .define_model(...) first.")
        if not isinstance(self.dataset, Dataset4d):
            raise ValueError("Dataset must be Dataset4d or Dataset4dstem.")
        if reset not in ("initialized", "mean_refined"):
            raise ValueError("reset must be 'initialized' or 'mean_refined'.")

        # Choose initialization state and load into the live model so we can
        # read current parameter values + constraint settings off the modules.
        if reset == "mean_refined":
            if self.state_mean_refined is None:
                raise RuntimeError("mean_refined state is unavailable. Run fit_mean_diffraction_pattern first.")
            init_state = self._clone_state_dict(self.state_mean_refined)
        else:
            if self.state_initialized is None:
                raise RuntimeError("initialized state is unavailable. Call define_model first.")
            init_state = self._clone_state_dict(self.state_initialized)
        self._load_model_state_dict_copy(init_state)

        if constraint_params is not None:
            self.model.apply_constraint_params(constraint_params, strict=True)
        if constraint_config_params is not None:
            self.model.apply_constraint_configs(constraint_config_params, strict=True)

        scan_r = int(self.dataset.shape[0])
        scan_c = int(self.dataset.shape[1])
        rows_arr, cols_arr = _resolve_rows_cols_for_batched(rows, cols, scan_r, scan_c)
        positions: list[tuple[int, int]] = [(int(r), int(c)) for r in rows_arr for c in cols_arr]
        if len(positions) == 0:
            return self

        if self.state_individual_refined is None or self.state_individual_refined.shape != (scan_r, scan_c):
            self.state_individual_refined = np.full(shape=(scan_r, scan_c), fill_value=None, dtype=object)

        ctx = self.ctx
        components_list = list(self.model.components)
        plan = _BatchedPlan.from_model(self.model, components_list, optimizer_params or {})
        plan.set_scheduler_params(scheduler_params)

        # Build per-position trainability arrays (default True everywhere).
        n_positions = len(positions)
        is_trainable_pos: dict[str, np.ndarray] = {
            key: np.ones(n_positions, dtype=bool) for key in plan.lrs
        }
        frozen_keys = plan.resolve_component_keys(frozen_components)
        for key in frozen_keys:
            is_trainable_pos[key][:] = False
        hard_skip_keys: set[str] = set(frozen_keys)
        if sample_trainability is not None:
            for key, arr in sample_trainability.items():
                if key not in is_trainable_pos:
                    raise KeyError(
                        f"sample_trainability key '{key}' is not a known stacked-param. "
                        f"Known: {sorted(is_trainable_pos)}"
                    )
                mask = np.asarray(arr, dtype=bool).reshape(-1)
                if mask.shape != (n_positions,):
                    raise ValueError(
                        f"sample_trainability['{key}'] has shape {mask.shape}, "
                        f"expected ({n_positions},)."
                    )
                is_trainable_pos[key] = mask

        loss_fn = self.loss_fn

        fidelity_mask: torch.Tensor | None = None
        fidelity_valid_count: torch.Tensor | None = None
        if ctx.mask is not None:
            fidelity_mask = ctx.mask.bool().to(ctx.device)
            fidelity_valid_count = fidelity_mask.sum().clamp(min=1).to(ctx.dtype)

        total_steps = len(positions) * n_steps
        pbar = tqdm(total=total_steps, desc="Fit individual (batched)", disable=not progress)


        dataset_gpu: torch.Tensor | None = None
        dev = self.ctx.device

        if save_to_gpu and str(dev) != "cpu":
            try:
                print(f"Loading data to {dev}...", flush=True)
                dataset_gpu = torch.as_tensor(
                    self.dataset.array,
                    device=dev,
                    dtype=ctx.dtype
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                dataset_gpu = None


        for start in range(0, len(positions), int(batch_size)):
            chunk = positions[start:start + int(batch_size)]
            B = len(chunk)

            if dataset_gpu is not None:
                rows_batch = [r for r, c in chunk]
                cols_batch = [c for r, c in chunk]
                targets = dataset_gpu[rows_batch, cols_batch]
            else:
                targets = torch.stack(
                    [
                        torch.as_tensor(self.dataset.array[r, c], device=ctx.device, dtype=ctx.dtype)
                        for (r, c) in chunk
                    ],
                    dim=0,
                )

            stacked = plan.build_stacked_params(B)

            opt_state: dict[str, dict[str, torch.Tensor]] = {
                name: (
                    {"m": torch.zeros_like(p.detach()), "v": torch.zeros_like(p.detach())}
                    if plan.opt_type in ("adam", "adamw")
                    else {"buf": torch.zeros_like(p.detach())}
                )
                for name, p in stacked.items()
            }

            # Per-chunk trainability slice as (B,) bool tensors on device.
            chunk_slice = slice(start, start + B)
            chunk_trainable: dict[str, torch.Tensor] = {}
            mixed_trainable_keys: list[str] = []
            init_snapshots: dict[str, torch.Tensor] = {}
            for key in stacked:
                if key in is_trainable_pos:
                    mask_np = is_trainable_pos[key][chunk_slice]
                    mask = torch.as_tensor(mask_np, device=ctx.device, dtype=torch.bool)
                    chunk_trainable[key] = mask
                    # A "mixed" key has some frozen samples and some trainable.
                    # We snapshot the init value so frozen samples can be
                    # restored after hard constraints / Adam.
                    if not bool(mask.all()) and not bool((~mask).all()):
                        mixed_trainable_keys.append(key)
                        init_snapshots[key] = stacked[key].detach().clone()

            for step in range(int(n_steps)):
                pred = plan.batched_forward(ctx, stacked)
                # Per-sample fidelity loss summed → scalar with per-sample grads
                diff2 = (pred.float() - targets.float())
                # Match SqrtMSELoss behavior approximately when loss_fn is SqrtMSELoss:
                # gamma-power transform of (x - min(x) + 1), per-sample independently.
                from quantem.core.fitting.base import SqrtMSELoss, LogMSELoss
                if isinstance(loss_fn, SqrtMSELoss):
                    gamma = float(loss_fn.gamma)
                    eps = 1.0
                    if fidelity_mask is not None:
                        pred_min = pred.masked_fill(~fidelity_mask, float("inf")).amin(dim=(1, 2), keepdim=True)
                        tgt_min = targets.masked_fill(~fidelity_mask, float("inf")).amin(dim=(1, 2), keepdim=True)
                    else:
                        pred_min = pred.amin(dim=(1, 2), keepdim=True)
                        tgt_min = targets.amin(dim=(1, 2), keepdim=True)
                    pred_mod = (pred - pred_min + eps) ** gamma
                    tgt_mod = (targets - tgt_min + eps) ** gamma
                    sq = (pred_mod - tgt_mod) ** 2
                    if fidelity_mask is not None:
                        per_sample_loss = (sq * fidelity_mask).sum(dim=(1, 2)) / fidelity_valid_count
                    else:
                        per_sample_loss = sq.mean(dim=(1, 2))
                elif isinstance(loss_fn, LogMSELoss):
                    sq = (torch.log1p(pred) - torch.log1p(targets)) ** 2
                    if fidelity_mask is not None:
                        per_sample_loss = (sq * fidelity_mask).sum(dim=(1, 2)) / fidelity_valid_count
                    else:
                        per_sample_loss = sq.mean(dim=(1, 2))
                elif isinstance(loss_fn, torch.nn.L1Loss):
                    ad = diff2.abs()
                    if fidelity_mask is not None:
                        per_sample_loss = (ad * fidelity_mask).sum(dim=(1, 2)) / fidelity_valid_count
                    else:
                        per_sample_loss = ad.mean(dim=(1, 2))
                else:
                    sq = diff2 * diff2
                    if fidelity_mask is not None:
                        per_sample_loss = (sq * fidelity_mask).sum(dim=(1, 2)) / fidelity_valid_count
                    else:
                        per_sample_loss = sq.mean(dim=(1, 2))

                # Add per-sample soft constraint loss.
                if float(constraint_weight) != 0.0:
                    constraint_per_sample = plan.batched_constraint_loss(ctx, stacked)
                    per_sample_loss = per_sample_loss + float(constraint_weight) * constraint_per_sample

                total_loss = per_sample_loss.sum()

                grads_list = list(torch.autograd.grad(total_loss, list(stacked.values()), allow_unused=True))

                # Apply trainability masks: zero out per-sample gradient entries
                # for frozen samples/params before the Adam step.
                for i, (name, g) in enumerate(zip(stacked.keys(), grads_list)):
                    if g is None:
                        continue
                    mask_b = chunk_trainable.get(name)
                    if mask_b is None:
                        continue
                    if bool(mask_b.all()):
                        continue
                    # Broadcast (B,) over remaining dims.
                    view_shape = (g.shape[0],) + (1,) * (g.ndim - 1)
                    grads_list[i] = g * mask_b.view(view_shape).to(dtype=g.dtype)

                t = step + 1
                current_lrs = plan.lr_at_step(t, int(n_steps))
                _optimizer_step_inplace(
                    plan.opt_type, stacked, tuple(grads_list), opt_state,
                    current_lrs, plan.opt_hparams, chunk_trainable, t,
                )

                with torch.no_grad():
                    plan.apply_hard_constraints(stacked, skip_keys=hard_skip_keys)
                    for key in stacked:
                        if key not in chunk_trainable:
                            continue
                        mask = chunk_trainable[key]
                        if bool(mask.all()): 
                            continue
                        if key not in init_snapshots:
                            init_snapshots[key] = stacked[key].detach().clone()
                        view_shape = (mask.shape[0],) + (1,) * (stacked[key].ndim - 1)
                        keep = mask.view(view_shape)
                        stacked[key].data.copy_(
                            torch.where(keep, stacked[key].data, init_snapshots[key])
                        )

                pbar.update(B)

            # Unstack each sample back into a state_dict and store
            for b, (r, c) in enumerate(chunk):
                sample_state = plan.build_sample_state_dict(init_state, stacked, b)
                self.state_individual_refined[r, c] = sample_state

        pbar.close()
        self.individual_refined = True
        return self

    def get_individual_uv_vectors(self) -> "ModelDiffraction":
        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]
        
        self.u_array = np.empty(shape=(scan_r, scan_c, 2))
        self.v_array = np.empty(shape=(scan_r, scan_c, 2))
        if self.state_individual_refined is None:
            raise RuntimeError("Call .fit_individual_diffraction_pattern(...) on all patterns first.")
        for r in range(scan_r):
            for c in range(scan_c):
                pos_state = self.state_individual_refined[r,c]
                if pos_state is None:
                    self.u_array[r,c,:] = None
                    self.v_array[r,c,:] = None
                    continue
                for key in pos_state.keys():
                    key_parts = key.split('.')
                    if(key_parts[-1] == 'u_row'):
                        self.u_array[r,c,0] = pos_state[key]
                    if(key_parts[-1] == 'u_col'):
                        self.u_array[r,c,1] = pos_state[key]
                    if(key_parts[-1] == 'v_row'):
                        self.v_array[r,c,0] = pos_state[key]
                    if(key_parts[-1] == 'v_col'):
                        self.v_array[r,c,1] = pos_state[key]

        return self
    
    def render_individual_pattern(self, row, col):
        if self.state_individual_refined is None:
            raise RuntimeError(
                "individual_refined_state is unavalible. Run fit_individual_diffraction_pattern(...) first."
            )
        if self.dataset.shape[0] <= row or self.dataset.shape[1] <= col:
            raise ValueError("individual row or column outside bounds of dataset")
        if row < 0 or col < 0:
            raise ValueError("individual row or column outside bounds of dataset")
        if self.state_individual_refined[row, col] is None:
            raise RuntimeError(
                "individual_refined_state is not avalible for given row and column. Run fit_individual_diffraction_pattern(...) for that row and column."
            )
        return self._render_state_array(self.state_individual_refined[row, col])

    
    def create_mask(
        self,
        use_radial_method: bool = False,
        exclusion_radius_fraction: float = 0.1,
        plot: bool = True,
        figsize: tuple[float, float] = (5, 4),
    ):
        """Compute the per-position weight :attr:`mask` from the fitted lattice signal.

        Builds a ``(scan_row, scan_col)`` weight in ``[0, 1]`` measuring the lattice
        signal at each position -- the model-fitting analogue of
        :attr:`BraggVectors.mask_weight` and the cepstral ``mask_weight``. It is passed to
        :meth:`calculate_strain_map`, where higher-weight positions contribute more to the
        reference lattice and weak/vacuum positions are down-weighted.

        The raw signal (summed intensity of the non-central fitted disks, or the
        diffracted intensity outside a central disk for the radial method) is min-max
        normalized to ``[0, 1]``; no contrast windowing or smoothing is applied. Display
        contrast is applied via the ``mask_range`` argument of
        :meth:`StrainMap.plot_strain`, :meth:`StrainMap.update_reference`, and
        :meth:`StrainMap.estimate_strain_precision` (e.g. ``mask_range=(0.37, 0.5)``):
        weights at/below ``low`` render black, at/above ``high`` render full color. The
        weight-map plot shows the distribution so an appropriate ``mask_range`` can be
        chosen.

        Parameters
        ----------
        use_radial_method : bool, default=False
            If ``True``, weight by the total diffracted intensity *outside* a central
            disk (radius ``exclusion_radius_fraction`` of the detector width). If
            ``False`` (default, recommended), weight by the summed intensity of the
            fitted non-central lattice disks -- requires
            :meth:`fit_individual_diffraction_pattern` to have been run.
        exclusion_radius_fraction : float, default=0.1
            Central-disk radius (fraction of detector width) excluded by the radial
            method.
        plot : bool, default=True
            If ``True``, show the resulting per-position weight map (so a separate
            ``plt.imshow(mask)`` cell is unnecessary).
        figsize : tuple of float, default=(5, 4)
            Figure size in inches for the weight-map plot.

        Returns
        -------
        ModelDiffraction
            ``self``, with :attr:`mask` set.
        """
        if not isinstance(self.dataset, (Dataset4d, Dataset4dstem)):
            raise ValueError("Dataset must be Dataset4d or Dataset4dstem.")

        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]
        self.i0_sum_array = np.zeros(shape=(scan_r, scan_c))

        if use_radial_method:
            center_y, center_x = np.array(self.dataset.shape[:-2]) / 2
            y, x = np.ogrid[: self.dataset.shape[-2], : self.dataset.shape[-1]]
            radius_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            exclusion_radius = exclusion_radius_fraction * self.dataset.shape[-1]
            outside_mask = radius_map > exclusion_radius
            for r in range(scan_r):
                for c in range(scan_c):
                    self.i0_sum_array[r, c] = np.sum(self.dataset.array[r, c][outside_mask])
        else:
            if self.state_individual_refined is None:
                raise RuntimeError("Call .fit_individual_diffraction_pattern(...) first.")
            for r in range(scan_r):
                for c in range(scan_c):
                    pos_state = self.state_individual_refined[r, c]
                    if pos_state is None:
                        continue
                    i0_raw = None
                    uv_indices = None
                    for key in pos_state.keys():
                        if key.endswith("i0_raw"):
                            i0_raw = pos_state[key].cpu().numpy()
                        if key.endswith("uv_indices"):
                            uv_indices = pos_state[key].cpu().numpy()
                    if i0_raw is None or uv_indices is None:
                        continue
                    is_not_center = ~((uv_indices[:, 0] == 0) & (uv_indices[:, 1] == 0))
                    self.i0_sum_array[r, c] = np.sum(i0_raw[is_not_center])

        # Min-max normalization to [0, 1], no contrast windowing (set mask_range at
        # display). Subtracting the floor -- rather than only dividing by the max -- keeps
        # the weight from saturating near 1.0 when every position carries a baseline
        # lattice intensity, and matches the cepstral _amplitude_mask_weight. Degenerate
        # (constant / non-finite) input falls back to uniform full weight.
        lo = np.nanmin(self.i0_sum_array)
        hi = np.nanmax(self.i0_sum_array)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            self.mask = (self.i0_sum_array - lo) / (hi - lo)
        else:
            self.mask = np.ones_like(self.i0_sum_array)

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            handle = ax.imshow(self.mask, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title("Per-position lattice weight (mask)")
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
        q_to_r_rotation_ccw_deg: float | None = None,
        q_transpose: bool | None = None,
    ) -> StrainMap:
        """Build a :class:`StrainMap` from the fitted per-position lattice vectors.

        Mirrors :meth:`BraggVectors.calculate_strain_map` and
        :meth:`StrainMapAutocorrelation.calculate_strain_map` so the downstream strain
        cells (``plot_strain``, ``update_reference``, ``estimate_strain_precision``) are
        identical across the correlation, cepstral, and model-fitting workflows.

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
            reference lattice. Defaults to :attr:`mask` from :meth:`create_mask` (the
            lattice signal strength), so strong, well-fit positions dominate the
            reference.

        Returns
        -------
        StrainMap
            A strain map initialized from the fitted lattice vectors.
        """
        if self.u_array is None or self.v_array is None:
            self.get_individual_uv_vectors()
        if not isinstance(self.dataset, (Dataset4d, Dataset4dstem)):
            raise ValueError("Dataset must be Dataset4d or Dataset4dstem.")

        if mask is None:
            mask = self.mask

        default_units = None
        default_sampling = None
        if hasattr(self.dataset, 'units'):
            if isinstance(self.dataset.units, (tuple, list)):
                default_units = str(self.dataset.units[0])
            else:
                default_units = str(self.dataset.units)
        if hasattr(self.dataset, 'sampling'):
            if isinstance(self.dataset.sampling, (tuple, list, np.ndarray)):
                default_sampling = float(self.dataset.sampling[0])
            else:
                default_sampling = float(self.dataset.sampling)
        
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

        return StrainMap(
            u_array = self.u_array,
            v_array = self.v_array,
            ds_shape = self.dataset.shape,
            real_space = self.real_space,
            u_ref = u_ref,
            v_ref = v_ref,
            mask = mask,
            ds_sampling=default_sampling,
            ds_units = default_units,
            q_to_r_rotation_ccw_deg = q_to_r_rotation_ccw_deg,
            q_transpose = q_transpose,
        )

    @property
    def render_mean_refined(self) -> np.ndarray:
        if self.state_mean_refined is None:
            raise RuntimeError(
                "mean_refined state is unavailable. Run .fit_mean_diffraction_pattern(...) first."
            )
        return self._render_state_array(self.state_mean_refined)


def _resolve_rows_cols_for_batched(
    rows: Any, cols: Any, scan_r: int, scan_c: int
) -> tuple[np.ndarray, np.ndarray]:
    if rows is None:
        rows = range(scan_r)
    if cols is None:
        cols = range(scan_c)
    if isinstance(rows, int):
        rows_arr = np.array([rows], dtype=int)
    else:
        rows_arr = np.asarray(list(rows), dtype=int)
    if isinstance(cols, int):
        cols_arr = np.array([cols], dtype=int)
    else:
        cols_arr = np.asarray(list(cols), dtype=int)
    return rows_arr, cols_arr


def _lr_for_component(
    component: RenderComponent,
    component_idx: int,
    optimizer_params: dict[str, Any],
    model: AdditiveRenderModel,
) -> float:
    name = model._component_constraint_name(component, component_idx)
    if name in optimizer_params:
        return float(optimizer_params[name].get("lr", component.DEFAULT_LR))
    class_name = component.__class__.__name__
    if class_name in optimizer_params:
        return float(optimizer_params[class_name].get("lr", component.DEFAULT_LR))
    return float(component.DEFAULT_LR)


def _opt_spec_for_component(
    component: RenderComponent,
    component_idx: int,
    optimizer_params: dict[str, Any],
    model: AdditiveRenderModel,
):
    """Resolve the full optimizer spec (type + hyperparameters) for one component.

    Mirrors ``_lr_for_component``'s name-then-class lookup, but returns the parsed
    ``OptimizerParams`` spec instead of just the learning rate, so batched fitting can
    honor a requested optimizer ``type`` (e.g. ``"sgd"``) instead of silently always
    running Adam.
    """
    name = model._component_constraint_name(component, component_idx)
    class_name = component.__class__.__name__
    if name in optimizer_params:
        d = dict(optimizer_params[name])
    elif class_name in optimizer_params:
        d = dict(optimizer_params[class_name])
    else:
        return OptimizerParams.Adam(lr=component.DEFAULT_LR)
    d.setdefault("type", "adam")
    d.setdefault("lr", component.DEFAULT_LR)
    return OptimizerParams.parse_dict(d)


def _masked(t: torch.Tensor, mask_expanded: torch.Tensor | None) -> torch.Tensor:
    return t if mask_expanded is None else t * mask_expanded


def _adam_step_inplace(
    stacked: dict[str, torch.Tensor],
    grads: tuple[torch.Tensor, ...],
    adam_state: dict[str, dict[str, torch.Tensor]],
    lrs: dict[str, float],
    hparams: dict[str, dict[str, Any]],
    chunk_trainable: dict[str, torch.Tensor],
    t: int,
    decoupled_wd: bool = False,
) -> None:
    """
    In-place Adam/AdamW step with per-sample trainability support.

    Frozen samples (mask=False) will:
    1. Not accumulate gradient statistics in moments
    2. Not receive parameter updates -- including from weight decay, which is
       explicitly masked too (unlike a plain zeroed gradient, weight decay would
       otherwise still pull frozen samples toward zero every step).
    """
    with torch.no_grad():
        for (name, p), g in zip(stacked.items(), grads):
            if g is None:
                continue

            hp = hparams.get(name, {})
            beta1, beta2 = hp.get("betas", (0.9, 0.999))
            eps = hp.get("eps", 1e-8)
            wd = hp.get("weight_decay", 0.0)

            st = adam_state[name]
            mask_b = chunk_trainable.get(name)
            mask_expanded = None
            if mask_b is not None and not bool(mask_b.all()):
                view_shape = (g.shape[0],) + (1,) * (g.ndim - 1)
                mask_expanded = mask_b.view(view_shape).to(dtype=g.dtype)

            g_eff = _masked(g, mask_expanded)
            if wd != 0:
                if decoupled_wd:
                    lr_ = float(lrs.get(name, 1e-2))
                    p.data.sub_(_masked(lr_ * wd * p.data, mask_expanded))
                else:
                    g_eff = g_eff + _masked(wd * p.data, mask_expanded)

            st["m"].mul_(beta1).add_(g_eff, alpha=1.0 - beta1)
            st["v"].mul_(beta2).addcmul_(g_eff, g_eff, value=1.0 - beta2)

            m_hat = st["m"] / (1.0 - beta1 ** t)
            v_hat = st["v"] / (1.0 - beta2 ** t)

            lr = float(lrs.get(name, 1e-2))
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def _sgd_step_inplace(
    stacked: dict[str, torch.Tensor],
    grads: tuple[torch.Tensor, ...],
    sgd_state: dict[str, dict[str, torch.Tensor]],
    lrs: dict[str, float],
    hparams: dict[str, dict[str, Any]],
    chunk_trainable: dict[str, torch.Tensor],
    t: int,
) -> None:
    """In-place SGD step (with optional momentum/dampening/nesterov/weight_decay),
    matching ``torch.optim.SGD`` exactly, with the same per-sample trainability
    masking as :func:`_adam_step_inplace` (including for the weight-decay term)."""
    with torch.no_grad():
        for (name, p), g in zip(stacked.items(), grads):
            if g is None:
                continue

            hp = hparams.get(name, {})
            momentum = hp.get("momentum", 0.0)
            dampening = hp.get("dampening", 0.0)
            wd = hp.get("weight_decay", 0.0)
            nesterov = hp.get("nesterov", False)

            mask_b = chunk_trainable.get(name)
            mask_expanded = None
            if mask_b is not None and not bool(mask_b.all()):
                view_shape = (g.shape[0],) + (1,) * (g.ndim - 1)
                mask_expanded = mask_b.view(view_shape).to(dtype=g.dtype)

            d_p = _masked(g, mask_expanded)
            if wd != 0:
                d_p = d_p + _masked(wd * p.data, mask_expanded)
            if momentum != 0:
                buf = sgd_state[name]["buf"]
                if t == 1:
                    buf.copy_(d_p)
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1.0 - dampening)
                d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

            lr = float(lrs.get(name, 1e-2))
            p.data.add_(d_p, alpha=-lr)


def _optimizer_step_inplace(
    opt_type: str,
    stacked: dict[str, torch.Tensor],
    grads: tuple[torch.Tensor, ...],
    opt_state: dict[str, dict[str, torch.Tensor]],
    lrs: dict[str, float],
    hparams: dict[str, dict[str, Any]],
    chunk_trainable: dict[str, torch.Tensor],
    t: int,
) -> None:
    """Dispatch a single in-place optimizer step for the batched fit loop."""
    if opt_type in ("adam", "adamw"):
        _adam_step_inplace(
            stacked, grads, opt_state, lrs, hparams, chunk_trainable, t,
            decoupled_wd=(opt_type == "adamw"),
        )
    elif opt_type == "sgd":
        _sgd_step_inplace(stacked, grads, opt_state, lrs, hparams, chunk_trainable, t)
    else:
        raise NotImplementedError(f"Batched fit does not support optimizer type '{opt_type}'.")

class _BatchedPlan:
    """Resolved layout for the batched per-pattern fit: component refs, lrs, and helpers."""

    def __init__(self) -> None:
        self.origin: OriginND | None = None
        self.disk: DiskTemplate | None = None
        self.dcbg: DCBackground | None = None
        self.disk_idx: int | None = None
        self.dcbg_idx: int | None = None
        self.lrs: dict[str, float] = {}
        self.opt_type: str = "adam"
        self.opt_hparams: dict[str, dict[str, Any]] = {}
        self.scheduler_specs: dict[str, dict[str, Any]] = {}
        self.component_keys: dict[str, list[str]] = {}
        self.lats: list[SyntheticDiskLattice] = []
        self.lat_idxs: list[int] = []
        self.lat_names: list[str] = []
        self.lat_disk_key: dict[str, str] = {}
        self.gaussbgs: list[GaussianBackground] = []
        self.gaussbg_idxs: list[int] = []
        self.gaussbg_names: list[str] = []
        
        self._trainable_flags: dict[str, bool] = {}

    @classmethod
    def from_model(
        cls,
        model: AdditiveRenderModel,
        components_list: list[Any],
        optimizer_params: dict[str, Any],
    ) -> "_BatchedPlan":
        self = cls()
        self.origin = cast(OriginND, model.origin)

        for idx, comp in enumerate(components_list):
            if isinstance(comp, DiskTemplate) and self.disk is None:
                self.disk = comp
                self.disk_idx = idx
            elif isinstance(comp, DCBackground) and self.dcbg is None:
                self.dcbg = comp
                self.dcbg_idx = idx
            elif isinstance(comp, GaussianBackground) :
                self.gaussbgs.append(comp)
                self.gaussbg_idxs.append(idx)
            elif isinstance(comp, SyntheticDiskLattice):
                self.lats.append(comp)
                self.lat_idxs.append(idx)  
            else:
                raise TypeError(
                    f"Batched fit does not yet support component type {type(comp).__name__} "
                    f"at index {idx} (or duplicate of an already-handled type)."
                )

        # Track trainability for each parameter
        if self.disk is not None and self.disk_idx is not None:
            lr = _lr_for_component(self.disk, self.disk_idx, optimizer_params, model)
            name = model._component_constraint_name(self.disk, self.disk_idx)
            
            # Check if parameters are trainable
            self._trainable_flags["disk.template_raw"] = self.disk.template_raw.requires_grad
            self._trainable_flags["disk.intensity_raw"] = self.disk.intensity_raw.requires_grad
            
            # Only add to lrs if trainable
            if self._trainable_flags["disk.template_raw"]:
                self.lrs["disk.template_raw"] = lr
            if self._trainable_flags["disk.intensity_raw"]:
                self.lrs["disk.intensity_raw"] = lr
            
            self.component_keys[name] = ["disk.template_raw", "disk.intensity_raw"]
            self.component_keys[self.disk.__class__.__name__] = list(self.component_keys[name])
            
        if self.dcbg is not None and self.dcbg_idx is not None:
            lr = _lr_for_component(self.dcbg, self.dcbg_idx, optimizer_params, model)
            name = model._component_constraint_name(self.dcbg, self.dcbg_idx)
            
            self._trainable_flags["dcbg.intensity_raw"] = self.dcbg.intensity_raw.requires_grad
            if self._trainable_flags["dcbg.intensity_raw"]:
                self.lrs["dcbg.intensity_raw"] = lr
            
            self.component_keys[name] = ["dcbg.intensity_raw"]
            self.component_keys[self.dcbg.__class__.__name__] = list(self.component_keys[name])
            
        if self.gaussbgs:
            seen_prefixes: set[str] = set()
            for gaussbg, gaussbg_idx in zip(self.gaussbgs, self.gaussbg_idxs):
                lr = _lr_for_component(gaussbg, gaussbg_idx, optimizer_params, model)
                canonical_name = model._component_constraint_name(gaussbg, gaussbg_idx)
                key_prefix = "gaussbg" if len(self.gaussbgs) == 1 else canonical_name
                if key_prefix in seen_prefixes:
                    raise ValueError(
                        f"Multiple GaussianBackground components resolve to the same name "
                        f"'{key_prefix}'. Pass a distinct name=... to each GaussianBackground."
                    )
                seen_prefixes.add(key_prefix)
                self.gaussbg_names.append(key_prefix)

                gaussbg_keys: list[str] = []

                sigma_key = f"{key_prefix}.sigma_raw"
                self._trainable_flags[sigma_key] = gaussbg.sigma_raw.requires_grad
                if self._trainable_flags[sigma_key]:
                    self.lrs[sigma_key] = lr
                    gaussbg_keys.append(sigma_key)

                if gaussbg.sigma_col_raw is not None:
                    sigma_col_key = f"{key_prefix}.sigma_col_raw"
                    self._trainable_flags[sigma_col_key] = gaussbg.sigma_col_raw.requires_grad
                    if self._trainable_flags[sigma_col_key]:
                        self.lrs[sigma_col_key] = lr
                        gaussbg_keys.append(sigma_col_key)

                intensity_key = f"{key_prefix}.intensity_raw"
                self._trainable_flags[intensity_key] = gaussbg.intensity_raw.requires_grad
                if self._trainable_flags[intensity_key]:
                    self.lrs[intensity_key] = lr
                    gaussbg_keys.append(intensity_key)

                self.component_keys[canonical_name] = gaussbg_keys
                cls_key = gaussbg.__class__.__name__
                bucket = self.component_keys.setdefault(cls_key, [])
                for k in gaussbg_keys:
                    if k not in bucket:
                        bucket.append(k)
            
        if self.lats:
            first_lr = None
            for lat, lat_idx in zip(self.lats, self.lat_idxs):
                lr = _lr_for_component(lat, lat_idx, optimizer_params, model)
                if first_lr is None:
                    first_lr = lr
                canonical_name = model._component_constraint_name(lat, lat_idx)
                key_prefix = "lat" if len(self.lats) == 1 else canonical_name
                self.lat_names.append(key_prefix)

                # Resolve which tracked disk this lattice shares (by identity).
                if self.disk is not None and lat.disk is self.disk:
                    self.lat_disk_key[key_prefix] = "disk"
                else:
                    raise NotImplementedError(
                        f"Batched fit requires lattice '{canonical_name}' to share the "
                        f"single tracked DiskTemplate component; found a different/untracked disk."
                    )

                lat_keys = []
                for k in ("u_row", "u_col", "v_row", "v_col", "i0_raw", "ir", "ic", "irr", "icc", "irc"):
                    full_key = f"{key_prefix}.{k}"
                    param = getattr(lat, k, None)
                    if param is not None:
                        is_trainable = param.requires_grad if isinstance(param, torch.nn.Parameter) else False
                        self._trainable_flags[full_key] = is_trainable
                        if is_trainable:
                            self.lrs[full_key] = lr
                            lat_keys.append(full_key)

                self.component_keys[canonical_name] = lat_keys
                cls_key = lat.__class__.__name__
                bucket = self.component_keys.setdefault(cls_key, [])
                for k in lat_keys:
                    if k not in bucket:
                        bucket.append(k)

            # Origin uses the first lattice's LR (origin is shared/global).
            self._trainable_flags["origin.coords"] = self.origin.coords.requires_grad
            if self._trainable_flags["origin.coords"]:
                self.lrs["origin.coords"] = first_lr
        elif self.gaussbgs:
            intensity_key = f"{self.gaussbg_names[0]}.intensity_raw"
            if self._trainable_flags.get(intensity_key, False):
                self.lrs["origin.coords"] = self.lrs[intensity_key]
        else:
            if self.origin.coords.requires_grad:
                self.lrs["origin.coords"] = 1e-2

        # Default schedulers: constant LR for every key
        self.scheduler_specs = {k: {"type": "none"} for k in self.lrs}

        # Resolve optimizer type/hyperparameters. All components must agree on the
        # optimizer class -- mirroring OptimizerMixin.set_optimizer's own constraint --
        # since a single manual step function runs once per training step over every
        # stacked parameter. Keys with no explicit spec fall back to that type's
        # library defaults inside _adam_step_inplace/_sgd_step_inplace (e.g.
        # "origin.coords", which never has its own optimizer_params entry).
        spec_types: set[type] = set()
        for idx, comp in enumerate(components_list):
            spec = _opt_spec_for_component(comp, idx, optimizer_params, model)
            spec_types.add(type(spec))
            canonical_name = model._component_constraint_name(comp, idx)
            for key in self.component_keys.get(canonical_name, []):
                self.opt_hparams[key] = spec.params()
        if len(spec_types) > 1:
            raise ValueError(
                "All components must use the same optimizer type for batched fitting; "
                f"got {sorted(t.__name__ for t in spec_types)}."
            )
        if spec_types:
            self.opt_type = next(iter(spec_types))._name

        return self

    # ... (keep set_scheduler_params, lr_at_step, batched_constraint_loss, resolve_component_keys as-is)

    def build_stacked_params(self, B: int) -> dict[str, torch.Tensor]:
        """Build stacked params only for trainable parameters, expand frozen ones as needed."""
        out: dict[str, torch.Tensor] = {}
        assert self.origin is not None
        
        # Only stack trainable params
        if self._trainable_flags.get("origin.coords", False):
            out["origin.coords"] = self._stack(self.origin.coords, B)

        if self.disk is not None:
            if self._trainable_flags.get("disk.template_raw", False):
                out["disk.template_raw"] = self._stack(self.disk.template_raw, B)
            if self._trainable_flags.get("disk.intensity_raw", False):
                out["disk.intensity_raw"] = self._stack(self.disk.intensity_raw, B)
                
        if self.dcbg is not None:
            if self._trainable_flags.get("dcbg.intensity_raw", False):
                out["dcbg.intensity_raw"] = self._stack(self.dcbg.intensity_raw, B)
                
        if self.gaussbgs is not None:
            for gaussbg_name, gaussbg in zip(self.gaussbg_names, self.gaussbgs):
                sigma_key = f"{gaussbg_name}.sigma_raw"
                if self._trainable_flags.get(sigma_key, False):
                    out[sigma_key] = self._stack(gaussbg.sigma_raw, B)
                sigma_col_key = f"{gaussbg_name}.sigma_col_raw"
                if self._trainable_flags.get(sigma_col_key, False) and gaussbg.sigma_col_raw is not None:
                    out[sigma_col_key] = self._stack(gaussbg.sigma_col_raw, B)
                intensity_key = f"{gaussbg_name}.intensity_raw"
                if self._trainable_flags.get(intensity_key, False):
                    out[intensity_key] = self._stack(gaussbg.intensity_raw, B)
                
        if self.lats is not None:
            for lat_name, lat in zip(self.lat_names, self.lats):
                for attr in ("u_row", "u_col", "v_row", "v_col", "i0_raw"):
                    t = getattr(lat, attr)
                    key = f"{lat_name}.{attr}"
                    if t is not None and self._trainable_flags.get(key, False):
                        out[key] = self._stack(t, B)
                for attr in ("ir", "ic", "irr", "icc", "irc"):
                    t = getattr(lat, attr, None)
                    key = f"{lat_name}.{attr}"
                    if t is not None and self._trainable_flags.get(key, False):
                        out[key] = self._stack(t, B)
                    
        return out

    @staticmethod
    def _stack(p: torch.Tensor, B: int) -> torch.Tensor:
        x = p.detach().clone().unsqueeze(0).expand(B, *p.shape).contiguous()
        x.requires_grad_(True)
        return x

    def batched_forward(
        self, ctx: RenderContext, stacked: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Batched forward with frozen parameter support."""
        B = next(iter(stacked.values())).shape[0] if stacked else 1
        
        # Get origin (might be frozen)
        origin_b = stacked.get("origin.coords")
        if origin_b is None:
            # Origin is frozen - expand from component
            origin_b = self.origin.coords.unsqueeze(0).expand(B, *self.origin.coords.shape).clone()

        pred = torch.zeros(B, ctx.shape[0], ctx.shape[1], device=ctx.device, dtype=ctx.dtype)

        if self.disk is not None:
            # Handle potentially frozen disk params
            template_b = stacked.get("disk.template_raw")
            if template_b is None:
                template_b = self.disk.template_raw.unsqueeze(0).expand(B, *self.disk.template_raw.shape).clone()
            
            intensity_b = stacked.get("disk.intensity_raw")
            if intensity_b is None:
                intensity_b = self.disk.intensity_raw.unsqueeze(0).expand(B).clone()
            
            pred = pred + self.disk.forward_batched(
                ctx,
                template_raw_b=template_b,
                intensity_raw_b=intensity_b,
                origin_coords_b=origin_b,
            )
            
        if self.dcbg is not None:
            intensity_b = stacked.get("dcbg.intensity_raw")
            if intensity_b is None:
                intensity_b = self.dcbg.intensity_raw.unsqueeze(0).expand(B).clone()
            pred = pred + self.dcbg.forward_batched(ctx, intensity_raw_b=intensity_b)
            
        for gaussbg_name, gaussbg in zip(self.gaussbg_names, self.gaussbgs):
            sigma_b = stacked.get(f"{gaussbg_name}.sigma_raw")
            if sigma_b is None:
                sigma_b = gaussbg.sigma_raw.unsqueeze(0).expand(B).clone()

            sigma_col_b = stacked.get(f"{gaussbg_name}.sigma_col_raw")
            if sigma_col_b is None and gaussbg.sigma_col_raw is not None:
                sigma_col_b = gaussbg.sigma_col_raw.unsqueeze(0).expand(B).clone()

            intensity_b = stacked.get(f"{gaussbg_name}.intensity_raw")
            if intensity_b is None:
                intensity_b = gaussbg.intensity_raw.unsqueeze(0).expand(B).clone()

            pred = pred + gaussbg.forward_batched(
                ctx,
                sigma_raw_b=sigma_b,
                sigma_col_raw_b=sigma_col_b,
                intensity_raw_b=intensity_b,
                origin_coords_b=origin_b,
            )
            
        if self.lats:
            # Template (from disk) — resolved once, shared by every lattice.
            template_b = stacked.get("disk.template_raw")
            if template_b is None and self.disk is not None:
                template_b = self.disk.template_raw.unsqueeze(0).expand(B, *self.disk.template_raw.shape).clone()

            for lat_name, lat in zip(self.lat_names, self.lats):
                u_row_b = stacked.get(f"{lat_name}.u_row")
                if u_row_b is None:
                    u_row_b = lat.u_row.unsqueeze(0).expand(B).clone()

                u_col_b = stacked.get(f"{lat_name}.u_col")
                if u_col_b is None:
                    u_col_b = lat.u_col.unsqueeze(0).expand(B).clone()

                v_row_b = stacked.get(f"{lat_name}.v_row")
                if v_row_b is None:
                    v_row_b = lat.v_row.unsqueeze(0).expand(B).clone()

                v_col_b = stacked.get(f"{lat_name}.v_col")
                if v_col_b is None:
                    v_col_b = lat.v_col.unsqueeze(0).expand(B).clone()

                i0_raw_b = stacked.get(f"{lat_name}.i0_raw")
                if i0_raw_b is None:
                    i0_raw_b = lat.i0_raw.unsqueeze(0).expand(B, *lat.i0_raw.shape).clone()

                ir_b = stacked.get(f"{lat_name}.ir")
                if ir_b is None and lat.ir is not None:
                    ir_b = lat.ir.unsqueeze(0).expand(B, *lat.ir.shape).clone()

                ic_b = stacked.get(f"{lat_name}.ic")
                if ic_b is None and lat.ic is not None:
                    ic_b = lat.ic.unsqueeze(0).expand(B, *lat.ic.shape).clone()

                irr_b = stacked.get(f"{lat_name}.irr")
                if irr_b is None and lat.irr is not None:
                    irr_b = lat.irr.unsqueeze(0).expand(B, *lat.irr.shape).clone()

                icc_b = stacked.get(f"{lat_name}.icc")
                if icc_b is None and lat.icc is not None:
                    icc_b = lat.icc.unsqueeze(0).expand(B, *lat.icc.shape).clone()

                irc_b = stacked.get(f"{lat_name}.irc")
                if irc_b is None and lat.irc is not None:
                    irc_b = lat.irc.unsqueeze(0).expand(B, *lat.irc.shape).clone()

                pred = pred + lat.forward_batched(
                    ctx,
                    u_row_b=u_row_b,
                    u_col_b=u_col_b,
                    v_row_b=v_row_b,
                    v_col_b=v_col_b,
                    i0_raw_b=i0_raw_b,
                    ir_b=ir_b,
                    ic_b=ic_b,
                    irr_b=irr_b,
                    icc_b=icc_b,
                    irc_b=irc_b,
                    template_raw_b=template_b,
                    origin_coords_b=origin_b,
                )
        return pred

    def apply_hard_constraints(
        self,
        stacked: dict[str, torch.Tensor],
        skip_keys: set[str] | None = None,
    ) -> None:
        """Apply batched hard constraints, respecting frozen parameters."""
        skip_keys = skip_keys or set()
        
        # Add frozen params to skip_keys
        frozen_keys = {k for k, trainable in self._trainable_flags.items() if not trainable}
        skip_keys = skip_keys | frozen_keys
        
        # Parameter bounds (always elementwise; safe on any shape)
        if self.disk is not None:
            for pname, (lo, hi) in self.disk.parameter_bounds.items():
                key = f"disk.{pname}"
                if key in stacked and key not in skip_keys:
                    self._clamp_bounds_inplace(stacked[key], lo, hi)
        if self.dcbg is not None:
            for pname, (lo, hi) in self.dcbg.parameter_bounds.items():
                key = f"dcbg.{pname}"
                if key in stacked and key not in skip_keys:
                    self._clamp_bounds_inplace(stacked[key], lo, hi)

        for gaussbg_name, gaussbg in zip(self.gaussbg_names, self.gaussbgs):
            for pname, (lo, hi) in gaussbg.parameter_bounds.items():
                key = f"{gaussbg_name}.{pname}"
                if key in stacked and key not in skip_keys:
                    self._clamp_bounds_inplace(stacked[key], lo, hi)

        if self.lats is not None:
            for lat_name, lat in zip(self.lat_names, self.lats):
                for pname, (lo, hi) in lat.parameter_bounds.items():
                    key = f"{lat_name}.{pname}"
                    if key in stacked and key not in skip_keys:
                        self._clamp_bounds_inplace(stacked[key], lo, hi)

        disk_template_frozen = "disk.template_raw" in skip_keys
        disk_intensity_frozen = "disk.intensity_raw" in skip_keys

        # DiskTemplate composite hard constraints
        if self.disk is not None:
            template = stacked.get("disk.template_raw")
            intensity = stacked.get("disk.intensity_raw")
            cfg = self.disk.constraint_config
            force_positive = bool(self.disk.hard_constraints.get("force_positive", False))
            if not disk_template_frozen and template is not None and intensity is not None:
                if bool(self.disk.hard_constraints.get("force_center", False)):
                    self._batched_center_disk(template)
                if bool(self.disk.hard_constraints.get("force_cutoff", False)):
                    self._batched_enforce_cutoff(template, cfg)
                if bool(self.disk.hard_constraints.get("force_circular_mask", False)):
                    self._batched_enforce_circular_mask(template, cfg)
                if bool(self.disk.hard_constraints.get("force_shrinkage", False)):
                    template.sub_(float(cfg.get("shrinkage_amount", 0.25)))
                if force_positive:
                    template.clamp_(min=0.0)
                    if not disk_intensity_frozen and intensity is not None:
                        intensity.clamp_(min=0.0)
                if bool(self.disk.hard_constraints.get("force_norm", False)):
                    self._batched_enforce_norm(template)
            if force_positive and not disk_intensity_frozen and intensity is not None:
                intensity.clamp_(min=0.0)

        for lat_name, lat in zip(self.lat_names, self.lats):
            key = f"{lat_name}.i0_raw"
            if key not in skip_keys and bool(lat.hard_constraints.get("force_positive_intensity", False)):
                i0 = stacked.get(key)
                if i0 is not None:
                    i0.clamp_(min=0.0)

    @staticmethod
    def _clamp_bounds_inplace(t: torch.Tensor, lo: float | None, hi: float | None) -> None:
        if lo is None and hi is None:
            return
        if lo is None:
            t.clamp_(max=float(hi))  # type: ignore[arg-type]
        elif hi is None:
            t.clamp_(min=float(lo))
        else:
            t.clamp_(min=float(lo), max=float(hi))

    @staticmethod
    def _batched_center_disk(template_b: torch.Tensor) -> None:
        # template_b: (B, H_t, W_t)
        B, h, w = template_b.shape
        weights = template_b.clamp(min=0.0)
        mass = weights.sum(dim=(1, 2))
        if not torch.any(mass > 1e-12):
            return
        rr = torch.arange(h, device=template_b.device, dtype=template_b.dtype).view(1, h, 1)
        cc = torch.arange(w, device=template_b.device, dtype=template_b.dtype).view(1, 1, w)
        safe_mass = mass.clamp(min=1e-12)
        com_r = (weights * rr).sum(dim=(1, 2)) / safe_mass
        com_c = (weights * cc).sum(dim=(1, 2)) / safe_mass
        target_r = (h - 1) * 0.5
        target_c = (w - 1) * 0.5
        shift_r = target_r - com_r  # (B,)
        shift_c = target_c - com_c
        denom_h = max(h - 1, 1)
        denom_w = max(w - 1, 1)
        ty = -2.0 * shift_r / float(denom_h)
        tx = -2.0 * shift_c / float(denom_w)
        zeros = torch.zeros_like(tx)
        ones = torch.ones_like(tx)
        theta = torch.stack(
            [
                torch.stack([ones, zeros, tx], dim=1),
                torch.stack([zeros, ones, ty], dim=1),
            ],
            dim=1,
        )  # (B, 2, 3)
        src = template_b.unsqueeze(1)  # (B, 1, H, W)
        grid = F.affine_grid(theta, [B, 1, h, w], align_corners=True)
        shifted = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
        # Only shift samples with nonzero mass; others left as-is
        do_shift = (mass > 1e-12).view(B, 1, 1)
        template_b.copy_(torch.where(do_shift, shifted, template_b))

    @staticmethod
    def _batched_enforce_norm(template_b: torch.Tensor) -> None:
        mins = template_b.amin(dim=(1, 2), keepdim=True)
        template_b.sub_(mins)
        maxs = template_b.amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
        template_b.div_(maxs)

    @staticmethod
    def _batched_enforce_cutoff(template_b: torch.Tensor, cfg: dict[str, Any]) -> None:
        thresh_ratio = float(cfg.get("hard_cutoff_threshold", 0.35))
        maxs = template_b.amax(dim=(1, 2), keepdim=True)
        thresh = maxs * thresh_ratio
        mask = template_b <= thresh
        template_b.masked_fill_(mask, 0.0)

    @staticmethod
    def _batched_enforce_circular_mask(template_b: torch.Tensor, cfg: dict[str, Any]) -> None:
        B, h, w = template_b.shape
        radius = (min(h, w) / 2.0) * float(cfg.get("circular_mask_radius_fraction", 0.95))
        r = torch.arange(-h / 2, h / 2, device=template_b.device, dtype=template_b.dtype)
        c = torch.arange(-w / 2, w / 2, device=template_b.device, dtype=template_b.dtype)
        rr, cc = torch.meshgrid(r, c, indexing="ij")
        circle = torch.sqrt(rr * rr + cc * cc)
        if bool(cfg.get("soft_circular_mask", False)):
            sharpness = float(cfg.get("circular_mask_sharpness", 0))
            mask2d = torch.sigmoid(sharpness * (radius - circle))
        else:
            mask2d = (circle <= radius).to(dtype=template_b.dtype)
        template_b.mul_(mask2d.view(1, h, w))

    def build_sample_state_dict(
        self,
        init_state: dict[str, torch.Tensor],
        stacked: dict[str, torch.Tensor],
        b: int,
    ) -> dict[str, torch.Tensor]:
        """Extract sample b from stacked params, handling frozen params."""
        out = {k: v.detach().clone() for k, v in init_state.items()}

        # Origin: top-level key plus any 'components.X.origin.coords' or
        # 'components.X.disk.origin.coords' that PyTorch state_dict registers
        if "origin.coords" in stacked:
            origin_val = stacked["origin.coords"][b].detach().clone()
        else:
            # Origin was frozen - use the init value
            origin_val = init_state.get("origin.coords")
            if origin_val is not None:
                origin_val = origin_val.detach().clone()
        
        if origin_val is not None:
            for key in list(out.keys()):
                if key == "origin.coords" or key.endswith(".origin.coords"):
                    out[key] = origin_val.clone()

        # DiskTemplate (shared with lat.disk)
        if self.disk is not None and self.disk_idx is not None:
            if "disk.template_raw" in stacked:
                t_val = stacked["disk.template_raw"][b].detach().clone()
            else:
                # Template was frozen
                t_val = init_state.get(f"components.{self.disk_idx}.template_raw")
                if t_val is not None:
                    t_val = t_val.detach().clone()
            
            if "disk.intensity_raw" in stacked:
                i_val = stacked["disk.intensity_raw"][b].detach().clone()
            else:
                # Intensity was frozen
                i_val = init_state.get(f"components.{self.disk_idx}.intensity_raw")
                if i_val is not None:
                    i_val = i_val.detach().clone()
            
            if t_val is not None:
                for key in list(out.keys()):
                    if key.endswith(".template_raw"):
                        out[key] = t_val.clone()
            
            if i_val is not None:
                keys = [f"components.{self.disk_idx}.intensity_raw"]
                for lat_name, lat_idx in zip(self.lat_names, self.lat_idxs):
                    if self.lat_disk_key.get(lat_name) == "disk":
                        keys.append(f"components.{lat_idx}.disk.intensity_raw")
                for key in keys:
                    if key in out:
                        out[key] = i_val.clone()

        # DCBackground
        if self.dcbg is not None and self.dcbg_idx is not None:
            if "dcbg.intensity_raw" in stacked:
                out[f"components.{self.dcbg_idx}.intensity_raw"] = stacked["dcbg.intensity_raw"][b].detach().clone()
            else:
                # Was frozen - keep init value (already in out)
                pass

        # GaussianBackground
        for gaussbg_name, gaussbg_idx in zip(self.gaussbg_names, self.gaussbg_idxs):
            sigma_key = f"{gaussbg_name}.sigma_raw"
            if sigma_key in stacked:
                out[f"components.{gaussbg_idx}.sigma_raw"] = stacked[sigma_key][b].detach().clone()
            sigma_col_key = f"{gaussbg_name}.sigma_col_raw"
            if sigma_col_key in stacked:
                out[f"components.{gaussbg_idx}.sigma_col_raw"] = stacked[sigma_col_key][b].detach().clone()
            intensity_key = f"{gaussbg_name}.intensity_raw"
            if intensity_key in stacked:
                out[f"components.{gaussbg_idx}.intensity_raw"] = stacked[intensity_key][b].detach().clone()

        # SyntheticDiskLattice scalar + tensor params
        if self.lats is not None and self.lat_idxs is not None:
            for lat_name, lat_idx in zip(self.lat_names, self.lat_idxs):
                for attr in ("u_row", "u_col", "v_row", "v_col", "i0_raw", "ir", "ic", "irr", "icc", "irc"):
                    key = f"components.{lat_idx}.{attr}"
                    stacked_key = f"{lat_name}.{attr}"
                    if key in out and stacked_key in stacked:
                        out[key] = stacked[stacked_key][b].detach().clone()
                    # else: was frozen, keep init value
                
        return out
    
    def set_scheduler_params(self, scheduler_params: Any) -> None:
        """Configure per-key scheduler specs."""
        if scheduler_params is None:
            return
        if not isinstance(scheduler_params, dict):
            return

        def _normalize(spec: dict[str, Any]) -> dict[str, Any]:
            out = dict(spec)
            t = str(out.pop("type", out.pop("name", "none"))).lower()
            if t in ("cosine", "cosineannealing"):
                t = "cosine_annealing"
            if t == "cosine_annealing":
                if "t_max" in out:
                    out["T_max"] = out.pop("t_max")
                if "T_max" in out and out["T_max"] is not None:
                    out["T_max"] = int(float(out["T_max"]))
                if "eta_min" in out:
                    out["eta_min"] = float(out["eta_min"])
            elif t == "exponential":
                if "gamma" in out:
                    out["gamma"] = float(out["gamma"])
            elif t == "linear":
                for k in ("start_factor", "end_factor", "total_iters"):
                    if k in out and out[k] is not None:
                        out[k] = float(out[k]) if k != "total_iters" else int(float(out[k]))
            out["type"] = t
            return out

        if "type" in scheduler_params or "name" in scheduler_params:
            spec = _normalize(cast(dict[str, Any], scheduler_params))
            for k in self.scheduler_specs:
                self.scheduler_specs[k] = dict(spec)
            return

        for comp_name, spec in scheduler_params.items():
            if not isinstance(spec, dict):
                continue
            norm = _normalize(spec)
            param_keys = self.component_keys.get(str(comp_name), [])
            for k in param_keys:
                if k in self.scheduler_specs:
                    self.scheduler_specs[k] = dict(norm)

    def lr_at_step(self, step: int, n_steps: int) -> dict[str, float]:
        """Evaluate the analytic LR schedule for each stacked-param key at step."""
        out: dict[str, float] = {}
        for key, base_lr in self.lrs.items():
            spec = self.scheduler_specs.get(key, {"type": "none"})
            t = str(spec.get("type", "none")).lower()
            if t == "cosine_annealing":
                T_max = int(spec.get("T_max") or n_steps)
                if T_max < 1:
                    T_max = 1
                eta_min = float(spec.get("eta_min", 1e-7))
                s = max(step - 1, 0)
                import math
                lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * s / T_max))
                out[key] = float(lr)
            elif t == "exponential":
                gamma = float(spec.get("gamma", 1.0))
                out[key] = float(base_lr * (gamma ** max(step - 1, 0)))
            elif t == "linear":
                start = float(spec.get("start_factor", 1.0 / 3.0))
                end = float(spec.get("end_factor", 1.0))
                total = int(spec.get("total_iters", n_steps) or n_steps)
                if total < 1:
                    total = 1
                s = min(max(step - 1, 0), total)
                factor = start + (end - start) * (s / total)
                out[key] = float(base_lr * factor)
            else:
                out[key] = float(base_lr)
        return out

    def batched_constraint_loss(
        self,
        ctx: RenderContext,
        stacked: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Per-sample soft-constraint loss summed across components."""
        B = next(iter(stacked.values())).shape[0] if stacked else 1
        out = torch.zeros(B, device=ctx.device, dtype=ctx.dtype)
        if self.disk is not None:
            template_b = stacked.get("disk.template_raw")
            if template_b is not None:
                out = out + self.disk.constraint_loss_batched(ctx, template_raw_b=template_b)
        return out

    def resolve_component_keys(self, components: Any) -> list[str]:
        """Resolve a name/list of component names to the stacked-param keys they own."""
        if components is None:
            return []
        if isinstance(components, str):
            components = [components]
        keys: list[str] = []
        for name in components:
            ks = self.component_keys.get(str(name))
            if ks is None:
                raise KeyError(
                    f"Unknown component name '{name}' for batched fit. "
                    f"Known: {sorted(self.component_keys)}"
                )
            for k in ks:
                if k not in keys:
                    keys.append(k)
        return keys