from __future__ import annotations

from typing import Any, Literal, Sequence, cast

import numpy as np
import torch
from scipy.ndimage import shift as ndi_shift
from scipy.signal.windows import tukey
from tqdm import tqdm

from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d, Dataset4dstem
from quantem.core.fitting.base import (
    AdditiveRenderModel,
    FitBase,
    OriginND,
    RenderComponent,
    RenderContext,
)
from quantem.core.fitting.diffraction import DiskTemplate, SyntheticDiskLattice
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.optimizer_mixin import OptimizerType, SchedulerType
from quantem.core.utils.imaging_utils import cross_correlation_shift
from quantem.diffraction.model_fitting_visualizations import ModelDiffractionVisualizations


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
    ) -> "ModelDiffraction":
        arr = np.asarray(self.dataset.array)
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
        progress: bool = True,
    ) -> "ModelDiffraction":

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
    
    def get_individual_uv_vectors(self) -> "ModelDiffraction":
        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]
        # print(scan_r)
        
        self.u_array = np.empty(shape=(scan_r, scan_c, 2))
        self.v_array = np.empty(shape=(scan_r, scan_c, 2))
        if self.state_individual_refined is None:
            raise RuntimeError("Call .fit_individual_diffraction_pattern(...) first.")
        for r in range(scan_r):
            for c in range(scan_c):
                pos_state = self.state_individual_refined[r,c]
                if pos_state is None:
                    self.u_array[r,c,:] = None
                    self.v_array[r,c,:] = None
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
    
    def render_indivdual_pattern(self, row, col):
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

    def fit_strain(
        self, 
        mask_reference=None,
    ) -> "ModelDiffraction":
        u_fit = self.u_array
        v_fit = self.v_array

        if mask_reference is None:
            u_ref = np.median(u_fit.reshape(-1, 2), axis=0)
            v_ref = np.median(v_fit.reshape(-1, 2), axis=0)
        else:
            m = np.asarray(mask_reference, dtype=bool)
            u_ref = np.array(
                (
                    np.median(u_fit[m, 0]),
                    np.median(u_fit[m, 1]),
                ),
                dtype=float,
            )
            v_ref = np.array(
                (
                    np.median(v_fit[m, 0]),
                    np.median(v_fit[m, 1]),
                ),
                dtype=float,
            )

        scan_r = self.dataset.shape[0]
        scan_c = self.dataset.shape[1]

        Uref = np.stack((u_ref, v_ref), axis=1).astype(float)
        strain_trans = np.zeros((scan_r, scan_c, 2, 2))
        for r in range(scan_r):
            for c in range(scan_c):
                U = np.stack((u_fit[r, c, :], v_fit[r, c, :]), axis=1)
                det = np.linalg.det(U)
                if not np.isfinite(det) or abs(det) < 1e-12:
                    U_inv = np.linalg.pinv(U)
                else:
                    U_inv = np.linalg.inv(U)
                strain_trans[r, c, :, :] = Uref @ U_inv

        self.strain_raw_err = Dataset2d.from_array(
            strain_trans[:, :, 0, 0] - 1,
            name="strain err",
            signal_units="fractional",
        )
        self.strain_raw_ecc = Dataset2d.from_array(
            strain_trans[:, :, 1, 1] - 1,
            name="strain ecc",
            signal_units="fractional",
        )
        self.strain_raw_erc = Dataset2d.from_array(
            strain_trans[:, :, 1, 0] * 0.5 + strain_trans[:, :, 0, 1] * 0.5,
            name="strain erc",
            signal_units="fractional",
        )
        self.strain_rotation = Dataset2d.from_array(
            strain_trans[:, :, 1, 0] * -0.5 + strain_trans[:, :, 0, 1] * 0.5,
            name="strain rotation",
            signal_units="fractional",
        )

        return self


    def plot_strain(
        self,
        rotation_angle=20,
        strain_range_percent=(-3.0, 3.0),
        rotation_range_degrees=(-2.0, 2.0),
        plot_rotation=True,
        cmap_strain="RdBu_r",
        cmap_rotation="PiYG",
        layout="horizontal",
        figsize=(6, 6),
    ):
        import matplotlib.pyplot as plt

        if cmap_rotation is None:
            cmap_rotation = cmap_strain

        angle = np.deg2rad(rotation_angle)
        c = np.cos(angle)
        s = np.sin(angle)

        err = self.strain_raw_err.array
        ecc = self.strain_raw_ecc.array
        erc = self.strain_raw_erc.array

        euu = err * (c * c) + 2.0 * erc * (c * s) + ecc * (s * s)
        evv = err * (s * s) - 2.0 * erc * (c * s) + ecc * (c * c)
        euv = (ecc - err) * (c * s) + erc * (c * c - s * s)

        strain_euu = self.strain_raw_err.copy()
        strain_evv = self.strain_raw_ecc.copy()
        strain_euv = self.strain_raw_erc.copy()
        strain_euu.array[...] = euu
        strain_evv.array[...] = evv
        strain_euv.array[...] = euv

        if layout != "horizontal":
            raise ValueError("layout must be 'horizontal'")

        ncols = 4 if plot_rotation else 3
        fig, ax = plt.subplots(1, ncols, figsize=figsize)

        cm_strain = plt.get_cmap(cmap_strain).copy()
        cm_strain.set_bad(color="black")
        cm_rot = plt.get_cmap(cmap_rotation).copy()
        cm_rot.set_bad(color="black")

        euu_pct = strain_euu.array * 100
        evv_pct = strain_evv.array * 100
        euv_pct = strain_euv.array * 100
        rot_deg = np.rad2deg(self.strain_rotation.array)

        title_fs = 16
        im0 = ax[0].imshow(
            euu_pct,
            vmin=strain_range_percent[0],
            vmax=strain_range_percent[1],
            cmap=cm_strain,
        )
        ax[1].imshow(
            evv_pct,
            vmin=strain_range_percent[0],
            vmax=strain_range_percent[1],
            cmap=cm_strain,
        )
        ax[2].imshow(
            euv_pct,
            vmin=strain_range_percent[0],
            vmax=strain_range_percent[1],
            cmap=cm_strain,
        )

        ax[0].set_title(r"$\epsilon_{uu}$", fontsize=title_fs)
        ax[1].set_title(r"$\epsilon_{vv}$", fontsize=title_fs)
        ax[2].set_title(r"$\epsilon_{uv}$", fontsize=title_fs)

        if plot_rotation:
            im3 = ax[3].imshow(
                rot_deg,
                vmin=rotation_range_degrees[0],
                vmax=rotation_range_degrees[1],
                cmap=cm_rot,
            )
            ax[3].set_title("Rotation", fontsize=title_fs)

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            a.set_facecolor("black")

        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, wspace=0.03)

        b0 = ax[0].get_position()
        b2 = ax[2].get_position()
        left = b0.x0
        right = b2.x1
        width = right - left

        b3 = ax[3].get_position() if plot_rotation else None

        cb_height = 0.04
        cb_pad = 0.03
        y = b0.y0 - cb_pad - cb_height

        cax1 = fig.add_axes([left, y, width, cb_height])
        cbar1 = fig.colorbar(im0, cax=cax1, orientation="horizontal")
        cbar1.set_label("Strain (%)", fontsize=title_fs)
        cbar1.ax.tick_params(labelsize=12)

        if plot_rotation:
            left_r = b3.x0
            width_r = b3.x1 - b3.x0
            cax2 = fig.add_axes([left_r, y, width_r, cb_height])
            cbar2 = fig.colorbar(im3, cax=cax2, orientation="horizontal")
            cbar2.set_label("Rotation (deg)", fontsize=title_fs)
            cbar2.ax.tick_params(labelsize=12)

        for a in ax:
            a.set_aspect("equal")

        return fig, ax

    @property
    def render_mean_refined(self) -> np.ndarray:
        if self.state_mean_refined is None:
            raise RuntimeError(
                "mean_refined state is unavailable. Run .fit_mean_diffraction_pattern(...) first."
            )
        return self._render_state_array(self.state_mean_refined)
