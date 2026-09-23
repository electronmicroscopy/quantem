from __future__ import annotations

import copy
import gc
from typing import Any, Callable, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from tqdm.auto import tqdm

from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.dataset_models import (
    PtychographyDatasetBase,
    PtychographyDatasetRaster,
)
from quantem.diffractive_imaging.ptycho_utils import (
    OptimizationParameter as OptimizationParameter,  # re-export: the documented path
)
from quantem.diffractive_imaging.ptychography_lite import PtychoLite, PtychoLiteDIP


def _suggest_from_spec(trial: optuna.trial.Trial, spec: OptimizationParameter, name: str) -> float:
    """Sample a value from an OptimizationParameter using Optuna trial."""
    if spec.log:
        return trial.suggest_float(name, low=spec.low, high=spec.high, log=True)
    else:
        return trial.suggest_float(name, low=spec.low, high=spec.high)


def _resolve_params_with_trial(trial, config_dict, path_prefix=""):
    """Recursively resolve OptimizationParameter instances using trial suggestions.

    Args:
        trial: Optuna trial or FixedTrial instance
        config_dict: Configuration dictionary to resolve
        path_prefix: Dotted path prefix for nested parameters
    """
    resolved = {}
    for key, value in config_dict.items():
        # Build full parameter path
        full_path = f"{path_prefix}.{key}" if path_prefix else key

        if isinstance(value, OptimizationParameter):
            # Suggest value using the full dotted path
            if value.log:
                resolved[key] = trial.suggest_float(full_path, value.low, value.high, log=True)
            else:
                resolved[key] = trial.suggest_float(full_path, value.low, value.high)
        elif isinstance(value, dict):
            # Recursively resolve nested dicts, passing along the path
            resolved[key] = _resolve_params_with_trial(trial, value, full_path)
        else:
            # Keep non-parameter values as-is
            resolved[key] = value
    return resolved


def _replace_opt_params_with_best(config, best_params):
    """Replace all OptimizationParameter specs with best values from previous study."""

    def replace_recursive(obj, path=()):
        if isinstance(obj, OptimizationParameter):
            param_name = ".".join(str(p) for p in path)
            if param_name in best_params:
                return best_params[param_name]
            else:
                raise ValueError(f"Parameter '{param_name}' not found in previous study.")

        if isinstance(obj, dict):
            return {k: replace_recursive(v, (*path, k)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(replace_recursive(v, (*path, i)) for i, v in enumerate(obj))
        return obj

    # trial params are named relative to each sub-config, not to the whole config
    updated = dict(config)
    for key in ("base_kwargs", "dataset_kwargs", "dataset_preprocess_kwargs"):
        sub_config = updated.get(key)
        if sub_config is not None:
            updated[key] = replace_recursive(sub_config)
    return updated


def _is_dataset_param(param_path):
    """Check if parameter belongs in dataset_preprocess_kwargs."""
    dataset_params = {
        "com_fit_function",
        "plot_rotation",
        "plot_com",
        "probe_energy",
        "force_com_rotation",
        "force_com_transpose",
        "rotation_angle",
    }
    return param_path.split(".")[-1] in dataset_params


def _set_nested_value(target_dict, param_path, value):
    """Set value in nested dict using dotted path."""
    parts = param_path.split(".")
    current = target_dict
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _merge_new_params(config, new_params):
    """Merge new OptimizationParameters into config."""
    for param_path, param_value in new_params.items():
        if _is_dataset_param(param_path):
            target = config.setdefault("dataset_preprocess_kwargs", {})
        else:
            target = config.setdefault("base_kwargs", {})
        _set_nested_value(target, param_path, param_value)


def _build_ptychography_instance(constructors, resolved_kwargs):
    """Build Ptychography instance."""
    obj_kwargs = resolved_kwargs.get("object", {})
    obj_model = constructors["object"](**obj_kwargs)

    probe_kwargs = resolved_kwargs.get("probe", {})
    probe_model = constructors["probe"](**probe_kwargs)

    detector_kwargs = resolved_kwargs.get("detector", {})
    detector_model = constructors["detector"](**detector_kwargs)

    init_kwargs = resolved_kwargs.get("init", {}).copy()
    init_kwargs["verbose"] = False
    _isolate_trial_dataset(init_kwargs)

    return constructors["ptychography_class"](
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        **init_kwargs,
    )


def _build_ptycholite_instance(constructors, resolved_kwargs):
    """Build PtychoLite instance."""
    init_kwargs = resolved_kwargs.get("init", {}).copy()
    init_kwargs["verbose"] = False
    _isolate_trial_dataset(init_kwargs)

    return constructors["ptychography_class"](**init_kwargs)


def _isolate_trial_dataset(init_kwargs: dict[str, Any]) -> None:
    """Give each optimization trial a private mutable dataset model."""
    dset = init_kwargs.get("dset")
    if not isinstance(dset, PtychographyDatasetBase):
        return

    dset = _clone_ptychography_dataset(dset)
    dset.reset()
    dset.zero_grad(set_to_none=True)
    dset.reset_optimizer()
    init_kwargs["dset"] = dset


def _clone_ptychography_dataset(dset: PtychographyDatasetBase) -> PtychographyDatasetBase:
    """Clone a ptychography dataset without using torch Module deepcopy."""
    if not isinstance(dset, PtychographyDatasetRaster):
        raise RuntimeError(
            "Could not copy the ptychography dataset for an optimization trial. "
            "Pass a dataset_constructor instead so each trial can build a fresh dataset."
        )

    detector_mask = dset.detector_mask.detach().cpu().clone()
    origin = np.array([0, 0, *dset.dset.origin[-2:]])
    sampling = np.array([*dset.scan_sampling, *dset.detector_sampling])
    units = [*dset.scan_units, *dset.detector_units]
    cloned = PtychographyDatasetRaster.from_array(
        array=dset.intensities_4d.copy(),
        name=dset.dset.name,
        origin=origin,
        sampling=sampling,
        units=units,
        signal_units=dset.dset.signal_units,
        detector_mask=detector_mask,
        verbose=dset.verbose,
        learn_descan=dset.learn_descan,
        learn_scan_positions=dset.learn_scan_positions,
    )
    cloned.constraints = copy.deepcopy(dset.constraints)
    cloned._preprocessing_params = copy.deepcopy(dset._preprocessing_params)
    cloned.com_rotation_rad = dset.com_rotation_rad
    if hasattr(dset, "_transpose"):
        cloned.com_transpose = dset.com_transpose
    if dset.probe_energy is not None:
        cloned.probe_energy = dset.probe_energy

    if not dset.preprocessed:
        return cloned

    cloned.diffraction_padding = dset.diffraction_padding.copy()
    cloned.com_measured = dset.com_measured.copy()
    cloned.com_fit = dset.com_fit.copy()
    cloned.centered_amplitudes = dset.centered_amplitudes.detach().cpu().clone()
    # amplitudes / intensities / centered_intensities are derived on demand from
    # intensities_4d; only carry them over if the source has them materialized
    for attr in ("_amplitudes", "_intensities", "_centered_intensities"):
        source = getattr(dset, attr, None)
        if source is not None:
            setattr(cloned, attr[1:], source.detach().cpu().clone())
    cloned.detector_mask = detector_mask
    cloned.mean_diffraction_intensity = dset.mean_diffraction_intensity
    if hasattr(dset, "mean_diffraction_amplitude"):
        cloned.mean_diffraction_amplitude = dset.mean_diffraction_amplitude
    cloned._pattern_crop_mask = copy.deepcopy(getattr(dset, "_pattern_crop_mask", None))
    mask_shape = getattr(dset, "_pattern_crop_mask_shape", None)
    cloned._pattern_crop_mask_shape = (
        copy.deepcopy(mask_shape)
        if mask_shape is not None
        else (int(dset.roi_shape[0]), int(dset.roi_shape[1]))
    )
    cloned.initial_descan_shifts = dset.initial_descan_shifts.detach().cpu().clone()
    cloned.initial_scan_positions_px = dset.initial_scan_positions_px.detach().cpu().clone()
    cloned.descan_shifts = cloned.initial_descan_shifts.clone()
    cloned.scan_positions_px = cloned.initial_scan_positions_px.clone()
    cloned._patch_indices = dset.patch_indices.detach().cpu().clone()
    cloned._last_patch_positions_px = cloned.scan_positions_px.detach().clone()
    cloned._targets = dset.targets.detach().cpu().clone()
    cloned._preprocessed = True
    return cloned


def _run_reconstruction_pipeline(recon_obj, resolved_kwargs):
    """Run the reconstruction pipeline for either class."""
    # Preprocess step
    preprocess_kwargs = resolved_kwargs.get("preprocess")
    if preprocess_kwargs:
        recon_obj.preprocess(**preprocess_kwargs)

    # Reconstruct step
    recon_obj.verbose = False
    reconstruct_kwargs = resolved_kwargs.get("reconstruct")
    if reconstruct_kwargs:
        reconstruct_kwargs = dict(reconstruct_kwargs)
        # only PtychoLite/PtychoLiteDIP.reconstruct take verbose, and they reset recon_obj.verbose
        if isinstance(recon_obj, (PtychoLite, PtychoLiteDIP)):
            reconstruct_kwargs.setdefault("verbose", False)
        recon_obj.reconstruct(**reconstruct_kwargs)


def _extract_default_loss(recon_obj, class_type):
    """Extract loss from reconstruction object."""
    if class_type == "ptycholite":
        losses = getattr(recon_obj, "_losses", None) or getattr(recon_obj, "_iter_losses", None)
    else:
        losses = getattr(recon_obj, "_iter_losses", None)

    if not losses:
        msg = f"No losses available on {class_type} object. Provide a loss_getter."
        raise RuntimeError(msg)
    return float(losses[-1])


def _OptimizePtychographyObjective(
    constructors: Mapping[str, Callable[..., Any]],
    base_kwargs: Mapping[str, Any],
    loss_getter: Optional[Callable[[Any], float]] = None,
    dataset_constructor: Optional[Callable[..., Any]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    dataset_preprocess_kwargs: Optional[Mapping[str, Any]] = None,
    reconstruction_class: str = "auto",
) -> Callable[[optuna.trial.Trial], float]:
    """Build and return an Optuna objective for iterative ptychography or PtychoLite."""

    def objective(trial: optuna.trial.Trial) -> float:
        # 1) Resolve embedded OptimizationParameter specs to get sampled values
        resolved_kwargs = _resolve_params_with_trial(trial, base_kwargs)

        # 2) Handle dataset construction/preprocessing if optimizing dataset params
        if dataset_constructor is not None:
            resolved_dataset_kwargs = _resolve_params_with_trial(trial, dataset_kwargs or {})
            pdset = dataset_constructor(**resolved_dataset_kwargs)

            if dataset_preprocess_kwargs is not None:
                resolved_preprocess_kwargs = _resolve_params_with_trial(
                    trial, dataset_preprocess_kwargs
                )
                resolved_preprocess_kwargs["plot_rotation"] = False
                resolved_preprocess_kwargs["plot_com"] = False
                pdset.preprocess(**resolved_preprocess_kwargs)

            resolved_kwargs.setdefault("init", {})["dset"] = pdset

        # 3) Determine which class to use
        if reconstruction_class == "auto":
            main_constructor = constructors.get("ptychography_class")
            if main_constructor is None:
                raise ValueError("No ptychography_class constructor found.")

            constructor_name = str(main_constructor)
            if "PtychoLite" in constructor_name:
                class_type = "ptycholite"
            elif "Ptychography" in constructor_name:
                class_type = "ptychography"
            else:
                raise ValueError(
                    f"Could not auto-detect type from constructor: {constructor_name}"
                )
        else:
            class_type = reconstruction_class

        # 4) Build reconstruction object
        if class_type == "ptycholite":
            recon_obj = _build_ptycholite_instance(constructors, resolved_kwargs)
        else:
            recon_obj = _build_ptychography_instance(constructors, resolved_kwargs)

        # 5) Run the reconstruction pipeline
        _run_reconstruction_pipeline(recon_obj, resolved_kwargs)

        # 6) Extract loss
        if loss_getter is not None:
            return float(loss_getter(recon_obj))
        return _extract_default_loss(recon_obj, class_type)

    return objective


class OptimizePtychography:
    """Bayesian optimization for ptychography and PtychoLite reconstruction pipelines."""

    _token = object()

    def __init__(
        self,
        n_trials: int = 50,
        direction: str = "minimize",
        study_kwargs: Optional[Dict[str, Any]] = None,
        unit: str = "trial",
        verbose: bool = True,
        _token: object | None = None,
    ):
        """Initialize optimizer settings."""
        if _token is not self._token:
            raise RuntimeError("Use a factory method to instantiate this class.")
        self.objective_func = None
        self.n_trials = n_trials
        self.direction = direction
        self.study_kwargs = study_kwargs or {}
        self.unit = unit
        self.verbose = verbose
        self._config: Dict[str, Any] | None = None
        self.study = optuna.create_study(direction=direction, **self.study_kwargs)

    @classmethod
    def from_constructors(
        cls,
        constructors: Mapping[str, Callable[..., Any]],
        base_kwargs: Mapping[str, Any],
        dataset_constructor: Optional[Callable[..., Any]] = None,
        dataset_kwargs: Optional[Mapping[str, Any]] = None,
        dataset_preprocess_kwargs: Optional[Mapping[str, Any]] = None,
        loss_getter: Optional[Callable[[Any], float]] = None,
        reconstruction_class: str = "auto",
        n_trials: int = 50,
        direction: str = "minimize",
        study_kwargs: Optional[Dict[str, Any]] = None,
        unit: str = "trial",
        verbose: bool = True,
    ):
        """Create optimizer from constructor functions and parameter specifications."""
        instance = cls(
            n_trials=n_trials,
            direction=direction,
            study_kwargs=study_kwargs,
            unit=unit,
            verbose=verbose,
            _token=cls._token,
        )

        instance._config = {
            "constructors": constructors,
            "base_kwargs": base_kwargs,
            "dataset_constructor": dataset_constructor,
            "dataset_kwargs": dataset_kwargs,
            "dataset_preprocess_kwargs": dataset_preprocess_kwargs,
            "loss_getter": loss_getter,
            "reconstruction_class": reconstruction_class,
        }

        instance.objective_func = _OptimizePtychographyObjective(
            constructors=constructors,
            base_kwargs=base_kwargs,
            loss_getter=loss_getter,
            dataset_constructor=dataset_constructor,
            dataset_kwargs=dataset_kwargs,
            dataset_preprocess_kwargs=dataset_preprocess_kwargs,
            reconstruction_class=reconstruction_class,
        )

        return instance

    @classmethod
    def from_optimizer(
        cls,
        previous_study: optuna.study.Study,
        new_params: Optional[Mapping[str, OptimizationParameter]] = None,
        n_trials: int = 50,
        direction: str = "minimize",
        study_kwargs: Optional[Dict[str, Any]] = None,
        unit: str = "trial",
        verbose: bool = False,
    ):
        """Create optimizer from previous study, automatically using best values."""
        if "config" not in previous_study.user_attrs:
            raise ValueError("Previous study missing config. Use from_constructors().")

        prev_config = previous_study.user_attrs["config"]
        best_params = previous_study.best_params

        updated_config = _replace_opt_params_with_best(prev_config, best_params)

        if new_params:
            _merge_new_params(updated_config, new_params)

        instance = cls(n_trials, direction, study_kwargs, unit, verbose, _token=cls._token)
        instance._config = updated_config
        instance.objective_func = _OptimizePtychographyObjective(**updated_config)

        return instance

    def optimize(self) -> "OptimizePtychography":
        """Run the optimization study with progress bar."""
        if self.objective_func is None:
            raise RuntimeError(
                "No objective function set. Use a factory method like from_constructors()."
            )

        # Store config for chaining
        if hasattr(self, "_config") and self._config:
            self.study.set_user_attr("config", self._config)

        prev_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(
            optuna.logging.INFO if self.verbose else optuna.logging.WARNING
        )

        try:
            with tqdm(total=self.n_trials, desc="optimizing", unit=self.unit) as pbar:

                def _on_trial_end(
                    study_: optuna.study.Study, trial: optuna.trial.FrozenTrial
                ) -> None:
                    pbar.update(1)

                    torch.cuda.empty_cache()
                    gc.collect()

                self.study.optimize(
                    self.objective_func,
                    n_trials=self.n_trials,
                    callbacks=[_on_trial_end],
                    show_progress_bar=False,
                )
        finally:
            optuna.logging.set_verbosity(prev_verbosity)

        return self

    def visualize(self, figsize=None):
        """Visualize optimization results showing parameter values vs loss."""
        if not self.study.trials:
            raise RuntimeError("No trials to plot. Run optimize() first.")

        trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not trials:
            raise RuntimeError("No completed trials to plot.")

        # trials may not all sample the same parameters, so take the union
        param_names = list(dict.fromkeys(name for trial in trials for name in trial.params))
        best_trial = self.study.best_trial
        best_value = best_trial.value

        # Special case: 2 parameters - add 2D scatter plot
        if len(param_names) == 2:
            fig, axes = plt.subplots(1, 3, figsize=figsize or (15, 5))

            ax_2d = axes[0]
            param1, param2 = param_names

            pair_trials = [t for t in trials if param1 in t.params and param2 in t.params]
            param1_values = np.array([trial.params[param1] for trial in pair_trials])
            param2_values = np.array([trial.params[param2] for trial in pair_trials])
            losses = np.array([trial.value for trial in pair_trials])

            scatter = ax_2d.scatter(
                param1_values,
                param2_values,
                c=losses,
                s=100,
                cmap="magma",
                edgecolors="black",
                linewidth=0.5,
                alpha=0.8,
            )

            # Highlight best trial
            best_param1 = best_trial.params.get(param1)
            best_param2 = best_trial.params.get(param2)
            if best_param1 is not None and best_param2 is not None:
                ax_2d.scatter(
                    [best_param1],
                    [best_param2],
                    color="red",
                    s=300,
                    marker="*",
                    edgecolors="black",
                    linewidth=2,
                    zorder=5,
                )

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax_2d)
            cbar.set_label("Loss", fontsize=11, fontweight="bold")

            # Labels
            clean_name1 = param1.split(".")[-1]
            clean_name2 = param2.split(".")[-1]
            ax_2d.set_xlabel(clean_name1, fontsize=11, fontweight="bold")
            ax_2d.set_ylabel(clean_name2, fontsize=11, fontweight="bold")
            ax_2d.set_title("2D Parameter Space", fontsize=12, fontweight="bold")
            ax_2d.grid(True, alpha=0.3)

            # Second and third subplots: individual parameter plots
            for idx, param_name in enumerate(param_names):
                self._plot_param_panel(axes[idx + 1], trials, param_name, best_trial, best_value)

            plt.tight_layout()
            return fig, axes

        # General case: any number of parameters
        n_params = len(param_names)
        n_cols = min(3, n_params)  # Max 3 columns
        n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (10, 6), squeeze=False)
        axes = axes.flatten()

        # Plot each parameter
        for idx, param_name in enumerate(param_names):
            self._plot_param_panel(axes[idx], trials, param_name, best_trial, best_value)

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def _plot_param_panel(self, ax, trials, param_name, best_trial, best_value):
        """Scatter one parameter's values vs loss, highlighting the best trial."""
        param_trials = [t for t in trials if param_name in t.params]
        param_values = np.array([trial.params[param_name] for trial in param_trials])
        losses = np.array([trial.value for trial in param_trials])

        ax.scatter(param_values, losses, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)

        best_param_value = best_trial.params.get(param_name)
        if best_param_value is not None:
            ax.scatter(
                [best_param_value],
                [best_value],
                color="red",
                s=200,
                marker="*",
                edgecolors="black",
                linewidth=1.5,
                zorder=5,
            )
            # Vertical line at optimal parameter value
            ax.axvline(best_param_value, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

        clean_name = param_name.split(".")[-1]
        ax.set_xlabel(clean_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
        ax.set_title(f"{param_name}", fontsize=10)
        ax.grid(True, alpha=0.3)

    def _extract_optimization_params(self):
        """Extract OptimizationParameter specs from stored config."""
        param_info = {}

        def extract_recursive(obj, path=()):
            if isinstance(obj, OptimizationParameter):
                param_name = ".".join(str(p) for p in path)
                param_info[param_name] = obj
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    extract_recursive(v, (*path, k))

        if hasattr(self, "_config") and self._config:
            # Extract from base_kwargs
            extract_recursive(self._config.get("base_kwargs", {}))

            # Extract from dataset_preprocess_kwargs
            dataset_preprocess = self._config.get("dataset_preprocess_kwargs")
            if dataset_preprocess:
                extract_recursive(dataset_preprocess)

            # Extract from dataset_kwargs
            dataset_kw = self._config.get("dataset_kwargs")
            if dataset_kw:
                extract_recursive(dataset_kw)

        return param_info

    def grid_search(self, plot_objects=True, figsize=None, return_results=False):
        """Run grid search and plot reconstructed objects at each parameter value.
        Args:
            plot_objects: Whether to plot the reconstructed objects
            figsize: Figure size (auto if None)
            return_results: if True, returns 'results', 'best_result',
                        'param_grids', 'reconstructions'
        Returns:
            dict with 'results', 'best_result', 'param_grids', 'reconstructions'
        """
        from itertools import product

        if self.objective_func is None:
            raise RuntimeError("No objective function set. Use from_constructors() first.")

        # Extract optimization parameters
        param_info = self._extract_optimization_params()
        if not param_info:
            raise RuntimeError("No OptimizationParameter found in base_kwargs.")

        # Create grid of values using the parameter's grid_values() method
        param_grids = {}
        for param_name, spec in param_info.items():
            # Assuming spec is now an OptimizationParameter instance or dict
            if hasattr(spec, "grid_values"):
                param_grids[param_name] = spec.grid_values()
            elif isinstance(spec, dict):
                # If still a dict, convert to OptimizationParameter
                from quantem.diffractive_imaging.optimize_hyperparameters import (
                    OptimizationParameter,
                )

                param = OptimizationParameter(**spec)
                param_grids[param_name] = param.grid_values()
            else:
                raise ValueError(f"Invalid parameter spec for {param_name}")

        param_names = list(param_grids.keys())
        all_combinations = list(product(*param_grids.values()))

        # Run trials and capture reconstructions
        print("\nRunning reconstructions...")
        results = []

        with tqdm(total=len(all_combinations), desc="Grid search", unit="point") as pbar:
            for combo in all_combinations:
                params = dict(zip(param_names, combo))

                # Manually run reconstruction to capture the object
                recon_obj, loss = self._run_reconstruction_with_params(params)

                results.append(
                    {
                        "params": params.copy(),
                        "loss": loss,
                        "reconstruction": recon_obj,
                    }
                )

                pbar.update(1)

                torch.cuda.empty_cache()
                gc.collect()

        # Find best
        best_idx = self._best_index([r["loss"] for r in results])
        best_result = results[best_idx]

        # Plot objects
        if plot_objects:
            self._plot_grid_objects(results, param_names, figsize)

        if return_results:
            return {
                "results": results,
                "best_result": best_result,
                "param_grids": param_grids,
            }

    def _best_index(self, losses) -> int:
        """Index of the best loss given the study direction."""
        argfn = np.argmax if self.direction == "maximize" else np.argmin
        return int(argfn(losses))

    def _run_reconstruction_with_params(self, params):
        """Run a single reconstruction with given parameters and return the object.

        Args:
            params: Dict of parameter values

        Returns:
            tuple: (reconstruction_object, loss)
        """
        from quantem.diffractive_imaging.optimize_hyperparameters import _resolve_params_with_trial

        trial = optuna.trial.FixedTrial(params)

        config = self._config
        if config is None:
            raise RuntimeError("Optimizer is not configured; use a factory method first.")

        # Resolve parameters
        resolved_kwargs = _resolve_params_with_trial(trial, config["base_kwargs"])

        # Handle dataset construction if needed
        if config.get("dataset_constructor") is not None:
            resolved_dataset_kwargs = _resolve_params_with_trial(
                trial, config.get("dataset_kwargs", {})
            )
            pdset = config["dataset_constructor"](**resolved_dataset_kwargs)

            if config.get("dataset_preprocess_kwargs") is not None:
                resolved_preprocess_kwargs = _resolve_params_with_trial(
                    trial, config["dataset_preprocess_kwargs"]
                )
                pdset.preprocess(**resolved_preprocess_kwargs)

            resolved_kwargs.setdefault("init", {})["dset"] = pdset

        # Determine reconstruction class
        reconstruction_class = config.get("reconstruction_class", "auto")
        constructors = config["constructors"]

        if reconstruction_class == "auto":
            main_constructor = constructors.get("ptychography_class")
            if main_constructor is None:
                raise ValueError("No ptychography_class constructor found.")

            constructor_name = str(main_constructor)
            if "PtychoLite" in constructor_name:
                class_type = "ptycholite"
            elif "Ptychography" in constructor_name:
                class_type = "ptychography"
            else:
                raise ValueError(
                    f"Could not auto-detect type from constructor: {constructor_name}"
                )
        else:
            class_type = reconstruction_class

        # Build reconstruction object
        if class_type == "ptycholite":
            from quantem.diffractive_imaging.optimize_hyperparameters import (
                _build_ptycholite_instance,
            )

            recon_obj = _build_ptycholite_instance(constructors, resolved_kwargs)
        else:
            from quantem.diffractive_imaging.optimize_hyperparameters import (
                _build_ptychography_instance,
            )

            recon_obj = _build_ptychography_instance(constructors, resolved_kwargs)

        # Run reconstruction pipeline
        from quantem.diffractive_imaging.optimize_hyperparameters import (
            _run_reconstruction_pipeline,
        )

        _run_reconstruction_pipeline(recon_obj, resolved_kwargs)

        # Extract loss
        loss_getter = config.get("loss_getter")
        if loss_getter is not None:
            loss = float(loss_getter(recon_obj))
        else:
            from quantem.diffractive_imaging.optimize_hyperparameters import _extract_default_loss

            loss = _extract_default_loss(recon_obj, class_type)

        return recon_obj, loss

    def _plot_grid_objects(self, results, param_names, figsize):
        """Plot reconstructed objects from grid search."""

        n_results = len(results)

        # Auto-calculate figure size
        if figsize is None:
            n_cols = min(5, n_results)
            n_rows = (n_results + n_cols - 1) // n_cols
            figsize = (n_cols * 3, n_rows * 3.5)
        else:
            n_cols = min(5, n_results)
            n_rows = (n_results + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Find best result
        best_idx = self._best_index([r["loss"] for r in results])

        for idx, result in enumerate(results):
            ax = axes[idx]

            recon_obj = result["reconstruction"]

            obj = recon_obj.obj_cropped
            if recon_obj.obj_type == "potential":
                obj = np.abs(obj).sum(0)
            elif recon_obj.obj_type == "pure_phase":
                # pure_phase obj_cropped is a real phase array — plot directly
                obj = obj.sum(0)
            else:
                obj = np.angle(obj).sum(0)

            show_2d(obj, cmap="magma", figax=(fig, ax))

            # Title with parameters and loss
            param_str = ", ".join(
                [f"{k.split('.')[-1]}={v:.1f}" for k, v in result["params"].items()]
            )
            title = f"{param_str}\nLoss: {result['loss']:.2e}"

            # Highlight best
            if idx == best_idx:
                ax.set_title(title, fontweight="bold", color="red", fontsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)
            else:
                ax.set_title(title, fontsize=8)

            ax.axis("off")

        # Hide unused subplots
        for idx in range(n_results, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Grid Search: Reconstructed Objects", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()
        plt.show()
