from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
from tqdm.auto import tqdm

from quantem.core.visualization import show_2d

_OPT_SPEC_MARKER = "__opt_param__"


def OptimizationParameter(
    low: Optional[Union[int, float]] = None,
    high: Optional[Union[int, float]] = None,
    *,
    choices: Optional[Sequence[Any]] = None,
    step: Optional[Union[int, float]] = None,
    log: bool = False,
    kind: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an embedded optimization spec."""
    if choices is None and (low is None or high is None):
        raise ValueError("OptimizationParameter requires either choices or both low and high.")
    if choices is not None and (low is not None or high is not None):
        raise ValueError("Provide either choices or low/high, not both.")
    if log and step is not None:
        raise ValueError("step is not supported with log=True.")

    return {
        _OPT_SPEC_MARKER: True,
        "low": low,
        "high": high,
        "choices": list(choices) if choices is not None else None,
        "step": step,
        "log": bool(log),
        "kind": kind,
        "name": name,
    }


def _replace_opt_params_with_best(config, best_params):
    """Replace all OptimizationParameter specs with best values from previous study."""

    def replace_recursive(obj, path=()):
        if _is_opt_spec(obj):
            # Generate parameter name from path
            param_name = obj.get("name") or ".".join(str(p) for p in path)
            if param_name in best_params:
                return best_params[param_name]
            else:
                raise ValueError(f"Parameter '{param_name}' not found in previous study.")

        if isinstance(obj, dict):
            return {k: replace_recursive(v, (*path, k)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(replace_recursive(v, (*path, i)) for i, v in enumerate(obj))
        return obj

    return replace_recursive(config)


def _merge_new_params(config, new_params):
    """Merge new OptimizationParameters into config."""
    for param_path, param_value in new_params.items():
        # Auto-detect placement
        if _is_dataset_param(param_path):
            target = config.setdefault("dataset_preprocess_kwargs", {})
        else:
            target = config.setdefault("base_kwargs", {})

        _set_nested_value(target, param_path, param_value)


def _is_opt_spec(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get(_OPT_SPEC_MARKER) is True


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


def _suggest_from_spec(trial, spec: Dict[str, Any], name: str) -> Any:
    # Categorical
    if spec.get("choices") is not None:
        choices = spec["choices"]
        return trial.suggest_categorical(name, choices)

    low = spec.get("low")
    high = spec.get("high")
    step = spec.get("step")
    log = bool(spec.get("log", False))
    kind = spec.get("kind")

    if low is None or high is None:
        msg = f"OptimizationParameter '{name}' requires low/high or choices."
        raise ValueError(msg)

    # Infer kind if not set
    if kind is None:
        ints = all(isinstance(v, int) for v in (low, high)) and (
            step is None or isinstance(step, int)
        )
        kind = "int" if ints else "float"

    if kind == "int":
        low_i, high_i = int(low), int(high)
        if step is not None:
            return trial.suggest_int(name, low=low_i, high=high_i, step=int(step))
        return trial.suggest_int(name, low=low_i, high=high_i)

    if kind == "float":
        low_f, high_f = float(low), float(high)
        if log:
            return trial.suggest_float(name, low=low_f, high=high_f, log=True)
        if step is not None:
            return trial.suggest_float(name, low=low_f, high=high_f, step=float(step))
        return trial.suggest_float(name, low=low_f, high=high_f)

    if kind == "categorical":
        msg = "kind='categorical' requires 'choices'."
        raise ValueError(msg)

    msg = f"Unsupported kind='{kind}' for OptimizationParameter '{name}'."
    raise ValueError(msg)


def _resolve_params_with_trial(
    trial: optuna.trial.Trial,
    obj: Any,
    path: Iterable[Union[str, int]] = (),
) -> Any:
    """Recursively traverse a nested structure and replace OptimizationParameter specs.

    - Dicts/lists/tuples are reconstructed with the same shape.
    - Leaves (e.g., tensors, datasets, callables) are passed by reference.
    - Parameter name defaults to the dotted path; can be overridden via spec['name'].
    """
    # Optimization spec leaf - check for the special dict marker
    if _is_opt_spec(obj):
        pname = obj.get("name") or ".".join(str(p) for p in path)
        return _suggest_from_spec(trial, obj, pname)

    # Dict-like
    if isinstance(obj, dict):
        return {k: _resolve_params_with_trial(trial, v, (*path, k)) for k, v in obj.items()}

    # List/tuple
    if isinstance(obj, list):
        return [_resolve_params_with_trial(trial, v, (*path, i)) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(_resolve_params_with_trial(trial, v, (*path, i)) for i, v in enumerate(obj))

    # Other leaves unchanged
    return obj


def _build_ptychography_instance(constructors, resolved_kwargs):
    """Build Ptychography instance (existing logic)."""
    obj_kwargs = resolved_kwargs.get("object", {})
    obj_model = constructors["object"](**obj_kwargs)

    probe_kwargs = resolved_kwargs.get("probe", {})
    probe_model = constructors["probe"](**probe_kwargs)

    detector_kwargs = resolved_kwargs.get("detector", {})
    detector_model = constructors["detector"](**detector_kwargs)

    init_kwargs = resolved_kwargs.get("init", {}).copy()
    init_kwargs["verbose"] = False

    return constructors["ptychography_class"](
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        **init_kwargs,
    )


def _build_ptycholite_instance(constructors, resolved_kwargs):
    """Build PtychoLite instance."""
    # FIX: Changed from "ptycholite" to "init"
    init_kwargs = resolved_kwargs.get("init", {}).copy()
    init_kwargs["verbose"] = False

    return constructors["ptychography_class"](**init_kwargs)


def _run_reconstruction_pipeline(recon_obj, resolved_kwargs, class_type):
    """Run the reconstruction pipeline for either class."""
    # Preprocess step
    preprocess_kwargs = resolved_kwargs.get("preprocess")

    if preprocess_kwargs:
        recon_obj.preprocess(**preprocess_kwargs)

    # Reconstruct step
    reconstruct_kwargs = resolved_kwargs.get("reconstruct", {})
    reconstruct_kwargs["verbose"] = False
    if reconstruct_kwargs:
        recon_obj.reconstruct(**reconstruct_kwargs)


def _extract_default_loss(recon_obj, class_type):
    """Extract loss from reconstruction object."""
    if class_type == "ptycholite":
        # Adjust based on how Ptychography stores losses
        losses = getattr(recon_obj, "_losses", None) or getattr(recon_obj, "_epoch_losses", None)
    else:
        losses = getattr(recon_obj, "_epoch_losses", None)

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
    reconstruction_class: str = "auto",  # "ptychography", "ptycholite", or "auto"
) -> Callable[[optuna.trial.Trial], float]:
    """Build and return an Optuna objective for iterative ptychography or Ptycholite.

    Args:
        reconstruction_class: Which class to use - "ptychography", "ptycholite", or "auto" to detect
    """

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
                pdset.preprocess(**resolved_preprocess_kwargs)

            resolved_kwargs.setdefault("init", {})["dset"] = pdset

        # 3) Determine which class to use
        if reconstruction_class == "auto":
            # Auto-detect directly from constructor name
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

        # 4) Build reconstruction object based on class type
        if class_type == "ptycholite":
            recon_obj = _build_ptycholite_instance(constructors, resolved_kwargs)
        else:
            recon_obj = _build_ptychography_instance(constructors, resolved_kwargs)

        # 5) Run the reconstruction pipeline
        _run_reconstruction_pipeline(recon_obj, resolved_kwargs, class_type)

        # 6) Extract loss
        if loss_getter is not None:
            return float(loss_getter(recon_obj))

        return _extract_default_loss(recon_obj, class_type)

    return objective


class OptimizePtychography:
    """Bayesian optimization for ptychography and Ptycholite reconstruction pipelines."""

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

        self.objective_func = None  # Will be set by factory methods
        self.n_trials = n_trials
        self.direction = direction
        self.study_kwargs = study_kwargs or {}
        self.unit = unit
        self.verbose = verbose
        self._config = None

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
        reconstruction_class: str = "auto",  # NEW: "ptychography", "ptycholite", or "auto"
        n_trials: int = 50,
        direction: str = "minimize",
        study_kwargs: Optional[Dict[str, Any]] = None,
        unit: str = "trial",
        verbose: bool = True,
    ):
        """Create optimizer from constructor functions and parameter specifications.

        Args:
            reconstruction_class: Which class to use - "ptychography", "ptycholite", or "auto"

        Examples:
            # For Ptychography
            constructors = {
                "object": ObjectPixelated.from_uniform,
                "probe": ProbePixelated.from_params,
                "detector": DetectorPixelated,
                "ptycho": Ptychography.from_models,
            }

            # For Ptycholite
            constructors = {
                "ptycholite": PtychoLite.from_dataset,
            }
        """
        # Create instance with basic settings
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
        # Set the objective function with Ptycholite support
        instance.objective_func = _OptimizePtychographyObjective(
            constructors=constructors,
            base_kwargs=base_kwargs,
            loss_getter=loss_getter,
            dataset_constructor=dataset_constructor,
            dataset_kwargs=dataset_kwargs,
            dataset_preprocess_kwargs=dataset_preprocess_kwargs,
            reconstruction_class=reconstruction_class,  # Ptycholite support restored
        )

        return instance

    @classmethod
    def from_optimizer(
        cls,
        previous_study: optuna.study.Study,
        new_params: Optional[Mapping[str, Any]] = None,
        n_trials: int = 50,
        direction: str = "minimize",
        study_kwargs: Optional[Dict[str, Any]] = None,
        unit: str = "trial",
        verbose: bool = False,
    ):
        """Create optimizer from previous study, automatically using best values.

        Args:
            previous_study: Completed study with user_attrs['config']
            new_params: Only NEW parameters to optimize (optional)

        Example:
            # First optimization
            study1 = OptimizeIterativePtychography.from_constructors(
                base_kwargs={"probe": {"probe_params": {"defocus": OptimizationParameter(-500, 500)}}},
                n_trials=20,
            ).optimize()

            # Second optimization - defocus automatically uses best value from study1
            study2 = OptimizeIterativePtychography.from_optimizer(
                previous_study=study1,
                new_params={
                    "probe.probe_params.C12": OptimizationParameter(0, 50),  # NEW optimization
                },
                n_trials=15,
            ).optimize()
        """
        # Get previous config and best params
        if "config" not in previous_study.user_attrs:
            raise ValueError("Previous study missing config. Use from_constructors().")

        prev_config = previous_study.user_attrs["config"]
        best_params = previous_study.best_params

        # Replace all OptimizationParameters with best values, then add new ones
        updated_config = _replace_opt_params_with_best(prev_config, best_params)

        # Add any new parameters to optimize
        if new_params:
            _merge_new_params(updated_config, new_params)

        instance = cls(n_trials, direction, study_kwargs, unit, verbose)
        instance._config = updated_config
        instance.objective_func = _OptimizePtychographyObjective(**updated_config)  # type: ignore
        return instance

    def optimize(self) -> OptimizePtychography:
        """Run the optimization study with progress bar."""
        if self.objective_func is None:
            msg = "No objective function set. Use a factory method like from_constructors()."
            raise RuntimeError(msg)

        # Control Optuna logging verbosity
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        # Run with embedded tqdm progress bar
        with tqdm(total=self.n_trials, desc="optimizing", unit=self.unit) as pbar:

            def _on_trial_end(study_: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
                pbar.update(1)

            self.study.optimize(
                self.objective_func,
                n_trials=self.n_trials,
                callbacks=[_on_trial_end],
                show_progress_bar=self.verbose,
            )

        # Restore original logging level
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        return self

    def visualize(self, figsize=(10, 6)):
        """Visualize optimization results showing parameter values vs loss."""

        if not self.study.trials:
            raise RuntimeError("No trials to plot. Run optimize() first.")

        # Get completed trials
        trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not trials:
            raise RuntimeError("No completed trials to plot.")

        # Get parameter names (all parameters that were optimized)
        param_names = list(trials[0].params.keys())

        # Get best trial info
        best_trial = self.study.best_trial
        best_value = best_trial.value

        # Special case: 2 parameters - add 2D scatter plot
        if len(param_names) == 2:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # First subplot: 2D scatter plot
            ax_2d = axes[0]
            param1, param2 = param_names

            # Extract data
            param1_values = np.array([trial.params[param1] for trial in trials])
            param2_values = np.array([trial.params[param2] for trial in trials])
            losses = np.array([trial.value for trial in trials])

            # 2D scatter with color-coded loss
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
            best_param1 = best_trial.params[param1]
            best_param2 = best_trial.params[param2]
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
                ax = axes[idx + 1]

                # Extract data
                param_values = np.array([trial.params[param_name] for trial in trials])
                losses = np.array([trial.value for trial in trials])

                # Scatter plot
                ax.scatter(
                    param_values, losses, alpha=0.6, s=50, edgecolors="black", linewidth=0.5
                )

                # Highlight best trial
                best_param_value = best_trial.params[param_name]
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

                # Clean up parameter name for label
                clean_name = param_name.split(".")[-1]

                # Labels
                ax.set_xlabel(clean_name, fontsize=11, fontweight="bold")
                ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
                ax.set_title(f"{param_name}", fontsize=10)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig, axes

        # General case: any number of parameters
        n_params = len(param_names)
        n_cols = min(3, n_params)  # Max 3 columns
        n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Plot each parameter
        for idx, param_name in enumerate(param_names):
            ax = axes[idx]

            # Extract data
            param_values = np.array([trial.params[param_name] for trial in trials])
            losses = np.array([trial.value for trial in trials])

            # Scatter plot
            ax.scatter(param_values, losses, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)

            # Highlight best trial
            best_param_value = best_trial.params[param_name]
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

            # Clean up parameter name for label
            clean_name = param_name.split(".")[-1]

            # Labels
            ax.set_xlabel(clean_name, fontsize=11, fontweight="bold")
            ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
            ax.set_title(f"{param_name}", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def _extract_optimization_params(self):
        """Extract OptimizationParameter specs from stored config."""
        param_info = {}

        def extract_recursive(obj, path=()):
            if _is_opt_spec(obj):
                param_name = obj.get("name") or ".".join(str(p) for p in path)
                param_info[param_name] = obj
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    extract_recursive(v, (*path, k))

        if hasattr(self, "_config") and self._config:
            extract_recursive(self._config.get("base_kwargs", {}))

        return param_info

    def grid_search(self, n_points=5, plot_objects=True, figsize=None, return_results=False):
        """Run grid search and plot reconstructed objects at each parameter value.

        Args:
            n_points: Number of evenly spaced points for each parameter
            plot_objects: Whether to plot the reconstructed objects
            figsize: Figure size (auto if None)
            return_results: if True, returns 'results', 'best_result',
            'param_grids', 'reconstructions'

        Returns:
            dict with 'results', 'best_result', 'param_grids', 'reconstructions'
        """
        from itertools import product

        import numpy as np

        if self.objective_func is None:
            raise RuntimeError("No objective function set. Use from_constructors() first.")

        # Extract optimization parameters
        param_info = self._extract_optimization_params()

        if not param_info:
            raise RuntimeError("No OptimizationParameter found in base_kwargs.")

        # Create grid of values
        param_grids = {}
        for param_name, spec in param_info.items():
            if spec.get("choices") is not None:
                param_grids[param_name] = spec["choices"]
            else:
                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    param_grids[param_name] = np.logspace(np.log10(low), np.log10(high), n_points)
                else:
                    param_grids[param_name] = np.linspace(low, high, n_points)

        param_names = list(param_grids.keys())
        all_combinations = list(product(*param_grids.values()))

        def objective_with_capture(trial):
            """Modified objective that captures the reconstruction object."""
            # Call the original objective
            loss = self.objective_func(trial)

            return loss

        # Enqueue all grid points
        for combo in all_combinations:
            params = dict(zip(param_names, combo))
            self.study.enqueue_trial(params)

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

        # Find best
        best_idx = np.argmin([r["loss"] for r in results])
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

    def _run_reconstruction_with_params(self, params):
        """Run a single reconstruction with given parameters and return the object.

        Args:
            params: Dict of parameter values

        Returns:
            tuple: (reconstruction_object, loss)
        """
        from quantem.diffractive_imaging.optimize_hyperparameters import _resolve_params_with_trial

        # Create a mock trial that returns our fixed parameters
        class FixedTrial:
            def __init__(self, fixed_params):
                self.params = fixed_params
                self.number = 0

            def suggest_float(self, name, low, high, **kwargs):
                return self.params.get(name, (low + high) / 2)

            def suggest_int(self, name, low, high, **kwargs):
                return int(self.params.get(name, (low + high) // 2))

            def suggest_categorical(self, name, choices):
                return self.params.get(name, choices[0])

        trial = FixedTrial(params)

        # Resolve parameters
        resolved_kwargs = _resolve_params_with_trial(trial, self._config["base_kwargs"])

        # Handle dataset construction if needed
        if self._config.get("dataset_constructor") is not None:
            resolved_dataset_kwargs = _resolve_params_with_trial(
                trial, self._config.get("dataset_kwargs", {})
            )
            pdset = self._config["dataset_constructor"](**resolved_dataset_kwargs)

            if self._config.get("dataset_preprocess_kwargs") is not None:
                resolved_preprocess_kwargs = _resolve_params_with_trial(
                    trial, self._config["dataset_preprocess_kwargs"]
                )
                pdset.preprocess(**resolved_preprocess_kwargs)

            resolved_kwargs.setdefault("init", {})["dset"] = pdset

        # Determine reconstruction class
        reconstruction_class = self._config.get("reconstruction_class", "auto")
        constructors = self._config["constructors"]

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

        _run_reconstruction_pipeline(recon_obj, resolved_kwargs, class_type)

        # Extract loss
        loss_getter = self._config.get("loss_getter")
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
        losses = [r["loss"] for r in results]
        best_idx = np.argmin(losses)

        for idx, result in enumerate(results):
            ax = axes[idx]

            recon_obj = result["reconstruction"]

            obj = recon_obj._to_numpy(recon_obj.obj_cropped)
            if recon_obj.obj_type == "potential":
                obj = np.abs(obj).sum(0)
            elif recon_obj.obj_type == "pure_phase":
                obj = np.angle(obj).sum(0)
            else:
                obj = np.angle(obj).sum(0)

            if obj is not None:
                show_2d(obj, cmap="magma", figax=(fig, ax))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No object\navailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_facecolor("#f0f0f0")

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
