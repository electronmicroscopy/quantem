from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

import optuna
from tqdm.auto import tqdm

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
    ptycholite_kwargs = resolved_kwargs.get("ptycholite", {}).copy()
    ptycholite_kwargs["verbose"] = False  # Force verbose=False during optimization

    return constructors["ptychography_class"](**ptycholite_kwargs)


def _run_reconstruction_pipeline(recon_obj, resolved_kwargs, class_type):
    """Run the reconstruction pipeline for either class."""
    # Preprocess step
    preprocess_kwargs = resolved_kwargs.get("preprocess")

    if preprocess_kwargs:
        recon_obj.preprocess(**preprocess_kwargs)

    # Reconstruct step
    reconstruct_kwargs = resolved_kwargs.get("reconstruct", {})
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
            if "Ptycholite" in constructor_name:
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
        verbose: bool = True,
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
