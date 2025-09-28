from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

import optuna
from tqdm.auto import tqdm

_OPT_SPEC_MARKER = "__opt_param__"


def OptimizationParameter(
    low: Optional[Union[int, float]] = None,
    high: Optional[Union[int, float]] = None,
    choices: Optional[Sequence[Any]] = None,
    step: Optional[Union[int, float]] = None,
    log: bool = False,
    kind: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an embedded optimization spec to place directly in your kwargs.

    Examples:
        OptimizationParameter(-500, 500)                      # float (linear)
        OptimizationParameter(1e-5, 1e-2, log=True)           # float (log-uniform)
        OptimizationParameter(64, 256, step=64)               # int (inferred by ints/step)
        OptimizationParameter(choices=["adam", "sgd"])        # categorical
        OptimizationParameter(-5, 5, name="probe.defocus")    # custom Optuna name

    Args:
        low, high: numeric bounds for float/int (ignored for categorical)
        choices: categorical choices (mutually exclusive with low/high)
        step: step for float/int. If log=True, step must be None.
        log: use log-uniform float sampling
        kind: "float", "int", or "categorical". If None, inferred automatically.
        name: optional explicit Optuna parameter name; defaults to dotted path.

    Returns:
        A dict spec that the resolver will recognize and sample via Optuna.
    """
    if choices is None and (low is None or high is None):
        msg = "OptimizationParameter requires either choices or both low and high."
        raise ValueError(msg)
    if choices is not None and (low is not None or high is not None):
        # Allow but ignore low/high if choices provided? Better to enforce exclusivity:
        msg = "Provide either choices or low/high, not both."
        raise ValueError(msg)
    if log and step is not None:
        msg = "step is not supported with log=True."
        raise ValueError(msg)

    return {
        _OPT_SPEC_MARKER: True,
        "low": low,
        "high": high,
        "choices": list(choices) if choices is not None else None,
        "step": step,
        "log": bool(log),
        "kind": kind,  # optional override: "float" | "int" | "categorical"
        "name": name,  # optional override for parameter name
    }


def _is_opt_spec(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get(_OPT_SPEC_MARKER) is True


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

    ptycho_kwargs = resolved_kwargs.get("ptycho", {})
    return constructors["ptycho"](
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        **ptycho_kwargs,
    )


def _build_pftm_instance(constructors, resolved_kwargs):
    """Build PFTM instance."""
    # Assuming PFTM has a simpler constructor - adjust based on your PFTM API
    pftm_kwargs = resolved_kwargs.get("pftm", {})
    return constructors["pftm"](**pftm_kwargs)


def _run_reconstruction_pipeline(recon_obj, resolved_kwargs, class_type):
    """Run the reconstruction pipeline for either class."""
    # Initialize step (if applicable)
    init_kwargs = resolved_kwargs.get("initialize")
    if init_kwargs and hasattr(recon_obj, "initialize"):
        recon_obj.initialize(**init_kwargs)

    # Preprocess step
    preprocess_kwargs = resolved_kwargs.get("preprocess")
    if preprocess_kwargs:
        recon_obj.preprocess(**preprocess_kwargs)

    # Reconstruct step
    reconstruct_kwargs = resolved_kwargs.get("reconstruct")
    if reconstruct_kwargs:
        recon_obj.reconstruct(**reconstruct_kwargs)


def _extract_default_loss(recon_obj, class_type):
    """Extract loss from reconstruction object."""
    if class_type == "pftm":
        # Adjust based on how PFTM stores losses
        losses = getattr(recon_obj, "_losses", None) or getattr(recon_obj, "_epoch_losses", None)
    else:
        losses = getattr(recon_obj, "_epoch_losses", None)

    if not losses:
        msg = f"No losses available on {class_type} object. Provide a loss_getter."
        raise RuntimeError(msg)
    return float(losses[-1])


def _OptimizeIterativePtychographyObjective(
    constructors: Mapping[str, Callable[..., Any]],
    base_kwargs: Mapping[str, Any],
    loss_getter: Optional[Callable[[Any], float]] = None,
    dataset_constructor: Optional[Callable[..., Any]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    dataset_preprocess_kwargs: Optional[Mapping[str, Any]] = None,
    reconstruction_class: str = "auto",  # "ptychography", "pftm", or "auto"
) -> Callable[[optuna.trial.Trial], float]:
    """Build and return an Optuna objective for iterative ptychography or PFTM.

    Args:
        reconstruction_class: Which class to use - "ptychography", "pftm", or "auto" to detect
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

            resolved_kwargs.setdefault("ptycho", {})["dset"] = pdset

        # 3) Determine which class to use
        if reconstruction_class == "auto":
            # Auto-detect based on constructor name or other heuristic
            ptycho_constructor = constructors.get("ptycho")
            pftm_constructor = constructors.get("pftm")

            if pftm_constructor is not None:
                class_type = "pftm"
            elif ptycho_constructor is not None:
                class_type = "ptychography"
            else:
                msg = "Could not auto-detect reconstruction class. Provide 'ptycho' or 'pftm' constructor."
                raise ValueError(msg)
        else:
            class_type = reconstruction_class

        # 4) Build reconstruction object based on class type
        if class_type == "pftm":
            recon_obj = _build_pftm_instance(constructors, resolved_kwargs)
        else:
            recon_obj = _build_ptychography_instance(constructors, resolved_kwargs)

        # 5) Run the reconstruction pipeline
        _run_reconstruction_pipeline(recon_obj, resolved_kwargs, class_type)

        # 6) Extract loss
        if loss_getter is not None:
            return float(loss_getter(recon_obj))

        return _extract_default_loss(recon_obj, class_type)

    return objective


class OptimizeIterativePtychography:
    """Bayesian optimization for ptychography and PFTM reconstruction pipelines."""

    def __init__(
        self,
        n_trials: int = 50,
        direction: str = "minimize",
        study: Optional[optuna.study.Study] = None,
        study_kwargs: Optional[Dict[str, Any]] = None,
        unit: str = "trial",
        verbose: bool = True,
    ):
        """Initialize optimizer settings."""
        self.objective_func = None  # Will be set by factory methods
        self.n_trials = n_trials
        self.direction = direction
        self.study_kwargs = study_kwargs or {}
        self.unit = unit
        self.verbose = verbose

        # Create or use provided study
        if study is None:
            self.study = optuna.create_study(direction=direction, **self.study_kwargs)
        else:
            self.study = study

    @classmethod
    def from_constructors(
        cls,
        constructors: Mapping[str, Callable[..., Any]],
        base_kwargs: Mapping[str, Any],
        dataset_constructor: Optional[Callable[..., Any]] = None,
        dataset_kwargs: Optional[Mapping[str, Any]] = None,
        dataset_preprocess_kwargs: Optional[Mapping[str, Any]] = None,
        loss_getter: Optional[Callable[[Any], float]] = None,
        reconstruction_class: str = "auto",  # NEW: "ptychography", "pftm", or "auto"
        n_trials: int = 50,
        direction: str = "minimize",
        study: Optional[optuna.study.Study] = None,
        study_kwargs: Optional[Dict[str, Any]] = None,
        unit: str = "trial",
        verbose: bool = True,
    ):
        """Create optimizer from constructor functions and parameter specifications.

        Args:
            reconstruction_class: Which class to use - "ptychography", "pftm", or "auto"

        Examples:
            # For Ptychography
            constructors = {
                "object": ObjectPixelated.from_uniform,
                "probe": ProbePixelated.from_params,
                "detector": DetectorPixelated,
                "ptycho": Ptychography.from_models,
            }

            # For PFTM
            constructors = {
                "pftm": PFTM.from_dataset,
            }
        """
        # Create instance with basic settings
        instance = cls(
            n_trials=n_trials,
            direction=direction,
            study=study,
            study_kwargs=study_kwargs,
            unit=unit,
            verbose=verbose,
        )

        # Set the objective function with PFTM support
        instance.objective_func = _OptimizeIterativePtychographyObjective(
            constructors=constructors,
            base_kwargs=base_kwargs,
            loss_getter=loss_getter,
            dataset_constructor=dataset_constructor,
            dataset_kwargs=dataset_kwargs,
            dataset_preprocess_kwargs=dataset_preprocess_kwargs,
            reconstruction_class=reconstruction_class,  # PFTM support restored
        )

        return instance

    def optimize(self) -> optuna.study.Study:
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

        return self.study
