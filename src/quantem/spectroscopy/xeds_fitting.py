from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from quantem.core import config
from quantem.core.ml.optimizer_mixin import OptimizerMixin, OptimizerParams
from quantem.spectroscopy.spectroscopy_models import (
    CubeSpectrumModel,
    MeanSpectrumModel,
    total_variation_in_plane,
    xeds_data_loss,
)


def _normalize_choice(name, param_name, allowed_values):
    name_norm = str(name).lower()
    if name_norm not in allowed_values:
        allowed_display = "', '".join(sorted(allowed_values))
        raise ValueError(f"{param_name} must be '{allowed_display}'")
    return name_norm


def _resolve_torch_device(device):
    device_name, _ = config.validate_device(config.get("device") if device is None else device)
    return torch.device(device_name)


class XEDSFitter(torch.nn.Module, OptimizerMixin):
    """Small framework-backed fitter for XEDS spectrum models."""

    def __init__(self, model):
        torch.nn.Module.__init__(self)
        OptimizerMixin.__init__(self)
        self.model = model

    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        return {key: list(params) for key, params in self.model.get_params().items()}

    def _normalize_optimizer_params(self, params):
        norm = super()._normalize_optimizer_params(params)
        if set(norm) == {self.DEFAULT_OPTIMIZER_KEY}:
            spec = norm[self.DEFAULT_OPTIMIZER_KEY]
            return {key: deepcopy(spec) for key in self.model.param_keys}

        expected = set(self.model.param_keys)
        got = set(norm)
        if got != expected:
            raise ValueError(
                f"optimizer_params keys must match model.param_keys: got {got}, expected {expected}"
            )
        return norm

    def run(self, num_iters, loss_fn, verbose=False):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")

        loss_history = []
        for i in range(num_iters):
            if isinstance(self.optimizer, torch.optim.LBFGS):

                def closure():
                    self.zero_optimizer_grad()
                    loss_value = loss_fn()
                    loss_value.backward()
                    return loss_value

                loss_value = self.optimizer.step(closure)
                if not torch.is_tensor(loss_value):
                    with torch.no_grad():
                        loss_value = loss_fn()
            else:
                self.zero_optimizer_grad()
                loss_value = loss_fn()
                loss_value.backward()
                self.step_optimizer()

            loss_scalar = float(loss_value.detach().cpu().item())
            loss_history.append(loss_scalar)
            if verbose and ((i + 1) % max(1, num_iters // 10) == 0 or i == 0):
                print(f"iter {i + 1:4d}/{num_iters}: loss={loss_scalar:.6g}")

        return loss_history


def _slice_spectra_to_energy_range(self, energy_range, device):
    energy_axis = torch.tensor(self.energy_axis, dtype=torch.float32, device=device)
    spectra = torch.tensor(self.array, dtype=torch.float32, device=device)

    if energy_range is None:
        fit_range = [float(energy_axis.min().item()), float(energy_axis.max().item())]
        return energy_axis, spectra, fit_range

    emin, emax = map(float, energy_range)
    mask = (energy_axis >= emin) & (energy_axis <= emax)
    if not torch.any(mask):
        raise ValueError("energy_range does not overlap the dataset energy axis")
    return energy_axis[mask], spectra[:, :, mask], [emin, emax]


def _resolve_element_filters(self, elements_to_fit):
    if elements_to_fit is not None:
        return type(self)._parse_element_selectors(
            elements_to_fit,
            param_name="elements_to_fit",
        )

    if not self.model_elements:
        raise ValueError("elements_to_fit must be specified")

    all_info = type(self)._ensure_element_info()
    element_filters = {}
    for element_name, selected_lines in self.model_elements.items():
        element_name = str(element_name)
        if not isinstance(selected_lines, dict) or not selected_lines:
            element_filters[element_name] = None
            continue

        all_lines = all_info.get(element_name) or {}
        if all_lines and len(selected_lines) >= len(all_lines):
            element_filters[element_name] = None
            continue

        suffixes = {
            str(type(self)._canonical_line_name(line_name))
            for line_name in selected_lines
            if str(line_name).strip()
        }
        element_filters[element_name] = suffixes or None

    return element_filters


def _collect_element_basis_data(self, energy_axis, elements_to_fit):
    element_filters = _resolve_element_filters(self, elements_to_fit)
    energy_min = float(torch.min(energy_axis).item())
    energy_max = float(torch.max(energy_axis).item())
    all_info = type(self)._ensure_element_info()

    element_names = []
    line_labels = []
    line_energies = []
    line_weights = []
    line_element_indices = []

    for element_index, element_name in enumerate(element_filters):
        lines = all_info.get(str(element_name)) or {}
        selected_rows = []
        for line_name, line_info in lines.items():
            if not type(self)._line_allowed_for_element(element_name, line_name, element_filters):
                continue
            try:
                energy = float(line_info.get("energy (keV)", line_info.get("energy")))
            except (AttributeError, TypeError, ValueError):
                continue
            if not (energy_min <= energy <= energy_max):
                continue
            try:
                weight = float(line_info.get("weight", 0.0))
            except (AttributeError, TypeError, ValueError):
                weight = 0.0
            label = f"{element_name}{type(self)._canonical_line_name(line_name)}"
            selected_rows.append((label, energy, max(weight, 0.0)))

        if not selected_rows:
            raise ValueError(
                f"No X-ray lines from '{element_name}' are inside the selected energy range."
            )

        raw_weights = np.asarray([row[2] for row in selected_rows], dtype=float)
        if np.all(raw_weights <= 0):
            normalized_weights = np.full(raw_weights.shape, 1.0 / raw_weights.size, dtype=float)
        else:
            normalized_weights = raw_weights / raw_weights.sum()

        element_names.append(str(element_name))
        for (label, energy, _weight), normalized_weight in zip(selected_rows, normalized_weights):
            line_labels.append(label)
            line_energies.append(energy)
            line_weights.append(float(normalized_weight))
            line_element_indices.append(element_index)

    return {
        "element_names": element_names,
        "line_labels": line_labels,
        "line_energies": torch.tensor(
            line_energies,
            dtype=energy_axis.dtype,
            device=energy_axis.device,
        ),
        "line_weights": torch.tensor(
            line_weights,
            dtype=energy_axis.dtype,
            device=energy_axis.device,
        ),
        "line_element_indices": torch.tensor(
            line_element_indices,
            dtype=torch.long,
            device=energy_axis.device,
        ),
    }


def _build_optimizer_params(
    optimizer_name,
    lr,
    default_lr_adam,
    default_lr_lbfgs,
):
    if optimizer_name == "adam":
        return OptimizerParams.Adam(lr=default_lr_adam if lr is None else lr)
    return OptimizerParams.LBFGS(
        lr=default_lr_lbfgs if lr is None else lr,
        line_search_fn="strong_wolfe",
    )


def _print_top_elements(element_names, amplitudes, max_items=10):
    amplitudes_np = amplitudes.detach().cpu().numpy()
    order = np.argsort(amplitudes_np)[::-1]
    print("\nTop fitted elements:")
    for rank, idx in enumerate(order[:max_items], start=1):
        print(f"{rank:2d}. {element_names[idx]:>3s}: {amplitudes_np[idx]:.4g}")


def _plot_loss_history(ax, loss_history, *, x=None, label=None):
    """Plot a loss curve on a log scale, shifting it only when needed."""
    loss_history = np.asarray(loss_history, dtype=float)
    if x is None:
        x = np.arange(len(loss_history))
    else:
        x = np.asarray(x)

    finite = np.isfinite(loss_history)
    if not np.any(finite):
        ax.plot(
            x, np.zeros_like(x, dtype=float), color="k" if label is None else None, label=label
        )
        ax.set_ylabel("Loss")
        return

    loss_min = float(np.min(loss_history[finite]))
    shifted = loss_history.copy()
    ylabel = "Loss"
    legend_label = label
    if loss_min <= 0:
        offset = abs(loss_min) + 1e-8
        shifted = shifted + offset
        ylabel = f"Shifted loss (+{offset:.3g})"
        if legend_label is not None:
            legend_label = f"{legend_label} (shifted)"

    ax.plot(x, shifted, color="k" if label is None else None, label=legend_label)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")


def _plot_fit_summary(
    energy_axis,
    input_spectrum,
    fitted_spectrum,
    background_spectrum,
    loss_history,
    fit_range,
    title,
    comparison_spectrum=None,
    comparison_label="Comparison",
):
    if loss_history is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        _plot_loss_history(axes[0], loss_history)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Iteration")

    spec_ax = axes[-1]
    spec_ax.plot(energy_axis, input_spectrum, "k-", label="Data", linewidth=1.0)
    spec_ax.plot(energy_axis, fitted_spectrum, "r-", label="Fit", linewidth=2.0)
    spec_ax.plot(energy_axis, background_spectrum, "b--", label="Background", linewidth=1.5)
    if comparison_spectrum is not None:
        spec_ax.plot(
            energy_axis,
            comparison_spectrum,
            color="cyan",
            label=comparison_label,
            linewidth=2.0,
        )
    spec_ax.set_xlim(fit_range[0], fit_range[1])
    spec_ax.set_title(title)
    spec_ax.set_xlabel("Energy (keV)")
    spec_ax.set_ylabel("Counts")
    spec_ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_global_and_local_losses(global_loss_history, local_loss_history):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    global_x = np.arange(len(global_loss_history))
    local_x = np.arange(len(local_loss_history)) + len(global_loss_history)
    _plot_loss_history(ax, global_loss_history, x=global_x, label="Mean stage")
    _plot_loss_history(ax, local_loss_history, x=local_x, label="Cube stage")
    ax.axvline(len(global_loss_history) - 0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Loss")
    ax.set_xlabel("Iteration")
    ax.legend()
    plt.tight_layout()
    plt.show()


def _fit_mean_model_pytorch(
    self,
    energy_axis,
    spectrum_raw,
    elements_to_fit,
    peak_width,
    polynomial_background_degree,
    num_iters,
    optimizer,
    lr,
    loss_name,
    normalize_target,
    default_lr_adam,
    default_lr_lbfgs,
    verbose=False,
):
    """Fit one mean spectrum with a small non-negative basis model."""
    optimizer_name = _normalize_choice(optimizer, "optimizer", {"adam", "lbfgs"})
    loss_name = _normalize_choice(loss_name, "loss_name", {"mse", "poisson"})
    basis_data = _collect_element_basis_data(self, energy_axis, elements_to_fit)

    target = spectrum_raw
    spectrum_offset = torch.tensor(0.0, dtype=target.dtype, device=target.device)
    spectrum_scale = torch.tensor(1.0, dtype=target.dtype, device=target.device)
    if normalize_target:
        spectrum_offset = torch.min(target)
        spectrum_scale = torch.clamp(torch.max(target) - spectrum_offset, min=1e-8)
        target = (target - spectrum_offset) / spectrum_scale

    baseline = torch.quantile(target.detach(), 0.05) if target.numel() else target.new_tensor(0.0)
    residual = torch.clamp(torch.sum(torch.clamp(target - baseline, min=0.0)), min=1e-3)
    amplitude_init = torch.full(
        (len(basis_data["element_names"]),),
        residual / max(len(basis_data["element_names"]), 1),
        dtype=target.dtype,
        device=target.device,
    )
    background_init = torch.zeros(
        polynomial_background_degree + 1,
        dtype=target.dtype,
        device=target.device,
    )
    background_init[0] = torch.clamp(baseline, min=1e-6)

    model = MeanSpectrumModel(
        energy_axis=energy_axis,
        line_energies=basis_data["line_energies"],
        line_weights=basis_data["line_weights"],
        line_element_indices=basis_data["line_element_indices"],
        n_elements=len(basis_data["element_names"]),
        background_degree=polynomial_background_degree,
        peak_width_init=float(peak_width),
        amplitude_init=amplitude_init,
        background_init=background_init,
    ).to(device=energy_axis.device, dtype=energy_axis.dtype)
    model.element_names = list(basis_data["element_names"])
    model.line_labels = list(basis_data["line_labels"])

    fitter = XEDSFitter(model)
    fitter.set_optimizer(
        _build_optimizer_params(
            optimizer_name=optimizer_name,
            lr=lr,
            default_lr_adam=default_lr_adam,
            default_lr_lbfgs=default_lr_lbfgs,
        )
    )
    loss_history = fitter.run(
        num_iters=num_iters,
        loss_fn=lambda: xeds_data_loss(model(), target, loss=loss_name),
        verbose=verbose,
    )

    with torch.no_grad():
        fitted_target = model()
        peak_target = model.peak_spectrum()
        background_target = model.background_spectrum()
        if normalize_target:
            final_pred_raw = fitted_target * spectrum_scale + spectrum_offset
            final_peak_raw = peak_target * spectrum_scale
            final_background_raw = background_target * spectrum_scale + spectrum_offset
            element_amplitudes = model.element_amplitudes() * spectrum_scale
            background_coeffs = model.background_coeffs() * spectrum_scale
            background_coeffs[0] = background_coeffs[0] + spectrum_offset
        else:
            final_pred_raw = fitted_target
            final_peak_raw = peak_target
            final_background_raw = background_target
            element_amplitudes = model.element_amplitudes()
            background_coeffs = model.background_coeffs()

    return {
        "model": model,
        "loss_history": np.asarray(loss_history),
        "final_pred_raw": final_pred_raw.detach(),
        "final_peak_raw": final_peak_raw.detach(),
        "final_background_raw": final_background_raw.detach(),
        "element_amplitudes": element_amplitudes.detach(),
        "background_coeffs": background_coeffs.detach(),
        "peak_width": float(model.peak_width().detach().cpu().item()),
        "element_names": list(basis_data["element_names"]),
        "line_labels": list(basis_data["line_labels"]),
        "line_energies": basis_data["line_energies"].detach(),
        "line_weights": basis_data["line_weights"].detach(),
        "line_element_indices": basis_data["line_element_indices"].detach(),
        "spectrum_offset": spectrum_offset.detach(),
        "spectrum_scale": spectrum_scale.detach(),
    }


def fit_spectrum_mean_pytorch(
    self,
    energy_range=None,
    elements_to_fit=None,
    peak_width=0.1,
    num_iters=1000,
    lr=None,
    polynomial_background_degree=3,
    optimizer="lbfgs",
    device=None,
    loss="mse",
    normalize_target=None,
    show_plot=True,
    verbose=True,
):
    """Fit the mean spectrum first. This is the stable starting point."""
    optimizer_name = _normalize_choice(optimizer, "optimizer", {"adam", "lbfgs"})
    loss_name = _normalize_choice(loss, "loss", {"mse", "poisson"})
    if normalize_target is None:
        normalize_target = loss_name == "mse"

    device = _resolve_torch_device(device)
    energy_axis, spectra, fit_range = _slice_spectra_to_energy_range(
        self=self,
        energy_range=energy_range,
        device=device,
    )
    spectrum_raw = spectra.mean(dim=(0, 1))
    mean_fit = self._fit_mean_model_pytorch(
        energy_axis=energy_axis,
        spectrum_raw=spectrum_raw,
        elements_to_fit=elements_to_fit,
        peak_width=peak_width,
        polynomial_background_degree=polynomial_background_degree,
        num_iters=num_iters,
        optimizer=optimizer_name,
        lr=lr,
        loss_name=loss_name,
        normalize_target=bool(normalize_target),
        default_lr_adam=5e-2,
        default_lr_lbfgs=1.0,
        verbose=verbose,
    )

    if verbose:
        _print_top_elements(mean_fit["element_names"], mean_fit["element_amplitudes"])

    energy_axis_np = energy_axis.detach().cpu().numpy()
    input_spectrum = spectrum_raw.detach().cpu().numpy()
    fitted_spectrum = mean_fit["final_pred_raw"].detach().cpu().numpy()
    background_spectrum = mean_fit["final_background_raw"].detach().cpu().numpy()
    concentrations = mean_fit["element_amplitudes"].detach().cpu().numpy()
    element_names = list(mean_fit["element_names"])
    edge_indices = np.arange(len(element_names), dtype=int)
    peak_widths = np.asarray([mean_fit["peak_width"]], dtype=np.float32)

    if show_plot:
        _plot_fit_summary(
            energy_axis=energy_axis_np,
            input_spectrum=input_spectrum,
            fitted_spectrum=fitted_spectrum,
            background_spectrum=background_spectrum,
            loss_history=mean_fit["loss_history"],
            fit_range=fit_range,
            title="Mean spectrum fit",
        )

    return {
        "loss_history": mean_fit["loss_history"],
        "fitted_spectrum": fitted_spectrum,
        "input_spectrum": input_spectrum,
        "background_spectrum": background_spectrum,
        "concentrations": concentrations,
        "element_names": element_names,
        "edge_concentrations": concentrations.copy(),
        "edge_names": element_names.copy(),
        "edge_element_indices": edge_indices,
        "peak_widths": peak_widths,
        "peak_width": float(mean_fit["peak_width"]),
        "background_coeffs": mean_fit["background_coeffs"].detach().cpu().numpy(),
        "line_labels": list(mean_fit["line_labels"]),
        "energy_axis": energy_axis_np,
        "fit_range": fit_range,
    }


def fit_spectrum_pytorch(
    self,
    energy_range=None,
    elements_to_fit=None,
    peak_width=0.1,
    num_iters=300,
    num_iters_global=200,
    polynomial_background_degree=3,
    optimizer_global="lbfgs",
    optimizer_local="lbfgs",
    loss_global=None,
    loss_local="poisson",
    freeze_peak_width=True,
    spatial_lambda=0.0,
    min_total_counts=0.0,
    verbose=True,
    fit_mean_only=False,
    show_plot=True,
    lr_global=None,
    lr_local=None,
    device=None,
    constrain_background=0.1,
):
    """Fit XEDS with a mean-first then full-cube workflow."""
    effective_optimizer_global = _normalize_choice(
        optimizer_global, "optimizer_global", {"adam", "lbfgs"}
    )
    effective_optimizer_local = _normalize_choice(
        optimizer_local, "optimizer_local", {"adam", "lbfgs"}
    )
    effective_loss_global = (
        _normalize_choice(loss_global, "loss_global", {"poisson", "mse"})
        if loss_global is not None
        else "mse"
    )
    effective_loss_local = (
        _normalize_choice(loss_local, "loss_local", {"poisson", "mse"})
        if not fit_mean_only
        else None
    )

    if spatial_lambda < 0:
        raise ValueError("spatial_lambda must be >= 0")
    if isinstance(constrain_background, bool):
        raise TypeError("constrain_background must be a non-negative float.")
    try:
        background_prior_lambda = float(constrain_background)
    except (TypeError, ValueError) as exc:
        raise TypeError("constrain_background must be a non-negative float.") from exc
    if background_prior_lambda < 0:
        raise ValueError("constrain_background must be >= 0")

    if fit_mean_only:
        return self.fit_spectrum_mean_pytorch(
            energy_range=energy_range,
            elements_to_fit=elements_to_fit,
            peak_width=peak_width,
            num_iters=num_iters,
            lr=lr_global,
            polynomial_background_degree=polynomial_background_degree,
            optimizer=effective_optimizer_global,
            device=device,
            loss=effective_loss_global,
            normalize_target=effective_loss_global == "mse",
            show_plot=show_plot,
            verbose=verbose,
        )

    device = _resolve_torch_device(device)
    energy_axis, spectra, fit_range = _slice_spectra_to_energy_range(
        self=self,
        energy_range=energy_range,
        device=device,
    )

    total_counts = spectra.sum(dim=2)
    valid_pixel_mask = total_counts >= float(min_total_counts)
    if not torch.any(valid_pixel_mask):
        raise ValueError("No pixels satisfy min_total_counts. Lower threshold and retry.")

    mean_spectrum = spectra[valid_pixel_mask].mean(dim=0)
    global_fit = self._fit_mean_model_pytorch(
        energy_axis=energy_axis,
        spectrum_raw=mean_spectrum,
        elements_to_fit=elements_to_fit,
        peak_width=peak_width,
        polynomial_background_degree=polynomial_background_degree,
        num_iters=num_iters_global,
        optimizer=effective_optimizer_global,
        lr=lr_global,
        loss_name=effective_loss_global,
        normalize_target=effective_loss_global == "mse",
        default_lr_adam=5e-2,
        default_lr_lbfgs=1.0,
        verbose=verbose,
    )

    scan_row, scan_col, _ = spectra.shape
    n_elements = len(global_fit["element_names"])
    mean_total_counts = torch.clamp(mean_spectrum.sum(), min=1e-8)
    pixel_scale = (total_counts / mean_total_counts).clamp(min=1e-3)

    amplitude_init = torch.clamp(
        global_fit["element_amplitudes"][:, None, None] * pixel_scale[None, :, :],
        min=1e-6,
    )
    background_init = torch.clamp(
        global_fit["background_coeffs"][:, None, None] * pixel_scale[None, :, :],
        min=1e-6,
    )

    cube_model = CubeSpectrumModel(
        energy_axis=energy_axis,
        line_energies=global_fit["line_energies"],
        line_weights=global_fit["line_weights"],
        line_element_indices=global_fit["line_element_indices"],
        n_elements=n_elements,
        background_degree=polynomial_background_degree,
        peak_width_init=global_fit["peak_width"],
        amplitude_init=amplitude_init,
        background_init=background_init,
        train_peak_width=not freeze_peak_width,
    ).to(device=energy_axis.device, dtype=energy_axis.dtype)

    fitter = XEDSFitter(cube_model)
    fitter.set_optimizer(
        _build_optimizer_params(
            optimizer_name=effective_optimizer_local,
            lr=lr_local,
            default_lr_adam=1e-1,
            default_lr_lbfgs=1.0,
        )
    )
    background_reference = background_init.detach().clone()

    def local_loss():
        predicted, _peak_part, _background_part = cube_model()
        loss_value = xeds_data_loss(
            predicted[valid_pixel_mask],
            spectra[valid_pixel_mask],
            loss=effective_loss_local,
        )
        if background_prior_lambda > 0:
            coeff_scale = torch.clamp(background_reference.mean(), min=1e-6)
            background_term = (
                ((cube_model.background_coeffs() - background_reference) / coeff_scale)
                .pow(2)
                .mean()
            )
            loss_value = loss_value + background_prior_lambda * background_term
        if spatial_lambda > 0:
            loss_value = loss_value + spatial_lambda * total_variation_in_plane(
                cube_model.element_amplitudes()
            )
        return loss_value

    local_loss_history = fitter.run(
        num_iters=num_iters,
        loss_fn=local_loss,
        verbose=verbose,
    )

    with torch.no_grad():
        predicted, _peak_part, background_part = cube_model()
        abundance_maps = cube_model.element_amplitudes().detach().cpu().numpy()
        peak_widths = np.asarray([cube_model.peak_width().detach().cpu().item()], dtype=np.float32)
        mean_input_spectrum = spectra[valid_pixel_mask].mean(dim=0).detach().cpu().numpy()
        mean_fitted_spectrum = predicted[valid_pixel_mask].mean(dim=0).detach().cpu().numpy()
        mean_background_spectrum = (
            background_part[valid_pixel_mask].mean(dim=0).detach().cpu().numpy()
        )
        mean_input_spectrum_all = spectra.mean(dim=(0, 1)).detach().cpu().numpy()
        mean_fitted_spectrum_all = predicted.mean(dim=(0, 1)).detach().cpu().numpy()
        mean_background_spectrum_all = background_part.mean(dim=(0, 1)).detach().cpu().numpy()

    pytorch_spectrum_images = self._build_pytorch_spectrum_images(
        abundance_maps=abundance_maps,
        element_names=list(global_fit["element_names"]),
    )
    self._spectrum_images_pytorch = {
        **getattr(self, "_spectrum_images_pytorch", {}),
        **pytorch_spectrum_images,
    }

    energy_axis_np = energy_axis.detach().cpu().numpy()
    global_loss_history = np.asarray(global_fit["loss_history"])
    local_loss_history = np.asarray(local_loss_history)

    if show_plot:
        _plot_global_and_local_losses(global_loss_history, local_loss_history)
        _plot_fit_summary(
            energy_axis=energy_axis_np,
            input_spectrum=mean_input_spectrum,
            fitted_spectrum=mean_fitted_spectrum,
            background_spectrum=mean_background_spectrum,
            loss_history=None,
            fit_range=fit_range,
            title="Full-cube fit (valid-pixel mean)",
            comparison_spectrum=global_fit["final_pred_raw"].detach().cpu().numpy(),
            comparison_label="Mean-stage fit",
        )

    return {
        "abundance_maps": abundance_maps,
        "element_names": list(global_fit["element_names"]),
        "peak_widths": peak_widths,
        "loss_history": local_loss_history,
        "global_loss_history": global_loss_history,
        "valid_pixel_mask": valid_pixel_mask.detach().cpu().numpy(),
        "energy_axis": energy_axis_np,
        "input_spectrum": mean_input_spectrum,
        "fitted_spectrum": mean_fitted_spectrum,
        "background_spectrum": mean_background_spectrum,
        "input_spectrum_all_pixels": mean_input_spectrum_all,
        "fitted_spectrum_all_pixels": mean_fitted_spectrum_all,
        "background_spectrum_all_pixels": mean_background_spectrum_all,
        "fit_range": fit_range,
        "spectrum_images_pytorch": self._spectrum_images_pytorch,
    }
