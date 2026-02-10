from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from quantem.core.visualization import show_2d
from quantem.spectroscopy import Dataset3dspectroscopy
from quantem.spectroscopy.spectroscopy_models import (
    EDSModel,
    GaussianPeaks,
    PolynomialBackground,
    abundance_smoothness_l2,
    build_element_basis,
    eds_data_loss,
    inverse_softplus,
    polynomial_energy_basis,
)


class Dataset3deds(Dataset3dspectroscopy):
    """An EDS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (energy, scan_y, scan_x).
    The first dimension represents the energy, while the latter
    two dimensions represent real space sampling.

    """

    element_info = None
    element_info_path = "xray_lines.json"
    dataset_type = "EDS"

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        """Initialize a 3D EDS dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
        name : str
            A descriptive name for the dataset
        origin : NDArray | tuple | list | float | int
            The origin coordinates for each dimension
        sampling : NDArray | tuple | list | float | int
            The sampling rate/spacing for each dimension
        units : list[str] | tuple | list
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"
        _token : object | None, optional
            Token to prevent direct instantiation, by default None
        """
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )
        self._virtual_images = {}
        self.dataset_type = "EDS"

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
        target = spectrum_raw
        spectrum_offset = torch.tensor(0.0, dtype=spectrum_raw.dtype, device=spectrum_raw.device)
        spectrum_scale = torch.tensor(1.0, dtype=spectrum_raw.dtype, device=spectrum_raw.device)
        if normalize_target:
            spectrum_offset = spectrum_raw.min()
            spectrum_scale = torch.clamp(spectrum_raw.max() - spectrum_offset, min=1e-8)
            target = (spectrum_raw - spectrum_offset) / spectrum_scale

        background = PolynomialBackground(energy_axis, degree=polynomial_background_degree)
        peaks = GaussianPeaks(energy_axis, peak_width=peak_width, elements_to_fit=elements_to_fit)
        model = EDSModel(peaks, background, energy_axis=energy_axis)
        model = model.to(device=energy_axis.device, dtype=energy_axis.dtype)
        if len(model.peak_model.element_names) == 0:
            raise ValueError("No elements found in the selected energy range/elements_to_fit.")

        with torch.no_grad():
            model.peak_model.concentrations.fill_(1.0)

        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            if lr is None:
                lr = default_lr_adam
            optimizer_obj = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "lbfgs":
            if lr is None:
                lr = default_lr_lbfgs
            optimizer_obj = torch.optim.LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=1,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError("optimizer must be 'lbfgs' or 'adam'")

        loss_iter = []
        for i in range(num_iters):
            if optimizer_name == "lbfgs":

                def closure():
                    optimizer_obj.zero_grad()
                    predicted = model()
                    loss = eds_data_loss(predicted, target, loss=loss_name)
                    loss.backward()
                    return loss

                loss = optimizer_obj.step(closure)
                if not torch.is_tensor(loss):
                    with torch.no_grad():
                        loss = eds_data_loss(model(), target, loss=loss_name)
            else:
                optimizer_obj.zero_grad()
                predicted = model()
                loss = eds_data_loss(predicted, target, loss=loss_name)
                loss.backward()
                optimizer_obj.step()

            loss_iter.append(float(loss.detach().cpu().item()))
            if verbose and ((i + 1) % max(1, num_iters // 10) == 0 or i == 0):
                print(f"iter {i + 1:4d}/{num_iters}: loss={loss_iter[-1]:.6g}")

        with torch.no_grad():
            final_pred_target = model()
            if normalize_target:
                final_pred_raw = final_pred_target * spectrum_scale + spectrum_offset
            else:
                final_pred_raw = final_pred_target

        return {
            "model": model,
            "loss_history": np.asarray(loss_iter),
            "final_pred_raw": final_pred_raw.detach(),
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
    ):
        return self.fit_spectrum_pytorch(
            energy_range=energy_range,
            elements_to_fit=elements_to_fit,
            peak_width=peak_width,
            num_iters=num_iters,
            lr=lr,
            polynomial_background_degree=polynomial_background_degree,
            optimizer=optimizer,
            loss="mse",
            fit_mean_only=True,
            show_plot=True,
            device=device,
        )

    def fit_spectrum_pytorch(
        self,
        energy_range=None,
        elements_to_fit=None,
        peak_width=0.1,
        num_iters=300,
        num_iters_global=200,
        lr=None,
        polynomial_background_degree=3,
        optimizer=None,
        loss=None,
        optimizer_global=None,
        optimizer_local=None,
        loss_global=None,
        loss_local=None,
        freeze_peak_width=True,
        spatial_lambda=0.0,
        min_total_counts=0.0,
        verbose=True,
        fit_mean_only=False,
        show_plot=True,
        lr_global=None,
        lr_local=None,
        device=None,
    ):
        """Fit EDS spectra with one entrypoint for mean-only or full-cube fitting.

        Parameters
        ----------
        energy_range : list[float] | tuple[float, float] | None
            Energy range [emin, emax] to include in the fit.
        elements_to_fit : list[str] | None
            Element symbols to fit. If None, all available elements in range are used.
        peak_width : float, optional
            Initial peak FWHM in keV.
        num_iters : int, optional
            Number of optimization iterations for per-pixel fitting.
        num_iters_global : int, optional
            Number of mean-spectrum iterations used to initialize local fitting (3D mode).
        lr : float, optional
            Backward-compatible shared learning rate fallback. Used for global/local
            fitting only when lr_global/lr_local are not provided.
        lr_global : float, optional
            Learning rate for the global mean-spectrum stage. In mean-only mode, this
            is the learning rate used for that fit.
        lr_local : float, optional
            Learning rate for the position-by-position stage (3D mode).
        polynomial_background_degree : int, optional
            Degree of per-pixel polynomial background.
        optimizer : str | None, optional
            Backward-compatible optimizer selector. In mean-only mode it controls
            the global stage. In 3D mode it controls the local stage.
        optimizer_global : str | None, optional
            Global/mean-stage optimizer, "adam" or "lbfgs". In 3D mode, defaults
            to "lbfgs" unless explicitly set.
        optimizer_local : str | None, optional
            Local position-by-position optimizer, "adam" or "lbfgs". In 3D mode,
            defaults to optimizer if provided, otherwise "lbfgs".
        loss : str | None, optional
            Backward-compatible shared data term, "poisson" or "mse". If provided,
            it applies to both stages unless stage-specific losses are set.
        loss_global : str | None, optional
            Global/mean-stage data term, "poisson" or "mse".
        loss_local : str | None, optional
            Local position-by-position data term, "poisson" or "mse".
        freeze_peak_width : bool, optional
            If True, lock peak widths after global fit (3D mode).
        spatial_lambda : float, optional
            Weight for spatial smoothness on abundance maps (3D mode).
        min_total_counts : float, optional
            Ignore pixels with summed counts below this threshold in data loss (3D mode).
        verbose : bool, optional
            Print progress updates.
        fit_mean_only : bool, optional
            If True, fit only the summed spectrum over (x, y).
        show_plot : bool, optional
            Plot fit diagnostics in mean-only mode.
        device : str | torch.device | None, optional
            Compute device to run fitting on. If None, uses CUDA when available,
            otherwise CPU.

        Returns
        -------
        dict
            Mean-only mode keys include concentrations, fit, and diagnostics.
            3D mode keys include abundance maps and fit diagnostics.
        """

        def _normalize_choice(name, param_name, allowed_values):
            if name is None:
                return None
            name_norm = name.lower()
            if name_norm not in allowed_values:
                allowed_display = "', '".join(sorted(allowed_values))
                raise ValueError(f"{param_name} must be '{allowed_display}'")
            return name_norm

        def _resolve_stage_settings(
            fit_mean_only_mode,
            optimizer_default_global,
            loss_default_global,
            loss_default_local,
        ):
            if fit_mean_only_mode:
                return (
                    optimizer_global_name or optimizer_name or optimizer_default_global,
                    None,
                    loss_global_name or loss_name or loss_default_global,
                    None,
                )
            # Preserve historical behavior: global stage defaults to LBFGS.
            return (
                optimizer_global_name or optimizer_default_global,
                optimizer_local_name or optimizer_name or "lbfgs",
                loss_global_name or loss_name or loss_default_global,
                loss_local_name or loss_name or loss_default_local,
            )

        optimizer_name = _normalize_choice(optimizer, "optimizer", {"adam", "lbfgs"})
        optimizer_global_name = _normalize_choice(
            optimizer_global, "optimizer_global", {"adam", "lbfgs"}
        )
        optimizer_local_name = _normalize_choice(
            optimizer_local, "optimizer_local", {"adam", "lbfgs"}
        )

        loss_name = _normalize_choice(loss, "loss", {"poisson", "mse"})
        loss_global_name = _normalize_choice(loss_global, "loss_global", {"poisson", "mse"})
        loss_local_name = _normalize_choice(loss_local, "loss_local", {"poisson", "mse"})

        (
            effective_optimizer_global,
            effective_optimizer_local,
            effective_loss_global,
            effective_loss_local,
        ) = _resolve_stage_settings(
            fit_mean_only_mode=fit_mean_only,
            optimizer_default_global="lbfgs",
            loss_default_global="mse" if fit_mean_only else "poisson",
            loss_default_local="poisson",
        )

        if spatial_lambda < 0:
            raise ValueError("spatial_lambda must be >= 0")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

        effective_lr_global = lr if lr_global is None else lr_global
        effective_lr_local = lr if lr_local is None else lr_local

        energy_axis_np = np.arange(self.shape[0]) * self.sampling[0] + self.origin[0]
        energy_axis = torch.tensor(energy_axis_np, dtype=torch.float32, device=device)
        spectra = torch.tensor(self.array, dtype=torch.float32, device=device)  # (E, Y, X)

        if energy_range is not None:
            ind = (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            energy_axis = energy_axis[ind]
            spectra = spectra[ind]
        else:
            energy_range = [float(energy_axis.min().item()), float(energy_axis.max().item())]

        if fit_mean_only:
            if verbose:
                print("fitting spectrum globally")
            spectrum_raw = spectra.sum((-1, -2))
            mean_fit = self._fit_mean_model_pytorch(
                energy_axis=energy_axis,
                spectrum_raw=spectrum_raw,
                elements_to_fit=elements_to_fit,
                peak_width=peak_width,
                polynomial_background_degree=polynomial_background_degree,
                num_iters=num_iters,
                optimizer=effective_optimizer_global,
                lr=effective_lr_global,
                loss_name=effective_loss_global,
                normalize_target=True,
                default_lr_adam=1e-3,
                default_lr_lbfgs=1.0,
                verbose=verbose,
            )

            model = mean_fit["model"]
            loss_history = mean_fit["loss_history"]
            spectrum_offset = mean_fit["spectrum_offset"]
            spectrum_scale = mean_fit["spectrum_scale"]
            with torch.no_grad():
                final_pred = mean_fit["final_pred_raw"].cpu().numpy()
                concs = (
                    nn.functional.softplus(model.peak_model.concentrations).detach().cpu().numpy()
                )
                final_fwhm = (
                    torch.nn.functional.softplus(model.peak_model.peak_width_by_peak)
                    .detach()
                    .cpu()
                    .numpy()
                )
                background_fit = (
                    (model.background_model().detach() * spectrum_scale + spectrum_offset)
                    .cpu()
                    .numpy()
                )

            print(
                f"\nFinal: width median={np.median(final_fwhm):.3f} keV, "
                f"min={final_fwhm.min():.3f}, max={final_fwhm.max():.3f}"
            )

            top_n = max(10, len(elements_to_fit) if elements_to_fit is not None else 0)
            sorted_indices = np.argsort(concs)[::-1]
            print("\nTop elements:")
            for i, idx in enumerate(sorted_indices[:top_n], 1):
                elem = model.peak_model.element_names[idx]
                conc = concs[idx]
                print(f"{i:2d}. {elem:2s}: {conc:.3f}")

            if show_plot:
                energy_axis_plot = energy_axis.detach().cpu().numpy()
                spectrum_raw_plot = spectrum_raw.detach().cpu().numpy()
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].plot(np.arange(loss_history.shape[0]), loss_history, color="k")
                ax[0].set_title("loss")
                ax[0].set_xlabel("iterations")
                ax[0].set_ylabel("loss")
                ax[0].set_yscale("log")

                ax[1].plot(energy_axis_plot, spectrum_raw_plot, "k-", label="Data", linewidth=1)
                ax[1].plot(energy_axis_plot, final_pred, "r-", label="Fit", linewidth=2)
                ax[1].plot(
                    energy_axis_plot,
                    background_fit,
                    "b--",
                    label="Background",
                    linewidth=1.5,
                )
                ax[1].set_xlim(energy_range[0], energy_range[1])
                ax[1].legend()
                ax[1].set_title("fit spectrum")
                ax[1].set_xlabel("Energy (keV)")
                ax[1].set_ylabel("Counts")
                plt.tight_layout()
                plt.show()

            return {
                "loss_history": loss_history,
                "fitted_spectrum": final_pred,
                "input_spectrum": spectrum_raw.detach().cpu().numpy(),
                "background_spectrum": background_fit,
                "concentrations": concs,
                "element_names": model.peak_model.element_names,
                "peak_widths": final_fwhm,
                "energy_axis": energy_axis.detach().cpu().numpy(),
                "fit_range": energy_range,
            }

        n_energy, n_y, n_x = spectra.shape
        n_pixels = n_y * n_x
        spectra_flat = spectra.permute(1, 2, 0).reshape(n_pixels, n_energy)

        total_counts = spectra_flat.sum(dim=1)
        valid_pixel_mask = total_counts >= float(min_total_counts)
        if not torch.any(valid_pixel_mask):
            raise ValueError("No pixels satisfy min_total_counts. Lower threshold and retry.")

        mean_spectrum = spectra_flat[valid_pixel_mask].mean(dim=0)

        # Stage 1: global mean-spectrum fit to initialize per-pixel parameters.
        if verbose:
            print("fitting spectrum globally")
        global_fit = self._fit_mean_model_pytorch(
            energy_axis=energy_axis,
            spectrum_raw=mean_spectrum,
            elements_to_fit=elements_to_fit,
            peak_width=peak_width,
            polynomial_background_degree=polynomial_background_degree,
            num_iters=num_iters_global,
            optimizer=effective_optimizer_global,
            lr=effective_lr_global,
            loss_name=effective_loss_global,
            normalize_target=True,
            default_lr_adam=1e-3,
            default_lr_lbfgs=1.0,
            verbose=verbose,
        )
        global_model = global_fit["model"]
        global_loss_history = global_fit["loss_history"]
        global_scale = global_fit["spectrum_scale"].detach()
        global_offset = global_fit["spectrum_offset"].detach()
        global_fitted_spectrum = global_fit["final_pred_raw"].detach().cpu().numpy()

        with torch.no_grad():
            # If the global stage fit a normalized target, convert amplitude-like
            # parameters back to raw-count scale for local initialization.
            global_conc = (
                nn.functional.softplus(global_model.peak_model.concentrations).detach()
                * global_scale
            )
            global_bg_coeffs = global_model.background_model.coeffs.detach() * global_scale
            if global_bg_coeffs.numel() > 0:
                global_bg_coeffs = global_bg_coeffs.clone()
                global_bg_coeffs[0] = global_bg_coeffs[0] + global_offset
            global_peak_width_params = global_model.peak_model.peak_width_by_peak.detach().clone()

        # Stage 2: vectorized per-pixel fit with shared peak shapes.
        n_elements = len(global_model.peak_model.element_names)
        peak_energies = global_model.peak_model.peak_energies
        peak_weights = global_model.peak_model.peak_weights
        peak_element_indices = global_model.peak_model.peak_element_indices
        energy_step = float(global_model.peak_model.energy_step)

        background_basis = polynomial_energy_basis(
            energy_axis, degree=polynomial_background_degree
        )

        mean_total = torch.clamp(mean_spectrum.sum(), min=1e-8)
        pixel_scales = (total_counts / mean_total).unsqueeze(1)
        # Avoid near-zero concentration initialization that can cause vanishing
        # softplus gradients in local optimization (especially on normalized data).
        conc_init = torch.clamp(
            global_conc.unsqueeze(0) * pixel_scales,
            min=1e-3,
        )
        # Small random perturbation helps break symmetry across pixels.
        conc_init = torch.clamp(
            conc_init * (1.0 + 0.02 * torch.randn_like(conc_init)),
            min=1e-3,
        )

        conc_logits = nn.Parameter(inverse_softplus(conc_init))
        bg_coeffs = nn.Parameter(global_bg_coeffs.unsqueeze(0).repeat(n_pixels, 1) * pixel_scales)

        if freeze_peak_width:
            peak_width_params = global_peak_width_params
        else:
            peak_width_params = nn.Parameter(global_peak_width_params.clone())

        if freeze_peak_width:
            element_basis = build_element_basis(
                energy_axis=energy_axis,
                peak_energies=peak_energies,
                peak_weights=peak_weights,
                peak_element_indices=peak_element_indices,
                peak_width_by_peak=peak_width_params,
                n_elements=n_elements,
                energy_step=energy_step,
            )

        trainable_params = [conc_logits, bg_coeffs]
        if not freeze_peak_width:
            trainable_params.append(peak_width_params)

        local_lr = (
            effective_lr_local
            if effective_lr_local is not None
            else (0.05 if effective_optimizer_local == "adam" else 1.0)
        )

        if effective_optimizer_local == "adam":
            local_opt = torch.optim.Adam(trainable_params, lr=local_lr)
        else:
            local_opt = torch.optim.LBFGS(
                trainable_params,
                lr=local_lr,
                max_iter=1,
                line_search_fn="strong_wolfe",
                history_size=10,
            )

        loss_history = []

        def _forward_model():
            basis = (
                element_basis
                if freeze_peak_width
                else build_element_basis(
                    energy_axis=energy_axis,
                    peak_energies=peak_energies,
                    peak_weights=peak_weights,
                    peak_element_indices=peak_element_indices,
                    peak_width_by_peak=peak_width_params,
                    n_elements=n_elements,
                    energy_step=energy_step,
                )
            )
            conc = nn.functional.softplus(conc_logits)  # (P, n_elements)
            peaks_pred = conc @ basis.t()  # (P, E)
            bg_raw = bg_coeffs @ background_basis  # (P, E)
            # Keep local background parameterization consistent with global initialization.
            bg_pred = bg_raw
            predicted = torch.clamp(peaks_pred + bg_pred, min=1e-8, max=1e8)
            return predicted, conc

        def _prepare_local_loss_inputs(pred_local):
            pred_eval = pred_local[valid_pixel_mask]
            target_eval = spectra_flat[valid_pixel_mask]
            local_scale = torch.clamp(global_scale, min=1e-8)
            pred_eval = pred_eval / local_scale
            target_eval = target_eval / local_scale
            return pred_eval, target_eval

        def _local_loss(pred_local, conc_local):
            pred_eval, target_eval = _prepare_local_loss_inputs(pred_local)

            loss_data = eds_data_loss(
                pred_eval,
                target_eval,
                loss=effective_loss_local,
            )
            if spatial_lambda <= 0:
                return loss_data

            conc_maps = conc_local.view(n_y, n_x, n_elements).permute(2, 0, 1)
            conc_maps = conc_maps / torch.clamp(global_scale, min=1e-8)
            loss_smooth = abundance_smoothness_l2(conc_maps)
            return loss_data + spatial_lambda * loss_smooth

        if verbose:
            print("fitting spectrum position-by-position")
        for i in range(num_iters):
            if effective_optimizer_local == "lbfgs":

                def _local_closure():
                    local_opt.zero_grad()
                    pred_local, conc_local = _forward_model()
                    loss_total = _local_loss(pred_local, conc_local)
                    loss_total.backward()
                    return loss_total

                loss_value = local_opt.step(_local_closure)
                if not torch.is_tensor(loss_value):
                    with torch.no_grad():
                        pred_local, conc_local = _forward_model()
                        loss_value = _local_loss(pred_local, conc_local)
            else:
                local_opt.zero_grad()
                pred_local, conc_local = _forward_model()
                loss_value = _local_loss(pred_local, conc_local)
                loss_value.backward()
                local_opt.step()

            loss_history.append(float(loss_value.detach().cpu().item()))
            if verbose and ((i + 1) % max(1, num_iters // 10) == 0 or i == 0):
                print(f"iter {i + 1:4d}/{num_iters}: loss={loss_history[-1]:.6g}")

        with torch.no_grad():
            pred_final, conc_final = _forward_model()
            bg_final = bg_coeffs @ background_basis

            mean_input_spectrum = spectra_flat.mean(dim=0).cpu().numpy()
            mean_fitted_spectrum = pred_final.mean(dim=0).cpu().numpy()
            mean_background_spectrum = bg_final.mean(dim=0).cpu().numpy()

            abundance_maps = conc_final.view(n_y, n_x, n_elements).permute(2, 0, 1).cpu().numpy()
            peak_widths = nn.functional.softplus(peak_width_params).detach().cpu().numpy()
        loss_history_array = np.asarray(loss_history)
        energy_axis_np = energy_axis.cpu().numpy()

        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            global_x = np.arange(global_loss_history.shape[0])
            local_x = np.arange(loss_history_array.shape[0]) + global_loss_history.shape[0]
            ax.plot(
                global_x,
                global_loss_history,
                "b-",
                label="global",
            )
            ax.plot(
                local_x,
                loss_history_array,
                "r-",
                label="local",
            )
            ax.axvline(
                x=global_loss_history.shape[0] - 0.5,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label="switch",
            )
            ax.set_title("loss")
            ax.set_xlabel("iterations")
            ax.set_ylabel("loss")
            ax.set_yscale("log")
            ax.legend()
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(energy_axis_np, mean_input_spectrum, "k-", label="Data", linewidth=1)
            ax.plot(
                energy_axis_np,
                global_fitted_spectrum,
                color="cyan",
                label="Global fit",
                linewidth=2.5,
            )
            ax.plot(energy_axis_np, mean_fitted_spectrum, "r-", label="Fit", linewidth=2.5)
            ax.plot(
                energy_axis_np,
                mean_background_spectrum,
                "b--",
                label="Background",
                linewidth=2.5,
            )
            ax.set_xlim(energy_range[0], energy_range[1])
            ax.legend()
            ax.set_title("fit spectrum after local fitting (x/y-averaged)")
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Counts")
            plt.tight_layout()
            plt.show()

            map_titles = [f"{name}" for name in global_model.peak_model.element_names]
            show_2d(list(abundance_maps), title=map_titles)

        return {
            "abundance_maps": abundance_maps,
            "element_names": global_model.peak_model.element_names,
            "peak_widths": peak_widths,
            "loss_history": loss_history_array,
            "global_loss_history": np.asarray(global_loss_history),
            "valid_pixel_mask": valid_pixel_mask.view(n_y, n_x).cpu().numpy(),
            "energy_axis": energy_axis_np,
            "input_spectrum": mean_input_spectrum,
            "fitted_spectrum": mean_fitted_spectrum,
            "background_spectrum": mean_background_spectrum,
            "fit_range": energy_range,
        }

    def calculate_background_powerlaw(self, spectrum):
        import numpy as np

        """
            From input spectrum, calculate power-law background typical for EDS Bremsstrahlung.
            Uses a conservative approach with heavy smoothing to avoid creating artifacts.
            
            Parameters
            ----------
            spectrum : ndarray
                1D spectrum
            energy_axis : ndarray
                Energy axis corresponding to spectrum
                
            Returns
            -------
            ndarray
                1D array representing the calculated background
            """
        from scipy.ndimage import gaussian_filter

        # Use a larger window for more conservative background estimation
        window_size = 15  # Larger window = smoother, less aggressive
        background = np.zeros_like(spectrum)
        half_window = window_size // 2

        # Estimate background from sliding minimum
        for i in range(len(spectrum)):
            start = max(0, i - half_window)
            end = min(len(spectrum), i + half_window + 1)
            # Use percentile instead of minimum for more robustness
            background[i] = np.percentile(spectrum[start:end], 10)

        # Apply heavy smoothing to avoid creating artificial features
        background = gaussian_filter(background, sigma=5.0)

        # Be very conservative - only subtract 80% of estimated background
        # This prevents over-subtraction that creates artificial peaks
        background = background * 0.8

        # Ensure background doesn't exceed spectrum
        background = np.minimum(background, spectrum * 0.9)

        return background
