import torch
import torch.nn as nn
import torch.nn.functional as F

from quantem.core.ml.models.model_base import PPLR


def inverse_softplus(x: torch.Tensor, min_value: float = 1e-8) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive values."""
    x = torch.clamp(x, min=min_value)
    return torch.where(
        x > 20.0,
        x + torch.log1p(-torch.exp(-x)),
        torch.log(torch.expm1(x)),
    )


def xeds_data_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss: str = "poisson",
    min_value: float = 1e-8,
) -> torch.Tensor:
    """Return a simple positive-data loss for XEDS fitting."""
    pred_safe = torch.clamp(torch.nan_to_num(predicted, nan=min_value), min=min_value)
    target_safe = torch.clamp(torch.nan_to_num(target, nan=0.0), min=0.0)

    if loss == "poisson":
        return torch.mean(pred_safe - target_safe * torch.log(pred_safe))
    if loss == "mse":
        return F.mse_loss(pred_safe, target_safe)
    raise ValueError("loss must be 'poisson' or 'mse'")


def build_polynomial_background_basis(
    energy_axis: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Build a non-negative polynomial basis on the normalized energy axis."""
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if energy_axis.ndim != 1:
        raise ValueError("energy_axis must be 1D")

    energy_min = torch.min(energy_axis)
    energy_span = torch.clamp(torch.max(energy_axis) - energy_min, min=1e-8)
    x = (energy_axis - energy_min) / energy_span
    return torch.stack([x**power for power in range(degree + 1)], dim=0)


def build_element_peak_basis(
    energy_axis: torch.Tensor,
    line_energies: torch.Tensor,
    line_weights: torch.Tensor,
    line_element_indices: torch.Tensor,
    n_elements: int,
    peak_width: torch.Tensor,
) -> torch.Tensor:
    """Build one spectral template per element from weighted Gaussian lines."""
    if energy_axis.ndim != 1:
        raise ValueError("energy_axis must be 1D")
    if n_elements < 1:
        raise ValueError("n_elements must be >= 1")

    width = torch.clamp(peak_width, min=1e-4)
    sigma = width / 2.355

    centers = line_energies.unsqueeze(1)
    channels = energy_axis.unsqueeze(0)
    gaussians = torch.exp(-0.5 * ((channels - centers) / sigma) ** 2)
    gaussians = gaussians / torch.clamp(gaussians.sum(dim=1, keepdim=True), min=1e-8)
    weighted_lines = gaussians * line_weights.unsqueeze(1)

    basis = torch.zeros(
        (n_elements, energy_axis.numel()),
        dtype=energy_axis.dtype,
        device=energy_axis.device,
    )
    basis.index_add_(0, line_element_indices, weighted_lines)
    return basis


def total_variation_in_plane(maps: torch.Tensor) -> torch.Tensor:
    """L1 total variation across y/x for maps shaped (n_maps, y, x)."""
    if maps.ndim != 3:
        raise ValueError("maps must have shape (n_maps, y, x)")

    loss = maps.new_tensor(0.0)
    if maps.shape[1] > 1:
        loss = loss + torch.abs(maps[:, 1:, :] - maps[:, :-1, :]).mean()
    if maps.shape[2] > 1:
        loss = loss + torch.abs(maps[:, :, 1:] - maps[:, :, :-1]).mean()
    return loss


class MeanSpectrumModel(nn.Module, PPLR):
    """Non-negative mean-spectrum model with one shared peak width."""

    def __init__(
        self,
        energy_axis: torch.Tensor,
        line_energies: torch.Tensor,
        line_weights: torch.Tensor,
        line_element_indices: torch.Tensor,
        n_elements: int,
        background_degree: int,
        peak_width_init: float,
        amplitude_init: torch.Tensor,
        background_init: torch.Tensor,
    ):
        super().__init__()
        self.n_elements = int(n_elements)
        self.register_buffer("energy_axis", energy_axis)
        self.register_buffer("line_energies", line_energies)
        self.register_buffer("line_weights", line_weights)
        self.register_buffer("line_element_indices", line_element_indices)
        self.register_buffer(
            "background_basis",
            build_polynomial_background_basis(energy_axis, degree=background_degree),
        )

        peak_width_tensor = torch.as_tensor(
            peak_width_init,
            dtype=energy_axis.dtype,
            device=energy_axis.device,
        )
        self.peak_width_raw = nn.Parameter(inverse_softplus(peak_width_tensor))
        self.element_amplitudes_raw = nn.Parameter(inverse_softplus(amplitude_init))
        self.background_coeffs_raw = nn.Parameter(inverse_softplus(background_init))

    def peak_width(self) -> torch.Tensor:
        return F.softplus(self.peak_width_raw)

    def element_amplitudes(self) -> torch.Tensor:
        return F.softplus(self.element_amplitudes_raw)

    def background_coeffs(self) -> torch.Tensor:
        return F.softplus(self.background_coeffs_raw)

    def element_basis(self) -> torch.Tensor:
        return build_element_peak_basis(
            energy_axis=self.energy_axis,
            line_energies=self.line_energies,
            line_weights=self.line_weights,
            line_element_indices=self.line_element_indices,
            n_elements=self.n_elements,
            peak_width=self.peak_width(),
        )

    def peak_spectrum(self) -> torch.Tensor:
        return self.element_amplitudes() @ self.element_basis()

    def background_spectrum(self) -> torch.Tensor:
        return self.background_coeffs() @ self.background_basis

    def forward(self) -> torch.Tensor:
        return self.peak_spectrum() + self.background_spectrum()

    def get_params(self) -> dict[str, list[nn.Parameter]]:
        return {
            "amplitudes": [self.element_amplitudes_raw],
            "background": [self.background_coeffs_raw],
            "peak_width": [self.peak_width_raw],
        }

    @property
    def param_keys(self) -> list[str]:
        return ["amplitudes", "background", "peak_width"]


class CubeSpectrumModel(nn.Module, PPLR):
    """Non-negative full-cube XEDS model that reuses the mean-fit basis."""

    def __init__(
        self,
        energy_axis: torch.Tensor,
        line_energies: torch.Tensor,
        line_weights: torch.Tensor,
        line_element_indices: torch.Tensor,
        n_elements: int,
        background_degree: int,
        peak_width_init: float,
        amplitude_init: torch.Tensor,
        background_init: torch.Tensor,
        train_peak_width: bool,
    ):
        super().__init__()
        self.n_elements = int(n_elements)

        self.register_buffer("energy_axis", energy_axis)
        self.register_buffer("line_energies", line_energies)
        self.register_buffer("line_weights", line_weights)
        self.register_buffer("line_element_indices", line_element_indices)
        self.register_buffer(
            "background_basis",
            build_polynomial_background_basis(energy_axis, degree=background_degree),
        )

        peak_width_tensor = torch.as_tensor(
            peak_width_init,
            dtype=energy_axis.dtype,
            device=energy_axis.device,
        )
        if train_peak_width:
            self.peak_width_raw = nn.Parameter(inverse_softplus(peak_width_tensor))
            self.register_buffer("_peak_width_fixed", torch.empty(0, device=energy_axis.device))
        else:
            self.peak_width_raw = None
            self.register_buffer("_peak_width_fixed", peak_width_tensor.reshape(()))

        self.element_amplitudes_raw = nn.Parameter(inverse_softplus(amplitude_init))
        self.background_coeffs_raw = nn.Parameter(inverse_softplus(background_init))

    def peak_width(self) -> torch.Tensor:
        if self.peak_width_raw is None:
            return self._peak_width_fixed
        return F.softplus(self.peak_width_raw)

    def element_amplitudes(self) -> torch.Tensor:
        return F.softplus(self.element_amplitudes_raw)

    def background_coeffs(self) -> torch.Tensor:
        return F.softplus(self.background_coeffs_raw)

    def element_basis(self) -> torch.Tensor:
        return build_element_peak_basis(
            energy_axis=self.energy_axis,
            line_energies=self.line_energies,
            line_weights=self.line_weights,
            line_element_indices=self.line_element_indices,
            n_elements=self.n_elements,
            peak_width=self.peak_width(),
        )

    def peak_spectra(self) -> torch.Tensor:
        return torch.einsum("eyx,ec->yxc", self.element_amplitudes(), self.element_basis())

    def background_spectra(self) -> torch.Tensor:
        return torch.einsum("byx,bc->yxc", self.background_coeffs(), self.background_basis)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak_part = self.peak_spectra()
        background_part = self.background_spectra()
        return peak_part + background_part, peak_part, background_part

    def get_params(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "amplitudes": [self.element_amplitudes_raw],
            "background": [self.background_coeffs_raw],
        }
        if self.peak_width_raw is not None:
            groups["peak_width"] = [self.peak_width_raw]
        return groups

    @property
    def param_keys(self) -> list[str]:
        keys = ["amplitudes", "background"]
        if self.peak_width_raw is not None:
            keys.append("peak_width")
        return keys
