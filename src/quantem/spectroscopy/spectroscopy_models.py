import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class EDSModel(nn.Module):
    """Complete EDS forward model with optional fit range"""

    def __init__(self, peak_model, background_model=None, fit_range=None, energy_axis=None):
        super().__init__()
        self.peak_model = peak_model
        self.background_model = background_model

    def forward(self):
        spectrum = self.peak_model()
        if self.background_model is not None:
            spectrum = spectrum + self.background_model()
        return spectrum


class GaussianPeaks(nn.Module):
    """Generate Gaussian peaks from peak library"""

    def __init__(self, energy_axis, elements_to_fit=None):
        super().__init__()

        current_dir = Path(__file__).parent
        with open(current_dir / "xray_lines.json", "r") as f:
            data = json.load(f)

        self.energy_axis = torch.tensor(energy_axis, dtype=torch.float32)
        self.energy_min = self.energy_axis.min().item()
        self.energy_max = self.energy_axis.max().item()

        # Calculate energy step for later use
        self.energy_step = (self.energy_axis[1] - self.energy_axis[0]).item()

        # Parse and filter elements
        all_element_data = {}
        for elem, lines in data["elements"].items():
            if len(lines) > 0:
                energies = []
                weights = []

                for line_name, line_data in lines.items():
                    energy = line_data["energy (keV)"]
                    if self.energy_min - 0.5 <= energy <= self.energy_max + 0.5:
                        energies.append(energy)
                        weights.append(line_data["weight"])

                if len(energies) > 0:
                    all_element_data[elem] = {"energies": energies, "weights": weights}

        # Filter to specific elements
        if elements_to_fit is not None:
            self.element_data = {}
            for elem in elements_to_fit:
                if elem in all_element_data:
                    self.element_data[elem] = all_element_data[elem]
        else:
            self.element_data = all_element_data

        self.element_names = list(self.element_data.keys())
        n_elements = len(self.element_names)

        # Pre-compute all peak positions and weights as tensors
        all_peak_energies = []
        all_peak_weights = []
        all_peak_element_indices = []

        for elem_idx, elem in enumerate(self.element_names):
            energies = self.element_data[elem]["energies"]
            weights = self.element_data[elem]["weights"]

            all_peak_energies.extend(energies)
            all_peak_weights.extend(weights)
            all_peak_element_indices.extend([elem_idx] * len(energies))

        # Store as tensors for fast computation
        self.peak_energies = torch.tensor(all_peak_energies, dtype=torch.float32)
        self.peak_weights = torch.tensor(all_peak_weights, dtype=torch.float32)
        self.peak_element_indices = torch.tensor(all_peak_element_indices, dtype=torch.long)
        self.n_peaks = len(all_peak_energies)

        print(f"Fitting {n_elements} elements with {self.n_peaks} total peaks")

        # Learnable parameters
        self.concentrations = nn.Parameter((torch.ones(n_elements)))
        self.peak_width = nn.Parameter(torch.tensor(0.13))

    def forward(self):
        """Vectorized forward pass"""
        centers = self.peak_energies.unsqueeze(1)
        energies = self.energy_axis.unsqueeze(0)

        sigma = self.peak_width / 2.355

        all_peaks = torch.exp(-0.5 * ((energies - centers) / sigma) ** 2)

        all_peaks = all_peaks * self.energy_step / (torch.sqrt(torch.tensor(2 * np.pi)) * sigma)

        peak_concentrations = nn.functional.softplus(
            self.concentrations[self.peak_element_indices]
        )
        weighted_peaks = all_peaks * (peak_concentrations * self.peak_weights).unsqueeze(1)

        spectrum = weighted_peaks.sum(dim=0)

        return spectrum


class PolynomialBackground(nn.Module):
    """Polynomial background model"""

    def __init__(self, energy_axis, degree=3):
        super().__init__()
        self.energy_axis = torch.tensor(energy_axis, dtype=torch.float32)
        self.degree = degree

        # Normalize energy axis to [0, 1] for numerical stability
        self.energy_norm = (self.energy_axis - self.energy_axis.min()) / (
            self.energy_axis.max() - self.energy_axis.min()
        )

        self.coeffs = nn.Parameter(torch.randn(degree + 1) * 0.1)

    def forward(self):
        background = torch.zeros_like(self.energy_axis)
        for i, coeff in enumerate(self.coeffs):
            background += coeff * (self.energy_norm**i)
        return background


class ExponentialBackground(nn.Module):
    """Exponential background for bremsstrahlung"""

    def __init__(self, energy_axis):
        super().__init__()
        self.energy_axis = torch.tensor(energy_axis, dtype=torch.float32)

        self.amplitude = nn.Parameter(torch.tensor(1.0))
        self.decay = nn.Parameter(torch.tensor(0.5))
        self.offset = nn.Parameter(torch.tensor(0.1))

    def forward(self):
        return self.amplitude * torch.exp(-self.decay * self.energy_axis) + self.offset
