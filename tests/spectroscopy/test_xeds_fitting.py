import numpy as np
import torch
from matplotlib import pyplot as plt

from quantem.spectroscopy.spectroscopy_models import (
    build_element_peak_basis,
    total_variation_in_plane,
    xeds_data_loss,
)
from quantem.spectroscopy.xeds_fitting import (
    _collect_element_basis_data,
    _plot_loss_history,
)


def _make_basis(dataset, elements, peak_width=0.09):
    energy_axis = torch.tensor(dataset.energy_axis, dtype=torch.float32)
    basis_data = _collect_element_basis_data(dataset, energy_axis, elements)
    basis = build_element_peak_basis(
        energy_axis=energy_axis,
        line_energies=basis_data["line_energies"],
        line_weights=basis_data["line_weights"],
        line_element_indices=basis_data["line_element_indices"],
        n_elements=len(basis_data["element_names"]),
        peak_width=torch.tensor(float(peak_width), dtype=torch.float32),
    )
    return basis_data, basis.detach().cpu().numpy()


class TestXEDSFitting:
    def test_mean_fit_with_adam_recovers_simple_spectrum(self, xeds_factory):
        rng = np.random.default_rng(0)
        dataset = xeds_factory(np.zeros((3, 3, 512), dtype=np.float32))
        basis_data, basis = _make_basis(dataset, ["Fe", "Cu"])

        background = 3.0 + 2.0 * np.linspace(0.0, 1.0, 512, dtype=np.float32)
        amplitudes = np.array([900.0, 350.0], dtype=np.float32)
        mean_spectrum = amplitudes @ basis + background
        cube = np.broadcast_to(mean_spectrum, (3, 3, 512)).copy()
        cube += rng.normal(scale=0.2, size=cube.shape)
        cube = np.clip(cube, 0.0, None)
        dataset = xeds_factory(cube)

        result = dataset.fit_spectrum_mean_pytorch(
            elements_to_fit=["Fe", "Cu"],
            num_iters=80,
            optimizer="adam",
            device="cpu",
            show_plot=False,
            verbose=False,
        )

        assert result["element_names"] == basis_data["element_names"]
        assert result["loss_history"][0] > result["loss_history"][-1]
        assert result["concentrations"][0] > result["concentrations"][1]

        rmse = np.sqrt(np.mean((result["fitted_spectrum"] - mean_spectrum) ** 2))
        assert rmse / np.max(mean_spectrum) < 0.08

    def test_mean_fit_with_lbfgs_runs_and_keeps_outputs_finite(self, xeds_factory):
        dataset = xeds_factory(np.zeros((2, 2, 384), dtype=np.float32))
        _basis_data, basis = _make_basis(dataset, ["Si", "Fe"])

        background = 1.5 + 1.0 * np.linspace(0.0, 1.0, 384, dtype=np.float32)
        amplitudes = np.array([450.0, 120.0], dtype=np.float32)
        mean_spectrum = amplitudes @ basis + background
        cube = np.broadcast_to(mean_spectrum, (2, 2, 384)).copy()
        dataset = xeds_factory(cube)

        result = dataset.fit_spectrum_mean_pytorch(
            elements_to_fit=["Si", "Fe"],
            num_iters=20,
            optimizer="lbfgs",
            device="cpu",
            show_plot=False,
            verbose=False,
        )

        assert np.isfinite(result["fitted_spectrum"]).all()
        assert np.isfinite(result["background_spectrum"]).all()
        assert np.isfinite(result["concentrations"]).all()
        assert result["loss_history"][-1] <= result["loss_history"][0]

    def test_full_cube_fit_returns_maps_and_fit_images(self, xeds_factory):
        rng = np.random.default_rng(1)
        dataset = xeds_factory(np.zeros((4, 4, 384), dtype=np.float32))
        basis_data, basis = _make_basis(dataset, ["Si", "Cu"])

        si_map = np.array(
            [
                [800.0, 720.0, 640.0, 560.0],
                [780.0, 700.0, 620.0, 540.0],
                [760.0, 680.0, 600.0, 520.0],
                [740.0, 660.0, 580.0, 500.0],
            ],
            dtype=np.float32,
        )
        cu_map = np.array(
            [
                [120.0, 200.0, 280.0, 360.0],
                [140.0, 220.0, 300.0, 380.0],
                [160.0, 240.0, 320.0, 400.0],
                [180.0, 260.0, 340.0, 420.0],
            ],
            dtype=np.float32,
        )
        amplitude_maps = np.stack([si_map, cu_map], axis=0)
        background = 2.0 + 0.7 * np.linspace(0.0, 1.0, 384, dtype=np.float32)
        cube = np.einsum("eyx,ec->yxc", amplitude_maps, basis) + background[None, None, :]
        cube += rng.normal(scale=0.05, size=cube.shape)
        cube = np.clip(cube, 0.0, None).astype(np.float32)
        dataset = xeds_factory(cube)

        result = dataset.fit_spectrum_pytorch(
            elements_to_fit=["Si", "Cu"],
            num_iters_global=50,
            num_iters=40,
            optimizer_global="adam",
            optimizer_local="adam",
            loss_local="mse",
            spatial_lambda=0.02,
            device="cpu",
            show_plot=False,
            verbose=False,
        )

        assert result["element_names"] == basis_data["element_names"]
        assert result["abundance_maps"].shape == (2, 4, 4)
        assert result["valid_pixel_mask"].all()
        assert result["loss_history"][0] > result["loss_history"][-1]
        assert np.corrcoef(result["abundance_maps"][0].ravel(), si_map.ravel())[0, 1] > 0.5
        assert np.corrcoef(result["abundance_maps"][1].ravel(), cu_map.ravel())[0, 1] > 0.5
        assert result["spectrum_images_pytorch"]


class TestXEDSFitHelpers:
    def test_loss_and_tv_helpers_are_finite(self):
        predicted = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 2.0, 4.0])
        maps = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])

        assert torch.isfinite(xeds_data_loss(predicted, target, loss="mse"))
        assert torch.isfinite(xeds_data_loss(predicted, target, loss="poisson"))
        assert torch.isfinite(total_variation_in_plane(maps))

    def test_plot_loss_history_shifts_negative_values_for_log_plot(self):
        fig, ax = plt.subplots(1, 1)

        _plot_loss_history(ax, np.array([2.0, -3.0, -1.0, 0.5]), label="Poisson")

        assert ax.get_yscale() == "log"
        assert "Shifted loss" in ax.get_ylabel()
        legend = ax.get_legend_handles_labels()[1]
        assert legend == ["Poisson (shifted)"]
        plt.close(fig)
