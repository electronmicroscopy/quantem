from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from quantem.spectroscopy import Dataset3dspectroscopy
from quantem.spectroscopy.spectroscopy_models import EDSModel, GaussianPeaks, PolynomialBackground


class Dataset3deds(Dataset3dspectroscopy):
    """An EDS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (energy, scan_y, scan_x).
    The first dimension represents the energy, while the latter
    two dimensions represent real space sampling.

    """

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

    def fit_spectrum_pytorch(
        self,
        energy_range=None,
        elements_to_fit=None,
        peak_width=0.1,
        num_iters=1000,
        lr=0.01,
        polynomial_background_degree=3,
    ):
        energy_axis = np.arange(self.shape[0]) * self.sampling[0] + self.origin[0]
        energy_axis = torch.tensor(energy_axis, dtype=torch.float32)

        # TODO: make_more_flexible
        spectrum_raw = torch.tensor(self.array.sum((-1, -2)), dtype=torch.float32)

        if energy_range is not None:
            ind = (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            spectrum_raw = spectrum_raw[ind]
            energy_axis = energy_axis[ind]
        else:
            energy_range = [energy_axis.min().numpy(), energy_axis.max().numpy()]

        # rescale 0 to 1
        spectrum_min = spectrum_raw.min()
        spectrum_max = spectrum_raw.max()
        spectrum = spectrum_raw - spectrum_min
        spectrum = spectrum / (spectrum_max - spectrum_min)

        # initialize
        background = PolynomialBackground(energy_axis, degree=polynomial_background_degree)

        peaks = GaussianPeaks(energy_axis, peak_width=peak_width, elements_to_fit=elements_to_fit)
        model = EDSModel(peaks, background, energy_axis=energy_axis)

        with torch.no_grad():
            model.peak_model.concentrations.fill_((1))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        loss_iter = []
        for iters in range(num_iters):
            optimizer.zero_grad()

            predicted = model()

            loss = loss_fn(predicted, spectrum)

            loss.backward()
            optimizer.step()

            loss_iter.append(loss.detach().numpy())

        loss_iter = np.asarray(loss_iter)
        # plot_results
        with torch.no_grad():
            final_pred = model().detach().numpy() * spectrum_max.numpy() + spectrum_min.numpy()
            concs = nn.functional.softplus(model.peak_model.concentrations).detach().numpy()

            final_fwhm = (
                torch.nn.functional.softplus(model.peak_model.peak_width_by_peak)
                .detach()
                .cpu()
                .numpy()
            )
            print(
                f"\nFinal: width median={np.median(final_fwhm):.3f} keV, "
                f"min={final_fwhm.min():.3f}, max={final_fwhm.max():.3f}"
            )

            # Sort and show top N
            top_n = np.max((10, len(elements_to_fit)))
            sorted_indices = np.argsort(concs)[::-1]

            print("\nTop elements:")
            for i, idx in enumerate(sorted_indices[:top_n], 1):
                elem = model.peak_model.element_names[idx]
                conc = concs[idx]
                print(f"{i:2d}. {elem:2s}: {conc:.3f}")

            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].plot(np.arange(loss_iter.shape[0]), loss_iter, color="k")
            ax[0].set_title("loss")
            ax[0].set_xlabel("iterations")
            ax[0].set_ylabel("loss")
            ax[0].set_yscale("log")

            ax[1].plot(energy_axis, spectrum_raw.numpy(), "k-", label="Data", linewidth=1)
            ax[1].plot(energy_axis, final_pred, "r-", label="Fit", linewidth=2)

            if model.background_model is not None:
                background = (
                    model.background_model().detach().numpy() * spectrum_max.numpy()
                    + spectrum_min.numpy()
                )
                ax[1].plot(energy_axis, background, "b--", label="Background", linewidth=1.5)

            ax[1].set_xlim(energy_range[0], energy_range[1])
            ax[1].legend()

            ax[1].set_title("fit spectrum")
            ax[1].set_xlabel("Energy (keV)")
            ax[1].set_ylabel("Counts")

            plt.tight_layout()
            plt.show()

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