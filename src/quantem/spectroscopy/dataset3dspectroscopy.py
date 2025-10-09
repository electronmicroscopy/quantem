from typing import Any, Self, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.decomposition import PCA


from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.validators import ensure_valid_array


class Dataset3dspectroscopy(Dataset3d):
    """A 3D-STEM spectroscopy dataset class that inherits from Dataset3d.

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
        """Initialize a 3D-STEM spectroscopy dataset.

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

    @classmethod
    def from_array(
        cls,
        array: NDArray | Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """
        Create a new Dataset3dspectroscopy from an array.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "3D-STEM dataset"
        origin : NDArray | tuple | list | float | int | None, optional
            The origin coordinates for each dimension. If None, defaults to zeros
        sampling : NDArray | tuple | list | float | int | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones
        units : list[str] | tuple | list | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset3dspectroscopy
            A new Dataset3dspectroscopy instance
        """
        array = ensure_valid_array(array, ndim=3)
        return cls(
            array=array,
            name=name if name is not None else "3D-STEM dataset",
            origin=origin if origin is not None else np.zeros(3),
            sampling=sampling if sampling is not None else np.ones(3),
            units=units if units is not None else ["pixels"] * 3,
            signal_units=signal_units,
            _token=cls._token,
        )

    ## PCA

    def perform_pca(
        self,
        n_components: int = 10,
        standardize: bool = True,
        mask: Optional[NDArray] = None,
        plot_results: bool = True,
        random_state: Optional[int] = 42
    ) -> dict:
        """
        Perform Principal Component Analysis (PCA) on the spectroscopy dataset.
        
        Parameters
        ----------
        n_components : int
            Number of principal components to compute
        standardize : bool
            If True, standardize the data before PCA (zero mean, unit variance)
        mask : Optional[NDArray]
            Optional spatial mask to select pixels for analysis
        plot_results : bool
            If True, plot the explained variance and first few components
        random_state : Optional[int]
            Random state for reproducibility
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'pca': fitted PCA object
            - 'components': principal component spectra (n_components x n_energy)
            - 'loadings': spatial loadings (n_components x n_pixels)
            - 'explained_variance_ratio': explained variance for each component
            - 'reconstructed': reconstructed dataset using n_components
        """
        data = np.asarray(self.array, dtype=float)
        n_energy, ny, nx = data.shape
        
        # Reshape data to (n_pixels, n_energy) for PCA
        data_reshaped = data.reshape(n_energy, -1).T  # (n_pixels, n_energy)
        
        if mask is not None:
            mask_flat = mask.flatten()
            data_masked = data_reshaped[mask_flat]
        else:
            data_masked = data_reshaped
        
        if standardize:
            mean = np.mean(data_masked, axis=0)
            std = np.std(data_masked, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            data_processed = (data_masked - mean) / std
        else:
            data_processed = data_masked
        
        # Perform PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        loadings = pca.fit_transform(data_processed)  # (n_pixels, n_components)
        components = pca.components_  # (n_components, n_energy)
        
        # Reconstruct data
        if standardize:
            reconstructed = pca.inverse_transform(loadings) * std + mean
        else:
            reconstructed = pca.inverse_transform(loadings)
        
        if mask is None:
            loadings_spatial = loadings.T.reshape(n_components, ny, nx)
        else:
            loadings_spatial = np.zeros((n_components, ny * nx))
            loadings_spatial[:, mask_flat] = loadings.T
            loadings_spatial = loadings_spatial.reshape(n_components, ny, nx)
        
        if plot_results:
            self._plot_pca_results(
                components, loadings_spatial, pca.explained_variance_ratio_,
                n_show=min(4, n_components)
            )
        
        return {
            'pca': pca,
            'components': components,
            'loadings': loadings_spatial,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_,
            'reconstructed': reconstructed.T.reshape(n_energy, ny, nx) if mask is None else reconstructed
        }
    
    def _plot_pca_results(
        self,
        components: NDArray,
        loadings: NDArray,
        explained_variance_ratio: NDArray,
        n_show: int = 4
    ):
        """
        Plot PCA results including scree plot, components, and loadings.
        
        Parameters
        ----------
        components : NDArray
            Principal component spectra
        loadings : NDArray
            Spatial loadings for each component
        explained_variance_ratio : NDArray
            Explained variance ratios
        n_show : int
            Number of components to show
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, n_show + 1, width_ratios=[1.5] + [1] * n_show)
        
        # Plot 1: Scree plot (explained variance)
        ax_scree = fig.add_subplot(gs[0, 0])
        cumsum_var = np.cumsum(explained_variance_ratio)
        
        ax_scree.bar(range(1, len(explained_variance_ratio) + 1), 
                    explained_variance_ratio * 100, alpha=0.6, label='Individual')
        ax_scree.plot(range(1, len(explained_variance_ratio) + 1), 
                     cumsum_var * 100, 'ro-', label='Cumulative')
        ax_scree.set_xlabel('Component Number')
        ax_scree.set_ylabel('Explained Variance (%)')
        ax_scree.set_title('Scree Plot')
        ax_scree.legend()
        ax_scree.grid(True, alpha=0.3)
        
        # Get energy axis
        energy_sampling = float(self.sampling[0])
        energy_origin = float(self.origin[0])
        energy_axis = energy_origin + energy_sampling * np.arange(components.shape[1])
        
        # Plot components and loadings
        for i in range(n_show):
            ax_comp = fig.add_subplot(gs[1, i + 1])
            ax_comp.plot(energy_axis, components[i])
            ax_comp.set_title(f'PC{i+1} ({explained_variance_ratio[i]*100:.1f}%)')
            ax_comp.set_xlabel('Energy')
            if i == 0:
                ax_comp.set_ylabel('Component')
            ax_comp.grid(True, alpha=0.3)
            
            ax_load = fig.add_subplot(gs[2, i + 1])
            im = ax_load.imshow(loadings[i], cmap='RdBu_r', origin='lower')
            ax_load.set_title(f'Loading {i+1}')
            ax_load.axis('off')
            plt.colorbar(im, ax=ax_load, fraction=0.046, pad=0.04)
        
        ax_stats = fig.add_subplot(gs[1:, 0])
        ax_stats.axis('off')
        
        stats_text = f"PCA Summary\n" + "="*20 + "\n\n"
        stats_text += f"Total components: {len(explained_variance_ratio)}\n"
        stats_text += f"Components for 95% var: {np.argmax(cumsum_var >= 0.95) + 1}\n"
        stats_text += f"Components for 99% var: {np.argmax(cumsum_var >= 0.99) + 1}\n\n"
        
        for i in range(min(5, len(explained_variance_ratio))):
            stats_text += f"PC{i+1}: {explained_variance_ratio[i]*100:.2f}%\n"
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('PCA Analysis Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    ## imaging
    def plot_virtual_image(
        self,
        E0: float,
        dE: float,
        mask: Optional[NDArray] = None,
        normalize_spectrum: bool = True,
        cmap: str = 'viridis',
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 5)
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Generate a virtual image by integrating over an energy range.
        
        Creates a figure with two sub-panels:
        1. Full spectrum with highlighted energy range
        2. Resulting virtual image from the energy integration
        
        Parameters
        ----------
        E0 : float
            Starting energy for integration (in same units as energy axis)
        dE : float
            Energy range width for integration
        mask : Optional[NDArray]
            Optional spatial mask to apply to the image (same shape as spatial dims)
        normalize_spectrum : bool
            If True, normalize the spectrum display to [0, 1]
        cmap : str
            Colormap for the virtual image display
        title : Optional[str]
            Custom title for the figure
        figsize : Tuple[float, float]
            Figure size (width, height) in inches
        
        Returns
        -------
        fig : plt.Figure
            The matplotlib Figure object
        (ax1, ax2) : Tuple[plt.Axes, plt.Axes]
            The axes for spectrum and image subplots
        """
        # Get energy axis
        energy_sampling = float(self.sampling[0])
        energy_origin = float(self.origin[0])
        energy_axis = energy_origin + energy_sampling * np.arange(self.shape[0])
        
        #energy indices for integration
        E_end = E0 + dE
        energy_indices = np.where((energy_axis >= E0) & (energy_axis <= E_end))[0]
        
        if len(energy_indices) == 0:
            raise ValueError(f"No energy channels found in range [{E0}, {E_end}]")
        
        data = np.asarray(self.array, dtype=float)
        
        # Compute mean spectrum (averaged over all spatial pixels)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != data.shape[1:]:
                raise ValueError(f"Mask shape {mask.shape} doesn't match spatial dimensions {data.shape[1:]}")
            
            spectrum = np.zeros(data.shape[0])
            for i in range(data.shape[0]):
                masked_data = data[i][mask]
                spectrum[i] = masked_data.mean() if masked_data.size > 0 else 0
        else:
            spectrum = data.mean(axis=(1, 2))
        
        # Create virtual image by integrating over energy range
        virtual_image = data[energy_indices].sum(axis=0)
        
        if mask is not None:
            virtual_image = virtual_image * mask
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Full spectrum with highlighted energy range
        
        if normalize_spectrum and spectrum.max() > 0:
            spectrum_display = spectrum / spectrum.max()
        else:
            spectrum_display = spectrum

        ax1.plot(energy_axis, spectrum_display, 'b-', linewidth=1.5, label='Full spectrum')
        
        energy_min_idx = energy_indices[0]
        energy_max_idx = energy_indices[-1]
        ax1.axvspan(energy_axis[energy_min_idx], energy_axis[energy_max_idx], 
                   alpha=0.3, color='red', label=f'Selected range')
        
        ax1.axvline(x=E0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axvline(x=E_end, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel(f'Energy ({self.units[0] if hasattr(self, "units") else "eV"})')
        ax1.set_ylabel('Intensity (arb. units)')
        ax1.set_title('Energy Spectrum')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        y_pos = ax1.get_ylim()[1] * 0.9
        ax1.text(E0 + dE/2, y_pos, f'E: {E0:.1f} - {E_end:.1f}\nΔE: {dE:.1f}',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
        
        # Plot 2: Virtual image
        im = ax2.imshow(virtual_image, cmap=cmap, origin='lower')
        ax2.set_xlabel(f'X ({self.units[2] if hasattr(self, "units") else "pixels"})')
        ax2.set_ylabel(f'Y ({self.units[1] if hasattr(self, "units") else "pixels"})')
        ax2.set_title(f'Virtual Image (E: {E0:.1f} - {E_end:.1f})')

        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Integrated Intensity', rotation=270, labelpad=15)
        
        if title is None:
            title = f'Virtual Image Analysis - Energy Integration'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    ## specturm picking


class DataSpectroscopy:
    """
    Class for handling 3D spectroscopy data and extracting spectra from ROIs.
    """
    def __init__(self, array):
        # Use the underlying array attribute, do not sum over axis 0
        self.array = np.asarray(array.array, dtype=float)
        self.sampling = array.sampling
        self.origin = array.origin
        self.shape = self.array.shape

    def image_to_spec(self, y, x, dy=None, dx=None, title=None):
        """
        Make and show a spectrum plot from a spatial ROI in a 3D EDS cube (E, Y, X).

        Parameters
        ----------
        y, x : int
            Top-left pixel of the ROI.
        dy, dx : int, optional
            ROI size (height, width). Defaults to full image if None.
        title : str, optional
            Plot title (auto-filled if None).

        Returns
        -------
        (fig, ax) : tuple
            The Matplotlib Figure and Axes of the spectrum plot.
        """
        if dy is None:
            dy = self.shape[1]
        if dx is None:
            dx = self.shape[2]

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E  = E0 + dE * np.arange(self.shape[0])

        spec = np.empty(self.shape[0], dtype=float)
        for k in range(self.shape[0]):
            img = np.asarray(self.array[k], dtype=float)
            roi = img[y:y+dy, x:x+dx]
            if roi.size == 0:
                raise ValueError("ROI is empty; check y/x/dy/dx.")
            spec[k] = roi.mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(E, spec)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Intensity")
        ax.set_title(title or f"Spectrum ROI y={y}:{y+dy}, x={x}:{x+dx}")
        fig.tight_layout()
        plt.show()

        return fig, ax
