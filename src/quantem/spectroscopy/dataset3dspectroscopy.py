from typing import Any, Self, Optional, Tuple

import os
import json
from scipy.signal import find_peaks

import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.validators import ensure_valid_array



class Dataset3dspectroscopy(Dataset3d):
    """
    Class for handling 3D spectroscopy data and extracting spectra from ROIs.
    Accepts either a dataset-like object or explicit arguments, and works as a base class.
    """

    # stores the element line info so you don't need to reload each time
    element_info = None

    # loads the element info
    @classmethod
    def load_element_info(cls, path='xray_lines.json'):
        if cls.element_info is not None:
            # don't reload if already loaded
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, path)
        with open(full_path, 'r') as f:
            cls.element_info = json.load(f)['elements']

    def __init__(self, array, name=None, origin=None, sampling=None, units=None, signal_units="arb. units", _token=None):
        if (
            name is None and origin is None and sampling is None and units is None
            and hasattr(array, "array") and hasattr(array, "name") and hasattr(array, "origin") and hasattr(array, "sampling") and hasattr(array, "units")
        ):
            super().__init__(
                array=array.array,
                name=array.name,
                origin=getattr(array, "origin", np.zeros(3)),
                sampling=array.sampling,
                units=array.units,
                signal_units=getattr(array, "signal_units", signal_units),
                _token=type(self)._token if _token is None else _token,
            )
            # Initialize model elements storage
            self.model_elements = None
        else:
            super().__init__(
                array=array,
                name=name,
                origin=origin,
                sampling=sampling,
                units=units,
                signal_units=signal_units,
                _token=type(self)._token if _token is None else _token,
            )
        
        # Initialize model elements storage
        self.model_elements = None
    
    def add_elements_to_model(self, elements):
        """
        Add elements to the model for persistent use in show_mean_spectrum.
        
        Parameters
        ----------
        elements : list or str
            Element symbol(s) to add to the model. Can be a single string (e.g., 'Al') 
            or list of symbols (e.g., ['Au', 'Cu', 'Si']).
        """
        # Load element info if not already loaded
        if type(self).element_info is None:
            type(self).load_element_info()
        
        # Convert to list if single string provided
        if isinstance(elements, str):
            elements = [elements]
        
        # Convert list of element symbols to dict using class element_info
        if isinstance(elements, list):
            all_info = type(self).element_info
            if all_info is not None:
                # Initialize model_elements as dict if it doesn't exist
                if self.model_elements is None:
                    self.model_elements = {}
                
                # Add new elements to existing model
                for el in elements:
                    if el in all_info:
                        self.model_elements[el] = all_info[el]
    
    def clear_model_elements(self):
        """Clear all elements from the model."""
        self.model_elements = None

    ## PCA ANALYSIS METHODS
    
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

    '''
    def quantify_composition(self, roi=None, elements=None, k_factors=None, method='cliff_lorimer', mask=None):
        """
        Quantify elemental composition from EDS spectrum using Cliff-Lorimer approach.
        
        The Cliff-Lorimer equation relates atomic fractions to X-ray intensities:
        CA/CB = kAB * (IA/IB)
        
        Parameters
        ----------
        roi : list or tuple, optional
            Region of interest as [y, x, dy, dx]. If None, uses full image.
        elements : list, required
            List of element symbols to quantify (e.g., ['Pt', 'Co']).
        k_factors : dict, optional
            K-factors for element pairs relative to first element.
            Format: {'Pt': 1.0, 'Co': 1.23} where first element = 1.0
            If None, uses theoretical k-factors from element database.
        method : str, optional
            Quantification method. Currently supports 'cliff_lorimer'.
        mask : array, optional
            Boolean mask for energy channel selection.
            
        Returns
        -------
        dict : Composition results containing:
            - 'atomic_percent': dict of element -> atomic %
            - 'weight_percent': dict of element -> weight %  
            - 'intensities': dict of element -> integrated intensity
            - 'k_factors': dict of k-factors used
            
        Examples
        --------
        # Basic quantification with theoretical k-factors
        comp = dataset.quantify_composition(elements=['Pt', 'Co'])
        
        # With experimental k-factors
        k_factors = {'Pt': 1.0, 'Co': 1.23}
        comp = dataset.quantify_composition(elements=['Pt', 'Co'], k_factors=k_factors)
        
        # Access results
        print(f"Pt: {comp['atomic_percent']['Pt']:.1f} at%")
        print(f"Co: {comp['atomic_percent']['Co']:.1f} at%")
        """
        
        # Input validation
        if elements is None or len(elements) < 2:
            raise ValueError("At least 2 elements required for quantification")
        
        # Load element info if not available
        if type(self).element_info is None:
            type(self).load_element_info()
        
        # Extract spectrum from ROI
        spectrum_data = self._extract_spectrum_for_quantification(roi, mask)
        spec = spectrum_data['spectrum']
        E = spectrum_data['energy']
        
        # Get X-ray line intensities for each element
        intensities = {}
        for element in elements:
            intensity = self._integrate_element_intensity(element, spec, E)
            intensities[element] = intensity
            
        # Handle k-factors
        if k_factors is None:
            k_factors = self._calculate_theoretical_k_factors(elements)
        else:
            # Validate k-factors
            if not all(elem in k_factors for elem in elements):
                raise ValueError("k_factors must include all elements")
        
        # Apply Cliff-Lorimer quantification
        if method == 'cliff_lorimer':
            results = self._cliff_lorimer_quantification(elements, intensities, k_factors)
        else:
            raise ValueError(f"Unknown quantification method: {method}")
        
        return results
    
    def _extract_spectrum_for_quantification(self, roi, mask):
        """Extract spectrum data for quantification (similar to show_mean_spectrum)."""
        # Parse ROI (reuse logic from show_mean_spectrum)
        if roi is None:
            y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
        elif len(roi) == 2:
            y, x, dy, dx = int(roi[0]), int(roi[1]), 1, 1
        elif len(roi) == 4:
            y_val, x_val, dy_val, dx_val = roi
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
        else:
            raise ValueError("roi must be None, [y, x], or [y, x, dy, dx]")
        
        # Energy axis
        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])
        
        # Extract spectrum with mask handling
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.shape[0],):
                raise ValueError(f"Mask shape {mask.shape} doesn't match energy axis ({self.shape[0]},)")
            arr = np.asarray(self.array, dtype=float)[mask, :, :]
            spec = arr.sum(axis=(1,2)) if arr.shape[0] > 0 else np.zeros(0)
            E = E[mask]
        else:
            spec = np.empty(self.shape[0], dtype=float)
            for k in range(self.shape[0]):
                img = np.asarray(self.array[k], dtype=float)
                roi_data = img[y:y+dy, x:x+dx]
                if roi_data.size == 0:
                    raise ValueError("ROI is empty")
                spec[k] = roi_data.mean()
        
        return {'spectrum': spec, 'energy': E}
    
    def _integrate_element_intensity(self, element, spectrum, energy):
        """Integrate X-ray intensity for a specific element using its characteristic lines."""
        all_info = type(self).element_info
        if element not in all_info:
            raise ValueError(f"Element {element} not found in database")
        
        total_intensity = 0.0
        element_lines = all_info[element]
        
        # Get the most intense lines (K-alpha, L-alpha, etc.)
        weighted_lines = [(info['weight'], info['energy (keV)'], line_name) 
                         for line_name, info in element_lines.items() 
                         if info['energy (keV)'] <= 12.0]  # Ignore high energy lines
        weighted_lines.sort(reverse=True)  # Sort by weight (highest first)
        
        # Use top 3 most intense lines for integration
        for weight, line_energy, line_name in weighted_lines[:3]:
            if weight > 0.1:  # Only significant lines
                # Find integration window around the line
                # Use ±0.1 keV window or adaptive based on energy resolution
                window_width = max(0.1, line_energy * 0.01)  # 1% of energy or 0.1 keV minimum
                
                # Find energy indices for integration
                energy_mask = (energy >= line_energy - window_width) & (energy <= line_energy + window_width)
                
                if np.any(energy_mask):
                    # Simple background subtraction: use linear interpolation at edges
                    line_spectrum = spectrum[energy_mask]
                    if len(line_spectrum) > 2:
                        # Background level from edges of integration window
                        bg_level = (line_spectrum[0] + line_spectrum[-1]) / 2
                        # Integrate above background, weighted by line intensity
                        net_intensity = np.sum(line_spectrum - bg_level) * weight
                        total_intensity += max(0, net_intensity)  # No negative intensities
        
        return total_intensity
    
    def _calculate_theoretical_k_factors(self, elements):
        """Calculate theoretical k-factors using atomic number approximation."""
        # This is a simplified approach - in practice you'd use more sophisticated
        # quantum mechanical calculations or experimental values
        
        # Atomic numbers for common elements
        atomic_numbers = {
            'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14,
            'P': 15, 'S': 16, 'Cl': 17, 'K': 19, 'Ca': 20, 'Ti': 22, 'Cr': 24, 
            'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ag': 47,
            'Pt': 78, 'Au': 79
        }
        
        # Use first element as reference (k = 1.0)
        reference_element = elements[0]
        ref_z = atomic_numbers.get(reference_element, 26)  # Default to Fe
        
        k_factors = {reference_element: 1.0}
        
        # Rough approximation: k_factor scales with atomic number ratio
        # This is very approximate - real k-factors depend on X-ray cross sections,
        # fluorescence yields, detector efficiency, etc.
        for element in elements[1:]:
            element_z = atomic_numbers.get(element, 26)
            # Simplified relationship - should be replaced with proper theoretical calculation
            k_factors[element] = (ref_z / element_z) ** 0.7  # Approximate scaling
        
        print(f"Using theoretical k-factors: {k_factors}")
        print("Note: For accurate quantification, use experimentally determined k-factors")
        
        return k_factors
    
    def _cliff_lorimer_quantification(self, elements, intensities, k_factors):
        """Apply Cliff-Lorimer quantification method."""
        # Cliff-Lorimer equation: CA/CB = kAB * (IA/IB)
        # For multiple elements: CA = kA * IA / Σ(ki * Ii)
        
        # Calculate weighted intensities
        weighted_sum = 0.0
        weighted_intensities = {}
        
        for element in elements:
            weighted_intensity = k_factors[element] * intensities[element]
            weighted_intensities[element] = weighted_intensity
            weighted_sum += weighted_intensity
        
        # Calculate atomic percentages
        atomic_percent = {}
        for element in elements:
            if weighted_sum > 0:
                atomic_percent[element] = (weighted_intensities[element] / weighted_sum) * 100.0
            else:
                atomic_percent[element] = 0.0
        
        # Calculate weight percentages (requires atomic weights)
        atomic_weights = {
            'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00, 'Na': 22.99, 'Mg': 24.31,
            'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.07, 'Cl': 35.45, 'K': 39.10,
            'Ca': 40.08, 'Ti': 47.87, 'Cr': 52.00, 'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93,
            'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38, 'Ag': 107.87, 'Pt': 195.08, 'Au': 196.97
        }
        
        # Convert atomic % to weight %
        weight_sum = 0.0
        for element in elements:
            atomic_wt = atomic_weights.get(element, 55.85)  # Default to Fe
            weight_sum += (atomic_percent[element] / 100.0) * atomic_wt
        
        weight_percent = {}
        for element in elements:
            if weight_sum > 0:
                atomic_wt = atomic_weights.get(element, 55.85)
                weight_percent[element] = ((atomic_percent[element] / 100.0) * atomic_wt / weight_sum) * 100.0
            else:
                weight_percent[element] = 0.0
        
        # Print summary for verification
        print(f"\n=== Quantification Results ===")
        print(f"Method: {method}")
        print(f"Elements: {elements}")
        print(f"ROI: {'Full image' if roi is None else roi}")
        print(f"\nRaw Intensities:")
        for elem in elements:
            print(f"  {elem}: {intensities[elem]:.1f}")
        print(f"\nK-factors used:")
        for elem in elements:
            print(f"  {elem}: {k_factors[elem]:.3f}")
        print(f"\nAtomic Composition:")
        total_atomic = sum(atomic_percent.values())
        for elem in elements:
            print(f"  {elem}: {atomic_percent[elem]:.1f} at%")
        print(f"  Total: {total_atomic:.1f} at%")
        print(f"\nWeight Composition:")
        total_weight = sum(weight_percent.values())
        for elem in elements:
            print(f"  {elem}: {weight_percent[elem]:.1f} wt%")
        print(f"  Total: {total_weight:.1f} wt%")
        
        return {
            'atomic_percent': atomic_percent,
            'weight_percent': weight_percent,
            'intensities': intensities,
            'k_factors': k_factors,
            'method': 'cliff_lorimer',
            'total_atomic': total_atomic,
            'total_weight': total_weight
        }
  
    '''

    def _find_best_element_combinations(self, peak_energies, peak_intensities, tolerance=0.15):
        """
        Find the best combination of elements that explains the detected peaks using a cost function.
        
        Parameters:
        peak_energies : array-like
            Detected peak positions in keV
        peak_intensities : array-like  
            Detected peak intensities
        tolerance : float, default 0.15
            Energy tolerance for peak matching in keV
            
        Returns:
        set : Set of element symbols that best explain the detected peaks
        """
        from itertools import combinations
        
        # Get element database
        all_info = type(self).element_info
        if all_info is None:
            return set()
        
        # Consider combinations of 1-4 elements (reasonable for most samples)
        best_elements = set()
        best_score = float('inf')
        
        # Get commonly analyzed elements (general EDS candidates)
        general_elements = ['Fe', 'Pt', 'Cu', 'C', 'O', 'Ni', 'Co', 'Al', 'Si', 'Ti', 'Cr', 'Mn', 'Au', 'Ag', 'Zn', 'Ca', 'K', 'Na', 'Mg']
        available_elements = [el for el in general_elements if el in all_info]
        
        # Test combinations of different sizes
        top_combinations = []  # Store combinations for analysis
        for num_elements in range(1, min(5, len(available_elements)+1)):
            for element_combo in combinations(available_elements, num_elements):
                score = self._calculate_element_combo_score(
                    element_combo, peak_energies, peak_intensities, all_info, tolerance
                )
                
                top_combinations.append((score, element_combo))
                
                if score < best_score:
                    best_score = score
                    best_elements = set(element_combo)
        return best_elements
    
    def _calculate_element_combo_score(self, element_combo, peak_energies, peak_intensities, all_info, tolerance):
        """
        Calculate a cost function score for a given combination of elements.
        Lower scores are better.
        """
        score = 0.0
        explained_peaks = set()
        
        # General element categories (no specific element bias)
        # Only penalize obvious contaminants/artifacts, don't favor specific elements
        substrate_elements = {'Cu': 0.5, 'C': 0.3}  # Mild penalty for substrate/grid elements
        very_rare_elements = {'Ir': -1.0, 'Os': -1.0, 'Ru': -1.0}  # Small penalty for very unlikely elements
        
        # Apply minimal element category adjustments
        for element in element_combo:
            if element in substrate_elements:
                score += substrate_elements[element]   # Small penalty for substrate
            elif element in very_rare_elements:
                score -= very_rare_elements[element]   # Small penalty for very rare elements
        
        # For each detected peak, find if it can be explained by the element combination
        for i, (peak_energy, peak_intensity) in enumerate(zip(peak_energies, peak_intensities)):
            best_match_distance = float('inf')
            best_line_weight = 0.0
            found_match = False
            
            # Check all elements in the combination
            for element in element_combo:
                if element in all_info:
                    for line_name, line_info in all_info[element].items():
                        line_energy = line_info['energy (keV)']
                        line_weight = line_info.get('weight', 0.5)
                        distance = abs(peak_energy - line_energy)
                        
                        if distance <= tolerance:
                            found_match = True
                            if distance < best_match_distance:
                                best_match_distance = distance
                                best_line_weight = line_weight
            
            if found_match:
                explained_peaks.add(i)
                # Add distance penalty (smaller is better)
                score += best_match_distance
                # Bonus for high-weight lines (major lines like Kα vs minor lines like M-lines)
                score -= best_line_weight * 1.0
            else:
                # Heavy penalty for unexplained peaks
                score += 10.0
        
        # Add penalty for unused elements (prefer simpler explanations)
        unused_element_penalty = (len(element_combo) - 1) * 2.0
        score += unused_element_penalty
        
        # Add penalty for unexplained peaks
        unexplained_peaks = len(peak_energies) - len(explained_peaks)
        score += unexplained_peaks * 5.0
        
        # Bonus for explaining multiple peaks with common elements (like Fe Kα + Kβ)
        multi_peak_bonus = 0.0
        for element in element_combo:
            if element in all_info:
                element_peaks = 0
                major_peaks = 0  # Count major lines (weight > 0.5)
                for line_name, line_info in all_info[element].items():
                    line_energy = line_info['energy (keV)']
                    line_weight = line_info.get('weight', 0.5)
                    for peak_energy in peak_energies:
                        if abs(peak_energy - line_energy) <= tolerance:
                            element_peaks += 1
                            if line_weight > 0.5:
                                major_peaks += 1
                
                if element_peaks > 1:
                    multi_peak_bonus += 2.0  # Bonus for elements with multiple matched peaks
                if major_peaks > 0:
                    multi_peak_bonus += 1.0  # Additional bonus for major line matches
        
        score -= multi_peak_bonus
        
        return score

    def show_mean_spectrum(self, roi=None, energy_range=None, elements=None, ignore_range=None, threshold=5.0, tolerance=0.15, mask=None, show_lines=True):
        """
        Make and show a spectrum plot from a spatial ROI in a 3D EDS cube (E, Y, X).

        Parameters
        ----------
        roi : list or tuple, optional
            Region of interest as [y, x, dy, dx] where:
            - y, x: top-left pixel coordinates
            - dy, dx: height and width of ROI
            Use None for default values:
            - [y, None, dy, None] → row y with height dy, full width
            - [None, x, None, dx] → column x with width dx, full height  
            - [y, x, None, None] → from (y,x) to bottom-right corner
            If roi=None, uses full image. Can also be [y, x] for single pixel.
        energy_range : list or tuple, optional
            Energy range to display as [min_energy, max_energy] in keV.
        elements : list or dict, optional
            Element symbols to plot as X-ray lines (e.g., ['Fe', 'Pt']).
            If None, automatically detects elements from statistically significant peaks.
        ignore_range : list or tuple, optional
            Energy range to ignore during peak detection as [min_energy, max_energy] in keV. 
            E.g., [0, 2.5] ignores 0-2.5 keV during auto-detection.
        threshold : float, optional
            Statistical significance threshold (multiple of background noise). Default: 5.0
        tolerance : float, optional
            Energy tolerance for X-ray line matching in keV. Default: 0.15
        mask : array, optional
            Boolean mask for pixel selection.
        show_lines : bool, optional
            Whether to show element lines and/or auto-detected peaks. 
            Auto-enabled if elements are specified or auto-detection is used.

        Returns
        -------
        (fig, ax) : tuple
            The Matplotlib Figure and Axes of the spectrum plot.
        """
 

    # ADJUST ROI BASED ON GIVEN FLAGS -----------------------------------------------
        # Parse ROI parameter
        if roi is None:
            # Full image
            y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
        elif len(roi) == 2:
            # Single pixel [y, x]
            y, x, dy, dx = int(roi[0]), int(roi[1]), 1, 1
        elif len(roi) == 4:
            # Full ROI [y, x, dy, dx] with None support for defaults
            y_val, x_val, dy_val, dx_val = roi
            
            # Handle None values with defaults
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
        else:
            raise ValueError("roi must be None, [y, x], or [y, x, dy, dx] (with None for defaults)")


    # ERROR HANDLING -------------------------------------------------------------------
        errs = []
        Ymax = int(self.shape[1])
        Xmax = int(self.shape[2])

        # type/NaN checks (optional if you already cast to int above)
        for name, val in (("y", y), ("x", x), ("dy", dy), ("dx", dx)):
            if val is None:
                errs.append(f"{name} is None (missing after normalization).")
       
        # if any None, bail early to avoid arithmetic errors
        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

        # basic constraints
        if y < 0:            errs.append(f"y={y} < 0")
        if x < 0:            errs.append(f"x={x} < 0")
        if dy < 1:           errs.append(f"dy={dy} < 1")
        if dx < 1:           errs.append(f"dx={dx} < 1")

        # starts within image
        if y >= Ymax:        errs.append(f"y start {y} out of bounds [0, {Ymax-1}]")
        if x >= Xmax:        errs.append(f"x start {x} out of bounds [0, {Xmax-1}]")

        # ends within image
        end_y = y + dy
        end_x = x + dx
        if end_y > Ymax:     errs.append(f"y+dy = {end_y} exceeds height {Ymax}")
        if end_x > Xmax:     errs.append(f"x+dx = {end_x} exceeds width {Xmax}")

        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))
        

    # SPECTRUM CALCULATION --------------------------------------------------------------

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E  = E0 + dE * np.arange(self.shape[0])


        # MASK HANDLING ---------------------------------------------------------------------
        if mask is not None:
            # Convert to ndarray and validate
            mask = np.asarray(mask)
            
            # Check that it's a proper ndarray
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Mask must be a numpy ndarray, got {type(mask)}")
            
            # Check dimensions - must be 1D
            if mask.ndim != 1:
                raise ValueError(f"Mask must be 1-dimensional, got {mask.ndim}D array with shape {mask.shape}")
            
            # Convert to bool dtype and validate
            if mask.dtype != bool:
                try:
                    mask = mask.astype(bool)
                except (ValueError, TypeError):
                    raise TypeError(f"Mask cannot be converted to boolean dtype from {mask.dtype}")
            
            # Check shape matches energy axis
            arr = np.asarray(self.array, dtype=float)
            if mask.shape != (arr.shape[0],):
                raise ValueError(f"Mask shape {mask.shape} does not match energy axis shape ({arr.shape[0]},)")
            
            arr = arr[mask, :, :]  # select only masked energy channels
            spec = arr.sum(axis=(1,2)) if arr.shape[0] > 0 else np.zeros(0)
            E = E[mask]  # Mask the energy axis as well
        else:
            spec = np.empty(self.shape[0], dtype=float)
            for k in range(self.shape[0]):
                img = np.asarray(self.array[k], dtype=float)
                roi = img[y:y+dy, x:x+dx]
                if roi.size == 0:
                    raise ValueError("ROI is empty; check y/x/dy/dx.")
                spec[k] = roi.mean()

        # Store ignore_range for later use in element line filtering
        if ignore_range is None:
            ignore_range = [0, 0.25]  # Default: ignore 0-0.25 keV for element lines only

   
    # PLOTTING ---------------------------------------------------------------------------
        
        # Create subplot layout: image on left, spectrum on right
        fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))
        
        # LEFT PLOT: Show sum image with ROI highlighted
        # Create sum image across all energy channels (or masked channels)
        if mask is not None:
            sum_img = np.asarray(self.array, dtype=float)[mask, :, :].sum(axis=0)
            title_suffix = " (masked energies)"
        else:
            sum_img = np.asarray(self.array, dtype=float).sum(axis=0)
            title_suffix = ""
        
        im = ax_img.imshow(sum_img, cmap='viridis', origin='lower')
        ax_img.set_title(f"EDS Sum Image{title_suffix}")
        ax_img.set_xlabel("X (pixels)")
        ax_img.set_ylabel("Y (pixels)")
        
        # Highlight the ROI with a rectangle
        rect = Rectangle((x-0.5, y-0.5), dx, dy, linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
        ax_img.add_patch(rect)
        
        # Add colorbar for the image
        plt.colorbar(im, ax=ax_img, label='Intensity')
        
        # RIGHT PLOT: Show spectrum
        ax_spec.plot(E, spec, linewidth=1.5)
        ax_spec.set_xlabel("Energy (keV)")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(f"Spectrum from ROI [{y}:{y+dy}, {x}:{x+dx}]")
        ax_spec.grid(True, alpha=0.1)
        
        # Use ax_spec for all subsequent peak/line plotting
        ax = ax_spec

        # HANDLE SHOW_LINES FLAG AND MODEL ELEMENTS ------------------------------------
        # Auto-enable show_lines if elements are specified or if auto-detection is needed
        if show_lines is None:
            show_lines = (elements is not None) or (hasattr(self, 'model_elements') and self.model_elements is not None)
        
        # Use model elements if no elements specified but model has elements
        if elements is None and hasattr(self, 'model_elements') and self.model_elements is not None:
            elements = list(self.model_elements.keys())

        # Skip all line plotting if show_lines is False
        if not show_lines:
            fig.tight_layout()
            plt.show()
            return fig, (ax_img, ax_spec)

        # AUTO-DETECT ELEMENTS FROM STATISTICALLY SIGNIFICANT PEAKS -------------------
        auto_peak_labels = []  # Store label positions to avoid overlap
        if elements is None:
            try:
                # Statistical peak detection based on intensity distribution
                # Step 1: Find all potential peaks
                peak_indices, peak_properties = find_peaks(spec, height=0, distance=5)
                peak_heights = peak_properties['peak_heights']
                
                # Step 2: Calculate background statistics
                # Use lower percentiles to estimate background level
                background_level = np.percentile(spec, 25)  # 25th percentile as background
                background_std = np.std(spec[spec <= np.percentile(spec, 50)])  # Std of lower half
                
                # Step 3: Statistical significance threshold
                # A peak is significant if it's above background + N*sigma
                significance_threshold = background_level + 3.0 * background_std
                
                # Step 4: Filter peaks by statistical significance
                significant_peaks = []
                for i, (peak_idx, height) in enumerate(zip(peak_indices, peak_heights)):
                    peak_energy = E[peak_idx]
                    
                    # Skip peaks in ignore range
                    if ignore_range is not None and len(ignore_range) == 2:
                        min_ignore, max_ignore = ignore_range
                        if min_ignore <= peak_energy <= max_ignore:
                            continue
                    
                    # Check statistical significance
                    if height > significance_threshold:
                        significant_peaks.append((peak_idx, height, peak_energy))
                
                # Step 5: Additional filtering based on relative prominence
                if len(significant_peaks) > 0:
                    # Calculate signal-to-noise ratio for each significant peak
                    max_intensity = max([height for _, height, _ in significant_peaks])
                    
                    # Use a more inclusive approach: take significant peaks with good SNR
                    # Lower prominence threshold to 5% and include more peaks
                    prominence_threshold = 0.05 * max_intensity
                    important_peaks = []
                    
                    for peak_idx, height, energy in significant_peaks:
                        snr = height / background_std if background_std > 0 else float('inf')
                        # Include peaks above prominence OR with good SNR
                        if height >= prominence_threshold or snr >= 5.0:
                            important_peaks.append((peak_idx, height, energy, snr))
                    
                    print(f"  {len(important_peaks)} peaks above prominence threshold (5% of max) or SNR >= 5")
                    print(f"  → ALL {len(important_peaks)} peaks used for element identification analysis")
                    
                    # Sort by signal-to-noise ratio for analysis
                    important_peaks.sort(key=lambda x: x[3], reverse=True)  # Sort by SNR
                    
                    # Use all important peaks for cost function analysis
                    analysis_peak_indices = [peak[0] for peak in important_peaks]
                    
                    # For display, limit to most significant peaks if there are many
                    display_limit = min(len(important_peaks), 15)  # Show max 15 in table
                    print(f"  → Showing top {display_limit} peaks in detailed table below")
                    top_peaks = [peak[0] for peak in important_peaks[:display_limit]]
                else:
                    analysis_peak_indices = []
                    top_peaks = []
                
                # Simple element line matching - only include significant lines
                element_lines_db = []
                all_info = type(self).element_info
                if all_info is not None:
                    for elem, lines in all_info.items():
                        # Get top 3 weighted lines per element
                        weighted_lines = [(info['weight'], info['energy (keV)'], elem, line) 
                                        for line, info in lines.items() 
                                        if info['energy (keV)'] <= 12.0]  # Ignore > 12 keV
                        # Sort by weight (highest first) and take top 3
                        weighted_lines.sort(reverse=True)
                        for weight, energy, elem, line in weighted_lines[:3]:
                            if weight > 0.1:  # Only include lines with significant weight
                                element_lines_db.append((energy, elem, line, weight))
                
                # Use cost function with all statistically significant peaks
                if len(analysis_peak_indices) > 0:
                    analysis_energies = E[analysis_peak_indices]
                    analysis_intensities = spec[analysis_peak_indices]
                    
                    # Find optimal element combination using all significant peaks
                    best_elements = self._find_best_element_combinations(
                        analysis_energies, analysis_intensities, tolerance
                    )
                    
                    peak_energies = E[top_peaks]  # For display table
                    peak_intensities = spec[top_peaks]
                else:
                    best_elements = set()
                    peak_energies = []
                    peak_intensities = []
                
                # Now create detailed peak matching report (only if we found significant peaks)
                if len(top_peaks) > 0:
                    peak_data = []
                    for idx in top_peaks:
                        peak_energy = E[idx]
                        peak_intensity = spec[idx]
                        # Find matches within the identified elements first, then others
                        matches = []
                        if element_lines_db:
                            distances = []
                            for el_energy, elem, line, weight in element_lines_db:
                                distance = abs(el_energy - peak_energy)
                                if distance <= tolerance:
                                    # Prioritize matches from identified elements
                                    priority = 0 if elem in best_elements else 1
                                    distances.append((priority, distance, elem, line, el_energy, weight))
                            
                            # Sort by priority (identified elements first), then distance
                            distances.sort(key=lambda x: (x[0], x[1]))
                            matches = [(d[1], d[2], d[3], d[4], d[5]) for d in distances[:3]]  # distance, elem, line, energy, weight
                        
                        peak_data.append((peak_energy, peak_intensity, matches, idx))
                    
                    # Sort by energy for table display
                    peak_data_sorted = sorted(peak_data, key=lambda x: x[0])
                    
                    # Print detailed peak summary table with backup options
                    print(f"\nIdentified Elements: {', '.join(sorted(best_elements)) if best_elements else 'None detected'}")
                    print(f"{'Energy (keV)':<12} {'Intensity':<12} {'Primary Match':<20} {'2nd Option':<20} {'3rd Option':<20}")
                    print("-"*84)
                    
                    for energy, intensity, matches, idx in peak_data_sorted:
                        # Primary match
                        if len(matches) > 0:
                            elem = matches[0][1]
                            line = matches[0][2]
                            if elem == 'Cu':
                                primary = f"{elem} {line} (grid)"
                            elif elem == 'C':
                                primary = f"{elem} {line} (carbon)"
                            else:
                                primary = f"{elem} {line}"
                        else:
                            primary = "Unknown"
                        
                        # Secondary match
                        if len(matches) > 1:
                            elem2 = matches[1][1]
                            line2 = matches[1][2]
                            if elem2 == 'Cu':
                                secondary = f"{elem2} {line2} (grid)"
                            elif elem2 == 'C':
                                secondary = f"{elem2} {line2} (carbon)"
                            else:
                                secondary = f"{elem2} {line2}"
                        else:
                            secondary = "-"
                        
                        # Tertiary match
                        if len(matches) > 2:
                            elem3 = matches[2][1]
                            line3 = matches[2][2]
                            if elem3 == 'Cu':
                                tertiary = f"{elem3} {line3} (grid)"
                            elif elem3 == 'C':
                                tertiary = f"{elem3} {line3} (carbon)"
                            else:
                                tertiary = f"{elem3} {line3}"
                        else:
                            tertiary = "-"
                        
                        print(f"{energy:<12.3f} {intensity:<12.1f} {primary:<20} {secondary:<20} {tertiary:<20}")
                    print("-"*84)
                    
                    # Plot lines for the identified elements
                    y_max = ax_spec.get_ylim()[1] 
                    element_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                    
                    # Plot lines for cost-function identified elements
                    for color_idx, elem in enumerate(list(best_elements)):
                        if elem in all_info:
                            elem_color = element_colors[color_idx % len(element_colors)]
                            
                            # Get top 3 weighted lines for this element (same logic as above)
                            weighted_lines = [(info['weight'], info['energy (keV)'], line_name) 
                                            for line_name, info in all_info[elem].items() 
                                            if info['energy (keV)'] <= 12.0]
                            weighted_lines.sort(reverse=True)
                            
                            # Plot only top 3 significant lines
                            for weight, line_energy, line_name in weighted_lines[:3]:
                                if weight > 0.1 and line_energy >= E[0] and line_energy <= E[-1]:
                                    # Skip lines in ignore range
                                    if ignore_range is not None and len(ignore_range) == 2:
                                        min_ignore, max_ignore = ignore_range
                                        if min_ignore <= line_energy <= max_ignore:
                                            continue
                                    
                                    # Plot line at theoretical position
                                    line_alpha = 0.3 + 0.2 * weight  # More prominent for higher weight
                                    ax_spec.axvline(line_energy, color=elem_color, linestyle='--', 
                                             alpha=min(line_alpha, 0.8), linewidth=1.5)
                                    
                                    # Add label with better positioning and spacing
                                    y_pos = y_max * (0.85 - color_idx * 0.08)  # Start higher, space down
                                    label_x = line_energy + 0.1  # More offset from line
                                    
                                    # Add special labeling for substrate elements
                                    if elem == 'Cu':
                                        label_text = f"{elem} {line_name} (grid)"
                                    elif elem == 'C':
                                        label_text = f"{elem} {line_name} (carbon)"
                                    else:
                                        label_text = f"{elem} {line_name}"
                                    
                                    ax_spec.text(label_x, y_pos, label_text, 
                                           rotation=90, va='bottom', ha='left', 
                                           fontsize=8, color=elem_color, weight='normal',
                                           alpha=0.8)
            except ImportError:
                print("scipy is required for auto peak labeling. Please install scipy.")

        # ELEMENT LINES ---------------------------------------------------------------------
        lines_to_plot = None
        if elements is not None:
            if isinstance(elements, list):
                # Convert list of element symbols to dict using class element_info
                all_info = type(self).element_info
                if all_info is not None:
                    lines_to_plot = {el: all_info[el] for el in elements if el in all_info}
            elif isinstance(elements, dict):
                lines_to_plot = elements
        elif hasattr(self, 'model_elements'):
            # Use model elements if available
            lines_to_plot = self.model_elements

        if lines_to_plot is not None:
            E_min = E[0] if len(E) > 0 else 0
            E_max = E[-1] if len(E) > 0 else 20
            element_labels = []  # Store element label positions (energy, y_position)
            element_line_data = []  # Store element line data for summary table
            y_max = ax.get_ylim()[1]
            colors = ['orange', 'red', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
            
            # Collect all lines to plot first for better positioning
            all_lines_to_plot = []
            
            # First pass: collect all lines that will be plotted and track elements
            elements_to_label = {}  # Track which element gets labeled and where
            
            for color_idx, (elem, lines) in enumerate(lines_to_plot.items()):
                color = colors[color_idx % len(colors)]
                
                # Get top 3 weighted lines for this element
                weighted_lines = [(info['weight'], info['energy (keV)'], line, info) 
                                for line, info in lines.items() 
                                if info['energy (keV)'] <= 12.0]  # Ignore > 12 keV
                weighted_lines.sort(reverse=True)
                
                element_lines = []  # Lines for this element that will be plotted
                
                # Process only top 3 significant lines
                for weight, energy, line, info in weighted_lines[:3]:
                    if weight <= 0.1:  # Skip lines with very low weight
                        continue
                    
                    # Skip lines outside energy range
                    if energy < E_min or energy > E_max:
                        continue
                    
                    # Skip element lines in ignore range
                    if ignore_range is not None and len(ignore_range) == 2:
                        min_ignore, max_ignore = ignore_range
                        if min_ignore <= energy <= max_ignore:
                            continue
                    
                    all_lines_to_plot.append((energy, elem, line, color, weight))
                    element_lines.append((energy, line, weight))
                
                # Determine which line should get the element label (highest weight)
                if element_lines:
                    # Sort by weight and choose the most prominent line for labeling
                    element_lines.sort(key=lambda x: x[2], reverse=True)
                    label_energy, label_line, _ = element_lines[0]
                    elements_to_label[elem] = (label_energy, label_line, color)
            
            # Sort all lines by energy for better positioning
            all_lines_to_plot.sort(key=lambda x: x[0])
            
            # Second pass: plot all lines (but only label once per element)
            for i, (energy, elem, line, color, weight) in enumerate(all_lines_to_plot):
                # Find the closest channel
                idx = np.abs(E - energy).argmin()
                
                # Weight-based alpha (more prominent for higher weights)
                line_alpha = 0.3 + 0.4 * weight
                ax.axvline(E[idx], color=color, linestyle='-', alpha=min(line_alpha, 0.8), linewidth=1.5)
                
                # Store element line data for summary table
                intensity = spec[idx] if 'spec' in locals() else 0
                element_line_data.append((E[idx], intensity, elem, line, weight))
            
            # Third pass: Add labels only for the most prominent line of each element
            # Sort elements by energy for systematic positioning
            sorted_elements = sorted(elements_to_label.items(), key=lambda x: x[1][0])
            
            for i, (elem, (label_energy, label_line, color)) in enumerate(sorted_elements):
                # Find the closest channel for the label energy
                idx = np.abs(E - label_energy).argmin()
                
                # Simple vertical spacing - each element gets its own height level
                base_y = y_max * 0.9  # Start near top
                vertical_spacing = y_max * 0.08  # 8% spacing between labels
                
                # Position labels at regular intervals going down
                final_y_pos = base_y - (i * vertical_spacing)
                
                # Keep within reasonable bounds
                final_y_pos = max(final_y_pos, y_max * 0.2)  # Don't go below 20%
                
                element_labels.append((E[idx], final_y_pos))
                
                # Add label with better offset and styling
                label_x = E[idx] - 0.1  # Offset from line
                
                # Add special labeling for substrate elements (element name only, no line designation)
                if elem == 'Cu':
                    label_text = f"{elem} (grid)"
                elif elem == 'C':
                    label_text = f"{elem} (carbon)"
                else:
                    label_text = elem  # Just the element symbol
                
                ax.text(label_x, final_y_pos, label_text, rotation=90, va='bottom', ha='right', 
                       fontsize=8, color=color, weight='normal', alpha=0.8, clip_on=True)
            
            # Print concise element lines summary
            if element_line_data:
                element_line_data_sorted = sorted(element_line_data, key=lambda x: x[0])  # Sort by energy
                print(f"\nElement Lines: {', '.join([f'{elem} {line}' for _, _, elem, line, _ in element_line_data_sorted])}")

        fig.tight_layout()
        plt.show()

        return fig, (ax_img, ax_spec)
    


Dataset3dspectroscopy.load_element_info()
