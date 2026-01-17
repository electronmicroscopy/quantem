from typing import Any, Self, Optional, Tuple

import os
import json
import csv
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

    # stores the element line info so you don't need to reload each time
    element_info = None

    # loads the xray lines dataset
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

# QUANTIFICATION -----------------------------------------------

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
        if k_factors is None: # if they arent provided, calculate from kfacs_Titan_300_keV.csv
            k_factors = self._calculate_theoretical_k_factors(elements)
        else:
            # Validate k-factors
            if not all(elem in k_factors for elem in elements):
                raise ValueError("k_factors must include all elements")
        
        # Apply Cliff-Lorimer quantification
        if method == 'cliff_lorimer':
            results = self._cliff_lorimer_quantification(elements, intensities, k_factors, method, roi)
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
                # Use +/- 0.1 keV window or adaptive based on energy resolution
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
        """Load k-factors from Titan 300 keV CSV file."""
        # Get the path to the CSV file (same directory as this Python file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'kfacs_Titan_300_keV.csv')
        
        # Load k-factors from CSV
        k_factor_data = {}
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    element = row['Element']
                    k_factor_data[element] = {
                        'K': float(row['K']),
                        'L': float(row['L']),
                        'M': float(row['M'])
                    }
        except FileNotFoundError:
            print(f"Warning: K-factor CSV file not found at {csv_path}")
            print("Using simplified k-factors (all set to 1.0)")
            return {elem: 1.0 for elem in elements}
        
        # Get element info database to determine which X-ray line to use
        all_info = type(self).element_info
        
        k_factors = {}
        for element in elements:
            if element not in k_factor_data:
                print(f"Warning: Element {element} not found in k-factor database, using 1.0")
                k_factors[element] = 1.0
                continue
            
            # Determine which X-ray line (K, L, or M) to use based on the element's primary lines
            if element in all_info:
                element_lines = all_info[element]
                
                # Check which X-ray series is most prominent for this element
                has_k_lines = any('Ka' in line or 'Kb' in line for line in element_lines.keys())
                has_l_lines = any('La' in line or 'Lb' in line for line in element_lines.keys())
                has_m_lines = any('Ma' in line or 'Mb' in line for line in element_lines.keys())
                
                # Prioritize K-lines, then L-lines, then M-lines
                if has_k_lines and k_factor_data[element]['K'] > 0:
                    k_factors[element] = k_factor_data[element]['K']
                    line_type = 'K'
                elif has_l_lines and k_factor_data[element]['L'] > 0:
                    k_factors[element] = k_factor_data[element]['L']
                    line_type = 'L'
                elif has_m_lines and k_factor_data[element]['M'] > 0:
                    k_factors[element] = k_factor_data[element]['M']
                    line_type = 'M'
                else:
                    # Default to K-line k-factor if available
                    if k_factor_data[element]['K'] > 0:
                        k_factors[element] = k_factor_data[element]['K']
                        line_type = 'K'
                    elif k_factor_data[element]['L'] > 0:
                        k_factors[element] = k_factor_data[element]['L']
                        line_type = 'L'
                    elif k_factor_data[element]['M'] > 0:
                        k_factors[element] = k_factor_data[element]['M']
                        line_type = 'M'
                    else:
                        k_factors[element] = 1.0
                        line_type = 'default'
            else:
                # Element not in database, use K-line if available
                if k_factor_data[element]['K'] > 0:
                    k_factors[element] = k_factor_data[element]['K']
                    line_type = 'K'
                elif k_factor_data[element]['L'] > 0:
                    k_factors[element] = k_factor_data[element]['L']
                    line_type = 'L'
                elif k_factor_data[element]['M'] > 0:
                    k_factors[element] = k_factor_data[element]['M']
                    line_type = 'M'
                else:
                    k_factors[element] = 1.0
                    line_type = 'default'
        
        print(f"Using k-factors from Titan 300 keV database: {csv_path}")
        for elem in elements:
            print(f"  {elem}: {k_factors[elem]:.3f}")
        
        return k_factors
    
    def _cliff_lorimer_quantification(self, elements, intensities, k_factors, method, roi):
        """Apply Cliff-Lorimer quantification method."""
        # Cliff-Lorimer equation: CA/CB = kAB * (IA/IB)
        # For multiple elements: CA = kA * IA / SUM(ki * Ii)
        
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
        
        # Print summary in Cliff-Lorimer format
        print(f"\n=== Quantification (Cliff-Lorimer) ===")
        print(f"ROI: {'Full image' if roi is None else roi}")
        print(f"Elements: {', '.join(elements)}")
        
        print(f"\nRaw Intensities:")
        for elem in elements:
            print(f"  {elem}: {intensities[elem]:.2f}")
        
        print(f"\nk-factors:")
        for elem in elements:
            print(f"  {elem}: {k_factors[elem]:.2f}")
        
        print(f"\nAtomic %:")
        for elem in elements:
            print(f"  {elem}: {atomic_percent[elem]:.1f} at%")
        
        print(f"\nWeight %:")
        for elem in elements:
            print(f"  {elem}: {weight_percent[elem]:.1f} wt%")
        
        return {
            'atomic_percent': atomic_percent,
            'weight_percent': weight_percent,
            'intensities': intensities,
            'k_factors': k_factors,
            'method': 'cliff_lorimer'
        }

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
        
        Strategy: Prioritize explaining ALL major peaks with the FEWEST elements.
        Only accept combinations that explain most peaks with high-weight lines.
        """
        score = 0.0
        explained_peaks = {}  # peak_idx -> (matched_distance, line_weight, element)
        
        # For each detected peak, find the BEST match in the element combination
        for i, (peak_energy, peak_intensity) in enumerate(zip(peak_energies, peak_intensities)):
            best_match_distance = float('inf')
            best_line_weight = 0.0
            best_element = None
            found_match = False
            
            # Check all elements in the combination
            for element in element_combo:
                if element in all_info:
                    for line_name, line_info in all_info[element].items():
                        line_energy = line_info['energy (keV)']
                        line_weight = line_info.get('weight', 0.5)
                        distance = abs(peak_energy - line_energy)
                        
                        # Only consider lines with significant weight (major lines only)
                        if line_weight > 0.2 and distance <= tolerance:
                            # Update best match if this line is better
                            if distance < best_match_distance or (distance == best_match_distance and line_weight > best_line_weight):
                                best_match_distance = distance
                                best_line_weight = line_weight
                                best_element = element
                                found_match = True
            
            if found_match:
                explained_peaks[i] = (best_match_distance, best_line_weight, best_element)
                # Penalty for distance (prefer closer matches)
                score += best_match_distance * 10.0
                # Bonus for high-weight lines (major lines score much better)
                score -= best_line_weight * 3.0
            else:
                # HEAVY penalty for unexplained peaks - this is the key constraint
                score += 50.0
        
        # Primary objective: explain ALL detected peaks
        unexplained_peaks = len(peak_energies) - len(explained_peaks)
        if unexplained_peaks > 0:
            score += unexplained_peaks * 100.0  # Very high penalty for unexplained peaks
        
        # Secondary objective: prefer simpler explanations (fewer elements)
        score += len(element_combo) * 5.0
        
        # Tertiary objective: prefer explanations with multiple peaks per element
        # This avoids one-off false matches and encourages coherent solutions
        peaks_per_element = {}
        for peak_idx, (dist, weight, elem) in explained_peaks.items():
            if elem not in peaks_per_element:
                peaks_per_element[elem] = []
            peaks_per_element[elem].append((dist, weight))
        
        # Bonus if each element explains multiple peaks (coherence - more likely correct)
        for elem, matches in peaks_per_element.items():
            if len(matches) > 1:
                # Elements with 2+ peak matches are much more likely correct
                score -= len(matches) * 2.0
        
        return score
        

    def show_mean_spectrum(self, roi=None, energy_range=None, elements=None, ignore_range=None, threshold=5.0, tolerance=0.15, mask=None, show_lines=True, show_text=True, snr_min=None, snr_threshold=None, distance_threshold_for_sample=0.05, contamination_elements=None, grid_peaks=None, background_subtraction='none', data_type='eds',peaks=15):
        """
        Make and show a spectrum plot from a spatial ROI in a 3D EDS cube (E, Y, X).

        Parameters
        ----------
        roi : list or tuple, optional
            Region of interest as [y, x, dy, dx] where:
            - y, x: top-left pixel coordinates
            - dy, dx: height and width of ROI
            Use None for default values:
            - [y, None, dy, None] = row y with height dy, full width
            - [None, x, None, dx] = column x with width dx, full height  
            - [y, x, None, None] = from (y,x) to bottom-right corner
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
        show_text : bool, optional
            Whether to show text labels for detected elements. Default: True.
            When False, vertical lines are still shown but element labels are hidden.
        snr_min : float, optional
            Minimum SNR threshold for detecting any peak. If None, automatically determined
            from peak distribution (typically 20-30 based on data characteristics).
            Lower values detect more peaks, higher values are more selective.
        snr_threshold : float, optional
            Minimum SNR for identifying a peak as a sample element (not contamination).
            If None, automatically determined based on peak statistics. For sparse spectra
            (few strong peaks), uses lower threshold (~30). For dense spectra (many peaks),
            uses higher threshold (~50-80) to filter noise.
        distance_threshold_for_sample : float, optional
            Maximum energy distance (keV) between detected peak and characteristic line
            for identifying as a sample element. Default: 0.05. Stricter values (smaller)
            reduce false positives.
        contamination_elements : set or list, optional
            Element symbols to exclude from sample detection (e.g., {'C', 'Cu', 'O'}).
            Default: {'C', 'N', 'O', 'Cu', 'Si', 'K', 'Kr', 'Po', 'Pb', 'Os', 'Ir', 'At', 'Do', 'Po'}
            These are common TEM support materials and artifacts.
        grid_peaks : dict, optional
            Dictionary of known grid/support peaks for labeling, e.g., {'C': 0.260, 'Cu': 8.020}.
            Default: {'C': 0.260, 'Cu': 8.020} for carbon support film and copper TEM grid.
        background_subtraction : str, optional
            Background subtraction method. Options:
            - 'none' (default): No background subtraction
            - 'auto': Automatically choose best method for data_type (EDS -> power-law, EELS -> iterative Gaussian)
            - 'powerlaw': Power-law background (best for EDS, suitable for Bremsstrahlung)
            - 'iterative': Iterative Gaussian fitting (best for EELS, isolates continuum)
        data_type : str, optional
            Type of spectroscopy data. Options: 'eds' (default) or 'eels'.
            Used with background_subtraction='auto' to select optimal method.
        peaks : int, optional
            Maximum number of peaks to display in the output table and plot as vertical lines.
            Default: 15. Limits output to peaks with highest SNR (most statistically significant).

        Returns
        -------
        (fig, ax) : tuple
            The Matplotlib Figure and Axes of the spectrum plot.
        """
        
        # Set defaults for detection parameters
        if contamination_elements is None:
            contamination_elements = {'C', 'N', 'O', 'Cu', 'Si', 'K', 'Kr', 'Po', 'Pb', 'Os', 'Ir', 'At', 'Do', 'Po'}
        else:
            contamination_elements = set(contamination_elements)
        
        if grid_peaks is None:
            grid_peaks = {'C': 0.260, 'Cu': 8.020}
 

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


    # VALIDATE ROI BOUNDS ---------------------------------------------------------------------------
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
        
        # BACKGROUND SUBTRACTION -------------------------------------------------------------------
        # Apply appropriate background subtraction method
        if background_subtraction == 'auto':
            # Automatically select best method for the data type
            if data_type.lower() == 'eels':
                background_subtraction = 'iterative'
            else:  # Default to EDS
                background_subtraction = 'powerlaw'
        
        if background_subtraction == 'powerlaw':
            # EDS: Power-law Bremsstrahlung background
            spec = self._subtract_background_eds(spec, E)
        elif background_subtraction == 'iterative':
            # EELS: Iterative Gaussian fitting for continuum
            spec = self._subtract_background_eels(spec, E)
        # else: 'none' - no subtraction

   
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

        # AUTO-DETECT PEAKS AND MATCH TO DATABASE -------------------
        if elements is None or (isinstance(elements, list) and len(elements) > 0):
            # elements is either None (full auto-detection) or a list of specific elements to search for
            try:
                # Step 1: Find all potential peaks
                peak_indices, peak_properties = find_peaks(spec, height=0, distance=5)
                peak_heights = peak_properties['peak_heights']
                
                # Step 2: Calculate background statistics
                # Use nanpercentile to handle any NaN values in the spectrum
                background_level = np.nanpercentile(spec, 25)
                background_std = np.nanstd(spec[spec <= np.nanpercentile(spec, 50)])
                
                # Step 3: Determine dynamic SNR thresholds if not provided
                # Calculate initial SNR for all peaks to assess data characteristics
                initial_snrs = []
                for peak_idx, height in zip(peak_indices, peak_heights):
                    snr = height / background_std if background_std > 0 else float('inf')
                    initial_snrs.append(snr)
                
                # Calculate statistics of SNR distribution
                if len(initial_snrs) > 0:
                    snr_median = np.nanmedian(initial_snrs)
                    snr_75th = np.nanpercentile(initial_snrs, 75)
                    snr_95th = np.nanpercentile(initial_snrs, 95)
                    num_high_snr_peaks = np.sum(np.array(initial_snrs) > 50)
                else:
                    snr_median = 0
                    snr_75th = 0
                    snr_95th = 0
                    num_high_snr_peaks = 0
                
                # Set snr_min (detection threshold) if not provided
                if snr_min is None:
                    # Use adaptive threshold based on SNR distribution
                    # For noisy data with many weak peaks, use higher threshold
                    if snr_median > 30:
                        min_snr = 25.0  # Many peaks -> slightly higher threshold
                    else:
                        min_snr = 20.0  # Sparse peaks -> standard threshold
                else:
                    min_snr = snr_min
                
                # Set snr_threshold (sample element threshold) if not provided
                if snr_threshold is None:
                    # Adaptive threshold based on peak density and SNR distribution
                    # Sparse spectra (few strong peaks) -> lower threshold
                    # Dense spectra (many peaks) -> higher threshold to filter noise
                    if num_high_snr_peaks > 50:  # Many high-SNR peaks (dense spectrum like map1)
                        snr_threshold_for_sample = min(80.0, snr_75th * 1.2)
                    elif num_high_snr_peaks > 20:  # Moderate number of peaks
                        snr_threshold_for_sample = min(60.0, snr_75th * 1.1)
                    elif num_high_snr_peaks < 10:  # Few peaks (sparse spectrum like Bare)
                        snr_threshold_for_sample = max(30.0, snr_75th * 0.8)
                    else:  # Default case
                        snr_threshold_for_sample = 40.0
                    
                    print(f"Auto-determined thresholds: snr_min={min_snr:.1f}, snr_threshold={snr_threshold_for_sample:.1f}")
                    print(f"  (Based on: median_snr={snr_median:.1f}, 75th_percentile={snr_75th:.1f}, high_snr_peaks={num_high_snr_peaks})")
                else:
                    snr_threshold_for_sample = snr_threshold
                
                # Step 4: Filter peaks by SNR
                significant_peaks = []
                for peak_idx, height in zip(peak_indices, peak_heights):
                    peak_energy = E[peak_idx]
                    
                    # Skip peaks in ignore range
                    if ignore_range is not None and len(ignore_range) == 2:
                        min_ignore, max_ignore = ignore_range
                        if min_ignore <= peak_energy <= max_ignore:
                            continue
                    
                    snr = height / background_std if background_std > 0 else float('inf')
                    
                    # Keep peaks with good SNR
                    if snr >= min_snr:
                        significant_peaks.append((peak_idx, height, peak_energy, snr))
                
                if len(significant_peaks) > 0:
                    # Sort by SNR (signal-to-noise ratio) for most statistically significant peaks
                    significant_peaks.sort(key=lambda x: x[3], reverse=True)
                    
                    # Limit to top N peaks for display
                    display_peaks = significant_peaks[:peaks]
                    
                    # Match detected peaks to xray_lines.json
                    all_info = type(self).element_info
                    peak_matches = []  # List of (peak_idx, height, peak_energy, snr, element, match_string, distance)
                    
                    # If specific elements are requested, filter the database to only those
                    if elements is not None and isinstance(elements, list):
                        search_elements = set(elements)
                        search_mode = f"for {search_elements}"
                    else:
                        search_elements = None
                        search_mode = "for all elements"
                    
                    print(f"\nDetected {len(significant_peaks)} peaks (SNR >= {min_snr:.1f}) {search_mode}")
                    if len(significant_peaks) > peaks:
                        print(f"Showing top {peaks} peaks by SNR (most statistically significant)")
                    print(f"{'Energy (keV)':<12} {'Intensity':<12} {'SNR':<8} {'Best Match':<25}")
                    print("-"*60)
                    
                    # For each detected peak, find the best match in the database
                    for peak_idx, height, peak_energy, snr in display_peaks:
                        best_match = None
                        best_distance = float('inf')
                        best_element = None
                        
                        # Search through elements in database
                        if all_info:
                            for elem, lines in all_info.items():
                                # If specific elements requested, only search those
                                if search_elements is not None and elem not in search_elements:
                                    continue
                                
                                for line_name, line_info in lines.items():
                                    line_energy = line_info['energy (keV)']
                                    line_weight = line_info.get('weight', 0.5)
                                    distance = abs(peak_energy - line_energy)
                                    
                                    # Prioritize K and L lines over M lines for element identification
                                    # M-lines are very weak and prone to false positives at low energies
                                    is_m_line = 'M' in line_name and not ('Ma' in line_name or 'Mb' in line_name)
                                    
                                    # Match to characteristic lines within tolerance
                                    # Require weight > 0.3 (filters weakest M-lines)
                                    # Penalize M-line matches by requiring closer distance
                                    effective_tolerance = tolerance * 0.5 if is_m_line else tolerance
                                    
                                    if line_weight > 0.3 and distance <= effective_tolerance and distance < best_distance:
                                        best_distance = distance
                                        best_match = f"{elem} {line_name}"
                                        best_element = elem
                        
                        if best_match:
                            peak_matches.append((peak_idx, height, peak_energy, snr, best_element, best_match, best_distance))
                            print(f"{peak_energy:<12.3f} {height:<12.1f} {snr:<8.1f} {best_match:<25}")
                        else:
                            print(f"{peak_energy:<12.3f} {height:<12.1f} {snr:<8.1f} {'Unknown':<25}")
                    
                    print("-"*60)
                    
                    # Detect elements: use only the strongest peaks that match VERY well
                    # Strategy: keep only peaks that:
                    # 1. Match a characteristic line within distance_threshold_for_sample (very tight tolerance)
                    # 2. Have SNR > snr_threshold_for_sample (strong peaks)
                    # 3. Are from non-contamination elements (or requested elements if specified)
                    detected_elements = set()
                    detected_sample_peaks = {}  # Map peak_energy -> is_sample_element for line styling
                    
                    for peak_idx, height, peak_energy, snr, element, match_str, distance in peak_matches:
                        # Very strict criteria for element detection
                        if (snr > snr_threshold_for_sample and                    # Strong peak
                            distance < distance_threshold_for_sample):            # Very close match to characteristic line
                            
                            # If specific elements requested, only keep those; otherwise exclude contamination
                            if search_elements is not None:
                                if element in search_elements:
                                    detected_elements.add(element)
                                    detected_sample_peaks[peak_energy] = True
                            else:
                                if element not in contamination_elements:  # Not a known contamination
                                    detected_elements.add(element)
                                    detected_sample_peaks[peak_energy] = True
                    
                    # MULTI-PEAK COHERENCE CHECK: Filter out elements with only single weak matches
                    # Count DISTINCT characteristic lines for each element (Ka vs Kb, La vs Lb, etc.)
                    element_line_types = {}  # element -> set of line types (e.g., 'Ka', 'Lb')
                    element_total_snr = {}
                    element_has_major_lines = {}  # Track if element has K or L lines (not just M)
                    
                    for peak_idx, height, peak_energy, snr, element, match_str, distance in peak_matches:
                        # Count ALL good matches for each element (not just sample-quality ones)
                        if distance < tolerance * 2:  # Within 2x tolerance for counting
                            if element not in element_line_types:
                                element_line_types[element] = set()
                                element_total_snr[element] = 0
                                element_has_major_lines[element] = False
                            
                            # Extract line type from match_str (e.g., "Pt La" -> "La")
                            line_type = match_str.split()[-1] if match_str else ""
                            element_line_types[element].add(line_type)
                            element_total_snr[element] += snr
                            
                            # Check if this is a major line (K or L series)
                            if any(x in line_type for x in ['Ka', 'Kb', 'La', 'Lb', 'Lg']):
                                element_has_major_lines[element] = True
                    
                    # Filter detected_elements: keep only if multiple DISTINCT lines OR very high SNR
                    # CRITICAL: Reject elements with only M-lines (no K or L confirmation)
                    filtered_detected_elements = set()
                    for element in detected_elements:
                        distinct_line_count = len(element_line_types.get(element, set()))
                        total_snr = element_total_snr.get(element, 0)
                        avg_snr = total_snr / distinct_line_count if distinct_line_count > 0 else 0
                        has_major_lines = element_has_major_lines.get(element, False)
                        
                        # Keep element if:
                        # - Has K or L lines (not just M-lines) - required for heavy elements
                        # - AND (has 2+ DISTINCT lines OR 1 line with very high SNR >70)
                        if has_major_lines and (distinct_line_count >= 2 or avg_snr > 70):
                            filtered_detected_elements.add(element)
                    
                    # Update detected_elements with filtered set
                    detected_elements = filtered_detected_elements
                    
                    # Update detected_sample_peaks to only include filtered elements
                    filtered_sample_peaks = {}
                    for peak_energy, is_sample in detected_sample_peaks.items():
                        # Find which element this peak belongs to
                        for peak_idx, height, pe, snr, element, match_str, distance in peak_matches:
                            if abs(pe - peak_energy) < 0.001 and element in detected_elements:
                                filtered_sample_peaks[peak_energy] = is_sample
                                break
                    detected_sample_peaks = filtered_sample_peaks
                    
                    # Plot detected peaks with appropriate line style (limit to display_peaks)
                    for peak_idx, height, peak_energy, snr in display_peaks:
                        # Use solid line for sample elements, dotted for others
                        is_sample = detected_sample_peaks.get(peak_energy, False)
                        linestyle = '-' if is_sample else ':'
                        
                        ax_spec.axvline(peak_energy, color='red', linestyle=linestyle, alpha=0.3, linewidth=1.5)
                        
                        # Add labels for grid artifacts and sample elements (if show_text enabled)
                        if show_text:
                            y_pos = height * 0.7  # Position label at 70% of peak height
                            
                            # Check if this is a grid/contamination peak
                            is_grid_peak = False
                            for grid_elem, grid_energy in grid_peaks.items():
                                if abs(peak_energy - grid_energy) < 0.1:  # Within 0.1 keV of known grid peak
                                    ax_spec.text(peak_energy, y_pos, f'{grid_elem}\n(grid)', 
                                               ha='center', va='bottom', fontsize=8, color='gray', style='italic')
                                    is_grid_peak = True
                                    break
                    
                    # If elements were detected, use them for element identification only (not for line plotting)
                    if detected_elements:
                        print(f"\nDetected elements: {', '.join(sorted(detected_elements))}")
                        
                        # Prepare labels with vertical orientation and offset handling
                        # Group labels by energy proximity (within 0.3 keV)
                        labels_to_plot = []  # List of (peak_energy, label_text, color, height)
                        colors_map = {'Fe': 'darkblue', 'Pt': 'darkred'}
                        
                        for peak_idx, height, peak_energy, snr, element, match_str, distance in peak_matches:
                            if element in detected_elements:
                                # Extract line name from match_str (e.g., "Fe Ka" -> "Ka")
                                line_name = match_str.split()[-1] if match_str else ""
                                label_text = f"{element} {line_name}" if line_name else element
                                color = colors_map.get(element, 'black')
                                labels_to_plot.append((peak_energy, label_text, color, height))
                        
                        # Sort by energy to group nearby peaks
                        labels_to_plot.sort(key=lambda x: x[0])
                        
                        # Offset overlapping labels vertically
                        label_offset_map = {}  # Map peak_energy -> vertical offset multiplier
                        proximity_threshold = 1.5  # 1.5 keV
                        
                        for i, (energy, label, color, height) in enumerate(labels_to_plot):
                            # Check if this label is close to previous labels
                            offset_count = 0
                            for j in range(i):
                                prev_energy, prev_label, prev_color, prev_height = labels_to_plot[j]
                                if abs(energy - prev_energy) < proximity_threshold:
                                    offset_count += 1
                            
                            label_offset_map[energy] = offset_count
                        
                        # Plot labels with vertical text and offsets (if show_text enabled)
                        if show_text:
                            for peak_energy, label_text, color, height in labels_to_plot:
                                # Position label above the peak
                                y_pos = height * 1.2
                                
                                ax_spec.text(peak_energy, y_pos, label_text, 
                                           ha='center', va='bottom', fontsize=10, color=color, 
                                           weight='bold', rotation=90)
                else:
                    print(f"\nNo peaks detected with SNR >= {min_snr:.1f}")
                    
            except ImportError:
                print("scipy is required for peak detection. Please install scipy.")
        
        # Skip element lines plotting - only show detected peaks
        # (Element characteristic lines are not plotted when using auto-detection)

        fig.tight_layout()
        plt.show()
        return fig, (ax_img, ax_spec)

# BACKGROND SUBTRACTION

    def subtract_background(self, roi=None, energy_range=None, ignore_range=None, mask=None, data_type='eds'):

        from quantem.spectroscopy import (
            Dataset3deds as Dataset3deds,
        )
        from quantem.spectroscopy import (
            Dataset3deels as Dataset3deels,
        )

        """
        Perform appropriate background subtraction routine on mean spectrum from a 3D spectroscopy dataset.
        """
        
        if data_type == 'eds':
            background = self.calculate_background_powerlaw(roi, energy_range, ignore_range, mask)
        elif data_type == 'eels':
            background = self.calculate_background_iterative(roi, energy_range, ignore_range, mask)

        spec3D_subtracted = np.empty(self.shape, dtype=float)


        for p in range(self.shape[1]):
            for q in range(self.shape[2]):
                spec3D_subtracted[:,p,q] = np.maximum(self.array[:,p,q] - background, 0)


        if data_type == 'eds':
            return Dataset3deds.from_array(
                array = spec3D_subtracted,
                sampling = self.sampling,
                origin = self.origin,
                units = self.units)

        elif data_type == 'eels':
            return Dataset3deels.from_array(
                array = spec3D_subtracted,
                sampling = self.sampling,
                origin = self.origin,
                units = self.units)




Dataset3dspectroscopy.load_element_info()