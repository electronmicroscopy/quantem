# from collections.abc import Sequence
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, label, sum as label_sum, center_of_mass
from tqdm import tqdm
import torch
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.cnn2d import MultiChannelCNN2d
from quantem.core.datastructures import Vector
from quantem.core.utils.polar import polar_transform_vector, cartesian_transform_vector
from quantem.diffraction.peak_detection import detect_blobs, find_central_beam_from_peaks
from quantem.diffraction.polymer_analytical_functions import add_to_histogram_bilinear
from quantem.core.utils.utils import parse_reciprocal_units, sample_average_from_image
from emdfile import tqdmnd
from scipy.ndimage import gaussian_filter1d

# TODO: Likely dataset4dSTEM rather than dataset4d input class
# Bragg peaks from crystalline vs polymer
# 
# TODO: "BraggPeaksPolymer" vs "BraggPeaksCrystal"
class BraggPeaksPolymer(AutoSerialize):
    """
    
    """

    _token = object()

    def __init__(
        self,
        dataset_cartesian: Dataset4dstem,
        compute_parameters: callable,
        normalize_data: callable,
        model: MultiChannelCNN2d = None,
        final_shape: Tuple[int, int] = (256, 256),
        device: str = 'cpu',
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use BraggPeaks.from_data() or .from_file() to instantiate this class."
            )

        self._dataset_cartesian = dataset_cartesian
        self._device = device
        self._final_shape = final_shape
        # Setting functions for normalization
        self.compute_parameters = compute_parameters
        self.normalize_data = normalize_data
        # To be set by class methods
        self.resized_cartesian_data = None
        self.peak_coordinates_cartesian = None
        self.peak_intensities = None
        self.image_centers = None
        self.polar_data = None
        self.polar_peaks = None
        self.max_radius = None
        self.num_radial_bins = None
        self.num_annular_bins = None

        if model is None:
            # Setup model
            input_channels = 1  # 1 for a greyscale image, 3 for RGB, 4 for RGBA, etc.
            k_size = 3
            # k_size = 7
            num_layers = 4
            start_filters = 16
            num_per_layer = 3
            # num_per_layer = 2
            use_skip_connections = True
            dtype = torch.float32
            dropout = 0.2     
            model = MultiChannelCNN2d(
                in_channels=input_channels,
                out_channels=2,
                start_filters=start_filters,
                num_layers=num_layers,
                num_per_layer=num_per_layer,
                use_skip_connections=use_skip_connections,
                dtype=dtype,
                dropout=dropout,
                final_activations=["sigmoid", "sigmoid"],
                conv_kernel_size=k_size,
            )
        self._model = model

    @property
    def model(self) -> MultiChannelCNN2d:
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def dataset_cartesian(self) -> Dataset4dstem:
        return self._dataset_cartesian

    @dataset_cartesian.setter
    def dataset_cartesian(self, dataset_cartesian):
        self._dataset_cartesian = dataset_cartesian
    @classmethod
    def from_file(
        cls,
        file_path: str,
        device: str,
        compute_parameters: callable,
        normalize_data: callable,
        file_type: str | None = None,
    ) -> "BraggPeaksPolymer":
        dataset_cartesian = Dataset4dstem.from_file(file_path, file_type=file_type)
        return cls.from_data(
            dataset_cartesian=dataset_cartesian,
            device=device,
            compute_parameters=compute_parameters,
            normalize_data=normalize_data,
        )

    @classmethod
    def from_data(
        cls,
        dataset_cartesian: Dataset4dstem,
        device: str,
        compute_parameters: callable,
        normalize_data: callable,
    ) -> "BraggPeaksPolymer":
        return cls(
            dataset_cartesian=dataset_cartesian,
            _token=cls._token,
            device=device,
            compute_parameters=compute_parameters,
            normalize_data=normalize_data,
        )

    def pixels_to_inv_A(self):
        # Get sampling conversion factor
        sampling_unit, sampling_angstrom_conversion_factor = parse_reciprocal_units(
            self.dataset_cartesian.units[2]
        )
        sampling_conversion_factor = self.dataset_cartesian.sampling[2] * sampling_angstrom_conversion_factor
        return sampling_conversion_factor
    
    def preprocess(self):
        print(self.device)
        self.resize_data(device=self.device)

    def resize_data(self, device:str = "cuda:0"):
        print(device)
        Ry, Rx, Qy, Qx = self._dataset_cartesian.shape
        scale_factor = (self._final_shape[0] * self._final_shape[1]) / (Qy * Qx)
        resized_data = np.zeros((Ry, Rx, self._final_shape[0], self._final_shape[1]))
        for i in tqdm(range(Ry), desc='rows'):
            inp = torch.tensor(self._dataset_cartesian[i].array, dtype=torch.float32).to(device)
            inp = torch.nn.functional.interpolate(inp[None, ...], size=self._final_shape, mode='bilinear', align_corners=False) * scale_factor
            resized_data[i, :, :, :] = inp.squeeze().detach().cpu().numpy()
        self.resized_cartesian_data = resized_data

    def set_model_weights(
        self,
        # path_to_model: str = None,
        path_to_weights: str = None,
    ) -> "BraggPeaksPolymer":
        # if path_to_model is None:
            # path_to_model = ""
        if path_to_weights is None:
            path_to_weights = ""  # TODO: Load weights from cloud
        self._model.load_state_dict(torch.load(path_to_weights, weights_only=True, map_location=self.device))
        self._model.to(self.device)

    def _postprocess_single(self, position_map, intensity_map, show=False):
        """Process a single 2D image"""
        # Find peaks with subpixel-refinement
        peak_coords, peak_position_signal_intensities, refinement_success = detect_blobs(
            position_map,
            sigma=1.0,  # Sigma for Gaussian smoothing used in processing
            threshold=0.25  # Threshold for strength of peak position signal to be valid peak
        )

        # If no peaks found, return empty lists
        if len(peak_coords) == 0:
            return np.array([]), np.array([])

        # map_coordinates expects coordinates in (row, col) = (y, x) order
        # peak_coords is already in [row, col] format from detect_blobs
        interpolated_intensities = map_coordinates(
            intensity_map, 
            peak_coords.T,  # Transpose to get [[all_y], [all_x]]
            order=1,  # 1 = bilinear interpolation
            mode='nearest'  # How to handle edges
        )
        
        # Optional: filter out peaks that were not successfully refined
        if np.any(refinement_success):
            pass
        
        if show:
            # Peak positions only
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(position_map, cmap='gray', alpha=0.8)
            ax.set_title("Input Position Map with Marked Peaks")
            ax.scatter(peak_coords[:, 1], peak_coords[:, 0], s=10, c='r', label="Peaks")
            ax.legend()
            plt.tight_layout()
            plt.show()

            # Peak positions with color representing intensity
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(position_map, cmap='gray', alpha=0.8)
            scatter = ax.scatter(
                peak_coords[:, 1],  # x coordinates
                peak_coords[:, 0],  # y coordinates
                c=interpolated_intensities,      # color by intensity
                s=10,
                cmap='turbo',    
                edgecolors='black', # white border for visibility
                linewidths=2,
                alpha=0.9,
                marker='o'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Intensity', fontsize=12)
            ax.set_title('Peak Positions and Intensities', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            plt.show()

        return peak_coords, interpolated_intensities

    def find_peaks_model(
        self, 
        device: str = "cuda:0",
        n_normalize_samples: int = 1000,
        show_plots=False,
    ):
        Ry, Rx, Qy, Qx = self.resized_cartesian_data.shape
        peaks = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["y_pixels", "x_pixels", "y_invA", "x_invA"],
            name="peaks_vector",
            units=["Pixels", "Pixels", "1/Å", "1/Å"],
        )
        # peaks = np.empty((Ry, Rx), dtype='object')
        intensities = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["intensities", "intensities_sampled_from_dp"],
            name="intensities_vector",
            units=["Normalized", "Normalized"],
        )
        # intensities = np.empty((Ry, Rx), dtype='object')
        
        # Normalize - compute parameters from sample
        scan_shape = (Ry, Rx)
        n_positions = scan_shape[0] * scan_shape[1]
        n_normalize_samples = min(n_normalize_samples, n_positions)
        
        # Get stats patterns and move to CPU for parameter computation
        # Sample indices (use numpy instead of torch for this)
        flat_indices = np.random.permutation(n_positions)[:n_normalize_samples]
        scan_y_indices = flat_indices // scan_shape[1]
        scan_x_indices = flat_indices % scan_shape[1]

        # Get stats patterns directly as numpy (never touches GPU)
        stats_patterns = self.resized_cartesian_data[scan_y_indices, scan_x_indices]

        # Move to CPU and convert to numpy for compute_parameters
        # stats_patterns = torch.squeeze(stats_patterns).detach().cpu().numpy()
        median, iqr = self.compute_parameters(stats_patterns)
        normalized_dps_array = np.zeros((Ry, Rx, Qy, Qx))
        # Process row by row
        for i in tqdm(range(Ry), desc="rows"):
            # Get row data
            ins = torch.tensor(
                self.resized_cartesian_data[i], 
                dtype=torch.float32
            ).to(device).squeeze()  # Shape: (Rx, Qy, Qx)
            
            # Normalize using new function (works with tensors on GPU)
            dps_norm = self.normalize_data(ins, median, iqr)
            normalized_dps_array[i, :, :, :] = dps_norm.detach().cpu().numpy()
            ins = dps_norm[:, None, ...]  # Add channel dimension: (Rx, 1, Qy, Qx)
            
            # Pass through model
            outs = self.model(ins).detach().cpu().numpy()
            
            # Post-process each pattern in the row
            for r0 in range(outs.shape[0]):
                peak_coords, peak_intensities = self._postprocess_single(
                    outs[r0, 0], 
                    outs[r0, 1],
                    show=show_plots,
                )
                if len(peak_coords) > 0:
                    # Get sampled intensities from DP
                    peak_intensity_averages = sample_average_from_image(ins[r0].squeeze().detach().cpu().numpy(), peak_coords)
                    peak_intensities_data = np.column_stack([
                        peak_intensities,
                        peak_intensity_averages,
                    ])
                    # First convert to inv A to add to Vector fields
                    peak_data = np.column_stack([
                        peak_coords,  # y_pixels, x_pixels
                        peak_coords * self.pixels_to_inv_A()  # y_invA, x_invA
                    ])
                    peaks.set_data(peak_data, i, r0)
                    # peaks.set_data(np.array(peak_coords), i, r0)
                    # peaks[i, r0] = np.array(peak_coords)
                    intensities.set_data(peak_intensities_data, i, r0)
                    # intensities[i, r0] = np.array(peak_intensities)
        
        print('done')
        self.peak_coordinates_cartesian = peaks
        self.peak_intensities = intensities
        self.normalized_dps_array = normalized_dps_array

    def save_peaks(self, filepath):
        np.save(filepath, self.peak_coordinates_cartesian)

    def load_peaks(self, filepath):
        peak_coordinates_cartesian = np.load(filepath, allow_pickle=True)
        if isinstance(peak_coordinates_cartesian, np.ndarray) and peak_coordinates_cartesian.dtype == object and peak_coordinates_cartesian.size == 1:
            peak_coordinates_cartesian = peak_coordinates_cartesian.item()
        self.peak_coordinates_cartesian = peak_coordinates_cartesian
    
    def save_polar_peaks(self, filepath):
        np.save(filepath, self.polar_peaks)

    def save_polar_data(self, filepath):
        np.save(filepath, self.polar_data)

    def load_polar_peaks(self, filepath):
        polar_peaks = np.load(filepath, allow_pickle=True)
        if isinstance(polar_peaks, np.ndarray) and polar_peaks.dtype == object and polar_peaks.size == 1:
            polar_peaks = polar_peaks.item()
        self.polar_peaks = polar_peaks

    def load_polar_data(self, filepath):
        polar_data = np.load(filepath, allow_pickle=True)
        if isinstance(polar_data, np.ndarray) and polar_data.dtype == object and polar_data.size == 1:
            polar_data = polar_data.item()
        self.polar_data = np.load(filepath, allow_pickle=True)

    def process_polar(self):
        """ Find center of image through brightest peak, return polar transform of data and peaks"""
        self.image_centers = self.find_central_beams_4d()
        self.polar_data = self.polar_transform_4d(self.resized_cartesian_data, centers=self.image_centers)
        self.polar_peaks = self.polar_transform_peaks(cartesian_peaks=self.peak_coordinates_cartesian, centers=self.image_centers)

    def find_central_beams_4d(self, intensity_threshold=0.3, distance_weight=0.5, sampling_radius=2, debug=False, use_tqdm=True):
        """
        Fast central beam finding for entire 4D dataset.
        
        Parameters:
        -----------
        use_tqdm : bool
            Show progress bar
        
        Returns:
        --------
        centers : ndarray, shape (scan_y, scan_x, 2)
            Center coordinates (y, x) for each scan position
        """
        from tqdm import tqdm
        
        scan_y, scan_x, det_y, det_x = self.resized_cartesian_data.shape
        centers = np.zeros((scan_y, scan_x, 2))
        
        iterator = tqdm(range(scan_y), disable=not use_tqdm, desc="Finding centers")
        
        for i in iterator:
            for j in range(scan_x):
                centers[i, j] = find_central_beam_from_peaks(
                    peak_coords=self.peak_coordinates_cartesian[i, j],
                    peak_intensities=None,
                    image_shape=self._final_shape,
                    intensity_threshold=intensity_threshold,
                    distance_weight=distance_weight,
                    debug=debug,
                    image=self.resized_cartesian_data[i, j].squeeze(),  # Provide actual DP
                    sampling_radius=sampling_radius  # Sample n-pixel radius around each peak
                )
        return centers

    def polar_transform_peaks(self, cartesian_peaks, centers, use_tqdm: bool=True):
        # Get sampling conversion factor
        sampling_conversion_factor = self.pixels_to_inv_A()
        polar_peaks = polar_transform_vector(cartesian_vector=cartesian_peaks, centers=centers, use_tqdm=use_tqdm, sampling_conversion_factor=sampling_conversion_factor)
        return polar_peaks

    def polar_transform_4d(self, data, centers, num_r=None, num_theta=360, use_tqdm: bool=True):
        """
        Perform polar transform on the last two axes of a 4D array.
        
        Parameters:
        -----------
        data : ndarray, shape (N, M, H, W)
            4D input array where H, W are the axes to transform
        centers : ndarray, shape (N, M, 2)
            Center of each diffraction pattern (usually determined by central beam)
        num_r : int, optional
            Number of radial bins. If None, uses max radius across all patterns
        num_theta : int, optional
            Number of angular bins (default: 360)
        
        Returns:
        --------
        polar_data : Vector
            Vector with shape (N, M) containing polar-transformed data.
            Each cell has fields:
            - 'r_pixels': radial coordinate in pixels
            - 'theta': angular coordinate in radians
            - 'r_invA': radial coordinate in 1/Å
            - 'intensity': transformed intensity value
        
        Notes:
        ------
        Also sets the following attributes on self:
        - self.max_radius_pixels : maximum radius in pixels
        - self.max_radius_invA : maximum radius in 1/Å
        - self.num_radial_bins : number of radial bins
        - self.num_annular_bins : number of angular bins
        """
        N, M, H, W = data.shape
        
        # Calculate consistent max_radius across entire dataset
        dist_to_origin = np.sqrt(centers[..., 0]**2 + centers[..., 1]**2).min()
        dist_to_corner = np.sqrt((H-1 - centers[..., 0])**2 + (W-1 - centers[..., 1])**2).max()
        max_radius_pixels = max(dist_to_origin, dist_to_corner)
        
        if num_r is None:
            num_r = int(np.ceil(max_radius_pixels))
        
        # Calculate maximum radius in inverse angstroms
        max_radius_invA = max_radius_pixels * self.pixels_to_inv_A()
        
        # Store metadata
        self.max_radius_pixels = max_radius_pixels
        self.max_radius_invA = max_radius_invA
        self.num_radial_bins = num_r
        self.num_annular_bins = num_theta
        
        # Pre-calculate coordinate arrays in both units
        r_pixels = np.linspace(0, max_radius_pixels, num_r)
        r_invA = r_pixels * self.pixels_to_inv_A()
        theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
        
        # Create Vector to store polar data with coordinates
        polar_data = Vector.from_shape(
            shape=(N, M),
            fields=["r_pixels", "theta", "r_invA", "intensity"],
            units=["Pixels", "Radians", "1/Å", "Intensity"],
            name="polar_transformed_data"
        )
        
        # Transform each 2D slice
        iterator = tqdm(range(N), disable=not use_tqdm, desc="Transforming data")
        for i in iterator:
            for j in range(M):
                center_y, center_x = centers[i, j]
                
                # Create meshgrid (always work in pixels for interpolation)
                r_grid, theta_grid = np.meshgrid(r_pixels, theta, indexing='ij')
                
                # Convert polar to Cartesian coordinates
                y_coords = center_y + r_grid * np.sin(theta_grid)
                x_coords = center_x + r_grid * np.cos(theta_grid)
                
                # Use map_coordinates for interpolation
                polar_image = map_coordinates(
                    data[i, j], 
                    [y_coords, x_coords], 
                    order=1,
                    mode='constant',
                    cval=0.0
                )
                
                # Flatten the 2D polar image to 1D for storage in Vector
                # Each point in the polar grid becomes a row
                r_flat = r_grid.ravel()
                theta_flat = theta_grid.ravel()
                r_invA_flat = r_flat * self.pixels_to_inv_A()
                intensity_flat = polar_image.ravel()
                
                # Create data array: [r_pixels, theta, r_invA, intensity]
                polar_data_array = np.column_stack([
                    r_flat,
                    theta_flat,
                    r_invA_flat,
                    intensity_flat
                ])
                
                # Store in Vector
                polar_data.set_data(polar_data_array, i, j)
        
        return polar_data

    def visualize_peak_detection(self, n_images=10, indices=None, images_per_row=5, figsize_per_image=(3.2, 3), vmax_polar=20, vmax_cartesian=1):
        """
        Visualize peak detection results for multiple diffraction patterns.
        
        Parameters:
        -----------
        self : BraggPeaksPolymer
            BraggPeaksPolymer object with processed data
        n_images : int
            Number of images to display (ignored if indices is provided)
        indices : list of tuples, optional
            List of (ind_y, ind_x) coordinates to visualize. If None, random indices are selected.
        images_per_row : int
            Number of images per row (default: 5)
        figsize_per_image : tuple
            Size of each subplot (width, height)
        vmax_polar : float
            Maximum value for polar data colormap
        vmax_cartesian : float
            Maximum value for cartesian data colormap
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        
        # Generate or validate indices
        if indices is None:
            Ry, Rx = self.resized_cartesian_data.shape[:2]
            # Generate random indices
            flat_indices = np.random.choice(Ry * Rx, size=min(n_images, Ry * Rx), replace=False)
            indices = [(idx // Rx, idx % Rx) for idx in flat_indices]
        else:
            n_images = len(indices)
        
        # Calculate grid dimensions
        n_rows = int(np.ceil(n_images / images_per_row))
        n_cols = 5  # 5 types of visualizations per pattern
        actual_cols = images_per_row * n_cols
        
        # Create figure
        fig_width = figsize_per_image[0] * actual_cols
        fig_height = figsize_per_image[1] * n_rows
        fig, axes = plt.subplots(n_rows, actual_cols, figsize=(fig_width, fig_height))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Column titles (only for first row)
        col_titles = [
            "Polar Transform",
            "Polar + Peaks",
            "Cartesian + Peaks",
            "Cartesian Original",
            "Cartesian Normalized"
        ]
        
        # Process each image
        for img_idx, (ind_y, ind_x) in enumerate(indices):
            row = img_idx // images_per_row
            col_offset = (img_idx % images_per_row) * n_cols
            
            # Check if peaks exist for this pattern
            has_peaks = (self.peak_coordinates_cartesian[ind_y, ind_x] is not None and 
                         len(self.peak_coordinates_cartesian[ind_y, ind_x]) > 0)
            
            # 1. Polar Transform
            ax = axes[row, col_offset]
            im = ax.matshow(self.polar_data[ind_y, ind_x], cmap='turbo', vmax=vmax_polar)
            if row == 0:
                ax.set_title(col_titles[0], fontsize=10, pad=10)
            ax.text(0.05, 0.95, f'({ind_y},{ind_x})', transform=ax.transAxes, 
                    fontsize=8, va='top', ha='left', color='white', 
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            ax.set_axis_off()
            
            # 2. Polar Transform with Peaks
            ax = axes[row, col_offset + 1]
            ax.matshow(self.polar_data[ind_y, ind_x], cmap='turbo', vmax=vmax_polar)
            if has_peaks and self.polar_peaks[ind_y, ind_x] is not None and len(self.polar_peaks[ind_y, ind_x]) > 0:
                # Convert radial coordinates to bin indices
                r_coords = self.polar_peaks[ind_y, ind_x][:, 0]
                theta_coords = self.polar_peaks[ind_y, ind_x][:, 1]
                
                # Convert theta from radians to angular bins (0 to num_annular_bins)
                theta_bins = theta_coords * (self.num_annular_bins / (2 * np.pi))
                
                ax.scatter(theta_bins, r_coords, c='red', s=15, alpha=0.8, edgecolors='white', linewidths=0.5)
            if row == 0:
                ax.set_title(col_titles[1], fontsize=10, pad=10)
            ax.set_axis_off()
            
            # 3. Cartesian with Peaks and Center
            ax = axes[row, col_offset + 2]
            ax.matshow(self.resized_cartesian_data[ind_y, ind_x], cmap="gray", vmax=vmax_cartesian)
            if has_peaks:
                ax.scatter(self.peak_coordinates_cartesian[ind_y, ind_x][:, 1], 
                          self.peak_coordinates_cartesian[ind_y, ind_x][:, 0], 
                          c='red', s=15, alpha=0.8, edgecolors='white', linewidths=0.5)
            ax.scatter(self.image_centers[ind_y, ind_x][1], 
                      self.image_centers[ind_y, ind_x][0], 
                      c='red', s=500, marker='x', linewidths=2)
            if row == 0:
                ax.set_title(col_titles[2], fontsize=10, pad=10)
            ax.set_axis_off()
            
            # 4. Original Cartesian
            ax = axes[row, col_offset + 3]
            im = ax.matshow(self.resized_cartesian_data[ind_y, ind_x], cmap="gray", vmax=vmax_cartesian)
            if row == 0:
                ax.set_title(col_titles[3], fontsize=10, pad=10)
            ax.set_axis_off()
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 5. Normalized Cartesian
            ax = axes[row, col_offset + 4]
            im = ax.matshow(self.normalized_dps_array[ind_y, ind_x], cmap="gray")
            if row == 0:
                ax.set_title(col_titles[4], fontsize=10, pad=10)
            ax.set_axis_off()
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        total_plots = n_images
        for idx in range(total_plots, n_rows * images_per_row):
            row = idx // images_per_row
            col_offset = (idx % images_per_row) * n_cols
            for col in range(n_cols):
                axes[row, col_offset + col].set_visible(False)
        
        fig.tight_layout()
        return fig, axes

    def peak_radial_intensity_plot(
        self,
        num_bins=200,
        q_min=None,
        q_max=None,
        ROI_xs=None,
        ROI_ys=None,
        vlines=None,
        vline_colors=None,
        vline_labels=None,
        plot=True,
        return_data=False,
    ):
        """
        Create radial intensity line plot summarizing polar peaks.
        
        Parameters
        ----------
        num_bins : int
            Number of radial bins
        q_min : float, optional
            Minimum q value for binning
        q_max : float, optional
            Maximum q value for binning
        ROI_xs : tuple, optional
            X range for region of interest (not yet implemented)
        ROI_ys : tuple, optional
            Y range for region of interest (not yet implemented)
        vlines : list of lists/arrays, optional
            Vertical lines to plot. Each element is a list/array of x-positions.
            Example: [[1.5, 2.0], [3.0, 3.5]] for two groups of lines
        vline_colors : list of colors, optional
            Colors for each group of vertical lines. Must match length of vlines.
            Example: ['red', 'blue'] or ['#FF0000', '#0000FF']
        vline_labels : list of str, optional
            Labels for each group of vertical lines (for legend)
        plot : bool
            Whether to display the plot
        return_data : bool
            Whether to return the binned data
            
        Returns
        -------
        r_centers : array (optional)
            Radial bin centers
        intensity_sum : array (optional)
            Integrated intensity per bin
        """
        # all_r = self.polar_peaks['r'].flatten() * self.dataset_cartesian.sampling[2]
        all_r = self.polar_peaks['r_invA'].flatten()
        all_intensity = self.peak_intensities['intensities_sampled_from_dp'].flatten()
        
        if q_min is None:
            q_min = 0
        if q_max is None:
            q_max = np.max(all_r)
        r_bins = np.linspace(q_min, q_max, num_bins + 1)
    
        # Histogram the data
        intensity_sum, _ = np.histogram(all_r, bins=r_bins, weights=all_intensity)
        counts, _ = np.histogram(all_r, bins=r_bins)
    
        # Bin centers
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
        if plot:
            # Create line plot
            fig, ax = plt.subplots()
            ax.plot(r_centers, intensity_sum, linewidth=2, label='Intensity')
            ax.set_xlabel('Radial Distance (1/Å)', fontsize=12)
            ax.set_ylabel('Integrated Intensity', fontsize=12)
            ax.set_title('Radial Intensity Profile (All Patterns)', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add vertical lines if provided
            if vlines is not None:
                # Convert to list of lists if needed
                if not isinstance(vlines[0], (list, np.ndarray)):
                    vlines = [vlines]
                
                # Default colors if not provided
                if vline_colors is None:
                    default_colors = plt.cm.tab10(np.linspace(0, 1, len(vlines)))
                    vline_colors = default_colors
                
                # Ensure vline_colors is a list
                if not isinstance(vline_colors, list):
                    vline_colors = [vline_colors]
                
                # Check length match
                if len(vline_colors) != len(vlines):
                    raise ValueError(
                        f"Number of vline_colors ({len(vline_colors)}) must match "
                        f"number of vline groups ({len(vlines)})"
                    )
                
                # Plot each group of vertical lines
                for i, (vline_group, color) in enumerate(zip(vlines, vline_colors)):
                    # Get label if provided
                    label = vline_labels[i] if vline_labels is not None and i < len(vline_labels) else None
                    
                    # Plot each line in the group
                    for j, x_pos in enumerate(vline_group):
                        # Only add label to first line in group (for legend)
                        line_label = label if j == 0 else None
                        ax.axvline(x_pos, color=color, linestyle='--', 
                                  linewidth=1.5, alpha=0.7, label=line_label)
            
                # Add legend if labels were provided
                if vline_labels is not None:
                    ax.legend()
            
            fig.tight_layout()
            plt.show()
        
        if return_data:
            return r_centers, intensity_sum

    def make_orientation_histogram(
        self,
        radial_ranges: np.ndarray = None,
        orientation_map=None,
        orientation_ind: int = 0,
        orientation_growth_angles: np.array = 0.0,
        orientation_separate_bins: bool = False,
        orientation_flip_sign: bool = False,
        upsample_factor=4.0,
        theta_step_deg=1.0,
        sigma_x=1.0,
        sigma_y=1.0,
        sigma_theta=3.0,
        normalize_intensity_image: bool = False,
        normalize_intensity_stack: bool = True,
        progress_bar: bool = True,
        r_field: str = "r",
        theta_field: str = "theta",
        intensity_field: str = "intensities_sampled_from_dp",
    ):
        """
        Create an 3D or 4D orientation histogram from a braggpeaks PointListArray
        from user-specified radial ranges, or from the Euler angles from a fiber
        texture OrientationMap generated by the ACOM module of py4DSTEM.
    
        Args:
            bragg_peaks (BraggVectors):         bragg_vectors containing centered peak locations.
            radial_ranges (np array):           Size (N x 2) array for N radial bins, or (2,) for a single bin.
            orientation_map (OrientationMap):   Class containing the Euler angles to generate a flowline map.
            orientation_ind (int):              Index of the orientation map (default 0)
            orientation_growth_angles (array):  Angles to place into histogram, relative to orientation.
            orientation_separate_bins (bool):   whether to place multiple angles into multiple radial bins.
            upsample_factor (float):            Upsample factor
            theta_step_deg (float):             Step size along annular direction in degrees
            sigma_x (float):                    Smoothing in x direction before upsample
            sigma_y (float):                    Smoothing in x direction before upsample
            sigma_theta (float):                Smoothing in annular direction (units of bins, periodic)
            normalize_intensity_image (bool):   Normalize to max peak intensity = 1, per image
            normalize_intensity_stack (bool):   Normalize to max peak intensity = 1, all images
            progress_bar (bool):                Enable progress bar
            r_field (str):                      Name of radial coordinate field (default: "r")
            theta_field (str):                  Name of angular coordinate field (default: "theta")
            intensity_field (str):              Name of intensity field (default: "intensity")
    
        Returns:
            orient_hist (array):                4D array containing Bragg peak intensity histogram
                                                [radial_bin x_probe y_probe theta]
        """
        # coordinates
        theta = np.arange(0, 180, theta_step_deg) * np.pi / 180.0
        dtheta = theta[1] - theta[0]
        dtheta_deg = dtheta * 180 / np.pi
        num_theta_bins = np.size(theta)
    
        if orientation_map is None:
            # Input bins
            radial_ranges = np.array(radial_ranges)
            if radial_ranges.ndim == 1:
                radial_ranges = radial_ranges[None, :]
            radial_ranges_2 = radial_ranges**2
            num_radii = radial_ranges.shape[0]
            size_input = self.polar_peaks.shape
        else:
            orientation_growth_angles = np.atleast_1d(orientation_growth_angles)
            num_angles = orientation_growth_angles.shape[0]
            size_input = [orientation_map.num_x, orientation_map.num_y]
            if orientation_separate_bins is False:
                num_radii = 1
            else:
                num_radii = num_angles
    
        size_output = np.round(
            np.array(size_input).astype("float") * upsample_factor
        ).astype("int")
    
        # output init
        orient_hist = np.zeros([num_radii, size_output[0], size_output[1], num_theta_bins])
    
        # Loop over all probe positions
        for a0 in range(num_radii):
            t = "Generating histogram " + str(a0)
            # for rx, ry in tqdmnd(
            #         *bragg_peaks.shape, desc=t,unit=" probe positions", disable=not progress_bar
            #     ):
            for rx, ry in tqdmnd(
                *size_input, desc=t, unit=" probe positions", disable=not progress_bar
            ):
                x = (rx + 0.5) * upsample_factor - 0.5
                y = (ry + 0.5) * upsample_factor - 0.5
                x = np.clip(x, 0, size_output[0] - 2)
                y = np.clip(y, 0, size_output[1] - 2)
    
                xF = np.floor(x).astype("int")
                yF = np.floor(y).astype("int")
                dx = x - xF
                dy = y - yF
    
                add_data = False
    
                if orientation_map is None:
                    p_r_invA = self.polar_peaks['r_invA'][rx, ry]
                    p_theta = self.polar_peaks['theta'][rx, ry]
                    
                    if p_r_invA is not None and len(p_r_invA) > 0:
                        # Extract columns: p is shape (N_peaks, 4)
                        # Column 0: y_pixels, Column 1: x_pixels
                        # Column 2: y_invA,   Column 3: x_invA
                        # p_y_invA = p[:, 2]
                        # p_x_invA = p[:, 3]
                        # r2 = p_y_invA**2 + p_x_invA**2
                        r2 = p_r_invA**2
                        sub = np.logical_and(
                            r2 >= radial_ranges_2[a0, 0], 
                            r2 < radial_ranges_2[a0, 1]
                        )
                        if np.any(sub):
                            intensity_data = self.peak_intensities['intensities_sampled_from_dp'][rx, ry]
                            if intensity_data is not None and len(intensity_data) > 0:
                                add_data = True
                                intensity = intensity_data[sub]
                                # Use theta directly from polar_peaks instead of recalculating
                                theta_radians = p_theta[sub]  # Already in radians
                                # Fold to 0-180° range (0 to π)
                                theta_folded = np.mod(theta_radians, np.pi)
                                t = theta_folded / dtheta
                else:
                    if orientation_map.corr[rx, ry, orientation_ind] > 0:
                        if orientation_separate_bins is False:
                            if orientation_flip_sign:
                                t = (
                                    np.array(
                                        [
                                            (
                                                -orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                - orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles
                                )
                            else:
                                t = (
                                    np.array(
                                        [
                                            (
                                                orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                + orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles
                                )
                            intensity = (
                                np.ones(num_angles)
                                * orientation_map.corr[rx, ry, orientation_ind]
                            )
                            add_data = True
                        else:
                            if orientation_flip_sign:
                                t = (
                                    np.array(
                                        [
                                            (
                                                -orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                - orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles[a0]
                                )
                            else:
                                t = (
                                    np.array(
                                        [
                                            (
                                                orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                + orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles[a0]
                                )
                            intensity = orientation_map.corr[rx, ry, orientation_ind]
                            add_data = True
    
                if add_data:
                    tF = np.floor(t).astype("int")
                    dt = t - tF
    
                    orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(1 - dx) * (1 - dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(1 - dx) * (1 - dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
                    orient_hist[a0, xF + 1, yF, :] = orient_hist[
                        a0, xF + 1, yF, :
                    ] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(dx) * (1 - dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF + 1, yF, :] = orient_hist[
                        a0, xF + 1, yF, :
                    ] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(dx) * (1 - dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
                    orient_hist[a0, xF, yF + 1, :] = orient_hist[
                        a0, xF, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(1 - dx) * (dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF, yF + 1, :] = orient_hist[
                        a0, xF, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(1 - dx) * (dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
                    orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                        a0, xF + 1, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(dx) * (dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                        a0, xF + 1, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(dx) * (dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
        # smoothing / interpolation
        if (sigma_x is not None) or (sigma_y is not None) or (sigma_theta is not None):
            if num_radii > 1:
                print("Interpolating orientation matrices ...", end="")
            else:
                print("Interpolating orientation matrix ...", end="")
            if sigma_x is not None and sigma_x > 0:
                orient_hist = gaussian_filter1d(
                    orient_hist,
                    sigma_x * upsample_factor,
                    mode="nearest",
                    axis=1,
                    truncate=3.0,
                )
            if sigma_y is not None and sigma_y > 0:
                orient_hist = gaussian_filter1d(
                    orient_hist,
                    sigma_y * upsample_factor,
                    mode="nearest",
                    axis=2,
                    truncate=3.0,
                )
            if sigma_theta is not None and sigma_theta > 0:
                orient_hist = gaussian_filter1d(
                    orient_hist, sigma_theta / dtheta_deg, mode="wrap", axis=3, truncate=2.0
                )
            print(" done.")
    
        # normalization
        if normalize_intensity_stack is True:
            orient_hist = orient_hist / np.max(orient_hist)
        elif normalize_intensity_image is True:
            for a0 in range(num_radii):
                orient_hist[a0, :, :, :] = orient_hist[a0, :, :, :] / np.max(
                    orient_hist[a0, :, :, :]
                )
    
        return orient_hist

    def plot_interactive_peak_map(self, radial_range=None, vmax_cartesian=7):
        """
        Interactive plot using sliders - more reliable than clicking in Jupyter.
        """
        from ipywidgets import interact, IntSlider
        
        Ry, Rx = self.peak_coordinates_cartesian.shape
        
        # Create intensity map
        intensity_map = np.zeros((Ry, Rx))
        for i in range(Ry):
            for j in range(Rx):
                peaks_r_invA = self.polar_peaks['r_invA'][i, j]
                if peaks_r_invA is not None and len(peaks_r_invA) > 0:
                    if radial_range is not None:
                        distances = peaks_r_invA
                        mask = (distances >= radial_range[0]) & (distances < radial_range[1])
                    else:
                        mask = np.ones(len(peaks_r_invA), dtype=bool)
                    
                    intensities = self.peak_intensities['intensities_sampled_from_dp'][i, j]
                    if intensities is not None and len(intensities) > 0:
                        intensity_map[i, j] = np.sum(intensities[mask])
        
        def show_pattern(ry, rx):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot intensity map with current position marked
            ax1.imshow(intensity_map, cmap='viridis', origin='lower', interpolation='nearest')
            ax1.plot(rx, ry, 'r+', markersize=15, markeredgewidth=2)
            ax1.set_title('Intensity Map')
            ax1.set_xlabel('Rx')
            ax1.set_ylabel('Ry')
            
            # Plot diffraction pattern
            ax2.imshow(self.resized_cartesian_data[ry, rx], cmap='gray', vmax=vmax_cartesian)
            
            peaks_r_invA = self.peak_coordinates_cartesian['r_invA'][ry, rx]
            peaks_y_invA = self.peak_coordinates_cartesian['y_invA'][ry, rx]
            peaks_x_invA = self.peak_coordinates_cartesian['x_invA'][ry, rx]
            if peaks is not None and len(peaks) > 0:
                if radial_range is not None:
                    distances = peaks_r_invA
                    mask = (distances >= radial_range[0]) & (distances < radial_range[1])
                    
                    ax2.scatter(peaks_y_invA, peaks_x_invA, c='gray', s=30, alpha=0.5,
                               edgecolors='white', linewidths=0.5, label='Other peaks')
                    
                    if np.any(mask):
                        ax2.scatter(peaks_y_invA[mask], peaks_x_invA[mask], c='red', s=30, alpha=0.8,
                                   edgecolors='white', linewidths=0.5, 
                                   label=f'Selected ({np.sum(mask)} peaks)')
                        ax2.legend(fontsize=8, loc='upper right')
                else:
                    ax2.scatter(peaks_y_invA, peaks_x_invA, c='red', s=30, alpha=0.8,
                               edgecolors='white', linewidths=0.5)
            
            if hasattr(self, 'image_centers') and self.image_centers is not None:
                ax2.scatter(self.image_centers[ry, rx, 1], self.image_centers[ry, rx, 0],
                           c='cyan', s=500, marker='x', linewidths=2, label='Center')
            
            title_str = f'Position (Ry={ry}, Rx={rx})'
            if radial_range is not None:
                title_str += f'\nRange: {radial_range[0]:.2f}-{radial_range[1]:.2f} 1/Å'
            ax2.set_title(title_str)
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        interact(show_pattern,
                 ry=IntSlider(min=0, max=Ry-1, step=1, value=Ry//2, description='Ry:', continuous_update=False),
                 rx=IntSlider(min=0, max=Rx-1, step=1, value=Rx//2, description='Rx:', continuous_update=False))


    def plot_peak_count_map(self, q_ranges, figsize_per_map=(5, 4), cmap='viridis'):
        """
        Plot 2D maps showing the number of peaks in specified q-ranges.
        
        Parameters:
        -----------
        q_ranges : list of tuples or single tuple
            Either a single (q_min, q_max) tuple or a list of tuples for multiple ranges.
            Example: (2.8, 3.2) or [(0.3, 0.7), (2.8, 3.2), (5.0, 5.4)]
        figsize_per_map : tuple
            Size of each subplot (width, height)
        cmap : str
            Colormap to use
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        count_maps : list of ndarrays
            The count maps for each q-range
        """
        # Handle single range or list of ranges
        if isinstance(q_ranges, tuple):
            q_ranges = [q_ranges]
        
        Ry, Rx = self.peak_coordinates_cartesian.shape
        n_ranges = len(q_ranges)
        
        # Create figure
        n_cols = min(3, n_ranges)  # Max 3 columns
        n_rows = int(np.ceil(n_ranges / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 figsize=(figsize_per_map[0]*n_cols, figsize_per_map[1]*n_rows))
        
        # Handle single subplot case
        if n_ranges == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        count_maps = []
        
        for idx, (q_min, q_max) in enumerate(q_ranges):
            # Create count map
            count_map = np.zeros((Ry, Rx))
            
            for i in range(Ry):
                for j in range(Rx):
                    peaks_r_invA = self.polar_peaks['r_invA'][i, j]
                    if peaks_r_invA is not None and len(peaks_r_invA) > 0:
                        # Get radial distances in 1/Å
                        distances = peaks_r_invA
                        # Count peaks in range
                        mask = (distances >= q_min) & (distances < q_max)
                        count_map[i, j] = np.sum(mask)
            
            count_maps.append(count_map)
            
            # Plot
            im = axes[idx].imshow(count_map, cmap=cmap, origin='lower', interpolation='nearest')
            axes[idx].set_title(f'Peak Count\n{q_min:.2f} - {q_max:.2f} 1/Å', fontsize=12)
            axes[idx].set_xlabel('Rx')
            axes[idx].set_ylabel('Ry')
            
            # Colorbar with integer ticks
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Number of Peaks', fontsize=10)
            
            # Set colorbar ticks to integers
            max_count = int(np.max(count_map))
            if max_count > 0:
                tick_spacing = max(1, max_count // 5)  # About 5 ticks
                ticks = np.arange(0, max_count + 1, tick_spacing)
                cbar.set_ticks(ticks)
            
            # Print statistics
            total_peaks = int(np.sum(count_map))
            positions_with_peaks = np.sum(count_map > 0)
            print(f"Range {q_min:.2f}-{q_max:.2f} 1/Å:")
            print(f"  Total peaks: {total_peaks}")
            print(f"  Positions with peaks: {positions_with_peaks}/{Ry*Rx}")
            print(f"  Max peaks at one position: {max_count}")
            print(f"  Mean peaks per position: {np.mean(count_map):.2f}")
            print()
        
        # Hide unused subplots
        for idx in range(n_ranges, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes, count_maps
