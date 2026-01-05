# from collections.abc import Sequence
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, label, sum as label_sum, center_of_mass
from tqdm import tqdm
import torch
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem, dataset4dstem_polar_transform
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.cnn2d import MultiChannelCNN2d
from quantem.core.datastructures import Vector
from quantem.core.visualization import show_2d
from quantem.core.utils.polar import polar_transform_vector, cartesian_transform_vector
from quantem.diffraction.peak_detection import detect_blobs, find_central_beam_from_peaks
from quantem.diffraction.polymer_analytical_functions import add_to_histogram_bilinear
from quantem.core.utils.utils import parse_reciprocal_units, sample_average_from_image
from quantem.diffraction.polar_new import RDF_new
from emdfile import tqdmnd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
import ipywidgets as widgets
from ipywidgets import interact, IntSlider

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
        # self.resized_cartesian_data = None
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

    @property
    def final_shape(self) -> str:
        return self._final_shape

    @final_shape.setter
    def final_shape(self, final_shape):
        self._final_shape = final_shape
        
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
        # self.resize_data(device=self.device)

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

    def resize_images(self, images, device: str = "cuda:0", initial_chunk_size: int = 100, show_progress=False):
        # Handle Dataset objects - extract array
        if hasattr(images, 'array'):
            images = images.array
        elif isinstance(images, Dataset3d):
            # If it's a Dataset3d, get the underlying array
            images = np.array([images[i].array for i in range(images.shape[0])])
        
        N, Qy, Qx = images.shape
        scale_factor = (self._final_shape[0] * self._final_shape[1]) / (Qy * Qx)
        resized_data = np.zeros((N, self._final_shape[0], self._final_shape[1]))
        
        chunk_size = initial_chunk_size
        i = 0
        
        with tqdm(total=N, desc='images', disable=not show_progress) as pbar:
            while i < N:
                try:
                    # Determine the end index for this chunk
                    end_idx = min(i + chunk_size, N)
                    chunk = images[i:end_idx]
                    
                    # Process chunk on GPU
                    inp = torch.tensor(chunk, dtype=torch.float32).to(device)
                    inp = torch.nn.functional.interpolate(
                        inp.unsqueeze(1),  # Add channel dimension
                        size=self._final_shape, 
                        mode='bilinear', 
                        align_corners=False
                    ) * scale_factor
                    
                    resized_data[i:end_idx, :, :] = inp.squeeze(1).detach().cpu().numpy()
                    
                    # Clear GPU cache
                    del inp
                    if 'cuda' in device:
                        torch.cuda.empty_cache()
                    
                    # Update progress and move to next chunk
                    pbar.update(end_idx - i)
                    i = end_idx
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        # Clear cache and reduce chunk size
                        if 'cuda' in device:
                            torch.cuda.empty_cache()
                        
                        chunk_size = max(1, chunk_size // 2)
                        print(f"\nGPU OOM! Reducing chunk size to {chunk_size}")
                        
                        if chunk_size == 1:
                            # If even single image fails, fall back to CPU
                            print("Falling back to CPU processing")
                            device = "cpu"
                    else:
                        raise e
        
        return resized_data

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

    def _postprocess_single(self, position_map, intensity_map, sigma=1.0, threshold=0.25, show=False):
        """Process a single 2D image"""
        # Find peaks with subpixel-refinement
        peak_coords, peak_position_signal_intensities, refinement_success = detect_blobs(
            position_map,
            sigma=sigma,  # Sigma for Gaussian smoothing used in processing
            threshold=threshold,  # Threshold for strength of peak position signal to be valid peak
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
        scan_mask: ArrayLike = None,
        n_normalize_samples: int = 1000,
        initial_chunk_size: int = 100,
        sigma_peak_blur: float = 1.0,
        threshold_peak: float = 0.5,
        show_plots=False,
    ):
        Ry, Rx, Qy, Qx = self.dataset_cartesian.shape
        total_positions = Ry * Rx
        
        # ============================================
        # Handle scan_mask
        # ============================================
        if scan_mask is None:
            scan_mask = np.ones((Ry, Rx), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
            if scan_mask.shape != (Ry, Rx):
                raise ValueError(f"scan_mask shape {scan_mask.shape} must match scan shape ({Ry}, {Rx})")
        
        # Get list of valid positions
        valid_positions = np.argwhere(scan_mask)  # Returns array of (ry, rx) pairs
        n_valid = len(valid_positions)
        
        peaks = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["y_pixels", "x_pixels", "y_invA", "x_invA"],
            name="peaks_vector",
            units=["Pixels", "Pixels", "1/Å", "1/Å"],
        )
        intensities = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["intensities", "intensities_sampled_from_dp"],
            name="intensities_vector",
            units=["Normalized", "Normalized"],
        )
        
        # ============================================
        # 1. Compute normalization parameters (only from valid positions)
        # ============================================
        n_normalize_samples = min(n_normalize_samples, n_valid)
        sample_indices = np.random.choice(n_valid, size=n_normalize_samples, replace=False)
        
        stats_patterns = np.array([
            self.dataset_cartesian[ry, rx].array 
            for ry, rx in valid_positions[sample_indices]
        ])
        
        stats_patterns_resized = self.resize_images(stats_patterns, device=device)
        median, iqr = self.compute_parameters(stats_patterns_resized)
    
        # ============================================
        # 2. Process only valid positions with chunking
        # ============================================
        chunk_size = initial_chunk_size
        pos_idx = 0
        
        with tqdm(total=n_valid, desc="Processing patterns") as pbar:
            while pos_idx < n_valid:
                try:
                    # ----------------------------------------
                    # 2a. Determine chunk boundaries
                    # ----------------------------------------
                    end_pos_idx = min(pos_idx + chunk_size, n_valid)
                    actual_chunk_size = end_pos_idx - pos_idx
                    
                    # ----------------------------------------
                    # 2b. Extract chunk data (only valid positions)
                    # ----------------------------------------
                    chunk_data = []
                    chunk_positions = []
                    
                    for i in range(pos_idx, end_pos_idx):
                        ry, rx = valid_positions[i]
                        chunk_data.append(self.dataset_cartesian[ry, rx].array)
                        chunk_positions.append((ry, rx))
                    
                    chunk_array = np.array(chunk_data)
                    
                    # ----------------------------------------
                    # 2c. Resize chunk
                    # ----------------------------------------
                    chunk_resized = self.resize_images(
                        chunk_array, 
                        device=device, 
                        initial_chunk_size=actual_chunk_size
                    )
                    
                    # ----------------------------------------
                    # 2d. Normalize and run model
                    # ----------------------------------------
                    ins = torch.tensor(chunk_resized, dtype=torch.float32).to(device)
                    dps_norm = self.normalize_data(ins, median, iqr)
                    ins_batch = dps_norm[:, None, ...]
                    
                    outs = self.model(ins_batch).detach().cpu().numpy()
                    
                    # ----------------------------------------
                    # 2e. Post-process each pattern in chunk
                    # ----------------------------------------
                    for k in range(outs.shape[0]):
                        ry, rx = chunk_positions[k]
                        
                        peak_coords, peak_intensities = self._postprocess_single(
                            outs[k, 0], 
                            outs[k, 1],
                            show=show_plots,
                            sigma=sigma_peak_blur,
                            threshold=threshold_peak,
                        )
                        
                        if len(peak_coords) > 0:
                            peak_intensity_averages = sample_average_from_image(
                                ins_batch[k].squeeze().detach().cpu().numpy(), 
                                peak_coords
                            )
                            peak_intensities_data = np.column_stack([
                                peak_intensities,
                                peak_intensity_averages,
                            ])
                            
                            peak_coords_original = peak_coords * (
                                self.dataset_cartesian.shape[2] / self.final_shape[0]
                            )
                            
                            peak_data = np.column_stack([
                                peak_coords_original,
                                peak_coords_original * self.pixels_to_inv_A()
                            ])
                            
                            peaks.set_data(peak_data, ry, rx)
                            intensities.set_data(peak_intensities_data, ry, rx)
                    
                    # ----------------------------------------
                    # 2f. Memory cleanup
                    # ----------------------------------------
                    del ins, dps_norm, ins_batch, outs, chunk_array, chunk_resized
                    if 'cuda' in device:
                        torch.cuda.empty_cache()
                    
                    # ----------------------------------------
                    # 2g. Update progress and move to next chunk
                    # ----------------------------------------
                    pbar.update(actual_chunk_size)
                    pos_idx = end_pos_idx
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if 'cuda' in device:
                            torch.cuda.empty_cache()
                        
                        chunk_size = max(1, chunk_size // 2)
                        print(f"\nGPU OOM! Reducing chunk size to {chunk_size}")
                        
                        if chunk_size == 1:
                            print("Falling back to CPU processing")
                            device = "cpu"
                    else:
                        raise e
        
        print('Done!')
        self.peak_coordinates_cartesian = peaks
        self.peak_intensities = intensities

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

    def process_polar(self, scan_mask: ArrayLike = None):
        """ Find center of image through brightest peak, return polar transform of data and peaks"""
        self.image_centers = self.find_central_beams_4d(scan_mask=scan_mask)
        self.polar_peaks = self.polar_transform_peaks(cartesian_peaks=self.peak_coordinates_cartesian, centers=self.image_centers, scan_mask=scan_mask)
        self.polar_data = self.polar_transform_4d(self.dataset_cartesian, centers=self.image_centers, scan_mask=scan_mask)
        # self.image_centers = self.find_central_beams_4d()
        # self.polar_peaks = self.polar_transform_peaks(cartesian_peaks=self.peak_coordinates_cartesian, centers=self.image_centers)
        # self.polar_data = self.polar_transform_4d(self.dataset_cartesian, centers=self.image_centers)

        # self.polar_data = dataset4dstem_polar_transform(self.dataset_cartesian, origin_row=self.image_centers[0], origin_col=self.image_centers[1])
        # self.polar_data = self.dataset_cartesian.polar_transform(
            # origin_row=self.image_centers[1],
            # origin_col=self.image_centers[0],
            # two_fold_rotation_symmetry=True,
            # name: str | None = None,
            # signal_units: str | None = None,
        # )
        
        # rdf_new = RDF_new.from_data(data=self.dataset_cartesian, origin_row=self.image_centers[1], origin_col=self.image_centers[0])
        # self.polar_data = self.polar_transform_4d(self.resized_cartesian_data, centers=self.image_centers)

    def find_central_beams_4d(self, scan_mask: ArrayLike = None, intensity_threshold=0.3, distance_weight=0.5, sampling_radius=2, debug=False, use_tqdm=True):
        """
        Fast central beam finding for entire 4D dataset.
        
        Parameters:
        -----------
        scan_mask : ArrayLike, optional
            Boolean mask (Ry, Rx) indicating which positions to process
        use_tqdm : bool
            Show progress bar
        
        Returns:
        --------
        centers : ndarray, shape (2, scan_y, scan_x)
            Center coordinates (y, x) for each scan position
        """        
        scan_y, scan_x, det_y, det_x = self.dataset_cartesian.shape
        centers = np.zeros((2, scan_y, scan_x))
        
        # Handle scan_mask
        if scan_mask is None:
            scan_mask = np.ones((scan_y, scan_x), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
        
        iterator = tqdm(range(scan_y), disable=not use_tqdm, desc="Finding centers")
        
        for i in iterator:
            for j in range(scan_x):
                if not scan_mask[i, j]:
                    continue
                if self.peak_coordinates_cartesian[i, j] is None:
                    print(f"None at i={i}, j={j}")
                centers[:, i, j] = find_central_beam_from_peaks(
                    peak_coords=self.peak_coordinates_cartesian[i, j],
                    peak_intensities=None,
                    image_shape=self._final_shape,
                    intensity_threshold=intensity_threshold,
                    distance_weight=distance_weight,
                    debug=debug,
                    image=self.dataset_cartesian[i, j].array.squeeze(),
                    sampling_radius=sampling_radius
                )
        return centers
    
    def polar_transform_peaks(self, cartesian_peaks, centers, scan_mask: ArrayLike = None, use_tqdm: bool=True):
        # Get sampling conversion factor
        sampling_conversion_factor = self.pixels_to_inv_A()
        polar_peaks = polar_transform_vector(
            cartesian_vector=cartesian_peaks,
            centers=centers,
            scan_mask=scan_mask,
            use_tqdm=use_tqdm,
            sampling_conversion_factor=sampling_conversion_factor,
            two_fold_symmetry=True,
        )
        return polar_peaks
    
    def polar_transform_4d(self, data, centers, scan_mask: ArrayLike = None, num_r=None, num_theta=360, use_tqdm: bool=True):
        """
        Perform polar transform on the last two axes of a 4D array.
        
        Parameters:
        -----------
        data : ndarray, shape (N, M, H, W)
            4D input array where H, W are the axes to transform
        centers : ndarray, shape (2, N, M)
            Center of each diffraction pattern (usually determined by central beam)
        scan_mask : ArrayLike, optional
            Boolean mask (N, M) indicating which positions to process
        num_r : int, optional
            Number of radial bins. If None, uses max radius across all patterns
        num_theta : int, optional
            Number of angular bins (default: 360)
        
        Returns:
        --------
        polar_data : Vector
            Vector with shape (N, M) containing polar-transformed data.
        
        Notes:
        ------
        Also sets the following attributes on self:
        - self.max_radius_pixels : maximum radius in pixels
        - self.max_radius_invA : maximum radius in 1/Å
        - self.num_radial_bins : number of radial bins
        - self.num_annular_bins : number of angular bins
        """
        N, M, H, W = data.shape
        
        # Handle scan_mask
        if scan_mask is None:
            scan_mask = np.ones((N, M), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
        
        # Calculate consistent max_radius across entire dataset (only from masked positions)
        valid_centers_0 = centers[0][scan_mask]
        valid_centers_1 = centers[1][scan_mask]
        dist_to_origin_sq = (valid_centers_0**2 + valid_centers_1**2).min()
        dist_to_corner_sq = ((H-1 - valid_centers_0)**2 + (W-1 - valid_centers_1)**2).max()
        max_radius_pixels = np.sqrt(max(dist_to_origin_sq, dist_to_corner_sq))
        
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
        # Create meshgrid (always work in pixels for interpolation)
        r_grid, theta_grid = np.meshgrid(r_pixels, theta, indexing='ij')
        r_invA_grid = r_grid * self.pixels_to_inv_A()
        polar_intensity = np.zeros((N, M, num_r, num_theta), dtype=np.float32)
        # Create Vector to store polar data with coordinates
        # polar_data = Vector.from_shape(
        #     shape=(N, M),
        #     fields=["r_pixels", "theta", "r_invA", "intensity"],
        #     units=["Pixels", "Radians", "1/Å", "Intensity"],
        #     name="polar_transformed_data"
        # )
        
        # Transform each 2D slice (only masked positions)
        iterator = tqdm(range(N), disable=not use_tqdm, desc="Transforming data")
        for i in iterator:
            for j in range(M):
                if not scan_mask[i, j]:
                    continue
                    
                center_y, center_x = centers[:, i, j]
                
                # Convert polar to Cartesian coordinates
                y_coords = center_y + r_grid * np.sin(theta_grid)
                x_coords = center_x + r_grid * np.cos(theta_grid)
                
                # Use map_coordinates for interpolation
                polar_intensity[i, j] = map_coordinates(
                    data[i, j].array, 
                    [y_coords, x_coords], 
                    order=1,
                    mode='constant',
                    cval=0.0
                )
                
                # Flatten the 2D polar image to 1D for storage in Vector
                    # Create data array: [r_pixels, theta, r_invA, intensity]
                # polar_data_array = np.stack([
                #     r_grid,
                #     theta_grid,
                #     r_invA_grid,
                #     intensity_grid
                # ], axis=-1)
                
                # Store in Vector
                # polar_data.set_data(polar_data_array, i, j)
        polar_data = {
            "r_pixels": r_grid,
            "theta": theta_grid,
            "r_invA": r_invA_grid,
            "intensity": polar_intensity,
        }
        
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
            Ry, Rx = self.dataset_cartesian.shape[:2]
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
            print(self.polar_data["intensity"][ind_y, ind_x].shape)
            im = ax.matshow(self.polar_data["intensity"][ind_y, ind_x], cmap='turbo', vmax=vmax_polar)
            if row == 0:
                ax.set_title(col_titles[0], fontsize=10, pad=10)
            ax.text(0.05, 0.95, f'({ind_y},{ind_x})', transform=ax.transAxes, 
                    fontsize=8, va='top', ha='left', color='white', 
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            ax.set_axis_off()
            
            # 2. Polar Transform with Peaks
            ax = axes[row, col_offset + 1]
            ax.matshow(self.polar_data["intensity"][ind_y, ind_x], cmap='turbo', vmax=vmax_polar)
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
            ax.matshow(self.dataset_cartesian[ind_y, ind_x].array, cmap="gray", vmax=vmax_cartesian)
            # ax.matshow(self.resized_cartesian_data[ind_y, ind_x], cmap="gray", vmax=vmax_cartesian)
            if has_peaks:
                ax.scatter(self.peak_coordinates_cartesian[ind_y, ind_x][:, 1], 
                          self.peak_coordinates_cartesian[ind_y, ind_x][:, 0], 
                          c='red', s=15, alpha=0.8, edgecolors='white', linewidths=0.5)
            ax.scatter(self.image_centers[1, ind_y, ind_x], 
                      self.image_centers[0, ind_y, ind_x], 
                      c='red', s=500, marker='x', linewidths=2)
            if row == 0:
                ax.set_title(col_titles[2], fontsize=10, pad=10)
            ax.set_axis_off()
            
            # 4. Original Cartesian
            ax = axes[row, col_offset + 3]
            im = ax.matshow(self.dataset_cartesian[ind_y, ind_x].array, cmap="gray", vmax=vmax_cartesian)
            # im = ax.matshow(self.resized_cartesian_data[ind_y, ind_x], cmap="gray", vmax=vmax_cartesian)
            if row == 0:
                ax.set_title(col_titles[3], fontsize=10, pad=10)
            ax.set_axis_off()
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # # 5. Normalized Cartesian
            # ax = axes[row, col_offset + 4]
            # im = ax.matshow(self.normalized_dps_array[ind_y, ind_x], cmap="gray")
            # if row == 0:
            #     ax.set_title(col_titles[4], fontsize=10, pad=10)
            # ax.set_axis_off()
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        total_plots = n_images
        for idx in range(total_plots, n_rows * images_per_row):
            row = idx // images_per_row
            col_offset = (idx % images_per_row) * n_cols
            for col in range(n_cols):
                axes[row, col_offset + col].set_visible(False)
        
        fig.tight_layout()
        return fig, axes

    def estimate_peak_windows(
        self,
        num_bins=200,
        q_min=None,
        q_max=None,
        n_peaks=5,
        height_percentile=10,
        prominence_factor=0.1,
        width_factor=2.0,
        min_width=0.05,
        smoothing_sigma=2.0,
    ):
        """
        Automatically detect the top N most prominent peaks and estimate their windows.
        
        Parameters
        ----------
        num_bins : int
            Number of radial bins
        q_min : float, optional
            Minimum q value for binning
        q_max : float, optional
            Maximum q value for binning
        n_peaks : int
            Number of top peaks to detect
        height_percentile : float
            Percentile threshold for peak height (peaks below this are ignored)
        prominence_factor : float
            Factor of max intensity for minimum peak prominence
        width_factor : float
            Multiplier for estimating peak window width from FWHM
        min_width : float
            Minimum window width in 1/Å
        smoothing_sigma : float
            Gaussian smoothing sigma for noise reduction before peak detection
            
        Returns
        -------
        peak_centers : array
            q-values for peak centers (shape: n_peaks)
        peak_windows : array
            Window boundaries for each peak (shape: n_peaks, 2)
            Each row is [q_min, q_max] for that peak
        peak_info : dict
            Additional information about detected peaks including:
            - 'heights': peak heights
            - 'prominences': peak prominences
            - 'widths': estimated peak widths (FWHM)
        """
        
        # Get radial intensity profile
        all_r = self.polar_peaks['r_invA'].flatten()
        all_intensity = self.peak_intensities['intensities_sampled_from_dp'].flatten()
        
        if q_min is None:
            q_min = 0
        if q_max is None:
            q_max = np.max(all_r)
        
        r_bins = np.linspace(q_min, q_max, num_bins + 1)
        intensity_sum, _ = np.histogram(all_r, bins=r_bins, weights=all_intensity)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        # Smooth the data to reduce noise
        if smoothing_sigma > 0:
            intensity_smooth = gaussian_filter1d(intensity_sum, smoothing_sigma)
        else:
            intensity_smooth = intensity_sum
        
        # Calculate thresholds
        height_threshold = np.percentile(intensity_smooth, height_percentile)
        prominence_threshold = prominence_factor * np.max(intensity_smooth)
        
        # Find peaks
        peaks_indices, properties = find_peaks(
            intensity_smooth,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=int(min_width / (r_centers[1] - r_centers[0]))  # Minimum separation
        )
        
        if len(peaks_indices) == 0:
            print("No peaks found with current parameters!")
            return np.array([]), np.array([]).reshape(0, 2), {}
        
        # Sort by prominence and take top N
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1][:n_peaks]
        top_peak_indices = peaks_indices[sorted_indices]
        top_peak_indices = np.sort(top_peak_indices)  # Re-sort by position
        
        # Get peak centers
        peak_centers = r_centers[top_peak_indices]
        
        # Calculate peak widths (FWHM)
        widths_data = peak_widths(intensity_smooth, top_peak_indices, rel_height=0.5)
        fwhm_bins = widths_data[0]  # Width in bins
        fwhm_invA = fwhm_bins * (r_centers[1] - r_centers[0])  # Convert to 1/Å
        
        # Estimate windows: center ± width_factor * FWHM/2, with minimum width
        half_widths = np.maximum(width_factor * fwhm_invA / 2, min_width / 2)
        peak_windows = np.column_stack([
            peak_centers - half_widths,
            peak_centers + half_widths
        ])
        
        # Clip windows to data range
        peak_windows[:, 0] = np.maximum(peak_windows[:, 0], q_min)
        peak_windows[:, 1] = np.minimum(peak_windows[:, 1], q_max)
        
        # Collect additional info
        peak_info = {
            'heights': intensity_smooth[top_peak_indices],
            'prominences': prominences[sorted_indices],
            'widths_fwhm': fwhm_invA,
            'intensity_profile': intensity_smooth,
            'r_centers': r_centers,
        }
        
        # Print summary
        print(f"Detected {len(peak_centers)} peaks:")
        for i, (center, window, height, prom, width) in enumerate(zip(
            peak_centers, peak_windows, peak_info['heights'], 
            peak_info['prominences'], peak_info['widths_fwhm']
        )):
            print(f"  Peak {i+1}: center={center:.3f} 1/Å, "
                  f"window=[{window[0]:.3f}, {window[1]:.3f}] 1/Å, "
                  f"height={height:.1f}, prominence={prom:.1f}, FWHM={width:.3f} 1/Å")
        
        return peak_centers, peak_windows, peak_info

    def peak_radial_intensity_plot(
        self,
        num_bins=200,
        q_min=None,
        q_max=None,
        ROI_xs=None,
        ROI_ys=None,
        peak_centers=None,
        peak_windows=None,
        vlines=None,
        vline_colors=None,
        vline_labels=None,
        window_alpha=0.3,
        window_color='red',
        fill_alpha=0.5,
        fill_color=None,
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
        peak_centers : array, optional
            1D array of peak center positions to mark with vertical lines
        peak_windows : array, optional
            2D array (N, 2) of [q_min, q_max] for each peak window to highlight
        vlines : list of lists/arrays, optional
            Additional vertical lines to plot. Each element is a list/array of x-positions.
        vline_colors : list of colors, optional
            Colors for each group of vertical lines
        vline_labels : list of str, optional
            Labels for each group of vertical lines (for legend)
        window_alpha : float
            Transparency for peak window background highlighting (0-1)
        window_color : str or color
            Color for peak window background highlighting
        fill_alpha : float
            Transparency for filled area under curve within windows (0-1)
        fill_color : str or color, optional
            Color for filled area under curve. If None, uses window_color
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
        
        # Use window_color for fill if not specified
        if fill_color is None:
            fill_color = window_color
    
        if plot:
            # Create line plot
            fig, ax = plt.subplots()
            ax.plot(r_centers, intensity_sum, linewidth=2, label='Intensity', color='black')
            ax.set_xlabel('Radial Distance (1/Å)', fontsize=12)
            ax.set_ylabel('Integrated Intensity', fontsize=12)
            ax.set_title('Radial Intensity Profile (All Patterns)', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add peak windows as filled regions and fill under curve
            if peak_windows is not None:
                peak_windows = np.atleast_2d(peak_windows)
                for i, (q_min_win, q_max_win) in enumerate(peak_windows):
                    # Background window highlight
                    ax.axvspan(q_min_win, q_max_win, alpha=window_alpha, 
                              color=window_color, zorder=0,
                              label='Peak windows' if i == 0 else None)
                    
                    # Fill under the curve within this window
                    # Find indices within the window
                    mask = (r_centers >= q_min_win) & (r_centers <= q_max_win)
                    if np.any(mask):
                        r_window = r_centers[mask]
                        intensity_window = intensity_sum[mask]
                        ax.fill_between(r_window, 0, intensity_window, 
                                       alpha=fill_alpha, color=fill_color,
                                       label='Peak intensity' if i == 0 else None,
                                       zorder=1)
            
            # Add peak centers as vertical lines
            if peak_centers is not None:
                peak_centers = np.atleast_1d(peak_centers)
                for i, center in enumerate(peak_centers):
                    ax.axvline(center, color=window_color, linestyle='-', 
                              linewidth=2, alpha=0.8,
                              label='Peak centers' if i == 0 else None, zorder=2)
            
            # Add additional vertical lines if provided
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
                                  linewidth=1.5, alpha=0.7, label=line_label, zorder=2)
            
                # Add legend
                ax.legend()
            elif peak_centers is not None or peak_windows is not None:
                # Add legend for peak markers if present
                ax.legend()
            
            fig.tight_layout()
            plt.show()
        
        if return_data:
            return r_centers, intensity_sum

    def peak_radial_count_plot(
        self,
        num_bins=200,
        q_min=None,
        q_max=None,
        ROI_xs=None,
        ROI_ys=None,
        peak_centers=None,
        peak_windows=None,
        vlines=None,
        vline_colors=None,
        vline_labels=None,
        window_alpha=0.3,
        window_color='red',
        fill_alpha=0.5,
        fill_color=None,
        plot=True,
        return_data=False,
    ):
        """
        Create radial peak count line plot summarizing polar peaks.
    
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
        peak_centers : array, optional
            1D array of peak center positions to mark with vertical lines
        peak_windows : array, optional
            2D array (N, 2) of [q_min, q_max] for each peak window to highlight
        vlines : list of lists/arrays, optional
            Additional vertical lines to plot. Each element is a list/array of x-positions.
        vline_colors : list of colors, optional
            Colors for each group of vertical lines
        vline_labels : list of str, optional
            Labels for each group of vertical lines (for legend)
        window_alpha : float
            Transparency for peak window background highlighting (0-1)
        window_color : str or color
            Color for peak window background highlighting
        fill_alpha : float
            Transparency for filled area under curve within windows (0-1)
        fill_color : str or color, optional
            Color for filled area under curve. If None, uses window_color
        plot : bool
            Whether to display the plot
        return_data : bool
            Whether to return the binned data
            
        Returns
        -------
        r_centers : array (optional)
            Radial bin centers
        peak_counts : array (optional)
            Number of peaks per bin
        """
        all_r = self.polar_peaks['r_invA'].flatten()
        
        if q_min is None:
            q_min = 0
        if q_max is None:
            q_max = np.max(all_r)
        r_bins = np.linspace(q_min, q_max, num_bins + 1)
    
        # Histogram the data - counts only, no weights
        peak_counts, _ = np.histogram(all_r, bins=r_bins)
    
        # Bin centers
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        # Use window_color for fill if not specified
        if fill_color is None:
            fill_color = window_color
    
        if plot:
            # Create line plot
            fig, ax = plt.subplots()
            ax.plot(r_centers, peak_counts, linewidth=2, label='Peak Count', color='black')
            ax.set_xlabel('Radial Distance (1/Å)', fontsize=12)
            ax.set_ylabel('Number of Peaks', fontsize=12)
            ax.set_title('Radial Peak Count Profile (All Patterns)', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add peak windows as filled regions and fill under curve
            if peak_windows is not None:
                peak_windows = np.atleast_2d(peak_windows)
                for i, (q_min_win, q_max_win) in enumerate(peak_windows):
                    # Background window highlight
                    ax.axvspan(q_min_win, q_max_win, alpha=window_alpha, 
                              color=window_color, zorder=0,
                              label='Peak windows' if i == 0 else None)
                    
                    # Fill under the curve within this window
                    # Find indices within the window
                    mask = (r_centers >= q_min_win) & (r_centers <= q_max_win)
                    if np.any(mask):
                        r_window = r_centers[mask]
                        counts_window = peak_counts[mask]
                        ax.fill_between(r_window, 0, counts_window, 
                                       alpha=fill_alpha, color=fill_color,
                                       label='Peak counts' if i == 0 else None,
                                       zorder=1)
            
            # Add peak centers as vertical lines
            if peak_centers is not None:
                peak_centers = np.atleast_1d(peak_centers)
                for i, center in enumerate(peak_centers):
                    ax.axvline(center, color=window_color, linestyle='-', 
                              linewidth=2, alpha=0.8,
                              label='Peak centers' if i == 0 else None, zorder=2)
            
            # Add additional vertical lines if provided
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
                                  linewidth=1.5, alpha=0.7, label=line_label, zorder=2)
            
                # Add legend
                ax.legend()
            elif peak_centers is not None or peak_windows is not None:
                # Add legend for peak markers if present
                ax.legend()
            
            fig.tight_layout()
            plt.show()
        
        if return_data:
            return r_centers, peak_counts

    # def make_orientation_histogram(
    #     self,
    #     radial_ranges: np.ndarray = None,
    #     orientation_map=None,
    #     orientation_ind: int = 0,
    #     orientation_growth_angles: np.array = 0.0,
    #     orientation_separate_bins: bool = False,
    #     orientation_flip_sign: bool = False,
    #     upsample_factor=4.0,
    #     theta_step_deg=1.0,
    #     sigma_x=1.0,
    #     sigma_y=1.0,
    #     sigma_theta=3.0,
    #     normalize_intensity_image: bool = False,
    #     normalize_intensity_stack: bool = True,
    #     progress_bar: bool = True,
    #     r_field: str = "r",
    #     theta_field: str = "theta",
    #     intensity_field: str = "intensities_sampled_from_dp",
    # ):
    #     """
    #     Create an 3D or 4D orientation histogram from a braggpeaks PointListArray
    #     from user-specified radial ranges, or from the Euler angles from a fiber
    #     texture OrientationMap generated by the ACOM module of py4DSTEM.
    
    #     Args:
    #         bragg_peaks (BraggVectors):         bragg_vectors containing centered peak locations.
    #         radial_ranges (np array):           Size (N x 2) array for N radial bins, or (2,) for a single bin.
    #         orientation_map (OrientationMap):   Class containing the Euler angles to generate a flowline map.
    #         orientation_ind (int):              Index of the orientation map (default 0)
    #         orientation_growth_angles (array):  Angles to place into histogram, relative to orientation.
    #         orientation_separate_bins (bool):   whether to place multiple angles into multiple radial bins.
    #         upsample_factor (float):            Upsample factor
    #         theta_step_deg (float):             Step size along annular direction in degrees
    #         sigma_x (float):                    Smoothing in x direction before upsample
    #         sigma_y (float):                    Smoothing in x direction before upsample
    #         sigma_theta (float):                Smoothing in annular direction (units of bins, periodic)
    #         normalize_intensity_image (bool):   Normalize to max peak intensity = 1, per image
    #         normalize_intensity_stack (bool):   Normalize to max peak intensity = 1, all images
    #         progress_bar (bool):                Enable progress bar
    #         r_field (str):                      Name of radial coordinate field (default: "r")
    #         theta_field (str):                  Name of angular coordinate field (default: "theta")
    #         intensity_field (str):              Name of intensity field (default: "intensity")
    
    #     Returns:
    #         orient_hist (array):                4D array containing Bragg peak intensity histogram
    #                                             [radial_bin x_probe y_probe theta]
    #     """
    #     # coordinates
    #     theta = np.arange(0, 180, theta_step_deg) * np.pi / 180.0
    #     dtheta = theta[1] - theta[0]
    #     dtheta_deg = dtheta * 180 / np.pi
    #     num_theta_bins = np.size(theta)
    
    #     if orientation_map is None:
    #         # Input bins
    #         radial_ranges = np.array(radial_ranges)
    #         if radial_ranges.ndim == 1:
    #             radial_ranges = radial_ranges[None, :]
    #         radial_ranges_2 = radial_ranges**2
    #         num_radii = radial_ranges.shape[0]
    #         size_input = self.polar_peaks.shape
    #     else:
    #         orientation_growth_angles = np.atleast_1d(orientation_growth_angles)
    #         num_angles = orientation_growth_angles.shape[0]
    #         size_input = [orientation_map.num_x, orientation_map.num_y]
    #         if orientation_separate_bins is False:
    #             num_radii = 1
    #         else:
    #             num_radii = num_angles
    
    #     size_output = np.round(
    #         np.array(size_input).astype("float") * upsample_factor
    #     ).astype("int")
    
    #     # output init
    #     orient_hist = np.zeros([num_radii, size_output[0], size_output[1], num_theta_bins])
    
    #     # Loop over all probe positions
    #     for a0 in range(num_radii):
    #         t = "Generating histogram " + str(a0)
    #         # for rx, ry in tqdmnd(
    #         #         *bragg_peaks.shape, desc=t,unit=" probe positions", disable=not progress_bar
    #         #     ):
    #         for rx, ry in tqdmnd(
    #             *size_input, desc=t, unit=" probe positions", disable=not progress_bar
    #         ):
    #             x = (rx + 0.5) * upsample_factor - 0.5
    #             y = (ry + 0.5) * upsample_factor - 0.5
    #             x = np.clip(x, 0, size_output[0] - 2)
    #             y = np.clip(y, 0, size_output[1] - 2)
    
    #             xF = np.floor(x).astype("int")
    #             yF = np.floor(y).astype("int")
    #             dx = x - xF
    #             dy = y - yF
    
    #             add_data = False
    
    #             if orientation_map is None:
    #                 p_r_invA = self.polar_peaks['r_invA'][rx, ry]
    #                 p_theta = self.polar_peaks['theta'][rx, ry]
                    
    #                 if p_r_invA is not None and len(p_r_invA) > 0:
    #                     # Extract columns: p is shape (N_peaks, 4)
    #                     # Column 0: y_pixels, Column 1: x_pixels
    #                     # Column 2: y_invA,   Column 3: x_invA
    #                     # p_y_invA = p[:, 2]
    #                     # p_x_invA = p[:, 3]
    #                     # r2 = p_y_invA**2 + p_x_invA**2
    #                     r2 = p_r_invA**2
    #                     sub = np.logical_and(
    #                         r2 >= radial_ranges_2[a0, 0], 
    #                         r2 < radial_ranges_2[a0, 1]
    #                     )
    #                     if np.any(sub):
    #                         intensity_data = self.peak_intensities['intensities_sampled_from_dp'][rx, ry]
    #                         if intensity_data is not None and len(intensity_data) > 0:
    #                             add_data = True
    #                             intensity = intensity_data[sub]
    #                             # Use theta directly from polar_peaks instead of recalculating
    #                             theta_radians = p_theta[sub]  # Already in radians
    #                             # Fold to 0-180° range (0 to π)
    #                             theta_folded = np.mod(theta_radians, np.pi)
    #                             t = theta_folded / dtheta
    #             else:
    #                 if orientation_map.corr[rx, ry, orientation_ind] > 0:
    #                     if orientation_separate_bins is False:
    #                         if orientation_flip_sign:
    #                             t = (
    #                                 np.array(
    #                                     [
    #                                         (
    #                                             -orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 0
    #                                             ]
    #                                             - orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 2
    #                                             ]
    #                                         )
    #                                         / dtheta
    #                                     ]
    #                                 )
    #                                 + orientation_growth_angles
    #                             )
    #                         else:
    #                             t = (
    #                                 np.array(
    #                                     [
    #                                         (
    #                                             orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 0
    #                                             ]
    #                                             + orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 2
    #                                             ]
    #                                         )
    #                                         / dtheta
    #                                     ]
    #                                 )
    #                                 + orientation_growth_angles
    #                             )
    #                         intensity = (
    #                             np.ones(num_angles)
    #                             * orientation_map.corr[rx, ry, orientation_ind]
    #                         )
    #                         add_data = True
    #                     else:
    #                         if orientation_flip_sign:
    #                             t = (
    #                                 np.array(
    #                                     [
    #                                         (
    #                                             -orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 0
    #                                             ]
    #                                             - orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 2
    #                                             ]
    #                                         )
    #                                         / dtheta
    #                                     ]
    #                                 )
    #                                 + orientation_growth_angles[a0]
    #                             )
    #                         else:
    #                             t = (
    #                                 np.array(
    #                                     [
    #                                         (
    #                                             orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 0
    #                                             ]
    #                                             + orientation_map.angles[
    #                                                 rx, ry, orientation_ind, 2
    #                                             ]
    #                                         )
    #                                         / dtheta
    #                                     ]
    #                                 )
    #                                 + orientation_growth_angles[a0]
    #                             )
    #                         intensity = orientation_map.corr[rx, ry, orientation_ind]
    #                         add_data = True
    
    #             if add_data:
    #                 tF = np.floor(t).astype("int")
    #                 dt = t - tF
    
    #                 orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
    #                     np.mod(tF, num_theta_bins),
    #                     weights=(1 - dx) * (1 - dy) * (1 - dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    #                 orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
    #                     np.mod(tF + 1, num_theta_bins),
    #                     weights=(1 - dx) * (1 - dy) * (dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    
    #                 orient_hist[a0, xF + 1, yF, :] = orient_hist[
    #                     a0, xF + 1, yF, :
    #                 ] + np.bincount(
    #                     np.mod(tF, num_theta_bins),
    #                     weights=(dx) * (1 - dy) * (1 - dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    #                 orient_hist[a0, xF + 1, yF, :] = orient_hist[
    #                     a0, xF + 1, yF, :
    #                 ] + np.bincount(
    #                     np.mod(tF + 1, num_theta_bins),
    #                     weights=(dx) * (1 - dy) * (dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    
    #                 orient_hist[a0, xF, yF + 1, :] = orient_hist[
    #                     a0, xF, yF + 1, :
    #                 ] + np.bincount(
    #                     np.mod(tF, num_theta_bins),
    #                     weights=(1 - dx) * (dy) * (1 - dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    #                 orient_hist[a0, xF, yF + 1, :] = orient_hist[
    #                     a0, xF, yF + 1, :
    #                 ] + np.bincount(
    #                     np.mod(tF + 1, num_theta_bins),
    #                     weights=(1 - dx) * (dy) * (dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    
    #                 orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
    #                     a0, xF + 1, yF + 1, :
    #                 ] + np.bincount(
    #                     np.mod(tF, num_theta_bins),
    #                     weights=(dx) * (dy) * (1 - dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    #                 orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
    #                     a0, xF + 1, yF + 1, :
    #                 ] + np.bincount(
    #                     np.mod(tF + 1, num_theta_bins),
    #                     weights=(dx) * (dy) * (dt) * intensity,
    #                     minlength=num_theta_bins,
    #                 )
    
    #     # smoothing / interpolation
    #     if (sigma_x is not None) or (sigma_y is not None) or (sigma_theta is not None):
    #         if num_radii > 1:
    #             print("Interpolating orientation matrices ...", end="")
    #         else:
    #             print("Interpolating orientation matrix ...", end="")
    #         if sigma_x is not None and sigma_x > 0:
    #             orient_hist = gaussian_filter1d(
    #                 orient_hist,
    #                 sigma_x * upsample_factor,
    #                 mode="nearest",
    #                 axis=1,
    #                 truncate=3.0,
    #             )
    #         if sigma_y is not None and sigma_y > 0:
    #             orient_hist = gaussian_filter1d(
    #                 orient_hist,
    #                 sigma_y * upsample_factor,
    #                 mode="nearest",
    #                 axis=2,
    #                 truncate=3.0,
    #             )
    #         if sigma_theta is not None and sigma_theta > 0:
    #             orient_hist = gaussian_filter1d(
    #                 orient_hist, sigma_theta / dtheta_deg, mode="wrap", axis=3, truncate=2.0
    #             )
    #         print(" done.")
    
    #     # normalization
    #     if normalize_intensity_stack is True:
    #         orient_hist = orient_hist / np.max(orient_hist)
    #     elif normalize_intensity_image is True:
    #         for a0 in range(num_radii):
    #             orient_hist[a0, :, :, :] = orient_hist[a0, :, :, :] / np.max(
    #                 orient_hist[a0, :, :, :]
    #             )
    
    #     return orient_hist

    def make_orientation_histogram(
        self,
        radial_ranges: np.ndarray = None,
        orientation_map=None,
        orientation_ind: int = 0,
        orientation_growth_angles: np.array = 0.0,
        orientation_separate_bins: bool = False,
        orientation_flip_sign: bool = False,
        orientation_offset_degrees: float = 0.0,
        upsample_factor: float = 4.0,
        theta_step_deg: float = 1.0,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        sigma_theta: float = 3.0,
        use_peak_sigma: bool = False,
        peak_sigma_samples: int = 6,
        normalize_intensity_image: bool = False,
        normalize_intensity_stack: bool = True,
        progress_bar: bool = True,
        r_field: str = "r_invA",
        theta_field: str = "theta",
        intensity_field: str = "intensities_sampled_from_dp",
    ):
        """
        Create a 3D or 4D orientation histogram from bragg peaks.
        
        Can generate histograms from either:
        1. Polar peak data with radial ranges
        2. Orientation map with Euler angles (for fiber textures)
    
        Parameters
        ----------
        radial_ranges : np.ndarray, optional
            Size (N x 2) array for N radial bins, or (2,) for a single bin.
        orientation_map : OrientationMap, optional
            Class containing Euler angles to generate a flowline map.
        orientation_ind : int
            Index of the orientation map (default 0)
        orientation_growth_angles : np.array
            Angles to place into histogram, relative to orientation.
        orientation_separate_bins : bool
            Whether to place multiple angles into multiple radial bins.
        orientation_flip_sign : bool
            Flip the direction of theta
        orientation_offset_degrees : float
            Offset for orientation angles in degrees
        upsample_factor : float
            Upsample factor for output histogram
        theta_step_deg : float
            Step size along annular direction in degrees
        sigma_x : float
            Smoothing in x direction before upsample
        sigma_y : float
            Smoothing in y direction before upsample
        sigma_theta : float
            Smoothing in annular direction (units of bins, periodic)
        use_peak_sigma : bool
            Spread signal along annular direction using measured peak width
        peak_sigma_samples : int
            Number of samples for peak sigma spreading
        normalize_intensity_image : bool
            Normalize to max peak intensity = 1, per image
        normalize_intensity_stack : bool
            Normalize to max peak intensity = 1, all images
        progress_bar : bool
            Enable progress bar
        r_field : str
            Name of radial coordinate field
        theta_field : str
            Name of angular coordinate field
        intensity_field : str
            Name of intensity field
    
        Returns
        -------
        orient_hist : np.ndarray
            4D array containing Bragg peak intensity histogram
            [radial_bin, x_probe, y_probe, theta]
        """
        # Coordinates
        theta = np.arange(0, 180, theta_step_deg) * np.pi / 180.0
        dtheta = theta[1] - theta[0]
        dtheta_deg = dtheta * 180 / np.pi
        num_theta_bins = np.size(theta)
        
        # Setup for peak sigma spreading
        if use_peak_sigma:
            v_sigma = np.linspace(-2, 2, 2 * peak_sigma_samples + 1)
            w_sigma = np.exp(-(v_sigma**2) / 2)
    
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
    
        # Output init
        orient_hist = np.zeros([num_radii, size_output[0], size_output[1], num_theta_bins])
    
        # Loop over all probe positions
        for a0 in range(num_radii):
            t = "Generating histogram " + str(a0)
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
                    p_r = self.polar_peaks[r_field][rx, ry]
                    p_theta = self.polar_peaks[theta_field][rx, ry]
                    
                    if p_r is not None and len(p_r) > 0:
                        r2 = p_r**2
                        sub = np.logical_and(
                            r2 >= radial_ranges_2[a0, 0], 
                            r2 < radial_ranges_2[a0, 1]
                        )
                        if np.any(sub):
                            intensity_data = self.peak_intensities[intensity_field][rx, ry]
                            if intensity_data is not None and len(intensity_data) > 0:
                                add_data = True
                                intensity = intensity_data[sub]
                                
                                # Get theta values
                                theta_radians = p_theta[sub]
                                if orientation_flip_sign:
                                    theta_radians *= -1
                                # Add offset
                                theta_radians += orientation_offset_degrees * np.pi / 180
                                
                                # Fold to 0-180° range (0 to π)
                                theta_folded = np.mod(theta_radians, np.pi)
                                t = theta_folded / dtheta
                                
                                # Spread signal using peak sigma if requested
                                if use_peak_sigma:
                                    # Try to get sigma values if available
                                    if 'sigma_theta' in self.polar_peaks.fields:
                                        theta_std = self.polar_peaks['sigma_theta'][rx, ry][sub] / dtheta
                                        t = (t[:, None] + theta_std[:, None] * v_sigma[None, :]).ravel()
                                        intensity = (intensity[:, None] * w_sigma[None, :]).ravel()
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
                            # Add offset
                            t += orientation_offset_degrees / dtheta_deg
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
                            # Add offset
                            t += orientation_offset_degrees / dtheta_deg
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
    
        # Smoothing / interpolation
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
    
        # Normalization
        if normalize_intensity_stack is True:
            orient_hist = orient_hist / np.max(orient_hist)
        elif normalize_intensity_image is True:
            for a0 in range(num_radii):
                orient_hist[a0, :, :, :] = orient_hist[a0, :, :, :] / np.max(
                    orient_hist[a0, :, :, :]
                )
    
        return orient_hist

    def plot_interactive_image_map(self, intensity_map=None, vmax_cartesian=7, 
                                    map_cmap='viridis', map_title='Intensity Map',
                                    norm_upper_quantile=None, norm_power=1.0,
                                    show_polar=True, vmax_polar=7):
        """
        Interactive plot for browsing diffraction patterns with optional intensity map.
        
        Parameters
        ----------
        intensity_map : array, optional
            2D array to display as reference map. Can be upsampled relative to dataset.
            If None, shows mean intensity at original resolution.
            Upsample factor is automatically detected from array dimensions.
        vmax_cartesian : float
            Maximum value for diffraction pattern display
        map_cmap : str
            Colormap for the intensity map
        map_title : str
            Title for the intensity map panel
        norm_upper_quantile : float, optional
            Upper quantile for normalization (0-1). If None, not used.
        norm_power : float
            Power law normalization exponent
        show_polar : bool
            Whether to show the polar transformed data panel
        vmax_polar : float, optional
            Maximum value for polar pattern display. If None, uses vmax_cartesian.
        """
        
        Ry, Rx = self.dataset_cartesian.shape[:2]
        
        # Check if polar data exists
        has_polar = hasattr(self, 'polar_data') and self.polar_data is not None
        if show_polar and not has_polar:
            print("Warning: polar_data not found. Set show_polar=False or run polar_transform_4d first.")
            show_polar = False
        
        # Create default intensity map if none provided
        if intensity_map is None:
            intensity_map = np.zeros((Ry, Rx))
            for i in range(Ry):
                for j in range(Rx):
                    intensity_map[i, j] = np.mean(self.dataset_cartesian[i, j].array)
            upsample_factor = 1
        else:
            # Auto-detect upsample factor from dimensions
            map_Ry, map_Rx = intensity_map.shape
            upsample_factor_y = map_Ry // Ry
            upsample_factor_x = map_Rx // Rx
            
            if upsample_factor_y != upsample_factor_x:
                raise ValueError(f"Inconsistent upsample factors: Y={upsample_factor_y}, X={upsample_factor_x}")
            
            if map_Ry % Ry != 0 or map_Rx % Rx != 0:
                raise ValueError(f"intensity_map shape {intensity_map.shape} is not an integer multiple "
                               f"of dataset shape ({Ry}, {Rx})")
            
            upsample_factor = upsample_factor_y
            print(f"Auto-detected upsample_factor: {upsample_factor}")
        
        # Set vmax_polar default
        if vmax_polar is None:
            vmax_polar = vmax_cartesian
        
        # Calculate slider ranges based on upsampled map
        slider_Ry = Ry * upsample_factor
        slider_Rx = Rx * upsample_factor
        
        def show_pattern(ry_slider, rx_slider):
            # Map slider position to original data indices
            ry_data = ry_slider // upsample_factor
            rx_data = rx_slider // upsample_factor
            
            # Calculate marker position in upsampled coordinates
            marker_ry = ry_slider
            marker_rx = rx_slider
            
            # Create figure
            if show_polar:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot intensity map (no normalization)
            im1 = ax1.imshow(intensity_map, cmap=map_cmap)
            ax1.plot(marker_rx, marker_ry, 'r+', markersize=15, markeredgewidth=2)
            ax1.set_title(map_title)
            ax1.set_xlabel('Rx (upsampled)' if upsample_factor > 1 else 'Rx')
            ax1.set_ylabel('Ry (upsampled)' if upsample_factor > 1 else 'Ry')
            plt.colorbar(im1, ax=ax1)
            
            # Plot diffraction pattern (with vmax)
            im2 = ax2.imshow(self.dataset_cartesian[ry_data, rx_data].array, cmap='gray', vmax=vmax_cartesian)
            ax2.set_title(f'Diffraction Pattern (Ry={ry_data}, Rx={rx_data})')
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2)
            
            # Plot polar transform if requested
            if show_polar:
                im3 = ax3.imshow(self.polar_data['intensity'][ry_data, rx_data].T, cmap='gray', vmax=vmax_polar, aspect='auto')
                ax3.set_title(f'Polar Transform (Ry={ry_data}, Rx={rx_data})')
                ax3.set_xlabel('Radius (bins)')
                ax3.set_ylabel('Theta (bins)')
                plt.colorbar(im3, ax=ax3)
            
            plt.tight_layout()
            plt.show()
        
        interact(show_pattern,
                 ry_slider=IntSlider(min=0, max=slider_Ry-1, step=1, value=slider_Ry//2, 
                             description='Ry:', continuous_update=False),
                 rx_slider=IntSlider(min=0, max=slider_Rx-1, step=1, value=slider_Rx//2, 
                             description='Rx:', continuous_update=False))
        
    def plot_interactive_peak_map(self, radial_range=None, intensity_map_override=None,
                                  vmax_cartesian=7, show_all_peaks=True, 
                                  show_center=True, selected_peak_color='red', 
                                  other_peak_color='gray', center_color='cyan', 
                                  norm_upper_quantile=None, norm_power=1.0, 
                                  peak_intensity_mode='size', peak_size_range=(30, 300), 
                                  peak_cmap='hot', peak_vmin=None, peak_vmax=None,
                                  show_polar=True, vmax_polar=None):
        """
        Interactive plot for browsing diffraction patterns with peak overlay.
        
        Parameters
        ----------
        radial_range : tuple, optional
            (q_min, q_max) in 1/Å to filter peaks. If None, shows all peaks.
        intensity_map_override : array, optional
            If provided, uses this 2D array instead of calculating peak intensity map.
            Can be upsampled relative to dataset_cartesian.
            Upsample factor is automatically detected from array dimensions.
        vmax_cartesian : float
            Maximum value for diffraction pattern display
        show_all_peaks : bool
            If True and radial_range is set, show non-selected peaks in gray
        show_center : bool
            Whether to show the beam center marker
        selected_peak_color : str or color
            Color for selected peaks (within radial_range) when peak_intensity_mode is None
        other_peak_color : str or color
            Color for non-selected peaks (outside radial_range)
        center_color : str or color
            Color for beam center marker
        norm_upper_quantile : float, optional
            Upper quantile for normalization (0-1). If None, not used.
        norm_power : float
            Power law normalization exponent
        peak_intensity_mode : str or None
            How to represent peak intensity: 'size', 'color', 'both', or None (fixed)
        peak_size_range : tuple
            (min_size, max_size) for marker sizes when using intensity-based sizing
        peak_cmap : str
            Colormap name for peak intensity coloring
        peak_vmin : float, optional
            Minimum intensity value for peak colormap normalization. If None, uses data min.
        peak_vmax : float, optional
            Maximum intensity value for peak colormap normalization. If None, uses data max.
        show_polar : bool
            Whether to show the polar transformed data panel
        vmax_polar : float, optional
            Maximum value for polar pattern display. If None, uses vmax_cartesian.
        """
        
        Ry, Rx = self.peak_coordinates_cartesian.shape
        
        # Check if polar data exists
        has_polar = hasattr(self, 'polar_data') and self.polar_data is not None
        if show_polar and not has_polar:
            print("Warning: polar_data not found. Set show_polar=False or run polar_transform_4d first.")
            show_polar = False
        
        # Use override if provided, otherwise create intensity map based on selected radial range
        if intensity_map_override is not None:
            intensity_map = intensity_map_override
            
            # Auto-detect upsample factor from dimensions
            map_Ry, map_Rx = intensity_map.shape
            upsample_factor_y = map_Ry // Ry
            upsample_factor_x = map_Rx // Rx
            
            if upsample_factor_y != upsample_factor_x:
                raise ValueError(f"Inconsistent upsample factors: Y={upsample_factor_y}, X={upsample_factor_x}")
            
            if map_Ry % Ry != 0 or map_Rx % Rx != 0:
                raise ValueError(f"intensity_map_override shape {intensity_map.shape} is not an integer multiple "
                               f"of dataset shape ({Ry}, {Rx})")
            
            upsample_factor = upsample_factor_y
            print(f"Auto-detected upsample_factor: {upsample_factor}")
            
            # Determine map title for override case
            if radial_range is not None:
                map_title = f'Custom Intensity Map\n({radial_range[0]:.2f}-{radial_range[1]:.2f} 1/Å)'
            else:
                map_title = 'Custom Intensity Map'
        else:
            # Calculate at original resolution
            intensity_map = np.zeros((Ry, Rx))
            for i in range(Ry):
                for j in range(Rx):
                    peaks_r_invA = self.polar_peaks['r_invA'][i, j]
                    if peaks_r_invA is not None and len(peaks_r_invA) > 0:
                        if radial_range is not None:
                            mask = (peaks_r_invA >= radial_range[0]) & (peaks_r_invA < radial_range[1])
                        else:
                            mask = np.ones(len(peaks_r_invA), dtype=bool)
                        
                        intensities = self.peak_intensities['intensities_sampled_from_dp'][i, j]
                        if intensities is not None and len(intensities) > 0:
                            intensity_map[i, j] = np.sum(intensities[mask])
            
            upsample_factor = 1
            
            # Determine map title for calculated case
            if radial_range is not None:
                map_title = f'Peak Intensity Map\n({radial_range[0]:.2f}-{radial_range[1]:.2f} 1/Å)'
            else:
                map_title = 'Peak Intensity Map (All Peaks)'
        
        # Set vmax_polar default
        if vmax_polar is None:
            vmax_polar = vmax_cartesian
        
        # Calculate slider ranges based on upsampled map
        slider_Ry = Ry * upsample_factor
        slider_Rx = Rx * upsample_factor
        
        def show_pattern(ry_slider, rx_slider):
            # Map slider position to original data indices
            ry_data = ry_slider // upsample_factor
            rx_data = rx_slider // upsample_factor
            
            # Calculate marker position in upsampled coordinates
            marker_ry = ry_slider
            marker_rx = rx_slider
            
            # Create figure
            if show_polar:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot intensity map (no normalization)
            im1 = ax1.imshow(intensity_map, cmap='viridis')
            ax1.plot(marker_rx, marker_ry, 'r+', markersize=15, markeredgewidth=2)
            ax1.set_title(map_title)
            ax1.set_xlabel('Rx (upsampled)' if upsample_factor > 1 else 'Rx')
            ax1.set_ylabel('Ry (upsampled)' if upsample_factor > 1 else 'Ry')
            plt.colorbar(im1, ax=ax1)
            
            # Plot diffraction pattern (with vmax)
            im2 = ax2.imshow(self.dataset_cartesian[ry_data, rx_data].array, cmap='gray', vmax=vmax_cartesian)
            ax2.set_title(f'Diffraction Pattern (Ry={ry_data}, Rx={rx_data})')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Overlay peaks on diffraction pattern
            peaks_r_invA = self.polar_peaks['r_invA'][ry_data, rx_data]
            peaks_y_pixels = self.peak_coordinates_cartesian['y_pixels'][ry_data, rx_data]
            peaks_x_pixels = self.peak_coordinates_cartesian['x_pixels'][ry_data, rx_data]
            peak_intensities = self.peak_intensities['intensities_sampled_from_dp'][ry_data, rx_data]
            
            if peaks_r_invA is not None and len(peaks_r_invA) > 0:
                if radial_range is not None:
                    mask = (peaks_r_invA >= radial_range[0]) & (peaks_r_invA < radial_range[1])
                    
                    if show_all_peaks and np.any(~mask):
                        ax2.scatter(peaks_x_pixels[~mask], peaks_y_pixels[~mask], 
                                   c=other_peak_color, s=30, alpha=0.5, marker='x',
                                   linewidths=1.5, label='Other peaks')
                    
                    if np.any(mask):
                        selected_x = peaks_x_pixels[mask]
                        selected_y = peaks_y_pixels[mask]
                        selected_intensities = peak_intensities[mask] if peak_intensities is not None else None
                        
                        if selected_intensities is not None and peak_intensity_mode is not None:
                            int_min = peak_vmin if peak_vmin is not None else np.min(selected_intensities)
                            int_max = peak_vmax if peak_vmax is not None else np.max(selected_intensities)
                            
                            if int_max > int_min:
                                norm_intensities = (selected_intensities - int_min) / (int_max - int_min)
                            else:
                                norm_intensities = np.ones_like(selected_intensities)
                            
                            if peak_intensity_mode == 'color':
                                colors = plt.cm.get_cmap(peak_cmap)(norm_intensities)
                                sizes = np.full(len(selected_intensities), 100)
                            elif peak_intensity_mode == 'size':
                                colors = selected_peak_color
                                sizes = peak_size_range[0] + norm_intensities * (peak_size_range[1] - peak_size_range[0])
                            elif peak_intensity_mode == 'both':
                                colors = plt.cm.get_cmap(peak_cmap)(norm_intensities)
                                sizes = peak_size_range[0] + norm_intensities * (peak_size_range[1] - peak_size_range[0])
                            else:
                                colors = selected_peak_color
                                sizes = np.full(len(selected_intensities), 100)
                            
                            ax2.scatter(selected_x, selected_y, c=colors, s=sizes, alpha=0.8, 
                                       marker='x', linewidths=2, label=f'Selected ({np.sum(mask)} peaks)')
                            
                            if peak_intensity_mode in ['color', 'both']:
                                sm = plt.cm.ScalarMappable(cmap=peak_cmap, 
                                                           norm=plt.Normalize(vmin=int_min, vmax=int_max))
                                sm.set_array([])
                                cbar = plt.colorbar(sm, ax=ax2, pad=0.02, fraction=0.046)
                                cbar.set_label('Peak Intensity', fontsize=8)
                        else:
                            ax2.scatter(selected_x, selected_y, c=selected_peak_color, s=100, 
                                       alpha=0.8, marker='x', linewidths=2, 
                                       label=f'Selected ({np.sum(mask)} peaks)')
                else:
                    if peak_intensities is not None and peak_intensity_mode is not None:
                        int_min = peak_vmin if peak_vmin is not None else np.min(peak_intensities)
                        int_max = peak_vmax if peak_vmax is not None else np.max(peak_intensities)
                        
                        if int_max > int_min:
                            norm_intensities = (peak_intensities - int_min) / (int_max - int_min)
                        else:
                            norm_intensities = np.ones_like(peak_intensities)
                        
                        if peak_intensity_mode == 'color':
                            colors = plt.cm.get_cmap(peak_cmap)(norm_intensities)
                            sizes = np.full(len(peak_intensities), 100)
                        elif peak_intensity_mode == 'size':
                            colors = selected_peak_color
                            sizes = peak_size_range[0] + norm_intensities * (peak_size_range[1] - peak_size_range[0])
                        elif peak_intensity_mode == 'both':
                            colors = plt.cm.get_cmap(peak_cmap)(norm_intensities)
                            sizes = peak_size_range[0] + norm_intensities * (peak_size_range[1] - peak_size_range[0])
                        else:
                            colors = selected_peak_color
                            sizes = np.full(len(peak_intensities), 100)
                        
                        ax2.scatter(peaks_x_pixels, peaks_y_pixels, c=colors, s=sizes, 
                                   alpha=0.8, marker='x', linewidths=2, 
                                   label=f'All peaks ({len(peaks_r_invA)})')
                        
                        if peak_intensity_mode in ['color', 'both']:
                            sm = plt.cm.ScalarMappable(cmap=peak_cmap, 
                                                       norm=plt.Normalize(vmin=int_min, vmax=int_max))
                            sm.set_array([])
                            cbar = plt.colorbar(sm, ax=ax2, pad=0.02, fraction=0.046)
                            cbar.set_label('Peak Intensity', fontsize=8)
                    else:
                        ax2.scatter(peaks_x_pixels, peaks_y_pixels, c=selected_peak_color, 
                                   s=100, alpha=0.8, marker='x', linewidths=2, 
                                   label=f'All peaks ({len(peaks_r_invA)})')
            
            if show_center and hasattr(self, 'image_centers') and self.image_centers is not None:
                ax2.scatter(self.image_centers[1, ry_data, rx_data], 
                           self.image_centers[0, ry_data, rx_data],
                           c=center_color, s=500, marker='x', linewidths=2, 
                           label='Center', zorder=10)
            
            handles, labels = ax2.get_legend_handles_labels()
            if labels:
                ax2.legend(fontsize=8, loc='upper right')
            
            if radial_range is not None:
                ax2.set_title(f'Diffraction Pattern (Ry={ry_data}, Rx={rx_data})\nRange: {radial_range[0]:.2f}-{radial_range[1]:.2f} 1/Å')
            
            # Plot polar transform if requested
            if show_polar:
                im3 = ax3.imshow(self.polar_data['intensity'][ry_data, rx_data].T, cmap='gray', vmax=vmax_polar, aspect='auto')
                ax3.set_title(f'Polar Transform (Ry={ry_data}, Rx={rx_data})')
                ax3.set_xlabel('Radius (bins)')
                ax3.set_ylabel('Theta (bins)')
                plt.colorbar(im3, ax=ax3)
            
            plt.tight_layout()
            plt.show()
        
        interact(show_pattern,
                 ry_slider=IntSlider(min=0, max=slider_Ry-1, step=1, value=slider_Ry//2, 
                             description='Ry:', continuous_update=False),
                 rx_slider=IntSlider(min=0, max=slider_Rx-1, step=1, value=slider_Rx//2, 
                             description='Rx:', continuous_update=False))

    def create_interactive_circular_mask(self, initial_x0=None, initial_y0=None, initial_r=None,
                                         reference_image=None, overlay_alpha=0.3):
        """
        Interactive mask creation with sliders for circular region selection.
        
        Parameters
        ----------
        initial_x0 : int, optional
            Initial x center position. If None, uses center of scan.
        initial_y0 : int, optional
            Initial y center position. If None, uses center of scan.
        initial_r : int, optional
            Initial radius. If None, uses 1/3 of minimum scan dimension.
        reference_image : array, optional
            2D array (Ry, Rx) to display as reference. If None, uses virtual image.
        overlay_alpha : float
            Transparency for mask overlay (0=transparent, 1=opaque)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'mask': final boolean mask array
            - 'x0', 'y0', 'r': final circle parameters
        """
        
        Ry, Rx = self.dataset_cartesian.shape[:2]
        
        # Set defaults
        if initial_x0 is None:
            initial_x0 = Ry // 2
        if initial_y0 is None:
            initial_y0 = Rx // 2
        if initial_r is None:
            initial_r = min(Ry, Rx) // 3
        
        # Get reference image and ensure it's a proper numpy array
        if reference_image is None:
            if hasattr(self.dataset_cartesian, 'virtual_images') and 'virtual_image' in self.dataset_cartesian.virtual_images:
                vimg = self.dataset_cartesian.virtual_images['virtual_image']
                # Extract array from Dataset2d object
                if hasattr(vimg, 'array'):
                    reference_image = vimg.array
                elif hasattr(vimg, 'data'):
                    reference_image = vimg.data
                else:
                    reference_image = np.array(vimg)
            else:
                # Create mean intensity image
                print("Creating reference image from mean intensities...")
                reference_image = np.zeros((Ry, Rx), dtype=float)
                for i in range(Ry):
                    for j in range(Rx):
                        dp = self.dataset_cartesian[i, j]
                        if hasattr(dp, 'array'):
                            reference_image[i, j] = np.mean(dp.array)
                        else:
                            reference_image[i, j] = np.mean(dp)
        
        # Ensure it's a float array
        reference_image = np.asarray(reference_image, dtype=float)
        
        # Verify reference_image is valid
        if reference_image.shape != (Ry, Rx):
            raise ValueError(f"reference_image shape {reference_image.shape} must match scan shape ({Ry}, {Rx})")
        
        # Store current state
        result = {'mask': None, 'x0': initial_x0, 'y0': initial_y0, 'r': initial_r}
        
        # Create sliders with proper orientation
        # X slider is inverted so bottom = 0, top = Ry-1
        x0_slider = widgets.IntSlider(
            min=0, max=Ry-1, step=1, value=Ry-1-initial_x0,  # Inverted initial value
            description='X (vert):',
            orientation='vertical',
            continuous_update=False,
            style={'description_width': '60px'},
            layout=widgets.Layout(height='300px'),
            readout=False  # We'll use custom label
        )
        
        # Custom label to show actual (inverted) value
        x0_label = widgets.Label(value=f'{initial_x0}')
        x0_label.layout.width = '60px'
        
        y0_slider = widgets.IntSlider(
            min=0, max=Rx-1, step=1, value=initial_y0,
            description='Y (horiz):',
            orientation='horizontal',
            continuous_update=False,
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
        
        r_slider = widgets.IntSlider(
            min=1, max=max(Ry, Rx), step=1, value=initial_r,
            description='Radius:',
            orientation='horizontal',
            continuous_update=False,
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
        
        output = widgets.Output()
        
        def update_mask(change=None):
            x0 = Ry - 1 - x0_slider.value  # Invert the slider value
            y0 = y0_slider.value
            r = r_slider.value
            
            # Update custom label
            x0_label.value = f'{x0}'
            
            # Create mask
            x = np.arange(Ry)[:, None]
            y = np.arange(Rx)[None, :]
            mask = (x - x0)**2 + (y - y0)**2 < r**2
            
            # Update result
            result['mask'] = mask
            result['x0'] = x0
            result['y0'] = y0
            result['r'] = r
            
            # Create matplotlib figure directly
            with output:
                output.clear_output(wait=True)
                fig, axs = plt.subplots(1, 3, figsize=(15, 4))
                
                # Plot 1: Reference image with circle outline
                im0 = axs[0].imshow(reference_image, cmap='gray')
                circle = plt.Circle((y0, x0), r, color='red', fill=False, linewidth=2, linestyle='--')
                axs[0].add_patch(circle)
                axs[0].plot(y0, x0, 'r+', markersize=15, markeredgewidth=2)
                axs[0].set_title('Reference Image')
                axs[0].set_xlabel('Rx')
                axs[0].set_ylabel('Ry')
                plt.colorbar(im0, ax=axs[0])
                
                # Plot 2: Mask only
                im1 = axs[1].imshow(mask.astype(float), cmap='Reds')
                axs[1].set_title('Mask')
                axs[1].set_xlabel('Rx')
                axs[1].set_ylabel('Ry')
                axs[1].text(0.02, 0.98, f'Center: ({x0}, {y0})\nRadius: {r}\nPixels: {mask.sum()}',
                           transform=axs[1].transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Plot 3: Overlay
                im2 = axs[2].imshow(reference_image, cmap='gray')
                axs[2].imshow(mask.astype(float), alpha=overlay_alpha, cmap='Reds')
                axs[2].set_title('Overlay')
                axs[2].set_xlabel('Rx')
                axs[2].set_ylabel('Ry')
                plt.colorbar(im2, ax=axs[2])
                
                plt.tight_layout()
                plt.show()
        
        # Link sliders to update function
        x0_slider.observe(update_mask, names='value')
        y0_slider.observe(update_mask, names='value')
        r_slider.observe(update_mask, names='value')
        
        # Create layout: vertical slider with label on left, horizontal sliders and output stacked on right
        x0_controls = widgets.VBox([
            x0_slider,
            x0_label
        ], layout=widgets.Layout(align_items='center'))
        
        ui = widgets.HBox([
            x0_controls,
            widgets.VBox([
                y0_slider,
                r_slider,
                output
            ])
        ])
        
        # Initial plot
        update_mask()
        
        # Display the widget
        display(ui)
        
        return result

    def plot_peak_histogram_map(
        self, 
        intensity_threshold=None,
        intensity_percentile=None,
        figsize=(8, 6), 
        cmap='viridis',
        return_values=False,
    ):
        """
        Plot 2D map showing the number of peaks found at each scan position.
        
        Parameters:
        -----------
        intensity_threshold : float, optional
            Absolute intensity threshold. Only count peaks above this value.
        intensity_percentile : float, optional
            Percentile threshold (0-100). Overrides intensity_threshold.
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap to use
        return_values : bool
            If True, return figure, axes, and count_map
        
        Returns:
        --------
        fig, ax, count_map : (optional) matplotlib figure, axes, and count array
        """
        Ry, Rx = self.peak_coordinates_cartesian.shape
        
        # Convert percentile to threshold if needed
        if intensity_percentile is not None:
            all_intensities = [
                self.peak_intensities['intensities_sampled_from_dp'][i, j]
                for i in range(Ry) for j in range(Rx)
                if self.peak_intensities['intensities_sampled_from_dp'][i, j] is not None
            ]
            if all_intensities:
                intensity_threshold = np.percentile(np.concatenate(all_intensities), intensity_percentile)
        
        # Build count map
        count_map = np.zeros((Ry, Rx))
        for i in range(Ry):
            for j in range(Rx):
                peaks = self.peak_coordinates_cartesian[i, j]
                if peaks is None or len(peaks) == 0:
                    continue
                
                if intensity_threshold is None:
                    count_map[i, j] = len(peaks)
                else:
                    intensities = self.peak_intensities['intensities_sampled_from_dp'][i, j]
                    if intensities is not None:
                        count_map[i, j] = np.sum(intensities >= intensity_threshold)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(count_map, cmap=cmap, origin='lower')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Peaks', fontsize=12)
        
        # Integer colorbar ticks
        max_count = int(np.max(count_map))
        if max_count > 0:
            ticks = np.arange(0, max_count + 1, max(1, max_count // 5))
            cbar.set_ticks(ticks)
        
        # Title
        title = 'Peak Count per Scan Position'
        if intensity_threshold is not None:
            title += f'\n(intensity ≥ {intensity_threshold:.3f})'
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Scan X', fontsize=12)
        ax.set_ylabel('Scan Y', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        if return_values:
            return fig, ax, count_map
            
    def plot_peak_count_map(self, q_ranges, figsize_per_map=(5, 4), cmap='viridis', return_values=False):
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
            
            # Calculate max_count early for use in both colorbar and statistics
            max_count = int(np.max(count_map))
            
            # Plot
            _, _ = show_2d(
                count_map,
                cmap=cmap,
                title=f'Peak Count\n{q_min:.2f} - {q_max:.2f} 1/Å',
                cbar=True,
                show_ticks=True,
                figax=(fig, axes[idx])
            )
            
            # Customize colorbar - get it from the axes image
            try:
                # Try to get colorbar from the image in the axes
                im = axes[idx].images[0]
                cbar = im.colorbar
                
                if cbar is not None:
                    cbar.set_label('Number of Peaks', fontsize=10)
                    
                    # Set colorbar ticks to integers
                    if max_count > 0:
                        tick_spacing = max(1, max_count // 5)  # About 5 ticks
                        ticks = np.arange(0, max_count + 1, tick_spacing)
                        cbar.set_ticks(ticks)
            except (AttributeError, IndexError):
                # If colorbar access fails, create it manually
                im = axes[idx].images[0]
                cbar = plt.colorbar(im, ax=axes[idx])
                cbar.set_label('Number of Peaks', fontsize=10)
                
                # Set colorbar ticks to integers
                if max_count > 0:
                    tick_spacing = max(1, max_count // 5)
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

        if return_values:
            return fig, axes, count_maps

    def make_flowline_map(
        self,
        orient_hist,
        thresh_seed=0.2,
        thresh_grow=0.05,
        thresh_collision=0.001,
        sep_seeds=None,
        sep_xy=6.0,
        sep_theta=5.0,
        sort_seeds="intensity",
        linewidth=2.0,
        step_size=0.5,
        min_steps=4,
        max_steps=1000,
        sigma_x=1.0,
        sigma_y=1.0,
        sigma_theta=2.0,
        progress_bar: bool = True,
    ):
        """
        Create an 3D or 4D orientation flowline map - essentially a pixelated "stream map" which represents diffraction data.
    
        Args:
            orient_hist (array):        Histogram of all orientations with coordinates
                                        [radial_bin x_probe y_probe theta]
                                        We assume theta bin ranges from 0 to 180 degrees and is periodic.
            thresh_seed (float):        Threshold for seed generation in histogram.
            thresh_grow (float):        Threshold for flowline growth in histogram.
            thresh_collision (float):   Threshold for termination of flowline growth in histogram.
            sep_seeds (float):          Initial seed separation in bins - set to None to use default value,
                                        which is equal to 0.5*sep_xy.
            sep_xy (float):             Search radius for flowline direction in x and y.
            sep_theta = (float):        Search radius for flowline direction in theta.
            sort_seeds (str):           How to sort the initial seeds for growth:
                                            None - no sorting
                                            'intensity' - sort by histogram intensity
                                            'random' - random order
            linewidth (float):          Thickness of the flowlines in pixels.
            step_size (float):          Step size for flowline growth in pixels.
            min_steps (int):            Minimum number of steps for a flowline to be drawn.
            max_steps (int):            Maximum number of steps for a flowline to be drawn.
            sigma_x (float):            Weighted sigma in x direction for direction update.
            sigma_y (float):            Weighted sigma in y direction for direction update.
            sigma_theta (float):        Weighted sigma in theta for direction update.
            progress_bar (bool):        Enable progress bar
    
        Returns:
            orient_flowlines (array):   4D array containing flowlines
                                        [radial_bin x_probe y_probe theta]
        """
    
        # Ensure sep_xy and sep_theta are arrays
        sep_xy = np.atleast_1d(sep_xy)
        sep_theta = np.atleast_1d(sep_theta)
    
        # number of radial bins
        num_radii = orient_hist.shape[0]
        if num_radii > 1 and len(sep_xy) == 1:
            sep_xy = np.ones(num_radii) * sep_xy
        if num_radii > 1 and len(sep_theta) == 1:
            sep_theta = np.ones(num_radii) * sep_theta
    
        # Default seed separation
        if sep_seeds is None:
            sep_seeds = np.round(np.min(sep_xy) / 2 + 0.5).astype("int")
        else:
            sep_seeds = np.atleast_1d(sep_seeds).astype("int")
            if num_radii > 1 and len(sep_seeds) == 1:
                sep_seeds = (np.ones(num_radii) * sep_seeds).astype("int")
    
        # coordinates
        theta = np.linspace(0, np.pi, orient_hist.shape[3], endpoint=False)
        dtheta = theta[1] - theta[0]
        size_3D = np.array(
            [
                orient_hist.shape[1],
                orient_hist.shape[2],
                orient_hist.shape[3],
            ]
        )
    
        # initialize weighting array
        vx = np.arange(-np.ceil(2 * sigma_x), np.ceil(2 * sigma_x) + 1)
        vy = np.arange(-np.ceil(2 * sigma_y), np.ceil(2 * sigma_y) + 1)
        vt = np.arange(-np.ceil(2 * sigma_theta), np.ceil(2 * sigma_theta) + 1)
        ay, ax, at = np.meshgrid(vy, vx, vt)
        k = (
            np.exp(ax**2 / (-2 * sigma_x**2))
            * np.exp(ay**2 / (-2 * sigma_y**2))
            * np.exp(at**2 / (-2 * sigma_theta**2))
        )
        k = k / np.sum(k)
        vx = vx[:, None, None].astype("int")
        vy = vy[None, :, None].astype("int")
        vt = vt[None, None, :].astype("int")
    
        # initalize flowline array
        orient_flowlines = np.zeros_like(orient_hist)
    
        # initialize output
        xy_t_int = np.zeros((max_steps + 1, 4))
        xy_t_int_rev = np.zeros((max_steps + 1, 4))
    
        # Loop over radial bins
        for a0 in range(num_radii):
            # initialize collision check array
            cr = np.arange(-np.ceil(sep_xy[a0]), np.ceil(sep_xy[a0]) + 1)
            ct = np.arange(-np.ceil(sep_theta[a0]), np.ceil(sep_theta[a0]) + 1)
            ay, ax, at = np.meshgrid(cr, cr, ct)
            c_mask = (
                (ax**2 + ay**2) / sep_xy[a0] ** 2 + at**2 / sep_theta[a0] ** 2
                <= (1 + 1 / sep_xy[a0]) ** 2
            )[None, :, :, :]
            cx = cr[None, :, None, None].astype("int")
            cy = cr[None, None, :, None].astype("int")
            ct = ct[None, None, None, :].astype("int")
    
            # Find all seed locations
            orient = orient_hist[a0, :, :, :]
            sub_seeds = np.logical_and(
                np.logical_and(
                    orient >= np.roll(orient, 1, axis=2),
                    orient >= np.roll(orient, -1, axis=2),
                ),
                orient >= thresh_seed,
            )
    
            # Separate seeds
            if sep_seeds > 0:
                for a1 in range(sep_seeds - 1):
                    sub_seeds[a1::sep_seeds, :, :] = False
                    sub_seeds[:, a1::sep_seeds, :] = False
    
            # Index seeds
            x_inds, y_inds, t_inds = np.where(sub_seeds)
            if sort_seeds is not None:
                if sort_seeds == "intensity":
                    inds_sort = np.argsort(orient[sub_seeds])[::-1]
                elif sort_seeds == "random":
                    inds_sort = np.random.permutation(np.count_nonzero(sub_seeds))
                x_inds = x_inds[inds_sort]
                y_inds = y_inds[inds_sort]
                t_inds = t_inds[inds_sort]
    
            # for a1 in tqdmnd(range(0,40), desc="Drawing flowlines",unit=" seeds", disable=not progress_bar):
            t = "Drawing flowlines " + str(a0)
            for a1 in tqdmnd(
                range(0, x_inds.shape[0]), desc=t, unit=" seeds", disable=not progress_bar
            ):
                # initial coordinate and intensity
                xy0 = np.array((x_inds[a1], y_inds[a1]))
                t0 = theta[t_inds[a1]]
    
                # init theta
                inds_theta = np.mod(
                    np.round(t0 / dtheta).astype("int") + vt, orient.shape[2]
                )
                orient_crop = (
                    k
                    * orient[
                        np.clip(
                            np.round(xy0[0]).astype("int") + vx, 0, orient.shape[0] - 1
                        ),
                        np.clip(
                            np.round(xy0[1]).astype("int") + vy, 0, orient.shape[1] - 1
                        ),
                        inds_theta,
                    ]
                )
                theta_crop = theta[inds_theta]
                t0 = np.sum(orient_crop * theta_crop) / np.sum(orient_crop)
    
                # forward direction
                t = t0
                v0 = np.array((-np.sin(t), np.cos(t)))
                v = v0 * step_size
                xy = xy0
                int_val = self.get_intensity(orient, xy0[0], xy0[1], t0 / dtheta)
                xy_t_int[0, 0:2] = xy0
                xy_t_int[0, 2] = t / dtheta
                xy_t_int[0, 3] = int_val
                # main loop
                grow = True
                count = 0
                while grow is True:
                    count += 1
    
                    # update position and intensity
                    xy = xy + v
                    int_val = self.get_intensity(orient, xy[0], xy[1], t / dtheta)
    
                    # check for collision
                    flow_crop = orient_flowlines[
                        a0,
                        np.clip(np.round(xy[0]).astype("int") + cx, 0, orient.shape[0] - 1),
                        np.clip(np.round(xy[1]).astype("int") + cy, 0, orient.shape[1] - 1),
                        np.mod(np.round(t / dtheta).astype("int") + ct, orient.shape[2]),
                    ]
                    int_flow = np.max(flow_crop[c_mask])
    
                    if (
                        xy[0] < 0
                        or xy[1] < 0
                        or xy[0] > orient.shape[0]
                        or xy[1] > orient.shape[1]
                        or int_val < thresh_grow
                        or int_flow > thresh_collision
                    ):
                        grow = False
                    else:
                        # update direction
                        inds_theta = np.mod(
                            np.round(t / dtheta).astype("int") + vt, orient.shape[2]
                        )
                        orient_crop = (
                            k
                            * orient[
                                np.clip(
                                    np.round(xy[0]).astype("int") + vx,
                                    0,
                                    orient.shape[0] - 1,
                                ),
                                np.clip(
                                    np.round(xy[1]).astype("int") + vy,
                                    0,
                                    orient.shape[1] - 1,
                                ),
                                inds_theta,
                            ]
                        )
                        theta_crop = theta[inds_theta]
                        t = np.sum(orient_crop * theta_crop) / np.sum(orient_crop)
                        v = np.array((-np.sin(t), np.cos(t))) * step_size
    
                        xy_t_int[count, 0:2] = xy
                        xy_t_int[count, 2] = t / dtheta
                        xy_t_int[count, 3] = int_val
    
                        if count > max_steps - 1:
                            grow = False
    
                # reverse direction
                t = t0 + np.pi
                v0 = np.array((-np.sin(t), np.cos(t)))
                v = v0 * step_size
                xy = xy0
                int_val = self.get_intensity(orient, xy0[0], xy0[1], t0 / dtheta)
                xy_t_int_rev[0, 0:2] = xy0
                xy_t_int_rev[0, 2] = t / dtheta
                xy_t_int_rev[0, 3] = int_val
                # main loop
                grow = True
                count_rev = 0
                while grow is True:
                    count_rev += 1
    
                    # update position and intensity
                    xy = xy + v
                    int_val = self.get_intensity(orient, xy[0], xy[1], t / dtheta)
    
                    # check for collision
                    flow_crop = orient_flowlines[
                        a0,
                        np.clip(np.round(xy[0]).astype("int") + cx, 0, orient.shape[0] - 1),
                        np.clip(np.round(xy[1]).astype("int") + cy, 0, orient.shape[1] - 1),
                        np.mod(np.round(t / dtheta).astype("int") + ct, orient.shape[2]),
                    ]
                    int_flow = np.max(flow_crop[c_mask])
    
                    if (
                        xy[0] < 0
                        or xy[1] < 0
                        or xy[0] > orient.shape[0]
                        or xy[1] > orient.shape[1]
                        or int_val < thresh_grow
                        or int_flow > thresh_collision
                    ):
                        grow = False
                    else:
                        # update direction
                        inds_theta = np.mod(
                            np.round(t / dtheta).astype("int") + vt, orient.shape[2]
                        )
                        orient_crop = (
                            k
                            * orient[
                                np.clip(
                                    np.round(xy[0]).astype("int") + vx,
                                    0,
                                    orient.shape[0] - 1,
                                ),
                                np.clip(
                                    np.round(xy[1]).astype("int") + vy,
                                    0,
                                    orient.shape[1] - 1,
                                ),
                                inds_theta,
                            ]
                        )
                        theta_crop = theta[inds_theta]
                        t = np.sum(orient_crop * theta_crop) / np.sum(orient_crop) + np.pi
                        v = np.array((-np.sin(t), np.cos(t))) * step_size
    
                        xy_t_int_rev[count_rev, 0:2] = xy
                        xy_t_int_rev[count_rev, 2] = t / dtheta
                        xy_t_int_rev[count_rev, 3] = int_val
    
                        if count_rev > max_steps - 1:
                            grow = False
    
                # write into output array
                if count + count_rev > min_steps:
                    if count > 0:
                        orient_flowlines[a0, :, :, :] = self.set_intensity(
                            orient_flowlines[a0, :, :, :], xy_t_int[1:count, :]
                        )
                    if count_rev > 1:
                        orient_flowlines[a0, :, :, :] = self.set_intensity(
                            orient_flowlines[a0, :, :, :], xy_t_int_rev[1:count_rev, :]
                        )
    
        # normalize to step size
        orient_flowlines = orient_flowlines * step_size
    
        # linewidth
        if linewidth > 1.0:
            s = linewidth - 1.0
    
            orient_flowlines = gaussian_filter1d(orient_flowlines, s, axis=1, truncate=3.0)
            orient_flowlines = gaussian_filter1d(orient_flowlines, s, axis=2, truncate=3.0)
            orient_flowlines = orient_flowlines * (s**2)
    
        return orient_flowlines
    
    
    def make_flowline_rainbow_image(
        self,
        orient_flowlines,
        int_range=[0, 0.2],
        sym_rotation_order=2,
        theta_offset=0.0,
        greyscale=False,
        greyscale_max=True,
        white_background=False,
        power_scaling=1.0,
        sum_radial_bins=False,
        plot_images=True,
        figsize=None,
    ):
        """
        Generate RGB output images from the flowline arrays.
    
        Args:
            orient_flowline (array):    Histogram of all orientations with coordinates [x y radial_bin theta]
                                        We assume theta bin ranges from 0 to 180 degrees and is periodic.
            int_range (float)           2 element array giving the intensity range
            sym_rotation_order (int):   rotational symmety for colouring
            theta_offset (float):       Offset the anglular coloring by this value in radians.
            greyscale (bool):           Set to False for color output, True for greyscale output.
            greyscale_max (bool):       If output is greyscale, use max instead of mean for overlapping flowlines.
            white_background (bool):    For either color or greyscale output, switch to white background (from black).
            power_scaling (float):      Power law scaling for flowline intensity output.
            sum_radial_bins (bool):     Sum all radial bins (alternative is to output separate images).
            plot_images (bool):         Plot the outputs for quick visualization.
            figsize (2-tuple):          Size of output figure.
    
        Returns:
            im_flowline (array):        3D or 4D array containing flowline images
        """
    
        # init array
        size_input = orient_flowlines.shape
        size_output = np.array([size_input[0], size_input[1], size_input[2], 3])
        im_flowline = np.zeros(size_output)
        theta_offset = np.atleast_1d(theta_offset)
    
        if greyscale is True:
            for a0 in range(size_input[0]):
                if greyscale_max is True:
                    im = np.max(orient_flowlines[a0, :, :, :], axis=2)
                else:
                    im = np.mean(orient_flowlines[a0, :, :, :], axis=2)
    
                sig = np.clip((im - int_range[0]) / (int_range[1] - int_range[0]), 0, 1)
    
                if power_scaling != 1:
                    sig = sig**power_scaling
    
                if white_background is False:
                    im_flowline[a0, :, :, :] = sig[:, :, None]
                else:
                    im_flowline[a0, :, :, :] = 1 - sig[:, :, None]
    
        else:
            # Color basis
            c0 = np.array([1.0, 0.0, 0.0])
            c1 = np.array([0.0, 0.7, 0.0])
            c2 = np.array([0.0, 0.3, 1.0])
    
            # angles
            theta = np.linspace(0, np.pi, size_input[3], endpoint=False)
            theta_color = theta * sym_rotation_order
    
            if size_input[0] > 1 and len(theta_offset) == 1:
                theta_offset = np.ones(size_input[0]) * theta_offset
    
            for a0 in range(size_input[0]):
                # color projections
                b0 = np.maximum(
                    1
                    - np.abs(
                        np.mod(theta_offset[a0] + theta_color + np.pi, 2 * np.pi) - np.pi
                    )
                    ** 2
                    / (np.pi * 2 / 3) ** 2,
                    0,
                )
                b1 = np.maximum(
                    1
                    - np.abs(
                        np.mod(
                            theta_offset[a0] + theta_color - np.pi * 2 / 3 + np.pi,
                            2 * np.pi,
                        )
                        - np.pi
                    )
                    ** 2
                    / (np.pi * 2 / 3) ** 2,
                    0,
                )
                b2 = np.maximum(
                    1
                    - np.abs(
                        np.mod(
                            theta_offset[a0] + theta_color - np.pi * 4 / 3 + np.pi,
                            2 * np.pi,
                        )
                        - np.pi
                    )
                    ** 2
                    / (np.pi * 2 / 3) ** 2,
                    0,
                )
    
                sig = np.clip(
                    (orient_flowlines[a0, :, :, :] - int_range[0])
                    / (int_range[1] - int_range[0]),
                    0,
                    1,
                )
                if power_scaling != 1:
                    sig = sig**power_scaling
    
                im_flowline[a0, :, :, :] = (
                    np.sum(sig * b0[None, None, :], axis=2)[:, :, None] * c0[None, None, :]
                    + np.sum(sig * b1[None, None, :], axis=2)[:, :, None]
                    * c1[None, None, :]
                    + np.sum(sig * b2[None, None, :], axis=2)[:, :, None]
                    * c2[None, None, :]
                )
    
                # clip limits
                im_flowline[a0, :, :, :] = np.clip(im_flowline[a0, :, :, :], 0, 1)
    
                # contrast flip
                if white_background is True:
                    im = rgb_to_hsv(im_flowline[a0])
                    im_v = im[:, :, 2]
                    im[:, :, 1] = im_v
                    im[:, :, 2] = 1
                    im_flowline[a0] = hsv_to_rgb(im)
    
        if sum_radial_bins is True:
            if white_background is False:
                im_flowline = np.clip(np.sum(im_flowline, axis=0), 0, 1)[None, :, :, :]
            else:
                # im_flowline = np.clip(np.sum(im_flowline,axis=0)+1-im_flowline.shape[0],0,1)[None,:,:,:]
                im_flowline = np.min(im_flowline, axis=0)[None, :, :, :]
    
        if plot_images is True:
            if figsize is None:
                fig, ax = plt.subplots(
                    im_flowline.shape[0], 1, figsize=(10, im_flowline.shape[0] * 10)
                )
            else:
                fig, ax = plt.subplots(im_flowline.shape[0], 1, figsize=figsize)
    
            if im_flowline.shape[0] > 1:
                for a0 in range(im_flowline.shape[0]):
                    ax[a0].imshow(im_flowline[a0])
                    # ax[a0].axis('off')
                plt.subplots_adjust(wspace=0, hspace=0.02)
            else:
                ax.imshow(im_flowline[0])
                # ax.axis('off')
            plt.show()
    
        return im_flowline
    
    
    def make_flowline_rainbow_legend(
        self,
        im_size=np.array([256, 256]),
        sym_rotation_order=2,
        theta_offset_degrees=0.0,
        white_background=False,
        return_image=False,
        radial_range=np.array([0.45, 0.9]),
        plot_legend=True,
        figsize=(4, 4),
    ):
        """
        This function generates a legend for a the rainbow colored flowline maps, and returns it as an RGB image.
    
        Parameters
        ----------
        im_size (np.array):
            Size of legend image in pixels.
        sym_rotation_order (int):
            rotational symmety for colouring
        theta_offset_degrees (float):
            Offset the anglular coloring by this value in degrees.
            Rotation is Q with respect to R, in the positive (counter clockwise) direction.
        white_background (bool):
            For either color or greyscale output, switch to white background (from black).
        return_image (bool):
            Return the image array.
        radial_range (np.array):
            Inner and outer radius for the legend ring.
        plot_legend (bool):
            Plot the generated legend.
        figsize (tuple or list):
            Size of the plotted legend.
    
        Returns
        ----------
    
        im_legend (array):
            Image array for the legend.
        """
    
        # Coordinates
        x = np.linspace(-1, 1, im_size[0])
        y = np.linspace(-1, 1, im_size[1])
        ya, xa = np.meshgrid(y, x)
        # TODO: Can replace with squared term? ra2? Faster
        # ra = np.sqrt(xa**2 + ya**2)
        ra2 = xa**2 + ya**2
        ta = np.arctan2(ya, xa) + np.deg2rad(theta_offset_degrees)
        ta_sym = ta * sym_rotation_order
    
        # mask
        mask = np.logical_and(ra2 > radial_range[0]**2, ra2 < radial_range[1]**2)
        # mask = np.logical_and(ra > radial_range[0], ra < radial_range[1])
    
        # rgb image
        z = mask * np.exp(1j * ta_sym)
        # hue_offset = 0
        amp = np.abs(z)
        vmin = np.min(amp)
        vmax = np.max(amp)
        ph = np.angle(z)  # + hue_offset
        h = np.mod(ph / (2 * np.pi), 1)
        s = 0.85 * np.ones_like(h)
        v = (amp - vmin) / (vmax - vmin)
        im_legend = hsv_to_rgb(np.dstack((h, s, v)))
    
        if white_background is True:
            im_legend[im_legend.sum(2) == 0] = 1
    
        # plotting
        if plot_legend:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(im_legend)
            ax.invert_yaxis()
            # ax.set_axis_off()
            ax.axis("off")
    
        if return_image:
            return im_legend
    
    
    def make_flowline_combined_image(
        self,
        orient_flowlines,
        int_range=[0, 0.2],
        cvals=np.array(
            [
                [0.0, 0.7, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.7, 1.0],
            ]
        ),
        white_background=False,
        power_scaling=1.0,
        sum_radial_bins=True,
        plot_images=True,
        figsize=None,
    ):
        """
        Generate RGB output images from the flowline arrays.
    
        Args:
            orient_flowline (array):    Histogram of all orientations with coordinates [x y radial_bin theta]
                                        We assume theta bin ranges from 0 to 180 degrees and is periodic.
            int_range (float)           2 element array giving the intensity range
            cvals (array):              Nx3 size array containing RGB colors for different radial ibns.
            white_background (bool):    For either color or greyscale output, switch to white background (from black).
            power_scaling (float):      Power law scaling for flowline intensities.
            sum_radial_bins (bool):     Sum outputs over radial bins.
            plot_images (bool):         Plot the output images for quick visualization.
            figsize (2-tuple):          Size of output figure.
    
        Returns:
            im_flowline (array):        flowline images
        """
    
        # init array
        size_input = orient_flowlines.shape
        size_output = np.array([size_input[0], size_input[1], size_input[2], 3])
        im_flowline = np.zeros(size_output)
        cvals = np.array(cvals)
    
        # Generate all color images
        for a0 in range(size_input[0]):
            sig = np.clip(
                (np.sum(orient_flowlines[a0, :, :, :], axis=2) - int_range[0])
                / (int_range[1] - int_range[0]),
                0,
                1,
            )
            if power_scaling != 1:
                sig = sig**power_scaling
    
            if white_background:
                im_flowline[a0, :, :, :] = 1 - sig[:, :, None] * (
                    1 - cvals[a0, :][None, None, :]
                )
            else:
                im_flowline[a0, :, :, :] = sig[:, :, None] * cvals[a0, :][None, None, :]
    
            # # contrast flip
            # if white_background is True:
            #     im = rgb_to_hsv(im_flowline[a0,:,:,:])
            #     # im_s = im[:,:,1]
            #     im_v = im[:,:,2]
            #     v_range = [np.min(im_v), np.max(im_v)]
            #     print(v_range)
    
            #     im[:,:,1] = im_v
            #     im[:,:,2] = 1
            #     im_flowline[a0,:,:,:] = hsv_to_rgb(im)
    
        if sum_radial_bins is True:
            if white_background is False:
                im_flowline = np.clip(np.sum(im_flowline, axis=0), 0, 1)[None, :, :, :]
            else:
                # im_flowline = np.clip(np.sum(im_flowline,axis=0)+1-im_flowline.shape[0],0,1)[None,:,:,:]
                im_flowline = np.min(im_flowline, axis=0)[None, :, :, :]
    
        if plot_images is True:
            if figsize is None:
                fig, ax = plt.subplots(
                    im_flowline.shape[0], 1, figsize=(10, im_flowline.shape[0] * 10)
                )
            else:
                fig, ax = plt.subplots(im_flowline.shape[0], 1, figsize=figsize)
    
            if im_flowline.shape[0] > 1:
                for a0 in range(im_flowline.shape[0]):
                    ax[a0].imshow(im_flowline[a0])
                    ax[a0].axis("off")
                plt.subplots_adjust(wspace=0, hspace=0.02)
            else:
                ax.imshow(im_flowline[0])
                ax.axis("off")
            plt.show()
    
        return im_flowline

    def get_intensity(
        self, 
        orient, 
        x, 
        y, 
        t
    ):
        # utility function to get histogram intensites
    
        x = np.clip(x, 0, orient.shape[0] - 2)
        y = np.clip(y, 0, orient.shape[1] - 2)
    
        xF = np.floor(x).astype("int")
        yF = np.floor(y).astype("int")
        tF = np.floor(t).astype("int")
        dx = x - xF
        dy = y - yF
        dt = t - tF
        t1 = np.mod(tF, orient.shape[2])
        t2 = np.mod(tF + 1, orient.shape[2])
    
        int_vals = (
            orient[xF, yF, t1] * ((1 - dx) * (1 - dy) * (1 - dt))
            + orient[xF, yF, t2] * ((1 - dx) * (1 - dy) * (dt))
            + orient[xF, yF + 1, t1] * ((1 - dx) * (dy) * (1 - dt))
            + orient[xF, yF + 1, t2] * ((1 - dx) * (dy) * (dt))
            + orient[xF + 1, yF, t1] * ((dx) * (1 - dy) * (1 - dt))
            + orient[xF + 1, yF, t2] * ((dx) * (1 - dy) * (dt))
            + orient[xF + 1, yF + 1, t1] * ((dx) * (dy) * (1 - dt))
            + orient[xF + 1, yF + 1, t2] * ((dx) * (dy) * (dt))
        )
    
        return int_vals
    
    
    def set_intensity(
        self, 
        orient, 
        xy_t_int
    ):
        # utility function to set flowline intensites
    
        xF = np.floor(xy_t_int[:, 0]).astype("int")
        yF = np.floor(xy_t_int[:, 1]).astype("int")
        tF = np.floor(xy_t_int[:, 2]).astype("int")
        dx = xy_t_int[:, 0] - xF
        dy = xy_t_int[:, 1] - yF
        dt = xy_t_int[:, 2] - tF
    
        inds_1D = np.ravel_multi_index(
            [xF, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            1 - dy
        ) * (1 - dt)
        inds_1D = np.ravel_multi_index(
            [xF, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            1 - dy
        ) * (dt)
        inds_1D = np.ravel_multi_index(
            [xF, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            dy
        ) * (1 - dt)
        inds_1D = np.ravel_multi_index(
            [xF, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            dy
        ) * (dt)
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
            1 - dy
        ) * (1 - dt)
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
            1 - dy
        ) * (dt)
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
            1 - dt
        )
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
            dt
        )
    
        return orient

    def interactive_probe_selector(self, probe_map=None, figsize=(14, 8), cmap='viridis'):
        """
        Interactive GUI to select probe positions and view diffraction patterns.
        
        Parameters
        ----------
        probe_map : ndarray, optional
            2D array to display as the probe position map. If None, uses mean diffraction intensity.
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap for the probe map
            
        Returns
        -------
        selected_positions : list of tuples
            List of (ry, rx) coordinates of selected positions
        """
        from matplotlib.widgets import Button
        from matplotlib.patches import Circle
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        Ry, Rx = self.dataset_cartesian.shape[:2]
        
        # Create default probe map if not provided
        if probe_map is None:
            probe_map = np.mean(self.dataset_cartesian.array, axis=(2, 3))
        
        # Storage for selected positions
        selected_positions = []
        markers = []
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[1, 1], 
                              hspace=0.3, wspace=0.3)
        
        # Probe map axis
        ax_probe = fig.add_subplot(gs[:, 0])
        im_probe = ax_probe.imshow(probe_map, cmap=cmap, origin='lower', 
                                    interpolation='nearest', aspect='auto')
        ax_probe.set_title('Probe Position Map\n(Click to add point)', fontsize=12)
        ax_probe.set_xlabel('Rx')
        ax_probe.set_ylabel('Ry')
        plt.colorbar(im_probe, ax=ax_probe, label='Intensity')
        
        # Diffraction pattern axes
        ax_dp1 = fig.add_subplot(gs[0, 1])
        ax_dp2 = fig.add_subplot(gs[1, 1])
        ax_dp1.set_title('Diffraction Pattern 1')
        ax_dp2.set_title('Diffraction Pattern 2')
        ax_dp1.axis('off')
        ax_dp2.axis('off')
        
        # Text area for position list
        ax_list = fig.add_subplot(gs[:, 2])
        ax_list.axis('off')
        ax_list.set_title('Selected Positions', fontsize=11, fontweight='bold')
        
        # Add clear all button
        ax_button = plt.axes([0.7, 0.02, 0.1, 0.04])
        btn_clear = Button(ax_button, 'Clear All')
        
        def update_display():
            """Update the position list and diffraction patterns."""
            # Clear position list
            ax_list.clear()
            ax_list.axis('off')
            ax_list.set_title('Selected Positions', fontsize=11, fontweight='bold')
            
            # Display positions
            y_pos = 0.95
            for idx, (ry, rx) in enumerate(selected_positions):
                text = f"{idx+1}. ({ry}, {rx})"
                ax_list.text(0.1, y_pos, text, fontsize=10, transform=ax_list.transAxes,
                            verticalalignment='top')
                y_pos -= 0.08
            
            # Update diffraction patterns
            if len(selected_positions) >= 1:
                ry, rx = selected_positions[-1]
                dp = self.dataset_cartesian[ry, rx].array
                ax_dp1.clear()
                ax_dp1.imshow(dp, cmap='gray')
                ax_dp1.set_title(f'DP at ({ry}, {rx})')
                ax_dp1.axis('off')
            
            if len(selected_positions) >= 2:
                ry, rx = selected_positions[-2]
                dp = self.dataset_cartesian[ry, rx].array
                ax_dp2.clear()
                ax_dp2.imshow(dp, cmap='gray')
                ax_dp2.set_title(f'DP at ({ry}, {rx})')
                ax_dp2.axis('off')
            
            fig.canvas.draw_idle()
        
        def onclick(event):
            """Handle click events on probe map."""
            if event.inaxes == ax_probe and event.button == 1:  # Left click
                rx = int(np.round(event.xdata))
                ry = int(np.round(event.ydata))
                
                # Check bounds
                if 0 <= ry < Ry and 0 <= rx < Rx:
                    selected_positions.append((ry, rx))
                    
                    # Add marker
                    marker = Circle((rx, ry), radius=0.5, color='red', 
                                   fill=True, zorder=10)
                    ax_probe.add_patch(marker)
                    markers.append(marker)
                    
                    # Add label
                    label = ax_probe.text(rx, ry, str(len(selected_positions)), 
                                         color='white', fontsize=8, ha='center', 
                                         va='center', fontweight='bold', zorder=11)
                    markers.append(label)
                    
                    update_display()
        
        def clear_all(event):
            """Clear all selected positions."""
            selected_positions.clear()
            for marker in markers:
                marker.remove()
            markers.clear()
            ax_dp1.clear()
            ax_dp1.axis('off')
            ax_dp2.clear()
            ax_dp2.axis('off')
            update_display()
        
        # Connect events
        fig.canvas.mpl_connect('button_press_event', onclick)
        btn_clear.on_clicked(clear_all)
        
        plt.show()
        
        return selected_positions
    
    
    def visualize_selected_patterns(self, positions, ncols=4, figsize_per_pattern=(3, 3), 
                                     cmap='gray', vmax=None):
        """
        Display diffraction patterns at selected probe positions in a grid.
        
        Parameters
        ----------
        positions : list of tuples
            List of (ry, rx) coordinates
        ncols : int
            Number of columns in the grid
        figsize_per_pattern : tuple
            Size of each subplot (width, height)
        cmap : str
            Colormap for diffraction patterns
        vmax : float, optional
            Maximum value for colormap normalization
            
        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        n_positions = len(positions)
        nrows = int(np.ceil(n_positions / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(figsize_per_pattern[0]*ncols, 
                                        figsize_per_pattern[1]*nrows))
        
        # Handle single subplot case
        if n_positions == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (ry, rx) in enumerate(positions):
            dp = self.dataset_cartesian[ry, rx].array
            
            im = axes[idx].imshow(dp, cmap=cmap, vmax=vmax)
            axes[idx].set_title(f'({ry}, {rx})', fontsize=10)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(n_positions, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    
    def interactive_probe_selector_widget(self, probe_map=None, cmap='viridis'):
        """
        Enhanced interactive GUI using ipywidgets for fine-tuning positions.
        
        Parameters
        ----------
        probe_map : ndarray, optional
            2D array to display as the probe position map
        cmap : str
            Colormap for the probe map
            
        Returns
        -------
        selected_positions : list of tuples
            List of (ry, rx) coordinates of selected positions
        """
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        Ry, Rx = self.dataset_cartesian.shape[:2]
        
        # Create default probe map if not provided
        if probe_map is None:
            probe_map = np.mean(self.dataset_cartesian.array, axis=(2, 3))
        
        # Storage
        selected_positions = []
        
        # Create output widgets
        output_plot = widgets.Output()
        output_list = widgets.Output()
        
        def update_plot():
            """Update the main plot with markers."""
            with output_plot:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                
                im = ax.imshow(probe_map, cmap=cmap, origin='lower', 
                              interpolation='nearest', aspect='auto')
                ax.set_title('Probe Position Map (Click to add point)', fontsize=12)
                ax.set_xlabel('Rx')
                ax.set_ylabel('Ry')
                plt.colorbar(im, ax=ax, label='Intensity')
                
                # Add markers
                for idx, (ry, rx) in enumerate(selected_positions):
                    circle = Circle((rx, ry), radius=0.5, color='red', 
                                   fill=True, zorder=10)
                    ax.add_patch(circle)
                    ax.text(rx, ry, str(idx+1), color='white', fontsize=8, 
                           ha='center', va='center', fontweight='bold', zorder=11)
                
                def onclick(event):
                    if event.inaxes == ax and event.button == 1:
                        rx = int(np.round(event.xdata))
                        ry = int(np.round(event.ydata))
                        if 0 <= ry < Ry and 0 <= rx < Rx:
                            selected_positions.append((ry, rx))
                            update_plot()
                            update_list()
                
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
        
        def update_list():
            """Update the position list with controls."""
            with output_list:
                clear_output(wait=True)
                
                if not selected_positions:
                    print("No positions selected")
                    return
                
                for idx, (ry, rx) in enumerate(selected_positions):
                    print(f"--- Position {idx+1} ---")
                    
                    # Create sliders for fine-tuning
                    ry_slider = widgets.IntSlider(
                        value=ry, min=0, max=Ry-1, step=1,
                        description=f'Ry {idx+1}:', continuous_update=False
                    )
                    rx_slider = widgets.IntSlider(
                        value=rx, min=0, max=Rx-1, step=1,
                        description=f'Rx {idx+1}:', continuous_update=False
                    )
                    
                    def make_update(i):
                        def update_position(change):
                            selected_positions[i] = (ry_slider.value, rx_slider.value)
                            update_plot()
                        return update_position
                    
                    ry_slider.observe(make_update(idx), names='value')
                    rx_slider.observe(make_update(idx), names='value')
                    
                    # Delete button
                    delete_btn = widgets.Button(description=f'Delete {idx+1}', 
                                                button_style='danger')
                    
                    def make_delete(i):
                        def delete_position(b):
                            del selected_positions[i]
                            update_plot()
                            update_list()
                        return delete_position
                    
                    delete_btn.on_click(make_delete(idx))
                    
                    display(widgets.HBox([ry_slider, rx_slider, delete_btn]))
                
                # Clear all button
                clear_btn = widgets.Button(description='Clear All', button_style='warning')
                def clear_all(b):
                    selected_positions.clear()
                    update_plot()
                    update_list()
                clear_btn.on_click(clear_all)
                
                display(clear_btn)
        
        # Layout
        ui = widgets.VBox([
            widgets.HBox([output_plot, output_list])
        ])
        
        display(ui)
        update_plot()
        update_list()
        
        return selected_positions