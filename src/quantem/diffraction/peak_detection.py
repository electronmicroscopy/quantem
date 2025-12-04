import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter, grey_dilation, map_coordinates

class DoGBlobDetector:
    def __init__(self, min_sigma: float = 1.0, max_sigma: float = 50.0, 
                 num_sigma: int = 10, threshold: float = 0.01, 
                 overlap: float = 0.5, device: str = 'cuda'):
        """
        Difference of Gaussians blob detector in PyTorch
        
        Args:
            min_sigma: Minimum standard deviation for Gaussian kernel
            max_sigma: Maximum standard deviation for Gaussian kernel
            num_sigma: Number of intermediate scales
            threshold: Minimum intensity of blobs
            overlap: Maximum overlap between blobs (for non-max suppression)
            device: 'cuda' or 'cpu'
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.device = device
        
        # Generate sigma values (logarithmic spacing)
        self.sigmas = np.logspace(
            np.log10(min_sigma), 
            np.log10(max_sigma), 
            num_sigma
        )
        
    def gaussian_kernel(self, sigma: float, kernel_size: int = None) -> torch.Tensor:
        """Create a 2D Gaussian kernel"""
        if kernel_size is None:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        
        # Create coordinate grids
        x = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        y = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Gaussian formula
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def apply_gaussian(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to image"""
        kernel = self.gaussian_kernel(sigma)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        
        # Add batch and channel dimensions if needed
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Apply convolution with padding
        pad = kernel.shape[-1] // 2
        blurred = F.conv2d(image, kernel, padding=pad)
        
        return blurred.squeeze()
    
    def compute_dog_pyramid(self, image: torch.Tensor) -> torch.Tensor:
        """Compute Difference of Gaussians at multiple scales"""
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        image = image.to(self.device)
        
        dog_pyramid = []
        
        for i in range(len(self.sigmas) - 1):
            sigma1 = self.sigmas[i]
            sigma2 = self.sigmas[i + 1]
            
            # Apply Gaussians
            gauss1 = self.apply_gaussian(image, sigma1)
            gauss2 = self.apply_gaussian(image, sigma2)
            
            # Difference of Gaussians (normalized by sigma for scale invariance)
            dog = (gauss2 - gauss1) * sigma1
            dog_pyramid.append(dog)
        
        # Stack into 3D tensor [scales, height, width]
        return torch.stack(dog_pyramid)
    
    def find_local_maxima(self, dog_pyramid: torch.Tensor) -> torch.Tensor:
        """Find local maxima in 3D space (scale + spatial)"""
        # Add batch and channel dimensions
        dog_3d = dog_pyramid.unsqueeze(0).unsqueeze(0)
        
        # 3D max pooling with kernel size 3
        maxpool = F.max_pool3d(dog_3d, kernel_size=3, stride=1, padding=1)
        
        # Find where original equals maxpool (local maxima)
        is_maxima = (dog_3d == maxpool) & (dog_3d > self.threshold)
        
        return is_maxima.squeeze()
    
    def detect_blobs(self, image: torch.Tensor) -> np.ndarray:
        """
        Detect blobs in image
        
        Returns:
            Array of shape (n_blobs, 3) containing (y, x, radius) for each blob
        """
        # Compute DoG pyramid
        dog_pyramid = self.compute_dog_pyramid(image)
        
        # Find local maxima
        is_maxima = self.find_local_maxima(dog_pyramid)
        
        # Get coordinates of maxima
        coords = torch.nonzero(is_maxima, as_tuple=False)
        
        if len(coords) == 0:
            return np.array([]).reshape(0, 3)
        
        # Convert to numpy
        coords = coords.cpu().numpy()
        
        # Get blob properties: (y, x, sigma)
        blobs = []
        for scale_idx, y, x in coords:
            sigma = self.sigmas[scale_idx]
            radius = sigma * np.sqrt(2)  # Convert sigma to radius
            intensity = dog_pyramid[scale_idx, y, x].item()
            blobs.append([y, x, radius, intensity])
        
        blobs = np.array(blobs)
        
        # Non-maximum suppression
        blobs = self._prune_blobs(blobs)
        
        return blobs[:, :3]  # Return only (y, x, radius)
    
    def _prune_blobs(self, blobs: np.ndarray) -> np.ndarray:
        """Remove overlapping blobs using non-maximum suppression"""
        if len(blobs) == 0:
            return blobs
        
        # Sort by intensity (descending)
        blobs = blobs[blobs[:, 3].argsort()[::-1]]
        
        keep = []
        for i, blob in enumerate(blobs):
            # Check overlap with already kept blobs
            overlaps = False
            for kept_blob in keep:
                dist = np.sqrt((blob[0] - kept_blob[0])**2 + 
                              (blob[1] - kept_blob[1])**2)
                if dist < self.overlap * (blob[2] + kept_blob[2]):
                    overlaps = True
                    break
            
            if not overlaps:
                keep.append(blob)
        
        return np.array(keep)


def visualize_blobs(image: np.ndarray, blobs: np.ndarray):
    """Visualize detected blobs"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    
    for y, x, r in blobs:
        circle = Circle((x, y), r, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
    
    ax.set_title(f'Detected {len(blobs)} blobs')
    plt.show()

def detect_blobs(image, sigma=1.0, threshold=None):
    """
    Detect strict local maxima (greater than 8 nearest neighbors) with subpixel quadratic refinement.
    
    Parameters:
    -----------
    image : 2D array
    sigma : float, for Gaussian smoothing
    threshold : float or None, minimum intensity for peak to be valid
    
    Returns:
    --------
    peaks : Nx2 array of (row, col) subpixel coordinates
    intensities : N array of signal intensities for peak position
    success : N array of booleans (True if refinement succeeded)
    """
    
    smoothed = gaussian_filter(image, sigma=sigma)
    local_max = maximum_filter(smoothed, size=3)
    # Make strict: exclude plateaus by checking inequality with neighbors
    # Use erosion to get image of maximum value in kernel convolution.
    # Used to check if strictly greater than nearest 8
    # Footprint to exclude center pixel. Evaluates nearest 8.
    footprint = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=bool)
    max_neighbors = grey_dilation(smoothed, footprint=footprint)
    peaks = (smoothed == local_max) & (smoothed > max_neighbors)
    
    # Remove borders and apply threshold
    peaks[:, 0] = peaks[:, -1] = peaks[0, :] = peaks[-1, :] = False
    if threshold is not None:
        peaks &= (smoothed > threshold)
    
    # Get integer coordinates
    peak_coords = np.argwhere(peaks)
    # If no peaks, return empty lists
    if len(peak_coords) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Subpixel refinement
    refined_coords, success = refine_peaks_quadratic(smoothed, peak_coords)
    # Get intensities of peak position signal
    intensities = map_coordinates(smoothed, refined_coords.T, order=1)
    
    return refined_coords, intensities, success

def refine_peaks_quadratic(smoothed, peak_coords):
    """
    Refine peak positions to subpixel accuracy using 2D quadratic fitting.
    
    Parameters:
    -----------
    smoothed : 2D array, image after Gaussian smoothing
    peak_coords : Nx2 array of (row, col) integer peak positions
    
    Returns:
    --------
    refined_coords : Nx2 array of (row, col) subpixel peak positions
    success : N array of booleans, True if refinement succeeded
    """
    refined = []
    success = []
    
    for y, x in peak_coords:
        # Skip peaks too close to border (need 3x3 neighborhood)
        if y < 1 or y >= smoothed.shape[0]-1 or x < 1 or x >= smoothed.shape[1]-1:
            refined.append([float(y), float(x)])
            success.append(False)
            continue
        
        # Get 3x3 neighborhood
        patch = smoothed[y-1:y+2, x-1:x+2]
        
        # Taylor expansion around the peak:
        # f(x+dx, y+dy) ≈ f(x,y) + g·[dx,dy] + 0.5·[dx,dy]·H·[dx,dy]
        # where g is gradient (1st power) and H is Hessian (2nd power)
        
        # First derivatives (gradient) using central differences
        dy = (patch[2, 1] - patch[0, 1]) / 2.0
        dx = (patch[1, 2] - patch[1, 0]) / 2.0
        
        # Second derivatives (Hessian) using finite differences
        dyy = patch[2, 1] - 2*patch[1, 1] + patch[0, 1]
        dxx = patch[1, 2] - 2*patch[1, 1] + patch[1, 0]
        dxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4.0
        
        # Build Hessian matrix
        H = np.array([[dyy, dxy],
                      [dxy, dxx]])
        
        # Gradient vector
        g = np.array([dy, dx])
        
        # At the peak, gradient should be zero: g + H·offset = 0
        # So: offset = -H^(-1)·g
        try:
            # Check if Hessian is negative definite (proper maximum)
            eigenvalues = np.linalg.eigvalsh(H)
            if np.all(eigenvalues < 0):  # Both eigenvalues negative = local maximum
                offset = -np.linalg.solve(H, g)
                
                # Sanity check: offset shouldn't be too large
                # (if it is, the quadratic approximation is probably bad and should just use integer coords)
                if np.all(np.abs(offset) <= 1.5):
                    refined.append([y + offset[0], x + offset[1]])
                    success.append(True)
                else:
                    # Offset too large, use integer position, as more accurate
                    refined.append([float(y), float(x)])
                    success.append(False)
            else:
                # Not a proper maximum (saddle point or minimum)
                refined.append([float(y), float(x)])
                success.append(False)
                
        except np.linalg.LinAlgError:
            # Singular matrix (flat region), use integer position
            refined.append([float(y), float(x)])
            success.append(False)
    
    return np.array(refined), np.array(success)

def pair_peaks(peaks_experimental, peaks_reference, radius_max):
    """
    Pair experimental Bragg peaks with reference peaks.
    
    Parameters:
    - peaks_experimental: np.array, shape (n, 2) for n experimental peaks
    - peaks_reference: np.array, shape (m, 2) for m reference peaks
    - radius_max: float, maximum distance for a match
    
    Returns:
    - matches: list of tuples (exp_index, ref_index, distance)
    - unmatched_exp: list of indices of unmatched experimental peaks
    """
    # Create KD-Tree for efficient nearest neighbor search
    tree = cKDTree(peaks_reference)
    
    # Find nearest neighbors for all experimental peaks
    distances, indices = tree.query(peaks_experimental, distance_upper_bound=radius_max)
    
    matches = []
    unmatched_exp = []
    
    for exp_index, (dist, ref_index) in enumerate(zip(distances, indices)):
        if dist <= radius_max:
            matches.append((exp_index, ref_index, dist))
        else:
            unmatched_exp.append(exp_index)
    
    return matches, unmatched_exp

def angle_difference(angle1, angle2):
    """Calculate the smallest difference between two angles in degrees with ML model coordinate system."""
    return np.mod(angle1 - angle2 + 180, 360) - 180

def pair_peaks_polar(peaks_experimental, peaks_reference, radius_max, angle_max=180, central_radius_threshold=5, filter_central_beam=False):
    """
    Pair experimental Bragg peaks with reference peaks in polar coordinates.
    
    Parameters:
    - peaks_experimental: np.array, shape (n, 2) for n experimental peaks (r, theta in degrees)
    - peaks_reference: np.array, shape (m, 2) for m reference peaks (r, theta in degrees)
    - radius_max: float, maximum radial distance for a match
    - angle_max: float, maximum angular difference for a match (in degrees)
    - central_radius_threshold: float, radius below which angles are ignored for matching
    - filter_central_beam: bool, if True return central beam info
    
    Returns:
    - matches: list of tuples (exp_index, ref_index, distance, delta_r, delta_phi)
    - unmatched_exp: list of indices of unmatched experimental peaks
    - unmatched_ref: list of indices of unmatched reference peaks
    - central_beam_info_exp: dict with keys 'exp_index', 'match_index' (ref_index if matched), 'in_unmatched_exp'
    - central_beam_info_ref: dict with keys 'ref_index', 'match_index' (exp_index if matched), 'in_unmatched_ref'
    """
    matches = []
    unmatched_exp = list(range(len(peaks_experimental)))
    unmatched_ref = list(range(len(peaks_reference)))
    
    # Find the central beams (smallest radius in both experimental and reference peaks)
    central_beam_exp_index = np.argmin(peaks_experimental[:, 0])
    central_beam_ref_index = np.argmin(peaks_reference[:, 0])
    
    central_beam_info_exp = {
        'exp_index': central_beam_exp_index,
        'match_index': None,  # ref_index if matched
        'in_unmatched_exp': None,  # Index in unmatched_exp list if unmatched
    }
    central_beam_info_ref = {
        'ref_index': central_beam_ref_index,
        'match_index': None,  # exp_index if matched
        'in_unmatched_ref': None,  # Index in unmatched_ref list if unmatched
    }

    for ref_index in unmatched_ref.copy():
        ref_peak = peaks_reference[ref_index]
        best_match = None
        best_distance = float('inf')
        
        for exp_index in unmatched_exp.copy():
            exp_peak = peaks_experimental[exp_index]
            
            delta_r = exp_peak[0] - ref_peak[0]
            delta_phi = angle_difference(exp_peak[1], ref_peak[1])
            delta_x = exp_peak[0] * np.cos(exp_peak[1] * np.pi/180) - ref_peak[0] * np.cos(ref_peak[1] * np.pi/180)
            delta_y = exp_peak[0] * np.sin(exp_peak[1] * np.pi/180) - ref_peak[0] * np.sin(ref_peak[1] * np.pi/180)
            
            # Check if either peak is within the central radius threshold
            if exp_peak[0] <= central_radius_threshold or ref_peak[0] <= central_radius_threshold:
                # For central peaks, only consider radial distance
                distance = np.sqrt(delta_x**2 + delta_y**2)
            else:
                # Use a combination of radial and angular difference for matching
                distance = np.sqrt(delta_x**2 + delta_y**2)
            
            if distance < radius_max and distance < best_distance and np.abs(delta_phi) < angle_max:
                best_match = (exp_index, ref_index, distance, delta_r, delta_phi, delta_x, delta_y)
                best_distance = distance
        
        if best_match:
            # Check if this match involves the experimental central beam
            if best_match[0] == central_beam_exp_index:
                central_beam_info_exp['match_index'] = best_match[1]  # Store the ref_index
            
            # Check if this match involves the reference central beam
            if best_match[1] == central_beam_ref_index:
                central_beam_info_ref['match_index'] = best_match[0]  # Store the exp_index
            
            matches.append(best_match)
            unmatched_exp.remove(best_match[0])
            unmatched_ref.remove(best_match[1])
    
    # Update central beam info for unmatched cases
    if central_beam_exp_index in unmatched_exp:
        central_beam_info_exp['in_unmatched_exp'] = unmatched_exp.index(central_beam_exp_index)
    
    if central_beam_ref_index in unmatched_ref:
        central_beam_info_ref['in_unmatched_ref'] = unmatched_ref.index(central_beam_ref_index)
    
    if filter_central_beam:
        return matches, unmatched_exp, unmatched_ref, central_beam_info_exp, central_beam_info_ref
    else:
        return matches, unmatched_exp, unmatched_ref

# def pair_peaks_polar(peaks_experimental, peaks_reference, radius_max, angle_max=180, central_radius_threshold=5, filter_central_beam=False):
#     """
#     Pair experimental Bragg peaks with reference peaks in polar coordinates.
    
#     Parameters:
#     - peaks_experimental: np.array, shape (n, 2) for n experimental peaks (r, theta in degrees)
#     - peaks_reference: np.array, shape (m, 2) for m reference peaks (r, theta in degrees)
#     - radius_max: float, maximum radial distance for a match
#     - angle_max: float, maximum angular difference for a match (in degrees)
#     - central_radius_threshold: float, radius below which angles are ignored for matching
    
#     Returns:
#     - matches: list of tuples (exp_index, ref_index, distance, delta_r, delta_phi)
#     - unmatched_exp: list of indices of unmatched experimental peaks
#     - unmatched_ref: list of indices of unmatched reference peaks
#     """
#     matches = []
#     unmatched_exp = list(range(len(peaks_experimental)))
#     unmatched_ref = list(range(len(peaks_reference)))

#     for ref_index in unmatched_ref.copy():
#         ref_peak = peaks_reference[ref_index]
#         best_match = None
#         best_distance = float('inf')
        
#         for exp_index in unmatched_exp.copy():
#             exp_peak = peaks_experimental[exp_index]
            
#             delta_r = exp_peak[0] - ref_peak[0]
#             delta_phi = angle_difference(exp_peak[1], ref_peak[1])
#             delta_x = exp_peak[0] * np.cos(exp_peak[1] * np.pi/180) - ref_peak[0] * np.cos(ref_peak[1] * np.pi/180)
#             delta_y = exp_peak[0] * np.sin(exp_peak[1] * np.pi/180) - ref_peak[0] * np.sin(ref_peak[1] * np.pi/180)
#             # Check if either peak is within the central radius threshold
#             if exp_peak[0] <= central_radius_threshold or ref_peak[0] <= central_radius_threshold:
#                 # For central peaks, only consider radial distance
#                 # distance = abs(delta_r)
#                 distance = np.sqrt(delta_x**2 + delta_y**2)
#             else:
#                 # Use a combination of radial and angular difference for matching
#                 distance = np.sqrt(delta_x**2 + delta_y**2)
#                 # distance = np.sqrt((delta_r / radius_max)**2 + (delta_phi / angle_max)**2)
            
#             if distance < radius_max and distance < best_distance and np.abs(delta_phi) < angle_max:  # '1' represents a normalized distance threshold
#                 best_match = (exp_index, ref_index, distance, delta_r, delta_phi, delta_x, delta_y)
#                 best_distance = distance
        
#         if best_match:
#             matches.append(best_match)
#             unmatched_exp.remove(best_match[0])
#             unmatched_ref.remove(best_match[1])

#     return matches, unmatched_exp, unmatched_ref
    