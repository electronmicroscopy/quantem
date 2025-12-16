import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter, grey_dilation, map_coordinates
from quantem.core.datastructures import Vector


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


def get_peak_intensity_from_image(peak_coord, image, radius=2):
    """
    Get average intensity in a circular region around a peak.
    
    Parameters:
    -----------
    peak_coord : tuple or array
        Peak coordinate (y, x)
    image : ndarray
        Original diffraction pattern
    radius : int
        Radius in pixels for sampling region
    
    Returns:
    --------
    intensity : float
        Average intensity in the circular region
    """
    y, x = peak_coord
    h, w = image.shape
    
    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:h, :w]
    
    # Calculate distance from peak
    distances = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
    
    # Create circular mask
    mask = distances <= radius
    
    # Get average intensity in the circular region
    if np.sum(mask) > 0:
        intensity = np.mean(image[mask])
    else:
        # Fallback: just use the pixel value at the peak
        intensity = image[int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))]
    
    return intensity


def find_central_beam_from_peaks(peak_coords, peak_intensities, image_shape, 
                                 intensity_threshold=0.5, distance_weight=0.3,
                                 debug=False, image=None, sampling_radius=2, 
                                 vector_x_field=['x_pixels', 'x'], 
                                 vector_y_field=['y_pixels', 'y']):
    """
    Find central beam from detected peaks with debugging visualization.
    
    Parameters:
    -----------
    peak_coords : ndarray, shape (N, 2) or (N, 4), or Vector
        Peak coordinates (y, x) or Vector with coordinate fields
    peak_intensities : ndarray, shape (N,) or None
        Peak intensities from model (ignored if image is provided)
    image_shape : tuple
        Shape of image (H, W)
    intensity_threshold : float
        Minimum intensity to consider (0-1)
    distance_weight : float
        Weight for distance vs intensity (0=only intensity, 1=only distance)
    debug : bool
        Show debug plots and print info
    image : ndarray, optional
        Original diffraction pattern for intensity sampling
    sampling_radius : int
        Radius in pixels for sampling intensity around each peak
    vector_x_field : str or list of str
        Field name(s) for x-coordinates. Default: ['x_pixels', 'x']
    vector_y_field : str or list of str
        Field name(s) for y-coordinates. Default: ['y_pixels', 'y']
    
    Returns:
    --------
    center : tuple
        (y, x) coordinates of central beam
    """
    # Helper function to find first matching field
    def find_field(field_options, available_fields):
        fields = [field_options] if isinstance(field_options, str) else field_options
        return next((f for f in fields if f in available_fields), None)
    
    # Handle Vector input
    if isinstance(peak_coords, Vector):
        if debug:
            print(f"Vector fields: {peak_coords.fields}")
        
        x_field = find_field(vector_x_field, peak_coords.fields)
        y_field = find_field(vector_y_field, peak_coords.fields)
        
        if not (x_field and y_field):
            raise ValueError(
                f"Missing fields in Vector. Available: {peak_coords.fields}\n"
                f"Looking for x in {[vector_x_field] if isinstance(vector_x_field, str) else vector_x_field}, "
                f"y in {[vector_y_field] if isinstance(vector_y_field, str) else vector_y_field}"
            )
        
        if debug:
            print(f"Using x='{x_field}', y='{y_field}'")
        
        if len(peak_coords.data) > 0:
            y_idx = peak_coords.fields.index(y_field)
            x_idx = peak_coords.fields.index(x_field)
            peak_coords = np.column_stack([peak_coords.data[:, y_idx],
                                           peak_coords.data[:, x_idx]])
        else:
            peak_coords = np.empty((0, 2))
    # Handle ndarray input
    elif isinstance(peak_coords, np.ndarray):
        if peak_coords.ndim == 2 and peak_coords.shape[1] == 4:
            if debug:
                print("ndarray with 4 columns, using first 2 (y, x)")
            peak_coords = peak_coords[:, :2]
        elif peak_coords.ndim == 2 and peak_coords.shape[1] == 2:
            pass  # Already correct
        elif peak_coords.ndim == 1 and len(peak_coords) == 0:
            peak_coords = np.empty((0, 2))
        else:
            raise ValueError(f"Array must be (N, 2) or (N, 4), got {peak_coords.shape}")
    else:
        raise TypeError(f"peak_coords must be Vector or ndarray, got {type(peak_coords)}")
        
    # Check for empty peaks
    if len(peak_coords) == 0:
        if debug:
            print("⚠️ No peaks! Using image center.")
        return (image_shape[0] / 2, image_shape[1] / 2)

    # Image center
    center_y, center_x = image_shape[0] / 2, image_shape[1] / 2
    
    # Determine which intensities to use
    if image is not None:
        # Sample intensities from actual diffraction pattern
        sampled_intensities = np.array([
            get_peak_intensity_from_image(coord, image, radius=sampling_radius)
            for coord in peak_coords
        ])
        intensities_to_use = sampled_intensities
        intensity_source = f"Sampled from DP (radius={sampling_radius}px)"
    else:
        # Use model-predicted intensities
        intensities_to_use = peak_intensities
        intensity_source = "Model predictions"
    
    # Normalize intensities to [0, 1]
    max_intensity = np.max(intensities_to_use)
    if max_intensity > 0:
        intensities_norm = intensities_to_use / max_intensity
    else:
        intensities_norm = intensities_to_use
    
    if debug:
        print(f"\n{'='*60}")
        print(f"DEBUG: Central Beam Detection")
        print(f"{'='*60}")
        print(f"Number of peaks detected: {len(peak_coords)}")
        print(f"Image shape: {image_shape}")
        print(f"Image center: ({center_y:.1f}, {center_x:.1f})")
        print(f"Intensity source: {intensity_source}")
        print(f"Intensity threshold: {intensity_threshold}")
        print(f"Distance weight: {distance_weight}")
        print(f"\nAll peaks:")
        for i, (coord, intensity, intensity_norm) in enumerate(zip(peak_coords, intensities_to_use, intensities_norm)):
            print(f"  Peak {i}: coord=({coord[0]:.2f}, {coord[1]:.2f}), "
                  f"intensity={intensity:.4f}, normalized={intensity_norm:.4f}")
    
    # Filter by intensity threshold
    intensity_mask = intensities_norm > intensity_threshold
    num_above_threshold = np.sum(intensity_mask)
    
    if debug:
        print(f"\nPeaks above intensity threshold ({intensity_threshold}): {num_above_threshold}/{len(peak_coords)}")
    
    if num_above_threshold == 0:
        if debug:
            print("⚠️ No peaks above intensity threshold! Using all peaks.")
        intensity_mask = np.ones(len(intensities_norm), dtype=bool)
    
    filtered_coords = peak_coords[intensity_mask]
    filtered_intensities = intensities_to_use[intensity_mask]
    filtered_intensities_norm = intensities_norm[intensity_mask]
    
    if debug:
        print(f"\nFiltered peaks ({len(filtered_coords)}):")
        for i, (coord, intensity, intensity_norm) in enumerate(zip(filtered_coords, filtered_intensities, filtered_intensities_norm)):
            print(f"  Peak {i}: coord=({coord[0]:.2f}, {coord[1]:.2f}), "
                  f"intensity={intensity:.4f}, normalized={intensity_norm:.4f}")
    
    # Calculate distance from image center
    distances = np.sqrt(
        (filtered_coords[:, 0] - center_y)**2 + 
        (filtered_coords[:, 1] - center_x)**2
    )
    
    if debug:
        print(f"\nDistances from center:")
        for i, dist in enumerate(distances):
            print(f"  Peak {i}: {dist:.2f} pixels")
    
    # Normalize distances
    if np.max(distances) > 0:
        distances_norm = distances / np.max(distances)
    else:
        distances_norm = distances
    
    if debug:
        print(f"\nNormalized values:")
        print(f"  Distance range: [{np.min(distances_norm):.3f}, {np.max(distances_norm):.3f}]")
        print(f"  Intensity range: [{np.min(filtered_intensities_norm):.3f}, {np.max(filtered_intensities_norm):.3f}]")
    
    # Score: high intensity, low distance wins
    # Lower score is better
    scores = (1 - filtered_intensities_norm) * (1 - distance_weight) + distances_norm * distance_weight
    
    if debug:
        print(f"\nScores (lower is better):")
        print(f"  Formula: (1 - intensity_norm) * {1-distance_weight:.2f} + distance_norm * {distance_weight:.2f}")
        for i, score in enumerate(scores):
            print(f"  Peak {i}: score={score:.4f} "
                  f"[intensity_term={(1-filtered_intensities_norm[i])*(1-distance_weight):.4f}, "
                  f"distance_term={distances_norm[i]*distance_weight:.4f}]")
    
    # Pick peak with best score
    best_idx = np.argmin(scores)
    central_beam_coords = filtered_coords[best_idx]
    
    # Map back to original peak index for reference
    original_indices = np.where(intensity_mask)[0]
    original_best_idx = original_indices[best_idx]
    
    if debug:
        print(f"\n{'='*60}")
        print(f"SELECTED CENTRAL BEAM:")
        print(f"  Peak index (filtered): {best_idx}")
        print(f"  Peak index (original): {original_best_idx}")
        print(f"  Coordinates: ({central_beam_coords[0]:.2f}, {central_beam_coords[1]:.2f})")
        print(f"  Intensity: {filtered_intensities[best_idx]:.4f}")
        print(f"  Normalized intensity: {filtered_intensities_norm[best_idx]:.4f}")
        print(f"  Distance from center: {distances[best_idx]:.2f} pixels")
        print(f"  Score: {scores[best_idx]:.4f}")
        print(f"{'='*60}\n")
    
    # Visualization
    if debug and image is not None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Top-left: Diffraction pattern with sampling circles
        ax = axes[0, 0]
        ax.imshow(image, cmap='viridis')
        ax.axhline(center_y, color='white', linestyle='--', alpha=0.5, linewidth=1, label='Image center')
        ax.axvline(center_x, color='white', linestyle='--', alpha=0.5, linewidth=1)
        
        # Draw sampling circles for all peaks
        for i, coord in enumerate(peak_coords):
            circle = Circle((coord[1], coord[0]), sampling_radius, 
                          fill=False, edgecolor='cyan', linewidth=1, alpha=0.5)
            ax.add_patch(circle)
            ax.text(coord[1] + sampling_radius + 2, coord[0], f'{i}', 
                   color='cyan', fontsize=8, alpha=0.7)
        
        # Highlight filtered peaks
        for i, coord in enumerate(filtered_coords):
            circle = Circle((coord[1], coord[0]), sampling_radius, 
                          fill=False, edgecolor='yellow', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
        
        # Highlight selected central beam
        circle = Circle((central_beam_coords[1], central_beam_coords[0]), sampling_radius, 
                       fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(circle)
        ax.scatter(central_beam_coords[1], central_beam_coords[0], 
                  s=500, c='red', marker='*', 
                  edgecolors='yellow', linewidths=3, zorder=10, label='Selected central beam')
        
        ax.set_title(f'Diffraction Pattern with Sampling Circles (radius={sampling_radius}px)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend(loc='upper right', fontsize=9)
        
        # Top-right: Peaks colored by sampled intensity
        ax = axes[0, 1]
        ax.imshow(image, cmap='viridis', alpha=0.6)
        
        scatter = ax.scatter(filtered_coords[:, 1], filtered_coords[:, 0], 
                           c=filtered_intensities, s=300, cmap='hot', marker='o',
                           edgecolors='black', linewidths=2, label='Filtered peaks')
        
        ax.scatter(central_beam_coords[1], central_beam_coords[0], 
                  s=500, c='red', marker='*', 
                  edgecolors='yellow', linewidths=3, label='Selected central beam',
                  zorder=10)
        
        plt.colorbar(scatter, ax=ax, label='Sampled Intensity')
        ax.set_title('Peaks Colored by Sampled Intensity', fontsize=12, fontweight='bold')
        ax.legend()
        
        # Bottom-left: Score visualization
        ax = axes[1, 0]
        ax.imshow(image, cmap='viridis', alpha=0.6)
        
        scatter = ax.scatter(filtered_coords[:, 1], filtered_coords[:, 0], 
                           c=scores, s=300, cmap='RdYlGn_r', marker='o',
                           edgecolors='black', linewidths=2, 
                           vmin=0, vmax=1, label='Filtered peaks (by score)')
        
        ax.scatter(central_beam_coords[1], central_beam_coords[0], 
                  s=500, c='red', marker='*', 
                  edgecolors='yellow', linewidths=3, label='Selected central beam',
                  zorder=10)
        
        # Add score labels
        for i, (coord, score) in enumerate(zip(filtered_coords, scores)):
            ax.annotate(f'{i}\n{score:.2f}', 
                       xy=(coord[1], coord[0]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.colorbar(scatter, ax=ax, label='Score (lower = better)')
        ax.set_title('Peaks Colored by Score', fontsize=12, fontweight='bold')
        ax.legend()
        
        # Bottom-right: Score breakdown
        ax = axes[1, 1]
        
        x = np.arange(len(filtered_coords))
        width = 0.35
        
        intensity_component = (1 - filtered_intensities_norm) * (1 - distance_weight)
        distance_component = distances_norm * distance_weight
        
        bars1 = ax.bar(x - width/2, intensity_component, width, 
                      label=f'Intensity term (weight={1-distance_weight:.2f})', 
                      alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, distance_component, width, 
                      label=f'Distance term (weight={distance_weight:.2f})', 
                      alpha=0.8, color='coral')
        
        # Highlight selected peak
        bars1[best_idx].set_color('darkblue')
        bars1[best_idx].set_edgecolor('yellow')
        bars1[best_idx].set_linewidth(3)
        bars2[best_idx].set_color('darkred')
        bars2[best_idx].set_edgecolor('yellow')
        bars2[best_idx].set_linewidth(3)
        
        ax.set_xlabel('Peak Index (filtered)', fontsize=12)
        ax.set_ylabel('Score Component', fontsize=12)
        ax.set_title('Score Breakdown by Component', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add total score line
        ax.plot(x, scores, 'ko-', linewidth=2, markersize=8, 
               label='Total score', zorder=5)
        ax.scatter([best_idx], [scores[best_idx]], s=300, c='red', 
                  marker='*', edgecolors='yellow', linewidths=2, 
                  zorder=10, label='Selected')
        ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Additional info table
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        peak_info = []
        for i in range(len(filtered_coords)):
            peak_info.append({
                'Peak': i,
                'Y': f"{filtered_coords[i, 0]:.1f}",
                'X': f"{filtered_coords[i, 1]:.1f}",
                'Intensity': f"{filtered_intensities[i]:.4f}",
                'Norm Int': f"{filtered_intensities_norm[i]:.3f}",
                'Distance': f"{distances[i]:.1f}",
                'Score': f"{scores[i]:.4f}",
                'Selected': '★' if i == best_idx else ''
            })
        
        # Create table
        table_data = [[info[key] for key in ['Peak', 'Y', 'X', 'Intensity', 'Norm Int', 'Distance', 'Score', 'Selected']] 
                     for info in peak_info]
        
        table = ax.table(cellText=table_data,
                        colLabels=['Peak', 'Y', 'X', 'Intensity', 'Norm Int', 'Distance', 'Score', ''],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for j in range(8):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # Highlight selected row
        for i in range(len(peak_info)):
            if i == best_idx:
                for j in range(8):
                    table[(i+1, j)].set_facecolor('#ffff99')
                    table[(i+1, j)].set_text_props(weight='bold')
        
        ax.axis('off')
        title_text = f'Peak Summary ({intensity_source})\n'
        title_text += f'distance_weight={distance_weight}, intensity_threshold={intensity_threshold}'
        if image is not None:
            title_text += f', sampling_radius={sampling_radius}px'
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    return (float(central_beam_coords[0]), float(central_beam_coords[1]))