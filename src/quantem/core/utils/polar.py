import numpy as np
from tqdm import tqdm
from typing import Optional, Union, List

from quantem.core.datastructures import Vector


# TODO: Elliptical polar transform
# TODO: Add offset options, such as rotation. Change of basis? if rotated basis, then maintain rotation?
# TODO: Add as method to vector? if method change names more
# TODO: Inheritence of units from input vector
# For input vector, when load dataset, get vector units from dataset

# "cartesian_to_polar_vector"
def polar_transform_vector(
    cartesian_vector: Vector,
    centers: Optional[np.ndarray] = None,
    x_field: Union[str, List[str]] = ['x_pixels', 'x'],
    y_field: Union[str, List[str]] = ['y_pixels', 'y'],
    sampling_conversion_factor: Optional[float] = None,
    r_unit: str = "pixels",
    theta_unit: str = "radians",
    name_suffix: str = "_polar",
    use_tqdm: bool = True,
) -> Vector:
    """
    Transform a Vector with Cartesian coordinates to polar coordinates.
    Returns new Vector with x field and y field converted to r, theta, and r_invA.
    Preserves all other fields.
    
    Parameters
    ----------
    cartesian_vector : Vector
        Input vector with Cartesian coordinates. Must have at least 2 fields
        for x and y coordinates.
    centers : np.ndarray, optional
        Array of center coordinates with shape matching cartesian_vector.shape + (2,).
        Last dimension should be [y, x] coordinates of the center for each position.
        If None, uses (0, 0) for all centers.
    x_field : str or list of str, optional
        Field name(s) for x-coordinates. If list, tries each in order.
        Default: ['x_pixels', 'x']
    y_field : str or list of str, optional
        Field name(s) for y-coordinates. If list, tries each in order.
        Default: ['y_pixels', 'y']
    sampling_conversion_factor : float, optional
        Conversion factor from pixels to 1/Å. If None, will be set to 1.0.
    r_unit : str, optional
        Unit for the radial coordinate in pixels (default: "pixels")
    theta_unit : str, optional
        Unit for the angular coordinate (default: "radians")
    name_suffix : str, optional
        Suffix to append to the vector name (default: "_polar")
    use_tqdm : bool, optional
        Whether to show progress bar (default: True)
    
    Returns
    -------
    Vector
        New vector with polar coordinates (r_pixels, theta, r_invA) and any additional 
        fields from the original vector (excluding x and y).
        Fields: ["r_pixels", "theta", "r_invA", ...additional_fields]
    
    Examples
    --------
    >>> # Simple 2-field vector (only x, y)
    >>> peaks = Vector.from_shape(shape=(10, 10), fields=["x", "y"])
    >>> centers = np.random.rand(10, 10, 2) * 100
    >>> polar_peaks = polar_transform_vector(peaks, centers, sampling_conversion_factor=0.01)
    
    >>> # Vector with 4 fields (x_pixels, y_pixels, x_invA, y_invA)
    >>> peaks = Vector.from_shape(
    ...     shape=(10, 10), 
    ...     fields=["y_pixels", "x_pixels", "y_invA", "x_invA"]
    ... )
    >>> # Automatically uses x_pixels and y_pixels
    >>> polar_peaks = polar_transform_vector(peaks, centers, sampling_conversion_factor=0.01)
    
    Notes
    -----
    - The theta angle is measured counter-clockwise from the positive x-axis
    - Theta is normalized to the range [0, 2π)
    - r_pixels is the radial distance in pixels
    - r_invA is the radial distance in inverse angstroms (r_pixels * sampling_conversion_factor)
    - Additional fields (beyond x, y) are preserved in the output vector
    - Empty cells (None or zero-length arrays) are handled gracefully
    """
    from tqdm import tqdm
    
    # Helper function to find first matching field
    def find_field(field_options, available_fields):
        fields = [field_options] if isinstance(field_options, str) else field_options
        return next((f for f in fields if f in available_fields), None)
    
    # Unwrap if vector is wrapped in numpy array
    if isinstance(cartesian_vector, np.ndarray) and cartesian_vector.dtype == object:
        cartesian_vector = cartesian_vector.item()
    
    # Validate inputs
    if not isinstance(cartesian_vector, Vector):
        raise TypeError(f"Expected Vector, got {type(cartesian_vector)}")
    
    # Find matching fields
    x_field_found = find_field(x_field, cartesian_vector.fields)
    y_field_found = find_field(y_field, cartesian_vector.fields)
    
    if not (x_field_found and y_field_found):
        missing = []
        if x_field_found is None:
            missing.append(f"x field (tried: {[x_field] if isinstance(x_field, str) else x_field})")
        if y_field_found is None:
            missing.append(f"y field (tried: {[y_field] if isinstance(y_field, str) else y_field})")
        
        raise ValueError(
            f"Could not find required coordinate fields in Vector.\n"
            f"Missing: {', '.join(missing)}\n"
            f"Available fields: {cartesian_vector.fields}"
        )
    
    # Get vector shape
    N, M = cartesian_vector.shape
    
    # Handle default centers
    if centers is None:
        centers = np.zeros((N, M, 2))
    
    # Validate centers shape
    expected_centers_shape = (N, M, 2)
    if centers.shape != expected_centers_shape:
        raise ValueError(
            f"Centers shape {centers.shape} doesn't match expected {expected_centers_shape}"
        )
    
    # Set default sampling conversion factor if not provided
    if sampling_conversion_factor is None:
        sampling_conversion_factor = 1.0
    
    # Determine which fields to keep (all except x and y)
    x_idx = cartesian_vector.fields.index(x_field_found)
    y_idx = cartesian_vector.fields.index(y_field_found)
    
    # Get indices and names of additional fields to preserve
    additional_fields = []
    additional_units = []
    additional_indices = []
    
    for idx, (field, unit) in enumerate(zip(cartesian_vector.fields, cartesian_vector.units)):
        if idx not in (x_idx, y_idx):
            additional_fields.append(field)
            additional_units.append(unit)
            additional_indices.append(idx)
    
    # Create output vector with r_pixels, theta, r_invA, and any additional fields
    output_fields = ["r_pixels", "theta", "r_invA"] + additional_fields
    output_units = [r_unit, theta_unit, "1/Å"] + additional_units
    
    polar_vector = Vector.from_shape(
        shape=(N, M),
        fields=output_fields,
        units=output_units,
        name=cartesian_vector.name + name_suffix,
    )
    
    # Transform coordinates
    iterator = tqdm(range(N), disable=not use_tqdm, desc="Polar transform")
    
    for i in iterator:
        for j in range(M):
            center_y, center_x = centers[i, j]
            cartesian_data = cartesian_vector[i, j]
            
            # Handle empty cells
            if cartesian_data is None or len(cartesian_data) == 0:
                empty_data = np.zeros((0, len(output_fields)))
                polar_vector.set_data(empty_data, i, j)
                continue
            
            # Extract x, y coordinates
            x_coords = cartesian_data[:, x_idx]
            y_coords = cartesian_data[:, y_idx]
            
            # Calculate polar coordinates
            dx = x_coords - center_x
            dy = y_coords - center_y
            r_pixels = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Normalize theta to [0, 2π)
            theta = np.mod(theta, 2 * np.pi)
            
            # Calculate r in inverse angstroms
            r_invA = r_pixels * sampling_conversion_factor
            
            # Build output array: [r_pixels, theta, r_invA, additional_field1, ...]
            polar_coords = np.column_stack([r_pixels, theta, r_invA])
            
            # Add any additional fields
            if additional_indices:
                additional_data = cartesian_data[:, additional_indices]
                polar_coords = np.column_stack([polar_coords, additional_data])
            
            polar_vector.set_data(polar_coords, i, j)
    
    return polar_vector

# TODO: Round-trip test
# "polar_to_cartesian_vector"
def cartesian_transform_vector(
    polar_vector: Vector,
    centers: np.ndarray,
    r_field: str = "r",
    theta_field: str = "theta",
    x_unit: str = "pixels",
    y_unit: str = "pixels",
    name_suffix: str = "_cartesian",
    use_tqdm: bool = True,
) -> Vector:
    """
    Transform a Vector with polar coordinates back to Cartesian coordinates.
    
    Parameters
    ----------
    polar_vector : Vector
        Input vector with polar coordinates. Must have at least 2 fields
        for r and theta coordinates.
    centers : np.ndarray
        Array of center coordinates with shape matching polar_vector.shape + (2,).
        Last dimension should be [y, x] coordinates of the center for each position.
    r_field : str, optional
        Name of the field containing radial coordinates (default: "r")
    theta_field : str, optional
        Name of the field containing angular coordinates (default: "theta")
    x_unit : str, optional
        Unit for the x coordinate (default: "Pixels")
    y_unit : str, optional
        Unit for the y coordinate (default: "Pixels")
    name_suffix : str, optional
        Suffix to append to the vector name (default: "_cartesian")
    use_tqdm : bool, optional
        Whether to show progress bar (default: True)
    
    Returns
    -------
    Vector
        New vector with Cartesian coordinates (x, y) and any additional fields
        from the original vector (excluding r and theta).
    
    Examples
    --------
    >>> polar_peaks = Vector.from_shape(shape=(10, 10), fields=["r", "theta"])
    >>> centers = np.random.rand(10, 10, 2) * 100
    >>> cartesian_peaks = cartesian_transform_vector(polar_peaks, centers)
    
    Notes
    -----
    - Assumes theta is in radians and measured counter-clockwise from positive x-axis
    - Additional fields (beyond r, theta) are preserved in the output vector
    """
    # Unwrap if vector is wrapped in numpy array
    if isinstance(polar_vector, np.ndarray) and polar_vector.dtype == object:
        polar_vector = polar_vector.item()
    
    # Validate inputs
    if not isinstance(polar_vector, Vector):
        raise TypeError(f"Expected Vector, got {type(polar_vector)}")
    
    if r_field not in polar_vector.fields:
        raise ValueError(f"Field '{r_field}' not found in vector. Available: {polar_vector.fields}")
    if theta_field not in polar_vector.fields:
        raise ValueError(f"Field '{theta_field}' not found in vector. Available: {polar_vector.fields}")
    
    # Get vector shape
    N, M = polar_vector.shape
    
    # Validate centers shape
    expected_centers_shape = (N, M, 2)
    if centers.shape != expected_centers_shape:
        raise ValueError(
            f"Centers shape {centers.shape} doesn't match expected {expected_centers_shape}"
        )
    
    # Determine which fields to keep (all except r and theta)
    r_idx = polar_vector.fields.index(r_field)
    theta_idx = polar_vector.fields.index(theta_field)
    
    # Get indices and names of additional fields to preserve
    additional_fields = []
    additional_units = []
    additional_indices = []
    
    for idx, (field, unit) in enumerate(zip(polar_vector.fields, polar_vector.units)):
        if idx not in (r_idx, theta_idx):
            additional_fields.append(field)
            additional_units.append(unit)
            additional_indices.append(idx)
    
    # Create output vector with x, y, and any additional fields
    output_fields = ["x", "y"] + additional_fields
    output_units = [x_unit, y_unit] + additional_units
    
    cartesian_vector = Vector.from_shape(
        shape=(N, M),
        fields=output_fields,
        units=output_units,
        name=polar_vector.name + name_suffix,
    )
    
    # Transform coordinates
    iterator = tqdm(range(N), disable=not use_tqdm, desc="Cartesian transform")
    
    for i in iterator:
        for j in range(M):
            center_y, center_x = centers[i, j]
            polar_data = polar_vector[i, j]
            
            # Handle empty cells
            if polar_data is None or len(polar_data) == 0:
                empty_data = np.zeros((0, len(output_fields)))
                cartesian_vector.set_data(empty_data, i, j)
                continue
            
            # Extract r, theta coordinates
            r = polar_data[:, r_idx]
            theta = polar_data[:, theta_idx]
            
            # Calculate Cartesian coordinates
            # Note: theta is measured counter-clockwise from positive x-axis
            # We need to adjust back from the [0, 2π) normalization
            theta_adjusted = theta - np.pi
            x = center_x + r * np.cos(theta_adjusted)
            y = center_y + r * np.sin(theta_adjusted)
            
            # Build output array: [x, y, additional_field1, additional_field2, ...]
            cartesian_coords = np.column_stack([x, y])
            
            # Add any additional fields
            if additional_indices:
                additional_data = polar_data[:, additional_indices]
                cartesian_coords = np.column_stack([cartesian_coords, additional_data])
            
            cartesian_vector.set_data(cartesian_coords, i, j)
    
    return cartesian_vector