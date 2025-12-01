import numpy as np
from tqdm import tqdm

from quantem.core.datastructures import Vector

def polar_transform_vector(
    cartesian_vector: Vector,
    centers: np.ndarray,
    x_field: str = "x",
    y_field: str = "y",
    r_unit: str = "Pixels",
    theta_unit: str = "Radians",
    name_suffix: str = "_polar",
    use_tqdm: bool = True,
) -> Vector:
    """
    Transform a Vector with Cartesian coordinates to polar coordinates.
    
    Parameters
    ----------
    cartesian_vector : Vector
        Input vector with Cartesian coordinates. Must have at least 2 fields
        for x and y coordinates.
    centers : np.ndarray
        Array of center coordinates with shape matching cartesian_vector.shape + (2,).
        Last dimension should be [y, x] coordinates of the center for each position.
    x_field : str, optional
        Name of the field containing x coordinates (default: "x")
    y_field : str, optional
        Name of the field containing y coordinates (default: "y")
    r_unit : str, optional
        Unit for the radial coordinate (default: "Pixels")
    theta_unit : str, optional
        Unit for the angular coordinate (default: "Radians")
    name_suffix : str, optional
        Suffix to append to the vector name (default: "_polar")
    use_tqdm : bool, optional
        Whether to show progress bar (default: True)
    
    Returns
    -------
    Vector
        New vector with polar coordinates (r, theta) and any additional fields
        from the original vector (excluding x and y).
    
    Examples
    --------
    >>> # Simple 2-field vector (only x, y)
    >>> peaks = Vector.from_shape(shape=(10, 10), fields=["x", "y"])
    >>> centers = np.random.rand(10, 10, 2) * 100
    >>> polar_peaks = polar_transform_vector(peaks, centers)
    
    >>> # Vector with additional fields (x, y, intensity, quality)
    >>> peaks = Vector.from_shape(
    ...     shape=(10, 10), 
    ...     fields=["x", "y", "intensity", "quality"]
    ... )
    >>> polar_peaks = polar_transform_vector(
    ...     peaks, 
    ...     centers,
    ...     x_field="x",
    ...     y_field="y"
    ... )
    >>> # Result has fields: ["r", "theta", "intensity", "quality"]
    
    Notes
    -----
    - The theta angle is measured counter-clockwise from the positive x-axis
    - Theta is normalized to the range [0, 2π)
    - Additional fields (beyond x, y) are preserved in the output vector
    - Empty cells (None or zero-length arrays) are handled gracefully
    """
    # Unwrap if vector is wrapped in numpy array
    if isinstance(cartesian_vector, np.ndarray) and cartesian_vector.dtype == object:
        cartesian_vector = cartesian_vector.item()
    
    # Validate inputs
    if not isinstance(cartesian_vector, Vector):
        raise TypeError(f"Expected Vector, got {type(cartesian_vector)}")
    
    if x_field not in cartesian_vector.fields:
        raise ValueError(f"Field '{x_field}' not found in vector. Available: {cartesian_vector.fields}")
    if y_field not in cartesian_vector.fields:
        raise ValueError(f"Field '{y_field}' not found in vector. Available: {cartesian_vector.fields}")
    
    # Get vector shape
    N, M = cartesian_vector.shape
    
    # Validate centers shape
    expected_centers_shape = (N, M, 2)
    if centers.shape != expected_centers_shape:
        raise ValueError(
            f"Centers shape {centers.shape} doesn't match expected {expected_centers_shape}"
        )
    
    # Determine which fields to keep (all except x and y)
    x_idx = cartesian_vector.fields.index(x_field)
    y_idx = cartesian_vector.fields.index(y_field)
    
    # Get indices and names of additional fields to preserve
    additional_fields = []
    additional_units = []
    additional_indices = []
    
    for idx, (field, unit) in enumerate(zip(cartesian_vector.fields, cartesian_vector.units)):
        if idx not in (x_idx, y_idx):
            additional_fields.append(field)
            additional_units.append(unit)
            additional_indices.append(idx)
    
    # Create output vector with r, theta, and any additional fields
    output_fields = ["r", "theta"] + additional_fields
    output_units = [r_unit, theta_unit] + additional_units
    
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
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Normalize theta to [0, 2π)
            theta = np.mod(theta + np.pi, 2 * np.pi)
            
            # Build output array: [r, theta, additional_field1, additional_field2, ...]
            polar_coords = np.column_stack([r, theta])
            
            # Add any additional fields
            if additional_indices:
                additional_data = cartesian_data[:, additional_indices]
                polar_coords = np.column_stack([polar_coords, additional_data])
            
            polar_vector.set_data(polar_coords, i, j)
    
    return polar_vector


def cartesian_transform_vector(
    polar_vector: Vector,
    centers: np.ndarray,
    r_field: str = "r",
    theta_field: str = "theta",
    x_unit: str = "Pixels",
    y_unit: str = "Pixels",
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