import numpy as np


def create_circle_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """
    Create a circular mask for virtual image formation.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the mask (rows, cols)
    center : tuple[float, float]
        Center coordinates (cy, cx) of the circle
    radius : float
        Radius of the circle

    Returns
    -------
    np.ndarray
        Boolean mask with True inside the circle
    """
    cy, cx = center
    y, x = np.ogrid[: shape[0], : shape[1]]

    # Calculate distance from center
    distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    return distance <= radius


def create_annular_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radii: tuple[float, float],
) -> np.ndarray:
    """
    Create an annular (ring-shaped) mask for virtual image formation.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the mask (rows, cols)
    center : tuple[float, float]
        Center coordinates (cy, cx) of the annulus
    radii : tuple[float, float]
        Inner and outer radii (r_inner, r_outer) of the annulus

    Returns
    -------
    np.ndarray
        Boolean mask with True inside the annular region
    """
    cy, cx = center
    r_inner, r_outer = radii
    y, x = np.ogrid[: shape[0], : shape[1]]

    # Calculate distance from center
    distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    return (distance >= r_inner) & (distance <= r_outer)
