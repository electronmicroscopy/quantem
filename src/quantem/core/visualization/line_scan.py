from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
from matplotlib.patches import Polygon

from quantem.core.visualization import show_2d


# TODO update sampling to allow for 3D and to plot with appropriate units along xy/z
def linescan(
    image: np.ndarray,
    center: tuple[int, int] | None = None,
    phi: float = 0,
    linewidth: int = 1,
    line_len: int = -1,
    show: bool = False,
    sampling: tuple[float, float] | np.ndarray | None = None,
    sampling_units: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, tuple[Any, Any]]:
    """
    Generate a line scan through an image.

    Args:
        image (np.ndarray): 2D or 3D image array. For 3D arrays, shape should be (depth, height, width).
        center (tuple, optional): Center point for the line scan (cy, cx). Defaults to image center.
        phi (float): Angle for the line scan in degrees.
        linewidth (int): Line width for the scan.
        line_len (int): Length of the line scan. If -1, uses full intersection with image bounds.
        show (bool): Whether to display the plot.
        **kwargs: Additional keyword arguments for plotting.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (positions, profile) where:
            - positions: 1D array of positions along the line, units are pixels if sampling is None
             otherwise units are the same as the sampling
            - profile: 1D array (for 2D input) or 2D array (for 3D input) of intensity values
    """
    image = np.asarray(image)

    if image.ndim == 2:
        scan_image = image
        image_shape = image.shape
    elif image.ndim == 3:
        # For 3D input, we'll scan along each depth slice
        scan_image = np.mean(image, axis=0)  # Use mean for visualization
        image_shape = image.shape[1:]  # (height, width)
    else:
        raise ValueError(f"Input image must be 2D or 3D, got {image.ndim}D")

    height, width = image_shape

    if center is None:
        center = (height // 2, width // 2)

    cy, cx = int(round(center[0])), int(round(center[1]))

    # Calculate start and end points
    start_point, end_point = _calculate_line_endpoints(image_shape, (cy, cx), phi, line_len)

    if image.ndim == 2:
        # For 2D images, use the original approach
        profile = skimage.measure.profile_line(
            scan_image,
            start_point,
            end_point,
            linewidth=linewidth,
            mode="constant",
            reduce_func=np.mean,
        )
    else:
        # For 3D images, extract profile from each depth slice
        profiles = []
        for depth_slice in image:
            slice_profile = skimage.measure.profile_line(
                depth_slice,
                start_point,
                end_point,
                linewidth=linewidth,
                mode="constant",
                reduce_func=np.mean,
            )
            profiles.append(slice_profile)
        profile = np.array(profiles)  # Shape: (depth, line_length)

    # Trim profile if line_len is specified
    if line_len > 0 and profile.shape[-1] > line_len:
        excess = profile.shape[-1] - line_len
        start_idx = excess // 2
        end_idx = start_idx + line_len
        profile = profile[..., start_idx:end_idx]

    # Create position array
    positions = np.arange(profile.shape[-1])
    if sampling is not None:
        if isinstance(sampling, float | int):
            s = sampling
        elif len(sampling) == 1:
            s = sampling[0]
        elif len(sampling) == 2:
            if sampling[0] != sampling[1]:
                raise ValueError("Sampling must be uniform across dimensions")
            s = sampling[0]
        else:
            raise ValueError(f"Sampling must be a single value or of len 2, got: {sampling}")
        positions = positions * s
        sampling_units = "A" if sampling_units is None else sampling_units
    else:
        sampling_units = "pixels"

    if show:
        fig, axs = _show_linescan(
            scan_image,
            profile,
            positions,
            start_point,
            end_point,
            linewidth,
            sampling_units,
            **kwargs,
        )
    else:
        fig, axs = None, None

    if kwargs.get("return_fig", False) and show:
        return positions, profile, (fig, axs)
    else:
        return positions, profile


def _calculate_line_endpoints(
    image_shape: tuple[int, int], center: tuple[int, int], phi: float, line_len: int = -1
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Calculate the start and end points of a line within image boundaries.

    Args:
        image_shape (tuple): Image dimensions (height, width).
        center (tuple): Center point (cy, cx).
        phi (float): Angle in degrees.
        line_len (int): Desired line length. If -1, extends to image boundaries.

    Returns:
        Tuple of start and end points: ((y1, x1), (y2, x2)).
    """
    height, width = image_shape
    cy, cx = center
    phi_rad = np.deg2rad(phi)

    if line_len > 0:
        # Use specified line length
        half_len = line_len / 2
        dy = np.sin(phi_rad) * half_len
        dx = np.cos(phi_rad) * half_len

        start_point = (cy - dy, cx - dx)
        end_point = (cy + dy, cx + dx)
    else:
        # Extend to image boundaries
        start_point, end_point = _find_boundary_intersections(image_shape, center, phi_rad)

    start_point = tuple(np.clip(start_point, 0, np.array(image_shape) - 1))
    end_point = tuple(np.clip(end_point, 0, np.array(image_shape) - 1))
    return start_point, end_point


def _find_boundary_intersections(
    image_shape: tuple[int, int], center: tuple[int, int], phi_rad: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find intersections of line with image boundaries."""
    height, width = image_shape
    cy, cx = center

    # Direction vector
    dx = np.cos(phi_rad)
    dy = np.sin(phi_rad)

    # Find intersections with all four boundaries
    intersections = []

    # Left boundary (x = 0)
    if dx != 0:
        t = -cx / dx
        y = cy + t * dy
        if 0 <= y < height:
            intersections.append((y, 0))

    # Right boundary (x = width - 1)
    if dx != 0:
        t = (width - 1 - cx) / dx
        y = cy + t * dy
        if 0 <= y < height:
            intersections.append((y, width - 1))

    # Top boundary (y = 0)
    if dy != 0:
        t = -cy / dy
        x = cx + t * dx
        if 0 <= x < width:
            intersections.append((0, x))

    # Bottom boundary (y = height - 1)
    if dy != 0:
        t = (height - 1 - cy) / dy
        x = cx + t * dx
        if 0 <= x < width:
            intersections.append((height - 1, x))

    if len(intersections) < 2:
        raise ValueError("Could not find valid line intersections with image boundaries")

    # Sort intersections by parameter t to get start and end points
    intersections_with_t = []
    for y, x in intersections:
        if dx != 0:
            t = (x - cx) / dx
        else:
            t = (y - cy) / dy
        intersections_with_t.append((t, (y, x)))

    intersections_with_t.sort()
    return intersections_with_t[0][1], intersections_with_t[-1][1]


def _show_linescan(
    image: np.ndarray,
    profile: np.ndarray,
    positions: np.ndarray,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    linewidth: int,
    sampling_units: str,
    **kwargs,
) -> tuple[Any, tuple[Any, Any]]:
    """Display the linescan results."""
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    if profile.ndim == 1:
        ax0.plot(positions, profile, linewidth=2)
    else:
        # For 3D input, show as image
        # im = ax0.imshow(profile, aspect="equal", origin="upper")
        im = ax0.imshow(profile, aspect="auto", origin="upper")
        plt.colorbar(im, ax=ax0)

    ax0.set_xlabel(f"Position ({sampling_units})")
    ax0.set_ylabel("Intensity" if profile.ndim == 1 else "Depth (pixels)")
    ax0.set_title("Line Profile")

    show_2d(image, figax=(fig, ax1), **kwargs)
    color = kwargs.get("color", "red")

    if linewidth > 1:
        _draw_line_with_width(ax1, start_point, end_point, linewidth, color)
    else:
        ax1.plot(
            [start_point[1], end_point[1]],
            [start_point[0], end_point[0]],
            color=color,
            linewidth=2,
        )

    ax1.set_xlim(0, image.shape[1] - 1)
    ax1.set_ylim(image.shape[0] - 1, 0)
    ax1.set_title("Image with Line Scan")

    plt.tight_layout()
    return fig, (ax0, ax1)


def _draw_line_with_width(
    ax,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    linewidth: int,
    color: str,
) -> None:
    """Draw a line with specified width as a filled polygon."""
    sy, sx = start_point
    ey, ex = end_point

    # Calculate perpendicular direction
    line_vec = np.array([ex - sx, ey - sy])
    line_length = np.linalg.norm(line_vec)

    if line_length > 0:
        perp_vec = np.array([-line_vec[1], line_vec[0]]) / line_length * (linewidth / 2)

        # Create polygon points
        p1 = (sx - perp_vec[0], sy - perp_vec[1])
        p2 = (ex - perp_vec[0], ey - perp_vec[1])
        p3 = (ex + perp_vec[0], ey + perp_vec[1])
        p4 = (sx + perp_vec[0], sy + perp_vec[1])

        polygon = Polygon([p1, p2, p3, p4], alpha=0.3, facecolor=color, edgecolor=None)
        ax.add_patch(polygon)

    # Draw center line
    ax.plot([sx, ex], [sy, ey], color=color, linewidth=1)
