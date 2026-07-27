"""Plotting and figure export for detected diffraction peaks.

Rendering helpers for peak data: radial profiles, per-position count and histogram
maps, an interactive pattern browser, and figure export. They take the arrays and
peak vectors they draw rather than an analysis object, so they work for any
4D-STEM dataset regardless of the material or how the peaks were found.

``BraggPeaksPolymer`` keeps same-named methods forwarding to these, so
``bp.plot_peak_count_map(...)`` and friends continue to work unchanged.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import HBox, IntSlider, VBox, interactive_output
from IPython.display import clear_output, display
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

from quantem.diffraction.vector_fields import vector_field_cell, vector_field_flat

__all__ = [
    "peak_radial_count_plot",
    "peak_radial_intensity_plot",
    "plot_interactive_image_map",
    "plot_peak_count_map",
    "plot_peak_histogram_map",
    "save_diffraction_figures",
    "visualize_selected_patterns",
]


def _mean_intensity_map(dataset_cartesian, scan_shape):
    Ry, Rx = scan_shape
    return np.array(
        [
            [np.mean(dataset_cartesian[i, j].array) for j in range(Rx)]
            for i in range(Ry)
        ]
    )


def _normalized_dp(
    dataset_cartesian,
    ry_data,
    rx_data,
    *,
    norm_upper_quantile=None,
    norm_power=1.0,
    copy_data=True,
):
    dp_data = dataset_cartesian[ry_data, rx_data].array
    if copy_data:
        dp_data = dp_data.copy()
    if norm_upper_quantile is not None:
        dp_data = np.clip(dp_data, 0, np.quantile(dp_data, norm_upper_quantile))
    if norm_power != 1.0:
        m = np.nanmax(dp_data)
        if np.isfinite(m) and m > 0:
            dp_data = (dp_data / m) ** norm_power * m
    return dp_data


def _resolve_intensity_map(
    dataset_cartesian,
    intensity_map,
    scan_shape,
    *,
    validate=True,
    announce_upsample=False,
):
    Ry, Rx = scan_shape
    if intensity_map is None:
        return _mean_intensity_map(dataset_cartesian, scan_shape), 1

    map_shape = intensity_map.shape[:2]
    upsample_factor = map_shape[0] // Ry
    if validate:
        if upsample_factor != map_shape[1] // Rx:
            raise ValueError("Inconsistent upsample factors")
        if map_shape[0] % Ry != 0 or map_shape[1] % Rx != 0:
            raise ValueError(
                f"intensity_map shape {intensity_map.shape} not integer multiple of ({Ry}, {Rx})"
            )
    if announce_upsample:
        print(f"Auto-detected upsample_factor: {upsample_factor}")
    return intensity_map, upsample_factor


def _intensity_display_limits(intensity_map):
    is_rgb_map = intensity_map.ndim == 3 and intensity_map.shape[2] in (3, 4)
    if is_rgb_map:
        return is_rgb_map, None, None
    finite = np.isfinite(intensity_map)
    if not np.any(finite):
        return is_rgb_map, 0.0, 1.0
    vmin, vmax = np.quantile(intensity_map[finite], [0.01, 0.99])
    return is_rgb_map, vmin, vmax


def peak_radial_intensity_plot(
    polar_peaks,
    peak_intensities,
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
    intensity_field='intensities',
    log_scale=False,
    show_d_spacing=False,
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
    all_r = vector_field_flat(polar_peaks, "r_invA")
    all_intensity = vector_field_flat(peak_intensities, intensity_field)
    
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

        fill_base = 0
        if log_scale:
            ax.set_yscale('log')
            _pos = intensity_sum[intensity_sum > 0]
            fill_base = (_pos.min() if _pos.size else 1e-9)

        if show_d_spacing:
            # top axis: real-space d-spacing (Å) = 1 / q (1/Å)
            secax = ax.secondary_xaxis(
                'top',
                functions=(lambda q: 1.0 / np.clip(q, 1e-12, None),
                           lambda d: 1.0 / np.clip(d, 1e-12, None)),
            )
            secax.set_xlabel('d-spacing (Å)', fontsize=12)
        
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
                    ax.fill_between(r_window, fill_base, intensity_window,
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
    polar_peaks,
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
    log_scale=False,
    show_d_spacing=False,
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
    all_r = vector_field_flat(polar_peaks, "r_invA")
    
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

        fill_base = 0
        if log_scale:
            ax.set_yscale('log')
            _pos = peak_counts[peak_counts > 0]
            fill_base = (_pos.min() if _pos.size else 1e-9)

        if show_d_spacing:
            # top axis: real-space d-spacing (Å) = 1 / q (1/Å)
            secax = ax.secondary_xaxis(
                'top',
                functions=(lambda q: 1.0 / np.clip(q, 1e-12, None),
                           lambda d: 1.0 / np.clip(d, 1e-12, None)),
            )
            secax.set_xlabel('d-spacing (Å)', fontsize=12)
        
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
                    ax.fill_between(r_window, fill_base, counts_window,
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


def plot_peak_count_map(
    peaks,
    polar_peaks,
    q_ranges, figsize_per_map=(5, 4), cmap='viridis', return_values=False):
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
    
    Ry, Rx = peaks.shape
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
                peaks_r_invA = vector_field_cell(polar_peaks, "r_invA", i, j)
                if peaks_r_invA is not None and len(peaks_r_invA) > 0:
                    # Get radial distances in 1/Å
                    distances = peaks_r_invA
                    # Count peaks in range
                    mask = (distances >= q_min) & (distances < q_max)
                    count_map[i, j] = np.sum(mask)
        
        count_maps.append(count_map)
        
        # Calculate max_count early for use in both colorbar and statistics
        max_count = int(np.max(count_map))
        
        # Plot as a true integer-count map. Avoid show_2d's default quantile
        # normalization here: count maps are discrete, not continuous images.
        boundaries = np.arange(-0.5, max_count + 1.5, 1)
        norm = BoundaryNorm(boundaries, ncolors=plt.get_cmap(cmap).N, clip=True)
        im = axes[idx].imshow(
            count_map,
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            origin='upper',
        )
        _dlo = (1.0 / q_max) if q_max > 0 else float('inf')  # d-spacing (Å) = 1 / q (1/Å)
        _dhi = (1.0 / q_min) if q_min > 0 else float('inf')
        axes[idx].set_title(
            f'Peak Count\n{q_min:.2f} - {q_max:.2f} 1/Å\n'
            f'd = {_dlo:.2f} - {_dhi:.2f} Å',
            fontsize=14)
        axes[idx].set_xlabel('Scan X', fontsize=12)
        axes[idx].set_ylabel('Scan Y', fontsize=12)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        cbar = plt.colorbar(im, ax=axes[idx], ticks=np.arange(max_count + 1))
        cbar.set_label('Number of Peaks', fontsize=10)
        
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


def plot_peak_histogram_map(
    peaks,
    peak_intensities,
    intensity_threshold=None,
    intensity_percentile=None,
    figsize=(8, 6), 
    cmap='viridis',
    return_values=False,
    intensity_field='intensities',
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
    Ry, Rx = peaks.shape
    
    # Convert percentile to threshold if needed
    if intensity_percentile is not None:
        all_intensities = [
            values
            for i in range(Ry)
            for j in range(Rx)
            if len(
                values := vector_field_cell(
                    peak_intensities, intensity_field, i, j
                )
            )
        ]
        if all_intensities:
            intensity_threshold = np.percentile(np.concatenate(all_intensities), intensity_percentile)
    
    # Build count map
    count_map = np.zeros((Ry, Rx))
    for i in range(Ry):
        for j in range(Rx):
            cell_peaks = peaks[i, j].array
            if len(cell_peaks) == 0:
                continue

            if intensity_threshold is None:
                count_map[i, j] = len(cell_peaks)
            else:
                intensities = vector_field_cell(
                    peak_intensities, intensity_field, i, j
                )
                if len(intensities):
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


def plot_interactive_image_map(
    dataset,
    polar_data=None,
    ry=None, rx=None, intensity_map=None, vmax_cartesian=None, vmin_cartesian=None, 
                                map_cmap='viridis', map_title='Intensity Map', dp_cmap="gray",
                                norm_upper_quantile=None, norm_power=1.0,
                                show_polar=True, vmax_polar=None, crosshair_color='r', figsize=None, 
                                crosshair_width=2, crosshair_size=15, gaussian_filter_sigma=None):
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
    vmin_cartesian : float
        Minimum value for diffraction pattern display
    map_cmap : str
        Colormap for the intensity map
    map_title : str
        Title for the intensity map panel
    dp_cmap : str
        Colormap for diffraction patterns
    norm_upper_quantile : float, optional
        Upper quantile for normalization (0-1). If None, not used.
    norm_power : float
        Power law normalization exponent
    show_polar : bool
        Whether to show the polar transformed data panel
    vmax_polar : float, optional
        Maximum value for polar pattern display. If None, uses vmax_cartesian.
    """
    
    Ry, Rx = dataset.shape[:2]
    
    # Check polar data availability
    if show_polar and not (polar_data is not None):
        print("Warning: polar_data not found. Set show_polar=False or run polar_transform_4d first.")
        show_polar = False
    
    intensity_map, upsample_factor = _resolve_intensity_map(
        dataset,
        intensity_map,
        (Ry, Rx),
        validate=True,
        announce_upsample=intensity_map is not None,
    )
    
    # Compute intensity map display limits
    _is_rgb_map, vmin_intensity_map, vmax_intensity_map = _intensity_display_limits(
        intensity_map
    )
    
    vmax_polar = vmax_polar or vmax_cartesian
    slider_Ry, slider_Rx = Ry * upsample_factor, Rx * upsample_factor
    
    # ---- Create figure and axes once ----
    if show_polar:
        if figsize is None:
            figsize=(15, 4)
        fig, (ax_map, ax_diff, ax_polar) = plt.subplots(1, 3, figsize=figsize)
    else:
        if figsize is None:
            figsize=(12, 5)
        fig, (ax_map, ax_diff) = plt.subplots(1, 2, figsize=figsize)
        ax_polar = None
    
    # Initialize image objects
    if vmin_intensity_map is None:
        im_map = ax_map.imshow(intensity_map, cmap=map_cmap)
    else:
        im_map = ax_map.imshow(intensity_map, cmap=map_cmap, 
                              vmin=vmin_intensity_map, vmax=vmax_intensity_map)
    line_marker, = ax_map.plot([], [], color=crosshair_color, marker='+', markersize=crosshair_size, markeredgewidth=crosshair_width)
    ax_map.set_title(map_title)
    ax_map.set_xlabel('Rx (upsampled)' if upsample_factor > 1 else 'Rx')
    ax_map.set_ylabel('Ry (upsampled)' if upsample_factor > 1 else 'Ry')
    cbar_map = plt.colorbar(im_map, ax=ax_map)
    
    # Diffraction pattern (initialize with zeros)
    im_diff = ax_diff.imshow(np.zeros((10, 10)), cmap=dp_cmap, vmin=vmin_cartesian, vmax=vmax_cartesian)
    ax_diff.set_title('Diffraction Pattern')
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    cbar_diff = plt.colorbar(im_diff, ax=ax_diff)
    
    # Polar transform
    if show_polar:
        # ax_polar.set_aspect('equal', adjustable='box')
        im_polar = ax_polar.imshow(np.zeros((10, 10)), cmap=dp_cmap, vmax=vmax_polar, aspect='auto')
        ax_polar.set_title('Polar Transform')
        ax_polar.set_xlabel('Radius (bins)')
        ax_polar.set_ylabel('Theta (bins)')
        cbar_polar = plt.colorbar(im_polar, ax=ax_polar)
    
    plt.tight_layout()
    plt.close(fig)
    
    # ---- Interactive display callback (updates only) ----
    def show_pattern(ry_slider, rx_slider):
        ry_data = ry_slider // upsample_factor
        rx_data = rx_slider // upsample_factor
        
        # Update marker
        line_marker.set_data([rx_slider], [ry_slider])
        
        # Update diffraction pattern
        dp_data = _normalized_dp(
            dataset,
            ry_data,
            rx_data,
            norm_upper_quantile=norm_upper_quantile,
            norm_power=norm_power,
            copy_data=False,
        )
        im_polar_data = polar_data['intensity'][ry_data, rx_data].T if show_polar else None
        if gaussian_filter_sigma is not None:
            dp_data = gaussian_filter(dp_data, gaussian_filter_sigma)
            if show_polar:
                im_polar_data = gaussian_filter(im_polar_data, gaussian_filter_sigma)
            
        im_diff.set_data(dp_data)
        ax_diff.set_title(f'Diffraction Pattern (Ry={ry_data}, Rx={rx_data})')
        
        # Update polar transform
        if show_polar:
            im_polar.set_data(im_polar_data)
            ax_polar.set_title(f'Polar Transform (Ry={ry_data}, Rx={rx_data})')
        
        clear_output(wait=True)
        display(fig)
    
    # Create widgets
    if ry is None:
        ry = slider_Ry//2
    if rx is None:
        rx = slider_Rx//2
    ry_slider = IntSlider(min=0, max=slider_Ry-1, value=ry, description='Ry:', continuous_update=False)
    rx_slider = IntSlider(min=0, max=slider_Rx-1, value=rx, description='Rx:', continuous_update=False)
    
    controls = VBox([HBox([ry_slider, rx_slider])])
    interactive_plot = interactive_output(show_pattern, {'ry_slider': ry_slider, 'rx_slider': rx_slider})
    display(controls, interactive_plot)


def save_diffraction_figures(
    dataset,
    polar_data,
    ry, rx, intensity_map=None, prefix='diffraction', save_dir='.', 
                            vmax_cartesian=None, vmin_cartesian=None,
                            map_cmap='viridis', map_title='Intensity Map', dp_cmap="gray",
                            norm_upper_quantile=None, norm_power=1.0,
                            show_polar=True, vmax_polar=None, crosshair_color='r', 
                            figsize_individual=None, figsize_combined=None, crosshair_width=2, crosshair_size=15,
                            gaussian_filter_sigma=None):
    """
    Save diffraction pattern figures for a specific scan position.
    
    Parameters
    ----------
    ry : int
        Y position in original dataset coordinates
    rx : int
        X position in original dataset coordinates
    intensity_map : array, optional
        2D array to display as reference map. If None, shows mean intensity.
    prefix : str
        Filename prefix for saved files
    save_dir : str
        Directory path for saving files
    vmax_cartesian : float
        Maximum value for diffraction pattern display
    vmin_cartesian : float
        Minimum value for diffraction pattern display
    map_cmap : str
        Colormap for the intensity map
    map_title : str
        Title for the intensity map panel
    dp_cmap : str
        Colormap for diffraction patterns
    norm_upper_quantile : float, optional
        Upper quantile for normalization (0-1). If None, not used.
    norm_power : float
        Power law normalization exponent
    show_polar : bool
        Whether to save the polar transformed data
    vmax_polar : float, optional
        Maximum value for polar pattern display. If None, uses vmax_cartesian.
    """
    
    from pathlib import Path
    
    Ry, Rx = dataset.shape[:2]
    
    # Validate coordinates
    if not (0 <= ry < Ry and 0 <= rx < Rx):
        raise ValueError(f"Coordinates ({ry}, {rx}) out of bounds for dataset shape ({Ry}, {Rx})")
    
    # Check polar data availability
    if show_polar and not (polar_data is not None):
        print("Warning: polar_data not found. Skipping polar transform save.")
        show_polar = False
    
    intensity_map, upsample_factor = _resolve_intensity_map(
        dataset,
        intensity_map,
        (Ry, Rx),
        validate=True,
    )
    
    # Compute intensity map display limits
    _is_rgb_map, vmin_intensity_map, vmax_intensity_map = _intensity_display_limits(
        intensity_map
    )
    
    vmax_polar = vmax_polar or vmax_cartesian
    
    # Create save directory
    save_path = Path(save_dir)
    try:
        save_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return
    
    # Calculate marker positions
    marker_ry = ry * upsample_factor
    marker_rx = rx * upsample_factor
    
    try:
        # Save intensity map
        if figsize_individual is None:
            figsize_individual = (6, 6)
        fig_map, ax = plt.subplots(figsize=figsize_individual)
        if vmin_intensity_map is None:
            im = ax.imshow(intensity_map, cmap=map_cmap)
        else:
            im = ax.imshow(intensity_map, cmap=map_cmap, 
                          vmin=vmin_intensity_map, vmax=vmax_intensity_map)
        ax.plot(marker_rx, marker_ry, color=crosshair_color, marker='+', markersize=crosshair_size, markeredgewidth=crosshair_width)
        ax.set_title(map_title)
        ax.set_xlabel('Rx (upsampled)' if upsample_factor > 1 else 'Rx')
        ax.set_ylabel('Ry (upsampled)' if upsample_factor > 1 else 'Ry')
        filename = save_path / f'{prefix}_ry{ry}_rx{rx}_intensity_map.pdf'
        fig_map.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_map)
        print(f'✓ Saved: {filename}')
        
        # Save diffraction pattern
        fig_diff, ax = plt.subplots(figsize=figsize_individual)
        dp_data = _normalized_dp(
            dataset,
            ry,
            rx,
            norm_upper_quantile=norm_upper_quantile,
            norm_power=norm_power,
        )
        polar_im_data = polar_data['intensity'][ry, rx].T if show_polar else None
        if gaussian_filter_sigma is not None:
            dp_data = gaussian_filter(dp_data, gaussian_filter_sigma)
            if show_polar:
                polar_im_data = gaussian_filter(polar_im_data, gaussian_filter_sigma)
        
        im = ax.imshow(dp_data, cmap=dp_cmap, vmin=vmin_cartesian, vmax=vmax_cartesian)
        ax.set_title(f'Diffraction Pattern (Ry={ry}, Rx={rx})')
        ax.set_xticks([])
        ax.set_yticks([])
        filename = save_path / f'{prefix}_ry{ry}_rx{rx}_diffraction.pdf'
        fig_diff.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_diff)
        print(f'✓ Saved: {filename}')
        
        # Save polar transform
        if show_polar:
            fig_polar, ax = plt.subplots(figsize=figsize_individual)
            im = ax.imshow(polar_im_data, cmap=dp_cmap, vmax=vmax_polar, aspect='auto')
            # ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'Polar Transform (Ry={ry}, Rx={rx})')
            ax.set_xlabel('Radius (bins)')
            ax.set_ylabel('Theta (bins)')
            filename = save_path / f'{prefix}_ry{ry}_rx{rx}_polar.pdf'
            fig_polar.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close(fig_polar)
            print(f'✓ Saved: {filename}')
        
        # Save combined figure
        if show_polar:
            if figsize_combined is None:
                figsize_combined = (15, 4)
            fig_combined, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize_combined)
        else:
            if figsize_combined is None:
                figszie_combined = (12, 5)
            fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_combined)
            ax3 = None
        
        # Plot intensity map
        if vmin_intensity_map is None:
            im1 = ax1.imshow(intensity_map, cmap=map_cmap)
        else:
            im1 = ax1.imshow(intensity_map, cmap=map_cmap, 
                            vmin=vmin_intensity_map, vmax=vmax_intensity_map)
        ax1.plot(marker_rx, marker_ry, color=crosshair_color, marker='+', markersize=crosshair_size, markeredgewidth=crosshair_width)
        ax1.set_title(map_title)
        ax1.set_xlabel('Rx (upsampled)' if upsample_factor > 1 else 'Rx')
        ax1.set_ylabel('Ry (upsampled)' if upsample_factor > 1 else 'Ry')
        
        # Plot diffraction pattern
        im2 = ax2.imshow(dp_data, cmap=dp_cmap, vmin=vmin_cartesian, vmax=vmax_cartesian)
        ax2.set_title(f'Diffraction Pattern (Ry={ry}, Rx={rx})')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Plot polar transform
        if show_polar:
            im3 = ax3.imshow(polar_im_data, 
                            cmap=dp_cmap, vmax=vmax_polar, aspect='auto')
            # ax3.set_aspect('equal', adjustable='box')
            ax3.set_title(f'Polar Transform (Ry={ry}, Rx={rx})')
            ax3.set_xlabel('Radius (bins)')
            ax3.set_ylabel('Theta (bins)')
        
        plt.tight_layout()
        filename = save_path / f'{prefix}_ry{ry}_rx{rx}_combined.pdf'
        fig_combined.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_combined)
        print(f'✓ Saved: {filename}')
        
        print(f'\nAll figures saved successfully to: {save_path}')
        
    except Exception as e:
        print(f"Error saving figures: {e}")


def visualize_selected_patterns(
    dataset,
    positions, ncols=4, figsize_per_pattern=(3, 3), 
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
        dp = dataset[ry, rx].array
        
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
