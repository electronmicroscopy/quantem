import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import norm

from quantem.core.visualization import show_2d


def plot_attached_spectrum(self, spectrum_index=0):
    fig, (ax_spec) = plt.subplots(1, 1, figsize=(12, 4))

    ds = self.attached_spectra[spectrum_index]
    energy = ds.origin[0] + ds.sampling[0] * np.arange(ds.shape[0])
    ax_spec.plot(energy, ds.array, linewidth=1.5)

    if self.dataset_type == "xeds":
        ax_spec.set_xlabel("Energy (keV)")
    elif self.dataset_type == "eels":
        ax_spec.set_xlabel("Energy (eV)")
    ax_spec.set_ylabel("Intensity")
    ax_spec.set_title(f"Spectrum in index {spectrum_index}")
    ax_spec.grid(True, alpha=0.1)

    fig.tight_layout()
    plt.show()


def _plot_pca_results(
    self,
    components,
    loadings,
    explained_variance_ratio,
    n_show: int = 4,
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
    fig, (ax_scree, ax_components) = plt.subplots(1, 2, figsize=(12, 4))
    cumsum_var = np.cumsum(explained_variance_ratio)
    component_numbers = np.arange(1, len(explained_variance_ratio) + 1)

    ax_scree.bar(
        component_numbers,
        explained_variance_ratio * 100,
        alpha=0.6,
        label="Individual",
    )
    ax_scree.plot(component_numbers, cumsum_var * 100, "ro-", label="Cumulative")
    ax_scree.set_xlabel("Component Number")
    ax_scree.set_ylabel("Explained Variance (%)")
    ax_scree.set_title("Scree Plot")
    ax_scree.legend()
    ax_scree.grid(True, alpha=0.3)

    energy_sampling = float(self.sampling[2])
    energy_origin = float(self.origin[2])
    energy_axis = energy_origin + energy_sampling * np.arange(components.shape[1])

    for i in range(n_show):
        ax_components.plot(
            energy_axis,
            components[i],
            label=f"PC{i + 1} ({explained_variance_ratio[i] * 100:.1f}%)",
        )
    ax_components.set_xlabel("Energy")
    ax_components.set_ylabel("Component")
    ax_components.set_title("Principal Component Spectra")
    ax_components.legend()
    ax_components.grid(True, alpha=0.3)

    fig.suptitle("PCA Analysis")
    fig.tight_layout()
    plt.show()

    show_2d(
        [loadings[i] for i in range(n_show)],
        title=[
            f"Loading {i + 1} ({explained_variance_ratio[i] * 100:.1f}%)" for i in range(n_show)
        ],
        cmap="RdBu_r",
        cbar=True,
        scalebar={
            "sampling": float(self.sampling[1]),
            "units": str(self.units[1]),
        },
    )
    plt.show()


def show_mean_spectrum(
    self,
    roi=None,
    roi_cal=None,
    energy_range=None,
    mask=None,
    intensity_range=None,
    normalize=False,
    **kwargs,
):
    """
    Plot the mean spectrum from a spatial ROI in a 3D spectroscopy cube (Y, X, E).

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
    mask : array, optional
        Boolean mask for pixel selection.
    intensity_range : 2-tuple, None
        If not None, sets intensity range on spectrum plot
    normalize : bool, optional
        If ``True``, scale the mean spectrum to the range [0, 1]. If
        ``False``, plot the mean spectrum in original intensity units.
    Returns
    -------
    (fig, ax) : tuple
        The Matplotlib Figure and Axes of the spectrum plot.
    """

    # CALCULATE MEAN SPECTRUM FOR GIVEN ROI AND ENERGY RANGE --------------------------

    y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)

    energy_range_for_calc = None if energy_range is None else list(energy_range)
    spec = self.calculate_mean_spectrum(
        roi=roi,
        roi_cal=roi_cal,
        energy_range=energy_range_for_calc,
        mask=mask,
        normalize=normalize,
    )

    E = np.asarray(self.energy_axis, dtype=float)

    if mask is not None:
        E = E[np.asarray(mask, dtype=bool)]

    if energy_range is not None:
        indices = np.where((E >= energy_range[0]) & (E <= energy_range[1]))[0]
        E = E[indices]

    # PLOTTING ---------------------------------------------------------------------------

    # Create subplot layout: image on left, spectrum on right
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))

    # LEFT PLOT: Show sum image with ROI highlighted
    # Create sum image across all energy channels (or masked channels)
    if mask is not None:
        sum_img = np.asarray(self.array, dtype=float)[:, :, np.asarray(mask, dtype=bool)].sum(
            axis=2
        )
        title_suffix = " (masked energies)"
    else:
        sum_img = np.asarray(self.array, dtype=float).sum(axis=2)
        title_suffix = ""

    map_title = f"Integrated Intensity Map{title_suffix}"
    show_2d(
        sum_img,
        figax=(fig, ax_img),
        title=map_title,
        cmap="viridis",
        cbar=True,
        show_ticks=True,
        scalebar={
            "sampling": float(self.sampling[1]),
            "units": str(self.units[1]),
        },
        **kwargs,
    )
    # Highlight the ROI with a rectangle
    rect = Rectangle(
        (x - 0.5, y - 0.5), dx, dy, linewidth=2, edgecolor="red", facecolor="none", alpha=0.8
    )
    ax_img.add_patch(rect)

    # RIGHT PLOT: Show spectrum
    ax_spec.plot(E, spec, linewidth=1.5, color="k")
    if self.dataset_type == "xeds":
        ax_spec.set_xlabel("Energy (keV)")
    else:
        ax_spec.set_xlabel("Energy (eV)")
    ax_spec.set_ylabel("Normalized intensity" if normalize else "Intensity")
    ax_spec.set_title(f"Spectrum from ROI [{y}:{y + dy}, {x}:{x + dx}]")
    ax_spec.grid(True, alpha=0.1)
    if intensity_range is not None:
        ax_spec.set_ylim([intensity_range[0], intensity_range[1]])

    fig.tight_layout()
    return fig, (ax_img, ax_spec)


def show_energy_window_map(
    self,
    energy_window=None,
    roi=None,
    roi_cal=None,
    mask=None,
    cmap="viridis",
    show=True,
):
    """Show a spatial map integrated over a selected energy window.

    This is a complementary view to ``show_mean_spectrum``:
    - ``show_mean_spectrum`` answers *what energies are present*.
    - ``show_energy_window_map`` answers *where a chosen energy range is present*.

    Parameters
    ----------
    energy_window : list[float] | tuple[float, float] | None
        Energy interval [emin, emax] to integrate. If None, use the
        full calibrated energy range of the dataset.
    roi : list | tuple | None, optional
        ROI as ``[y, x]`` or ``[y, x, dy, dx]`` (with ``None`` defaults),
        used only for overlay rectangle.
    mask : array-like | None, optional
        Optional boolean mask over energy channels. If provided, it is
        combined with ``energy_window``.
    cmap : str, optional
        Matplotlib colormap for the map.
    show : bool, optional
        If True, call ``plt.show()``.

    Returns
    -------
    tuple
        ``(fig, (ax_map, ax_spec), energy_map)`` where ``energy_map`` is the integrated 2D array.
    """
    y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)
    has_roi_overlay = any(val is not None for val in (roi, roi_cal))

    dE = float(self.sampling[2])
    E0 = float(self.origin[2]) if hasattr(self, "origin") else 0.0
    E = E0 + dE * np.arange(self.shape[2])

    if energy_window is None:
        emin = float(np.min(E))
        emax = float(np.max(E))
    else:
        if len(energy_window) != 2:
            raise ValueError("energy_window must be [min_energy, max_energy]")

        emin = float(energy_window[0])
        emax = float(energy_window[1])
        if not np.isfinite(emin) or not np.isfinite(emax) or emin >= emax:
            raise ValueError(
                "Invalid energy_window. Expected [min_energy, max_energy] with min < max"
            )

    window_mask = (E >= emin) & (E <= emax)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.shape[2],):
            raise ValueError(
                f"Mask shape {mask.shape} does not match energy axis shape ({self.shape[2]},)"
            )
        window_mask = window_mask & mask

    if not np.any(window_mask):
        raise ValueError("No energy channels selected. Adjust energy_window or mask")

    arr = np.asarray(self.array, dtype=float)
    energy_map = arr[:, :, window_mask].sum(axis=-1)

    spec = self.calculate_mean_spectrum(
        roi=roi,
        roi_cal=roi_cal,
        mask=mask,
        attach_mean_spectrum=False,
    )
    if mask is not None:
        E_spec = E[mask]
    else:
        E_spec = E

    unit_label = "keV" if str(self.dataset_type).lower() == "xeds" else "eV"
    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))
    show_2d(
        energy_map,
        figax=(fig, ax_map),
        title=f"Energy-Window Map [{emin:.3f}, {emax:.3f}] {unit_label}",
        cmap=cmap,
        cbar=True,
        show_ticks=True,
        scalebar={
            "sampling": float(self.sampling[1]),
            "units": str(self.units[1]),
        },
    )

    if has_roi_overlay:
        rect = Rectangle(
            (x - 0.5, y - 0.5),
            dx,
            dy,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            alpha=0.8,
        )
        ax_map.add_patch(rect)

    ax_spec.plot(E_spec, spec, linewidth=1.5, color="k")
    ax_spec.axvspan(emin, emax, color="orange", alpha=0.2, label="Selected window")
    ax_spec.set_xlabel(f"Energy ({unit_label})")
    ax_spec.set_ylabel("Intensity")
    ax_spec.set_title(f"Spectrum from ROI [{y}:{y + dy}, {x}:{x + dx}]")
    ax_spec.grid(True, alpha=0.1)
    ax_spec.legend(loc="best")

    fig.tight_layout()

    if show:
        plt.show()

    return fig, (ax_map, ax_spec), energy_map


def _plot_background_subtraction(
    self,
    energy_axis,
    input_spectrum,
    background_spectrum,
    subtracted_spectrum,
    fit_mode,
    show_subtracted,
):
    fig, (ax_specbacksub) = plt.subplots(1, 1, figsize=(12, 4))

    ax_specbacksub.plot(energy_axis, input_spectrum, linewidth=1.2, label="Input")
    ax_specbacksub.plot(energy_axis, background_spectrum, linewidth=1.2, label="Background")
    if show_subtracted:
        ax_specbacksub.plot(
            energy_axis,
            subtracted_spectrum,
            linewidth=1.5,
            label="Background-subtracted",
        )
    if self.dataset_type == "xeds":
        ax_specbacksub.set_xlabel("Energy (keV)")
    else:
        ax_specbacksub.set_xlabel("Energy (eV)")
    ax_specbacksub.set_ylabel("Intensity")
    ax_specbacksub.set_title(f"Background-subtracted spectrum from ROI ({fit_mode})")
    ax_specbacksub.grid(True, alpha=0.1)
    ax_specbacksub.legend()

    fig.tight_layout()
    plt.show()


def show_spectrum_images(
    self, x_ray_lines=None, return_fig=False, return_maps=False, method="integration", **kwargs
):
    """Display cached spectrum images.

    Parameters
    ----------
    x_ray_lines : str | sequence[str] | None, optional
        Selectors to filter which images are shown.  If ``None``, one
        panel per element is displayed.
    return_fig : bool, optional
        If ``True``, return ``(fig, ax)``.
    method : {"integration", "fit"}, optional
        Which cache to read from: integration-based maps or PyTorch
        fit-based maps.
    **kwargs
        Forwarded to :func:`show_2d` (e.g. ``cmap``).

    Returns
    -------
    tuple[Figure, Axes] | None
        Only returned when *return_fig* is ``True``.

    Raises
    ------
    ValueError
        If no cached spectrum images exist for the chosen *method*.
    """
    spectrum_images = self._get_spectrum_images(method)
    if not spectrum_images:
        raise ValueError("No spectrum images found. Run generate_spectrum_images(...) first.")

    line_map = {str(k): np.asarray(getattr(v, "array", v)) for k, v in spectrum_images.items()}
    labels = list(line_map)
    labels_by_element = type(self)._group_labels_by_element(labels)

    def sum_maps(lbls):
        return np.sum([line_map[lbl] for lbl in lbls], axis=0)

    specs = type(self)._normalize_specs(x_ray_lines, param_name="x_ray_lines", allow_none=True)
    if not specs:
        titles = sorted(labels_by_element)
        images = [sum_maps(labels_by_element[t]) for t in titles]
    else:
        selected = [
            type(self)._select_labels(str(raw), labels=labels, labels_by_element=labels_by_element)
            for raw in specs
        ]
        if any(not s for s in selected):
            bad = next(raw for raw, s in zip(specs, selected) if not s)
            raise ValueError(f"No spectrum images matched selector '{bad}'")
        images = [line_map[s[0]] if len(s) == 1 else sum_maps(s) for s in selected]
        titles = [s[0] if len(s) == 1 else str(raw).strip() for raw, s in zip(specs, selected)]

    fig, ax = show_2d(
        images,
        title=titles,
        cmap=kwargs.pop("cmap", "magma"),
        scalebar={"sampling": self.sampling[1], "units": self.units[1]},
        returnfig=True,
        **kwargs,
    )

    if return_maps and hasattr(self, "_map_to_dataset2d"):
        images = [
            self._map_to_dataset2d(image, name=str(title)) for image, title in zip(images, titles)
        ]

    if return_fig and return_maps:
        return (fig, ax), (images, titles)
    elif return_fig:
        return fig, ax
    elif return_maps:
        return images, titles


def plot_absolute_zlp_shift(dataset, search_window=(-10, 10)):
    """
    Calculates the ZLP shift per pixel and plots the absolute deviation from 0.0 eV.
    """
    data = dataset.array

    # Generate energy axis
    energies = np.asarray(dataset.energy_axis, dtype=float)

    # Mask energy window for peak finding
    mask = (energies > search_window[0]) & (energies < search_window[1])
    search_energies = energies[mask]

    # Calculate peak map and absolute deviation
    peak_indices = np.argmax(data[:, :, mask], axis=2)
    zlp_map_ev = search_energies[peak_indices]
    absolute_shift = np.abs(zlp_map_ev)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(absolute_shift, cmap="magma", origin="lower")

    plt.colorbar(im, ax=ax, label="Absolute Shift (eV)")
    ax.set_title(f"Absolute ZLP Deviation: {dataset.name}")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()

    return absolute_shift


def visualize_thickness_windows(dataset, zlp_window=(-3.0, 3.0), total_window=(-3.0, 75.0)):
    """
    Visualizes integration windows for I0 (ZLP) and It (Total).
    Returns a configuration dictionary for the calculation step.
    """
    # 1. Extract Energy and Mean Spectrum
    data = dataset.array
    mean_spec = np.mean(data, axis=(0, 1))

    # Use built-in energy axis if available, else generate from metadata
    if hasattr(dataset, "energy_axis"):
        energy = np.asarray(dataset.energy_axis, dtype=float)
    else:
        energy = dataset.origin[2] + np.arange(dataset.shape[2]) * dataset.sampling[2]

    # 2. Find indices for the windows
    zlp_idx = (
        np.argmin(np.abs(energy - zlp_window[0])),
        np.argmin(np.abs(energy - zlp_window[1])),
    )
    tot_idx = (
        np.argmin(np.abs(energy - total_window[0])),
        np.argmin(np.abs(energy - total_window[1])),
    )

    # 3. Create the Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy, mean_spec, "k-", lw=1.5, label="Mean Spectrum", zorder=5)

    # Highlight Windows
    z_mask = (energy >= zlp_window[0]) & (energy <= zlp_window[1])
    t_mask = (energy >= total_window[0]) & (energy <= total_window[1])

    ax.fill_between(
        energy[z_mask], 0, mean_spec[z_mask], color="red", alpha=0.3, label="$I_0$ (ZLP)"
    )
    ax.fill_between(
        energy[t_mask], 0, mean_spec[t_mask], color="blue", alpha=0.1, label="$I_t$ (Total)"
    )

    ax.axvline(0, color="green", lw=1.5, ls=":", label="0 eV")
    ax.set_title(f"QuantEM: Integration Windows ({dataset.name})", fontweight="bold")
    ax.set_xlabel("Energy Loss (eV)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_xlim(energy[0], total_window[1] + 20)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "zlp_idx": zlp_idx,
        "total_idx": tot_idx,
        "zlp_val": zlp_window,
        "total_val": total_window,
    }


def interpret_thickness_quality(t_over_lambda, a=0.3, b=1, c=2, dataset=None):
    """
    Performs a scientific quality assessment on the calculated t/lambda map.

    The Physical Meaning of the ThresholdsThe t/lambda value represents the average number of inelastic scattering events
    an electron undergoes.
    Vacuum (< a):
        (default a = 0.3)
        In pure vacuum, t/lambda should be 0. In practice, values up to ~0.3 often indicate the presence of thin carbon support films,
        surface contamination, or detector noise. Measurements in this regime are highly sensitive to ZLP (Zero Loss Peak) estimation errors.

    Thin (a <t/lambda < b):
        (default b = 1)
        The "Sweet Spot" for EELS. At t/lambda ~1, the probability of a single inelastic scattering event is maximized.
        In this regime, core-loss edges are sharp and clearly visible without the immediate need for complex mathematical
        deconvolution (e.g., Fourier-Log) to remove multiple scattering effects.

    Medium (b < t/lambda < c):
        (default c = 2)
        Multiple scattering begins to dominate the spectrum. The plural scattering of plasmons creates "ghost" peaks
        that overlap with higher-energy chemical edges. While data is still usable, quantitative analysis typically
        requires plural scattering correction for high accuracy.

    Thick (t/lambda > c):
        The "Multiple Scattering Regime.
        " Most electrons have undergone three or more scattering events, resulting in a "spectral soup"
        where fine-structure details and high-resolution chemical information are significantly broadened or lost.
    """

    name = dataset.name if dataset else "Dataset"

    # Classification Masks
    vacuum = t_over_lambda < a
    thin = (t_over_lambda >= a) & (t_over_lambda < b)
    medium = (t_over_lambda >= b) & (t_over_lambda < c)
    thick = t_over_lambda >= c

    print(f"\n{'=' * 20} QUANTEM INTERPRETATION: {name} {'=' * 20}")
    for label, mask in [
        ("Vacuum (<0.3)", vacuum),
        ("Thin (0.3-1.0)", thin),
        ("Medium (1.0-2.0)", medium),
        ("Thick (>2.0)", thick),
    ]:
        pct = 100 * np.sum(mask) / t_over_lambda.size
        print(f"  {label:20}: {pct:5.1f}%")

    # Plotting Classification
    classified = np.zeros_like(t_over_lambda)
    classified[thin] = 1
    classified[medium] = 2
    classified[thick] = 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(classified, cmap="RdYlGn_r", origin="lower")
    ax1.set_title("Region Classification")
    cbar = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Vacuum", "Thin", "Medium", "Thick"])

    t_masked = np.copy(t_over_lambda)
    t_masked[vacuum] = np.nan
    im2 = ax2.imshow(t_masked, cmap="viridis", origin="lower")
    ax2.set_title("Sample-Only Thickness")
    plt.colorbar(im2, ax=ax2, label=r"$t/\lambda$")

    plt.tight_layout()
    plt.show()


def plot_absolute_thickness(t_lambda_map, mfp_nm, dataset=None):
    """
    Converts relative thickness to nanometers and visualizes the absolute map.
    """
    thickness_nm = t_lambda_map * mfp_nm
    name = dataset.name if dataset else "Sample"

    # Mask vacuum for better visualization contrast
    display_map = np.copy(thickness_nm)
    display_map[t_lambda_map < 0.1] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Physical Analysis: {name}", fontsize=14)

    im = ax1.imshow(display_map, cmap="magma", origin="lower")
    ax1.set_title("Absolute Thickness (nm)")
    plt.colorbar(im, ax=ax1, label="nm")

    valid_data = thickness_nm[t_lambda_map >= 0.1].flatten()
    ax2.hist(valid_data, bins=50, color="firebrick", alpha=0.7, ec="k")
    ax2.axvline(
        np.nanmean(display_map),
        color="blue",
        ls="--",
        label=f"Mean: {np.nanmean(display_map):.1f} nm",
    )
    ax2.set_title("Physical Distribution")
    ax2.set_xlabel("Thickness (nm)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(
        f"\nQuantEM Absolute Report:\n  Mean: {np.nanmean(display_map):.2f} nm\n  MFP:  {mfp_nm:.2f} nm"
    )
    return thickness_nm


def plot_dual_eels_picker(ll, hl, coords=None, title="QuantEM: Dual-EELS Analysis"):
    """
    Dual-EELS Picker with starting coordinates.

    coords, when provided, is interpreted as (scan_row, scan_col).
    """
    # 1. Setup Data
    sum_ll = np.sum(ll.array, axis=2)
    sum_hl = np.sum(hl.array, axis=2)
    energy_ll = np.asarray(ll.energy_axis, dtype=float)
    energy_hl = np.asarray(hl.energy_axis, dtype=float)

    # 2. Handle Initial Coordinates
    if coords is not None:
        i_row, i_col = coords
    else:
        i_row, i_col = ll.shape[0] // 2, ll.shape[1] // 2

    # 3. Create Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{title}\n(Click on maps to update spectra)", fontsize=16)
    ax_map_ll, ax_spec_ll = axes[0, 0], axes[0, 1]
    ax_map_hl, ax_spec_hl = axes[1, 0], axes[1, 1]

    # Plot Maps & Markers
    ax_map_ll.imshow(sum_ll, cmap="viridis", origin="lower")
    (marker_ll,) = ax_map_ll.plot(i_col, i_row, "r+", ms=15, mew=2)

    ax_map_hl.imshow(sum_hl, cmap="magma", origin="lower")
    (marker_hl,) = ax_map_hl.plot(i_col, i_row, "r+", ms=15, mew=2)

    # Plot Initial Spectra
    (line_ll,) = ax_spec_ll.plot(energy_ll, ll.array[i_row, i_col, :], color="tab:blue")
    (line_hl,) = ax_spec_hl.plot(energy_hl, hl.array[i_row, i_col, :], color="tab:red")

    def update_plots(i_row, i_col):
        marker_ll.set_data([i_col], [i_row])
        marker_hl.set_data([i_col], [i_row])

        new_ll = ll.array[i_row, i_col, :]
        new_hl = hl.array[i_row, i_col, :]
        line_ll.set_ydata(new_ll)
        line_hl.set_ydata(new_hl)

        # Rescale
        ax_spec_ll.set_ylim(0, np.max(new_ll) * 1.1)
        ax_spec_hl.set_ylim(0, np.max(new_hl) * 1.1)

        ax_spec_ll.set_title(f"LL Spectrum at ({i_row}, {i_col})")
        ax_spec_hl.set_title(f"HL Spectrum at ({i_row}, {i_col})")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes in [ax_map_ll, ax_map_hl]:
            i_col, i_row = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= i_row < ll.shape[0] and 0 <= i_col < ll.shape[1]:
                update_plots(i_row, i_col)

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_spec_ll.set_title(f"LL Spectrum at ({i_row}, {i_col})")
    ax_spec_hl.set_title(f"HL Spectrum at ({i_row}, {i_col})")

    plt.tight_layout()
    plt.close(fig)  # Prevents double-plotting in VS Code
    return fig


def plot_quantem_diagnostic(dataset, zlp_window=5.0, title_suffix=""):
    """
    QuantEM Diagnostic Dashboard: Visualizes mean spectra, spatial variation,
    and Zero Loss Peak (ZLP) centering accuracy.

    1. Global Average Spectrum (Top Left): Shows the mean intensity across the entire scan.
    It is used to check the signal-to-noise ratio and see if the Zero Loss Peak (ZLP) is roughly centered at 0 eV.
    2. Spatial Variation (Top Right): Plots spectra from a 5x5 grid of pixels across your sample.
    This helps you see if the energy shift or intensity changes drastically from one side of the scan to the other
    (e.g., due to sample thickness changes or beam drift).
    3. Integrated Intensity Map (Bottom Left): A spatial image of the total counts.
    This is your "search image" to help you correlate the spectral data with the physical structure of your sample.
    4. ZLP Alignment Detail (Bottom Right): A high-zoom view of the energy region around 0 eV of the Mean Spectrum.
    It includes a dashed green line at the "Target 0" to show exactly how much residual calibration error remains
    after your alignment.

    Parameters:
    -----------
    dataset : QuantEM Object
        The EELS dataset containing .array, .origin, and .sampling attributes.
    zlp_window : float, optional
        The energy range (± eV) to display in the ZLP zoom plot. Default is 5.0.
    title_suffix : str, optional
        Additional text to append to the figure title (e.g., "(RAW)" or "(Aligned)").

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object for further manipulation or saving.
    """
    data = dataset.array
    energy = np.asarray(dataset.energy_axis, dtype=float)

    mean_spec = np.mean(data, axis=(0, 1))
    zlp_pos = energy[np.argmax(mean_spec)]
    sum_img = np.sum(data, axis=2)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    fig.suptitle(f"QuantEM Diagnostic: {dataset.name} {title_suffix}", fontsize=16)

    # 1. Mean Spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(energy, mean_spec, color="black", label="Mean")
    ax1.axvline(0, color="green", ls=":", label="Target")
    ax1.set_title("Global Average Spectrum")
    ax1.legend()

    # 2. Spatial Variability
    ax2 = fig.add_subplot(gs[0, 1])
    # Take a 5x5 grid for better representation than 3x3
    yy, xx = np.meshgrid(
        np.linspace(0, data.shape[0] - 1, 5, dtype=int),
        np.linspace(0, data.shape[1] - 1, 5, dtype=int),
    )
    for y, x in zip(yy.flatten(), xx.flatten()):
        ax2.plot(energy, data[y, x, :], alpha=0.3, lw=0.5)
    ax2.set_title("Spatial Variation (Grid Samples)")

    # 3. Map
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(sum_img, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax3)
    ax3.set_title("Integrated Intensity")

    # 4. ZLP Zoom
    ax4 = fig.add_subplot(gs[1, 1])
    mask = (energy > zlp_pos - zlp_window) & (energy < zlp_pos + zlp_window)
    ax4.plot(energy[mask], mean_spec[mask], lw=2)
    ax4.axvline(0, color="green", ls=":")
    ax4.set_title("ZLP Alignment Detail")
    plt.close(fig)

    return fig


def plot_zlp_drift_diagnostics(dataset, title="ZLP Drift Analysis"):
    """
    QuantEM Diagnostic: Maps the ZLP position and calculates the drift distribution.
    Uses scipy.stats for Gaussian fitting.
    """
    data = dataset.array
    energy = np.asarray(dataset.energy_axis, dtype=float)

    # 1. Mask and find peak per pixel
    search_mask = (energy > -2.0) & (energy < 2.0)
    search_energies = energy[search_mask]
    peak_indices = np.argmax(data[:, :, search_mask], axis=2)
    zlp_map = search_energies[peak_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"QuantEM: {dataset.name} - {title}", fontsize=16)

    # Plot A: Map
    im = ax1.imshow(zlp_map, cmap="RdYlBu_r", origin="lower")
    plt.colorbar(im, ax=ax1, label="Energy Shift (eV)")

    # Plot B: Histogram + Scipy Fit
    flat_pos = zlp_map.flatten()
    mu, std = norm.fit(flat_pos)  # Professional scipy fitting

    ax2.hist(flat_pos, bins=30, density=True, alpha=0.6, color="skyblue")
    x_range = np.linspace(np.min(flat_pos), np.max(flat_pos), 100)
    ax2.plot(
        x_range,
        norm.pdf(x_range, mu, std),
        color="darkred",
        lw=2,
        label=f"Fit: μ={mu:.3f} eV, σ={std:.3f} eV",
    )
    ax2.legend()

    plt.tight_layout()

    plt.close(fig)

    return fig
