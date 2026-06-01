from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.strain_visualization import (
    plot_strain_panels,
    plot_strain_precision_histogram,
)


class StrainMap(AutoSerialize):
    """Strain tensor maps fit from per-position lattice vectors.

    Stores the reference-frame strain components ``e_rr`` (row), ``e_cc`` (col),
    ``e_rc`` (shear), and ``phi`` (infinitesimal rotation). The reference lattice
    is the median of the fitted ``g_u``/``g_v`` over a mask/ROI; the strain tensor
    is recomputed by :meth:`update_reference`.

    For nanobeam 4D-STEM the lattice vectors are measured in reciprocal space, so
    the shear/rotation signs are flipped relative to real space (``real_space=False``).

    Parameters
    ----------
    u_array : np.ndarray
        Per-position first lattice vector, shape ``(scan_row, scan_col, 2)``.
    v_array : np.ndarray
        Per-position second lattice vector, shape ``(scan_row, scan_col, 2)``.
    ds_shape : tuple of int
        Shape of the parent scan grid, used to size the strain maps.
    real_space : bool
        ``True`` for real-space lattice vectors; ``False`` for reciprocal-space
        (nanobeam) data, which flips the shear/rotation signs.
    u_ref : np.ndarray, optional
        Fixed reference for ``u``; if omitted the median over the mask/ROI is used.
        A value supplied here persists across re-fits.
    v_ref : np.ndarray, optional
        Fixed reference for ``v``; if omitted the median over the mask/ROI is used.
        A value supplied here persists across re-fits.
    mask : np.ndarray, optional
        ``(scan_row, scan_col)`` weighting/ROI mask; defaults to all ones (the full
        scan). Normalized to ``[0, 1]`` on assignment.
    ds_sampling : float, optional
        Real-space scan sampling (step size); defaults to ``1.0``.
    ds_units : str, optional
        Units for ``ds_sampling``; defaults to ``"pixels"``.
    """

    mask: np.ndarray | None = None
    real_space: bool = False

    e_rr: Dataset2d
    e_cc: Dataset2d
    e_rc: Dataset2d
    phi: Dataset2d

    u_ref: np.ndarray | None = None
    v_ref: np.ndarray | None = None
    u_array: np.ndarray
    v_array: np.ndarray

    ds_sampling: float = 1.0
    ds_units: str = "pixels"
    ds_shape: tuple[int, ...]

    def __init__(
        self,
        u_array: np.ndarray,
        v_array: np.ndarray,
        ds_shape: tuple[int, ...],
        real_space: bool,
        u_ref: np.ndarray | None = None,
        v_ref: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        ds_sampling: float | None = None,
        ds_units: str | None = None,
    ):
        super().__init__()
        self.u_array = u_array
        self.v_array = v_array

        self.ds_shape = ds_shape
        self.real_space = real_space

        self.ds_sampling = 1.0 if ds_sampling is None else ds_sampling
        self.ds_units = "pixels" if ds_units is None else ds_units

        self.mask = np.ones(ds_shape[:2]) if mask is None else mask
        self.mask = (self.mask - np.min(self.mask)) / np.max(self.mask)

        # user-supplied reference vectors persist across re-fits (None = use median)
        self._u_ref_fixed = None if u_ref is None else np.asarray(u_ref, dtype=float)
        self._v_ref_fixed = None if v_ref is None else np.asarray(v_ref, dtype=float)
        self.u_ref = None
        self.v_ref = None

        self.update_reference()

    # ---- main methods ----

    def update_reference(
        self,
        strain_mask: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
        v_ref: np.ndarray | None = None,
        plot_strain_roi: bool = False,
        **plot_kwargs,
    ) -> "StrainMap":
        """(Re)compute the reference lattice and strain tensor maps.

        Reference precedence: explicit ``u_ref``/``v_ref`` argument > vectors fixed at
        construction > median over ``strain_mask`` (if given) else over ``self.mask``
        else the global median.

        Parameters
        ----------
        strain_mask : np.ndarray, optional
            ``(scan_row, scan_col)`` ROI selecting the positions used to compute the
            median reference lattice. If omitted, ``self.mask`` (else the global
            median) is used.
        u_ref : np.ndarray, optional
            Explicit reference for ``u``; overrides both the construction-time fixed
            value and the median.
        v_ref : np.ndarray, optional
            Explicit reference for ``v``; overrides both the construction-time fixed
            value and the median.
        plot_strain_roi : bool, default=False
            If ``True``, show the recomputed strain via :meth:`plot_strain_roi`
            (color-scaled to the ROI) so the chosen reference region can be checked
            for flatness.
        **plot_kwargs
            Forwarded to :meth:`plot_strain_roi` when ``plot_strain_roi=True``.

        Returns
        -------
        StrainMap
            ``self``, with the reference lattice and strain maps recomputed.
        """
        u_med, v_med = _reference_lattice(self.u_array, self.v_array, self.mask, strain_mask)

        if u_ref is not None:
            self.u_ref = np.asarray(u_ref, dtype=float)
        elif self._u_ref_fixed is not None:
            self.u_ref = self._u_ref_fixed
        else:
            self.u_ref = u_med

        if v_ref is not None:
            self.v_ref = np.asarray(v_ref, dtype=float)
        elif self._v_ref_fixed is not None:
            self.v_ref = self._v_ref_fixed
        else:
            self.v_ref = v_med

        e_rr, e_cc, e_rc, phi = _strain_tensor(
            self.u_array, self.v_array, self.u_ref, self.v_ref, self.real_space
        )
        self.e_rr = Dataset2d.from_array(e_rr, name="strain e_rr", signal_units="fractional")
        self.e_cc = Dataset2d.from_array(e_cc, name="strain e_cc", signal_units="fractional")
        self.e_rc = Dataset2d.from_array(e_rc, name="strain e_rc", signal_units="fractional")
        self.phi = Dataset2d.from_array(phi, name="strain rotation", signal_units="radians")

        if plot_strain_roi:
            self.plot_strain_roi(strain_mask=strain_mask, **plot_kwargs)
        return self

    def rotate_strain(
        self, rotation_angle: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tensor-rotate the strain into a frame rotated by ``rotation_angle`` (degrees).

        The rotation field ``phi`` is invariant under frame rotation and is not
        transformed.

        Parameters
        ----------
        rotation_angle : float, default=0.0
            Frame rotation angle, in degrees.

        Returns
        -------
        tuple of np.ndarray
            ``(e_uu, e_vv, e_uv)`` strain components in the rotated frame.
        """
        return _rotate_strain_tensor(
            self.e_rr.array, self.e_cc.array, self.e_rc.array, rotation_angle
        )

    def plot_strain_roi(
        self,
        strain_mask: np.ndarray | None = None,
        plot_rotation: bool = True,
        cmap_strain: str = "RdBu_r",
        cmap_rotation: str = "PiYG",
        layout: str = "horizontal",
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot the strain in the raw row/col reference frame, color-scaled to the ROI.

        The color range is symmetric about zero and set by the largest absolute
        strain (and rotation) *inside the reference ROI* — ``strain_mask`` if given,
        else ``self.mask`` — so a well-chosen, strain-free reference region reads as
        flat (near mid-color) and any residual gradient or tilt stands out. The ROI
        itself is drawn in color while everything outside it is shown in greyscale,
        so the chosen reference region is obvious at a glance. Unlike
        :meth:`plot_strain`, no display rotation is applied: the panels show the raw
        ``e_rr``/``e_cc``/``e_rc`` that :meth:`update_reference` just computed.

        Parameters
        ----------
        strain_mask : np.ndarray, optional
            ROI defining the color range (and the reference region). If omitted,
            ``self.mask`` is used.
        plot_rotation : bool, default=True
            Whether to include the rotation (``phi``) panel.
        cmap_strain : str, default="RdBu_r"
            Colormap for the strain panels.
        cmap_rotation : str, default="PiYG"
            Colormap for the rotation panel.
        layout : {"horizontal", "vertical"}, default="horizontal"
            Panel arrangement.
        figsize : tuple of float, optional
            Figure size in inches; if omitted it is derived from the layout.
        **kwargs
            Forwarded to
            :func:`~quantem.diffraction.strain_visualization.plot_strain_panels`.

        Returns
        -------
        tuple
            ``(fig, ax)`` from :func:`plot_strain_panels`.
        """
        roi_src = self.mask if strain_mask is None else strain_mask
        e_rr, e_cc, e_rc, phi = (
            self.e_rr.array,
            self.e_cc.array,
            self.e_rc.array,
            self.phi.array,
        )

        inside = np.asarray(roi_src) > 0 if roi_src is not None else np.ones(e_rr.shape, bool)
        if not inside.any():
            inside = np.ones(e_rr.shape, bool)

        strain_stack = np.stack([e_rr[inside], e_cc[inside], e_rc[inside]])
        smax = float(np.nanmax(np.abs(strain_stack))) * 100.0
        rmax = float(np.rad2deg(np.nanmax(np.abs(phi[inside]))))
        smax = smax if smax > 0 else 1e-6
        rmax = rmax if rmax > 0 else 1e-6

        return plot_strain_panels(
            e_rr,
            e_cc,
            e_rc,
            phi,
            self.mask,
            self.u_ref,
            self.v_ref,
            self.ds_shape,
            ds_sampling=self.ds_sampling,
            ds_units=self.ds_units,
            strain_range_percent=(-smax, smax),
            rotation_range_degrees=(-rmax, rmax),
            roi=inside,
            plot_rotation=plot_rotation,
            cmap_strain=cmap_strain,
            cmap_rotation=cmap_rotation,
            layout=layout,
            figsize=figsize,
            panel_titles=(
                r"$\epsilon_{rr}$ $\updownarrow$",
                r"$\epsilon_{cc}$ $\leftrightarrow$",
                r"$\epsilon_{rc}$ $\nwarrow\!\!\!\!\!\!\!\!\!\:\searrow$",
            ),
            **kwargs,
        )

    def plot_strain(
        self,
        rotation_angle: float = 20.0,
        strain_range_percent: tuple[float, float] = (-3.0, 3.0),
        rotation_range_degrees: tuple[float, float] = (-2.0, 2.0),
        mask_range: tuple[float, float] = (0.0, 1.0),
        plot_rotation: bool = True,
        plot_gvecs: bool = False,
        plot_scalebar: bool = False,
        cmap_strain: str = "RdBu_r",
        cmap_rotation: str = "PiYG",
        layout: str = "horizontal",
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot the strain (rotated into the display frame) and rotation panels.

        Parameters
        ----------
        rotation_angle : float, default=20.0
            Angle (degrees) by which the strain tensor is rotated into the display
            frame before plotting.
        strain_range_percent : tuple of float, default=(-3.0, 3.0)
            Symmetric color range for the strain panels, in percent.
        rotation_range_degrees : tuple of float, default=(-2.0, 2.0)
            Symmetric color range for the rotation panel, in degrees.
        mask_range : tuple of float, default=(0.0, 1.0)
            ``(low, high)`` window remapping the mask brightness: positions with
            mask ``>= high`` are shown at full color, ``<= low`` are black, and
            values between ramp linearly from black to full. The default leaves the
            normalized mask unchanged.
        plot_rotation : bool, default=True
            Whether to include the rotation (``phi``) panel.
        plot_gvecs : bool, default=False
            Whether to overlay the reference lattice vectors.
        plot_scalebar : bool, default=False
            Whether to draw a real-space scale bar.
        cmap_strain : str, default="RdBu_r"
            Colormap for the strain panels.
        cmap_rotation : str, default="PiYG"
            Colormap for the rotation panel.
        layout : {"horizontal", "vertical"}, default="horizontal"
            Panel arrangement.
        figsize : tuple of float, optional
            Figure size in inches; if omitted it is derived from the layout.
        **kwargs
            Forwarded to
            :func:`~quantem.diffraction.strain_visualization.plot_strain_panels`.

        Returns
        -------
        tuple
            ``(fig, ax)`` from :func:`plot_strain_panels`.
        """
        e_uu, e_vv, e_uv = self.rotate_strain(rotation_angle)
        return plot_strain_panels(
            e_uu,
            e_vv,
            e_uv,
            self.phi.array,
            self.mask,
            self.u_ref,
            self.v_ref,
            self.ds_shape,
            ds_sampling=self.ds_sampling,
            ds_units=self.ds_units,
            strain_range_percent=strain_range_percent,
            rotation_range_degrees=rotation_range_degrees,
            mask_range=mask_range,
            plot_rotation=plot_rotation,
            plot_gvecs=plot_gvecs,
            plot_scalebar=plot_scalebar,
            cmap_strain=cmap_strain,
            cmap_rotation=cmap_rotation,
            layout=layout,
            figsize=figsize,
            **kwargs,
        )

    def estimate_strain_precision(
        self,
        mask_range: tuple[float, float] = (0.0, 1.0),
        rotation_angle: float = 0.0,
        window: int = 5,
        mask_threshold: float = 0.5,
        min_neighbors: int = 3,
        component: str = "combined",
        bins: int = 50,
        bounds: tuple[float, float] | None = None,
        plot: bool = True,
        returnfig: bool = False,
    ):
        """Estimate strain *precision* (random scatter) from local median deviations.

        This measures repeatability, not accuracy. Without a ground truth (e.g. a
        simulation) it cannot detect systematic error — only how far each position
        scatters from its local neighborhood. For every position the deviation from
        the median of its surrounding well-indexed neighbors is

            ``error(r, c) = | strain(r, c) - median( strain over neighbors with
            scaled mask > mask_threshold ) |``

        computed for each tensor component (the center position is excluded from its
        own median). The three strain components are reduced to one rotation-invariant
        number via the Frobenius norm of the symmetric strain-tensor deviation,

            ``combined = sqrt(d_uu**2 + d_vv**2 + 2*d_uv**2)``,

        (equivalently the root-sum-square of the principal-strain deviations) so a
        single strain precision can be quoted and compared between datasets. Rotation
        precision is reported separately, not folded into ``combined``.

        Each component's precision is summarized as the mask-weighted **RMS** of its
        per-position deviations — a sigma-like scatter — and a weighted histogram of
        those deviations is shown. RMS combining is self-consistent under the
        Frobenius sum: ``rms(combined) == sqrt(rms_uu**2 + rms_vv**2 + 2*rms_uv**2)``.

        Parameters
        ----------
        mask_range : tuple of float, default=(0.0, 1.0)
            ``(low, high)`` window remapping :attr:`mask` to ``[0, 1]`` (same
            convention as :meth:`plot_strain`); the remapped mask both selects
            neighbors (``> mask_threshold``) and weights the histogram and mean.
        rotation_angle : float, default=0.0
            Frame rotation (degrees) applied before measuring per-component precision,
            matching :meth:`plot_strain`. ``0`` reports the raw row/col frame
            (``e_uu == e_rr`` ...). The combined number is rotation-invariant.
        window : int, default=5
            Odd edge length (px) of the neighborhood bounding box; the footprint is
            the inscribed disk of radius ``window / 2`` (3 -> 8 neighbors, 5 -> 20,
            7 -> 36). A pure linear strain ramp cancels in the (symmetric) median, so
            larger windows mostly just steady the median — at the cost of blurring
            *curved* strain and biasing the masked edges. ``5`` roughly halves the
            noise-floor over-estimate of ``3`` (~9% -> ~4%) while staying local.
        mask_threshold : float, default=0.5
            A neighbor contributes to the local median only if its scaled mask
            exceeds this value.
        min_neighbors : int, default=3
            Minimum number of valid neighbors required; positions with fewer get no
            precision estimate (``nan``, dropped from the statistics).
        component : {"combined","e_uu","e_vv","e_uv","rotation"}, default="combined"
            Which error distribution to histogram.
        bins : int, default=50
            Number of histogram bins (or a sequence of bin edges).
        bounds : tuple of float, optional
            ``(low, high)`` histogram range in display units (percent for strain,
            degrees for rotation). Fix it to compare datasets on the same axis. Mass
            outside the range (or outside an explicit ``bins`` edge array) is piled
            into the edge bins as overflow rather than dropped, so the histogram
            always integrates to the same weight as the reported RMS.
        plot : bool, default=True
            If ``True``, draw the weighted precision histogram.
        returnfig : bool, default=False
            If ``True``, return ``(fig, ax)`` instead of the results dict.

        Returns
        -------
        dict or tuple
            A results dict with the mask-weighted ``rms`` precision per component and
            the ``combined`` value (strain in percent, rotation in degrees), the
            normalized ``counts`` and ``edges`` of the histogrammed ``component`` (and
            ``counts_raw``, the weighted bin sums), ``out_of_range_fraction`` (weighted
            mass piled into the edge bins as overflow), and the chosen settings; or
            ``(fig, ax)`` when ``returnfig=True``.
        """
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be an odd integer >= 3.")
        valid_components = ("combined", "e_uu", "e_vv", "e_uv", "rotation")
        if component not in valid_components:
            raise ValueError(f"component must be one of {valid_components}.")

        # number of neighbors in the circular footprint (matches _local_masked_median)
        p = window // 2
        oy, ox = np.ogrid[-p : p + 1, -p : p + 1]
        n_neighbors = int(np.sum((oy ** 2 + ox ** 2) <= (window / 2.0) ** 2) - 1)

        # per-component fields in the (optionally rotated) display frame; phi is
        # rotation-invariant and is carried through unchanged
        e_uu, e_vv, e_uv = self.rotate_strain(rotation_angle)
        fields = {"e_uu": e_uu, "e_vv": e_vv, "e_uv": e_uv, "rotation": self.phi.array}

        # remap the mask exactly as plot_strain does, then use it both to select
        # neighbors (> mask_threshold) and to weight the histogram / mean
        low, high = float(mask_range[0]), float(mask_range[1])
        m = np.asarray(self.mask, dtype=float)
        if high > low:
            scaled = np.clip((m - low) / (high - low), 0.0, 1.0)
        else:
            scaled = (m >= high).astype(float)
        valid = scaled > float(mask_threshold)

        # per-component local-median deviation, native units (fractional / radians)
        dev = {
            name: np.abs(field - _local_masked_median(field, valid, window, min_neighbors))
            for name, field in fields.items()
        }
        # single rotation-invariant number: Frobenius norm of the symmetric
        # strain-tensor deviation (== root-sum-square of the principal-strain
        # deviations). Rotation is reported separately, not folded in: in nanobeam
        # data it is partly a systematic (tilt/descan) and would mix radians into a
        # percent figure.
        dev["combined"] = np.sqrt(
            dev["e_uu"] ** 2 + dev["e_vv"] ** 2 + 2.0 * dev["e_uv"] ** 2
        )

        # display-unit scaling: strain -> percent, rotation -> degrees
        scale = {
            "e_uu": 100.0,
            "e_vv": 100.0,
            "e_uv": 100.0,
            "rotation": float(np.rad2deg(1.0)),
            "combined": 100.0,
        }

        def _weighted_rms(err_native: np.ndarray, factor: float) -> float:
            e = err_native * factor
            finite = np.isfinite(e)
            w = scaled[finite]
            wsum = float(w.sum())
            return float(np.sqrt(np.sum(e[finite] ** 2 * w) / wsum)) if wsum > 0 else float("nan")

        # mask-weighted RMS deviation (a sigma-like scatter). Under the Frobenius
        # sum this is self-consistent: rms(combined) == sqrt(rms_uu**2 + rms_vv**2
        # + 2*rms_uv**2), so the combined number agrees with the per-component ones.
        rms = {name: _weighted_rms(dev[name], scale[name]) for name in scale}

        # weighted histogram of the chosen component, in display units. The headline
        # RMS above is computed over *all* finite positions, so the histogram must be
        # too: with an explicit bin-edge array (e.g. bins=np.arange(0, 1, 0.02))
        # np.histogram ignores `range` and silently DROPS everything outside the
        # edges, then renormalizing makes the plot look far tighter than the reported
        # RMS. Instead pile any out-of-range mass into the edge bins (overflow) so the
        # histogram integrates to the same weight the RMS sees, and report the
        # fraction that landed there.
        e = dev[component] * scale[component]
        finite = np.isfinite(e)
        e_f = e[finite]
        w_f = scaled[finite]
        edges = np.histogram_bin_edges(e_f, bins=bins, range=bounds)
        lo, hi = float(edges[0]), float(edges[-1])
        wtot = float(w_f.sum())
        frac_below = float(w_f[e_f < lo].sum()) / wtot if wtot > 0 else 0.0
        frac_above = float(w_f[e_f > hi].sum()) / wtot if wtot > 0 else 0.0
        out_of_range_fraction = frac_below + frac_above
        counts_raw, edges = np.histogram(np.clip(e_f, lo, hi), bins=edges, weights=w_f)
        total = float(counts_raw.sum())
        counts = counts_raw / total if total > 0 else counts_raw

        unit = "°" if component == "rotation" else "%"
        result = {
            "rms": rms,
            "component": component,
            "unit": unit,
            "counts": counts,
            "counts_raw": counts_raw,
            "edges": edges,
            "out_of_range_fraction": out_of_range_fraction,
            "window": int(window),
            "n_neighbors": n_neighbors,
            "mask_threshold": float(mask_threshold),
            "mask_range": (low, high),
            "rotation_angle": float(rotation_angle),
        }

        print("Strain precision  (RMS of local median deviation, weighted by scaled mask)")
        print(
            f"  reference={n_neighbors} neighbors (disk, window={window})  "
            f"mask>{mask_threshold:g}  min_neighbors={min_neighbors}  "
            f"rotation_angle={rotation_angle:g} deg"
        )
        for name in ("e_uu", "e_vv", "e_uv"):
            print(f"    {name:<9}: {rms[name]:7.4f} %")
        print(f"    {'rotation':<9}: {rms['rotation']:7.4f} deg")
        print(
            f"    {'combined':<9}: {rms['combined']:7.4f} %   "
            "(strain-only Frobenius norm; rotation excluded)"
        )
        if out_of_range_fraction > 0:
            print(
                f"  note: {100 * out_of_range_fraction:.1f}% of weighted mass fell "
                f"outside the histogram range [{lo:g}, {hi:g}] {unit} and was piled "
                "into the edge bins (overflow); widen `bins`/`bounds` to resolve the "
                "tail. The RMS above includes it."
            )

        if not (plot or returnfig):
            return result

        fig, ax = plot_strain_precision_histogram(edges, counts, rms, component, unit)
        if returnfig:
            return fig, ax
        return result


# ---- module-level fitting functions ----


def _local_masked_median(
    field: np.ndarray,
    valid: np.ndarray,
    window: int,
    min_neighbors: int,
) -> np.ndarray:
    """Median of each position's surrounding neighbors over valid (masked) pixels.

    The center position is excluded ("surrounding" only); a neighbor contributes
    only where ``valid`` is True and the field is finite. Neighbors are taken over a
    circular (isotropic) footprint of radius ``window / 2`` inscribed in the
    ``window`` x ``window`` box — a disk avoids the square's far corners, which
    over-weight the diagonals and sample the most strain-different points. Positions
    left with fewer than ``min_neighbors`` contributing neighbors return ``nan``.

    Parameters
    ----------
    field : np.ndarray
        ``(scan_row, scan_col)`` field to take local medians of.
    valid : np.ndarray
        ``(scan_row, scan_col)`` boolean mask of usable neighbor positions.
    window : int
        Odd edge length of the bounding box; the footprint is the disk of radius
        ``window / 2`` within it (3 -> 8 neighbors, 5 -> 20, 7 -> 36).
    min_neighbors : int
        Minimum contributing neighbors required, else ``nan``.

    Returns
    -------
    np.ndarray
        ``(scan_row, scan_col)`` local masked median (``nan`` where undefined).
    """
    p = window // 2
    fpad = np.pad(np.asarray(field, dtype=float), p, mode="constant", constant_values=np.nan)
    vpad = np.pad(np.asarray(valid, dtype=bool), p, mode="constant", constant_values=False)

    # writable per-position (window, window) neighborhoods
    fw = sliding_window_view(fpad, (window, window)).copy()
    vw = sliding_window_view(vpad, (window, window))
    fw[~vw] = np.nan
    fw[:, :, p, p] = np.nan  # exclude the center position from its own median
    # restrict the square box to a circular footprint of radius window/2
    oy, ox = np.ogrid[-p : p + 1, -p : p + 1]
    outside = (oy ** 2 + ox ** 2) > (window / 2.0) ** 2
    fw[:, :, outside] = np.nan

    flat = fw.reshape(fw.shape[0], fw.shape[1], -1)
    count = np.sum(np.isfinite(flat), axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(flat, axis=-1)
    med[count < min_neighbors] = np.nan
    return med


def _reference_lattice(
    u_array: np.ndarray,
    v_array: np.ndarray,
    mask: np.ndarray | None = None,
    strain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Median reference lattice vectors over an ROI / mask, else the global median.

    Parameters
    ----------
    u_array : np.ndarray
        Per-position first lattice vector, shape ``(scan_row, scan_col, 2)``.
    v_array : np.ndarray
        Per-position second lattice vector, shape ``(scan_row, scan_col, 2)``.
    mask : np.ndarray, optional
        ``(scan_row, scan_col)`` ROI; positions equal to 1 are included. Used when
        ``strain_mask`` is not given.
    strain_mask : np.ndarray, optional
        ``(scan_row, scan_col)`` ROI taking precedence over ``mask``.

    Returns
    -------
    tuple of np.ndarray
        ``(u_ref, v_ref)``, each a length-2 reference vector.
    """
    if strain_mask is not None:
        m = np.asarray(strain_mask == 1, dtype=bool)
    elif mask is not None:
        m = np.asarray(mask == 1, dtype=bool)
    else:
        m = None

    # nan-median: positions fit_lattice could not fit are NaN and must be ignored,
    # otherwise the reference (and hence the whole strain map) collapses to NaN.
    if m is None or not m.any():
        u_ref = np.nanmedian(u_array.reshape(-1, 2), axis=0)
        v_ref = np.nanmedian(v_array.reshape(-1, 2), axis=0)
    else:
        u_ref = np.array((np.nanmedian(u_array[m, 0]), np.nanmedian(u_array[m, 1])), dtype=float)
        v_ref = np.array((np.nanmedian(v_array[m, 0]), np.nanmedian(v_array[m, 1])), dtype=float)
    return u_ref, v_ref


def _strain_tensor(
    u_array: np.ndarray,
    v_array: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    real_space: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-position strain tensor from lattice vectors relative to a reference.

    For reciprocal-space (nanobeam) data the shear and rotation are sign-flipped via
    ``const = -1``.

    Parameters
    ----------
    u_array : np.ndarray
        Per-position first lattice vector, shape ``(scan_row, scan_col, 2)``.
    v_array : np.ndarray
        Per-position second lattice vector, shape ``(scan_row, scan_col, 2)``.
    u_ref : np.ndarray
        Reference first lattice vector (length 2).
    v_ref : np.ndarray
        Reference second lattice vector (length 2).
    real_space : bool
        ``True`` for real-space data; ``False`` (nanobeam, reciprocal space) flips
        the shear and rotation signs.

    Returns
    -------
    tuple of np.ndarray
        ``(e_rr, e_cc, e_rc, phi)``, each of shape ``(scan_row, scan_col)``.
    """
    scan_r, scan_c = u_array.shape[0], u_array.shape[1]
    Uref = np.stack((u_ref, v_ref), axis=1).astype(float)
    strain_trans = np.zeros((scan_r, scan_c, 2, 2))
    for r in range(scan_r):
        for c in range(scan_c):
            U = np.stack((u_array[r, c, :], v_array[r, c, :]), axis=1)
            # Positions fit_lattice could not fit are NaN; a degenerate (collinear)
            # fit is singular. Either way there is no meaningful inverse -- leave the
            # strain NaN (masked out downstream) rather than feeding NaN into pinv,
            # whose SVD does not converge and raises LinAlgError.
            if not np.all(np.isfinite(U)) or abs(np.linalg.det(U)) < 1e-12:
                strain_trans[r, c, :, :] = np.nan
                continue
            strain_trans[r, c, :, :] = Uref @ np.linalg.inv(U)

    const = 1 if real_space else -1
    e_rr = strain_trans[:, :, 0, 0] - 1
    e_cc = strain_trans[:, :, 1, 1] - 1
    e_rc = strain_trans[:, :, 1, 0] * 0.5 * const + strain_trans[:, :, 0, 1] * 0.5 * const
    phi = strain_trans[:, :, 1, 0] * -0.5 * const + strain_trans[:, :, 0, 1] * 0.5 * const
    return e_rr, e_cc, e_rc, phi


def _rotate_strain_tensor(
    e_rr: np.ndarray,
    e_cc: np.ndarray,
    e_rc: np.ndarray,
    rotation_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a 2D strain tensor by ``rotation_angle`` (degrees).

    Parameters
    ----------
    e_rr : np.ndarray
        Row-row (normal) strain component.
    e_cc : np.ndarray
        Column-column (normal) strain component.
    e_rc : np.ndarray
        Row-column (shear) strain component.
    rotation_angle : float
        Frame rotation angle, in degrees.

    Returns
    -------
    tuple of np.ndarray
        ``(e_uu, e_vv, e_uv)`` in the rotated frame.
    """
    angle = np.deg2rad(rotation_angle)
    c = np.cos(angle)
    s = np.sin(angle)
    e_uu = e_rr * (c * c) + 2.0 * e_rc * (c * s) + e_cc * (s * s)
    e_vv = e_rr * (s * s) - 2.0 * e_rc * (c * s) + e_cc * (c * c)
    e_uv = (e_cc - e_rr) * (c * s) + e_rc * (c * c - s * s)
    return e_uu, e_vv, e_uv
