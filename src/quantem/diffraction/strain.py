from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

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

    Two measurement modalities are supported and give identical strain for the same
    deformation, so correlation and cepstral maps can be compared directly:
    reciprocal-space Bragg vectors (``real_space=False``, nanobeam correlation) and
    real-space cepstral/autocorrelation vectors (``real_space=True``).

    Parameters
    ----------
    u_array : np.ndarray
        Per-position first lattice vector, shape ``(scan_row, scan_col, 2)``.
    v_array : np.ndarray
        Per-position second lattice vector, shape ``(scan_row, scan_col, 2)``.
    ds_shape : tuple of int
        Shape of the parent scan grid, used to size the strain maps.
    real_space : bool
        ``False`` for reciprocal-space (Bragg/correlation) lattice vectors; ``True``
        for real-space (cepstral autocorrelation / DPC) vectors. Both modalities are
        arranged to yield matching strain (see :func:`_strain_tensor`).
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
        q_to_r_rotation_ccw_deg: float = 0.0,
        q_transpose: bool = False,
    ):
        super().__init__()
        self.u_array = u_array
        self.v_array = v_array
        
        self.q_to_r_rotation_ccw_deg = q_to_r_rotation_ccw_deg
        self.q_transpose = q_transpose

        self.u_array = _raw_vec_to_display(self.u_array, rotation_ccw_deg = q_to_r_rotation_ccw_deg, transpose=q_transpose)
        self.v_array = _raw_vec_to_display(self.v_array, rotation_ccw_deg = q_to_r_rotation_ccw_deg, transpose=q_transpose)

        self.ds_shape = ds_shape
        self.real_space = real_space

        self.ds_sampling = 1.0 if ds_sampling is None else ds_sampling
        self.ds_units = "pixels" if ds_units is None else ds_units

        m = np.ones(ds_shape[:2], dtype=float) if mask is None else np.asarray(mask, dtype=float)
        m_lo = np.nanmin(m)
        m_hi = np.nanmax(m)
        if not (np.isfinite(m_lo) and np.isfinite(m_hi)) or m_hi <= m_lo:
            m = np.ones_like(m)
        elif m_lo < 0.0 or m_hi > 1.0:
            m = (m - m_lo) / (m_hi - m_lo)
        self.mask = m

        # user-supplied reference vectors persist across re-fits (None = use median)
        self._u_ref_fixed = None if u_ref is None else _raw_vec_to_display(np.asarray(u_ref, dtype=float), 
                                                                                                                        rotation_ccw_deg=q_to_r_rotation_ccw_deg, 
                                                                                                                        transpose=q_transpose)
        self._v_ref_fixed = None if v_ref is None else _raw_vec_to_display(np.asarray(v_ref, dtype=float), 
                                                                                                                        rotation_ccw_deg=q_to_r_rotation_ccw_deg, 
                                                                                                                        transpose=q_transpose)
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
        define_in_rotated_frame: bool = False,
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
        define_in_rotated_frame: bool, default = False
            If ''True'' means the u_ref and v_ref passed into the function is defined in the rotated detector frame
        **plot_kwargs
            Forwarded to :meth:`plot_strain_roi` when ``plot_strain_roi=True``.

        Returns
        -------
        StrainMap
            ``self``, with the reference lattice and strain maps recomputed.
        """
        u_med, v_med = _reference_lattice(self.u_array, self.v_array, self.mask, strain_mask)

        if u_ref is not None:
            if define_in_rotated_frame:
                self.u_ref = np.asarray(u_ref, dtype=float)
            else:
                self.u_ref = _raw_vec_to_display(
                                        np.asarray(u_ref, dtype=float),
                                        rotation_ccw_deg=self.q_to_r_rotation_ccw_deg, 
                                        transpose=self.q_transpose)
        elif self._u_ref_fixed is not None:
            self.u_ref = self._u_ref_fixed
        else:
            self.u_ref = u_med

        if v_ref is not None:
            if define_in_rotated_frame:
                self.v_ref = np.asarray(v_ref, dtype=float)
            else:
                self.v_ref = _raw_vec_to_display(
                                                    np.asarray(v_ref, dtype=float),
                                                    rotation_ccw_deg=self.q_to_r_rotation_ccw_deg, 
                                                    transpose=self.q_transpose)
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
        strain_range_percent: tuple[float, float] | None = None,
        rotation_range_degrees: tuple[float, float] | None = None,
        transpose_image: bool = False,
        rotate_title: bool = False,
        plot_dilation: bool = False,
        layout: str = "horizontal",
        arrow_style: str = "title",
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
        strain_range_percent : tuple of float, default=(-3.0, 3.0)
            Symmetric color range for the strain panels, in percent.
        rotation_range_degrees : tuple of float, default=(-2.0, 2.0)
            Symmetric color range for the rotation panel, in degrees.
        transpose_image: bool, default = False
            If ''True'' transpose the real space image before plotting strain
        rotate_title: bool, default = False
            If ''True'', rotates panel titles by 90 degrees.
        plot_dilation: bool, default = False
            If ''True'' plots euu + evv, and euv instead of euu, evv, euv
        layout : {"horizontal", "vertical"}, default="horizontal"
            Panel arrangement.
        arrow_style: str, default="title"
            Plots the directional arrows along with the strain titles. 
            Alternatively can be "legend" where it plots it on the side
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

        if arrow_style not in ("title", "legend"):
            raise ValueError("arrow_style must be 'title' or 'legend'")

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
            strain_range_percent=(-smax, smax) if strain_range_percent is None else strain_range_percent,
            rotation_range_degrees=(-rmax, rmax) if rotation_range_degrees is None else rotation_range_degrees,
            roi=inside,
            plot_rotation=plot_rotation,
            cmap_strain=cmap_strain,
            cmap_rotation=cmap_rotation,
            layout=layout,
            transpose_image = transpose_image,
            rotate_title = rotate_title,
            plot_dilation = plot_dilation,
            figsize=figsize,
            panel_titles=(
                r"$\epsilon_{rr}$ $\updownarrow$",
                r"$\epsilon_{cc}$ $\leftrightarrow$",
                r"$\epsilon_{rc}$ $\nwarrow\!\!\!\!\!\!\!\!\!\:\searrow$",
            ),
            arrow_style = arrow_style,
            **kwargs,
        )

    def plot_strain(
        self,
        rotation_angle: float = 0.0,
        strain_range_percent: tuple[float, float] = (-3.0, 3.0),
        rotation_range_degrees: tuple[float, float] = (-2.0, 2.0),
        mask_range: tuple[float, float] = (0.0, 1.0),
        plot_rotation: bool = True,
        plot_gvecs: bool = False,
        plot_scalebar: bool = False,
        cmap_strain: str = "RdBu_r",
        cmap_rotation: str = "PiYG",
        transpose_image: bool = False,
        transpose_strain: bool = False,
        rotate_title: bool = False,
        plot_dilation: bool = False,
        layout: str = "horizontal",
        arrow_style: str = "title",
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot the strain (rotated into the display frame) and rotation panels.

        Parameters
        ----------
        rotation_angle : float, default=0.0
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
        transpose_image: bool, default = False
            If ''True'' transpose the real space image before plotting strain
        transpose_strain : bool, default=False
            If ``True``, transpose the detector (row/col) axes before rotating,
            matching the DPC convention (see
            :func:`~quantem.diffraction.strain_autocorrelation._raw_vec_to_display`):
            transpose first, then rotate. This swaps the normal strain components,
            leaves the shear unchanged, and reverses the sign of the rotation field.
        rotate_title: bool, default = False
            If ''True'', rotates panel titles by 90 degrees.
        plot_dilation: bool, default = False
            If ''True'' plots euu + evv, and euv instead of euu, evv, euv
        layout : {"horizontal", "vertical"}, default="horizontal"
            Panel arrangement.
        figsize : tuple of float, optional
            Figure size in inches; if omitted it is derived from the layout.
        arrow_style: str, default="title"
            Plots the directional arrows along with the strain titles. 
            Alternatively can be "legend" where it plots it on the side
        **kwargs
            Forwarded to
            :func:`~quantem.diffraction.strain_visualization.plot_strain_panels`.

        Returns
        -------
        tuple
            ``(fig, ax)`` from :func:`plot_strain_panels`.
        """
        if arrow_style not in ("title", "legend"):
            raise ValueError("arrow_style must be 'title' or 'legend'")
        
        e_rr = self.e_rr.array
        e_cc = self.e_cc.array
        e_rc = self.e_rc.array
        phi = self.phi.array
        if transpose_strain:
            # Detector-axis transpose, applied BEFORE the rotation to match the DPC
            # convention shared across quantem (see _raw_vec_to_display): swapping the
            # (row, col) axes swaps the normal strains, keeps the shear unchanged, and
            # reverses the sense of the rotation field.
            e_rr, e_cc = e_cc, e_rr
            phi = -phi
        e_uu, e_vv, e_uv = _rotate_strain_tensor(e_rr, e_cc, e_rc, rotation_angle)
        return plot_strain_panels(
            e_uu,
            e_vv,
            e_uv,
            phi,
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
            transpose_image = transpose_image,
            rotate_title = rotate_title,
            plot_dilation = plot_dilation,
            figsize=figsize,
            strain_rotation_angle=rotation_angle,
            arrow_style = arrow_style,
            **kwargs,
        )

    def estimate_strain_precision(
        self,
        mask_range: tuple[float, float] = (0.0, 1.0),
        rotation_angle: float = 0.0,
        window: int = 5,
        mask_threshold: float = 0.5,
        min_neighbors: int = 3,
        require_full_neighborhood: bool = True,
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

        Each component's precision is summarized by the mask-weighted **median** of its
        per-position deviations — the center of the histogram bulk. The median is used
        (not the mean or RMS) because a handful of bad-fit pixels form a heavy tail that
        would drag a second moment far to the right of where the distribution actually
        sits, leaving the reported number disconnected from the histogram; the median
        ignores that tail. A weighted histogram of the chosen component is shown, marked
        with its median.

        Parameters
        ----------
        mask_range : tuple of float, default=(0.0, 1.0)
            ``(low, high)`` window remapping :attr:`mask` to ``[0, 1]`` (same
            convention as :meth:`plot_strain`); the remapped mask both selects which
            positions are trusted (``> mask_threshold`` -- used as neighbors *and* as
            the set the precision is computed over) and weights the histogram and the
            median.
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
            A position is trusted only if its scaled mask exceeds this value. Trusted
            positions are the ones used as local-median neighbors *and* the ones whose
            deviations enter the reported median and histogram; sub-threshold positions
            are excluded from both (not merely down-weighted), so a poorly-indexed
            pixel cannot leak its scatter into the precision.
        min_neighbors : int, default=3
            Minimum number of valid neighbors required; positions with fewer get no
            precision estimate (``nan``, dropped from the statistics).
        component : {"combined","e_uu","e_vv","e_uv","rotation"}, default="combined"
            Which error distribution to histogram.
        bins : int, default=50
            Number of histogram bins, or a sequence of explicit bin edges. With a
            bin *count* and no ``bounds``, the range defaults to ``[0, weighted 99th
            percentile]`` of the trusted deviations -- robust to the heavy outlier
            tail, which otherwise sets the range to its max and crushes the bulk into
            the first bin. Passing explicit edges (or ``bounds``) overrides this.
        bounds : tuple of float, optional
            ``(low, high)`` histogram range in display units (percent for strain,
            degrees for rotation). Fix it to compare datasets on the same axis, or to
            see the full tail. Values outside the range are left out of the bars (no
            overflow spike); the median is computed from all trusted positions
            regardless, and ``out_of_range_fraction`` records how much was off-range.
        plot : bool, default=True
            If ``True``, draw the weighted precision histogram.
        returnfig : bool, default=False
            If ``True``, return ``(fig, ax)`` instead of the results dict.

        Returns
        -------
        dict or tuple
            A results dict with the ``precision`` (mask-weighted median local
            deviation) per component and ``combined`` (strain in percent, rotation in
            degrees), the normalized ``counts`` and ``edges`` of the histogrammed
            ``component`` (and ``counts_raw``, the weighted bin sums),
            ``out_of_range_fraction`` (weighted mass outside the histogram range,
            excluded from the bars), and the chosen settings; or ``(fig, ax)`` when
            ``returnfig=True``.
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
        if require_full_neighborhood:
            min_neighbors = n_neighbors

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

        # Precision = the weighted MEDIAN of each per-position deviation distribution,
        # in display units. Restricted to trusted positions (valid == scaled >
        # mask_threshold, the SAME set used to pick neighbors) and mask-weighted within
        # it -- otherwise sub-threshold junk pixels, already excluded as neighbors,
        # would leak in. The median sits at the center of the histogram bulk and is
        # immune to the heavy outlier tail that a mean / RMS would chase out to the
        # right (a few bad-fit pixels dominate a second moment but not the median).
        def _weighted_median(err_native: np.ndarray, factor: float) -> float:
            e = err_native * factor
            use = np.isfinite(e) & valid
            return _weighted_quantile(e[use], scaled[use], 0.5)

        precision = {name: _weighted_median(dev[name], scale[name]) for name in scale}

        # weighted histogram of the chosen component, over the same trusted positions
        # as the median above. This is purely a picture of the common error values, so
        # anything beyond the bin range is left OUT of the bars -- no overflow spike at
        # the edge to crush the bulk. Nothing is lost: the median is computed from all
        # trusted positions regardless. Bars are normalized by the total trusted
        # weight, so each bar is the true fraction of all trusted positions and the
        # off-range mass simply isn't drawn (the bars sum to 1 - out_of_range_fraction).
        e = dev[component] * scale[component]
        use = np.isfinite(e) & valid  # trusted positions only, consistent with median
        e_f = e[use]
        w_f = scaled[use]
        # Default histogram range: a robust weighted upper percentile, NOT the raw
        # max. A handful of bad-fit positions can reach tens of percent; used as the
        # range they crush the entire bulk into the first bin and leave the rest of
        # the axis empty (a spurious "spike at 0" plus a far outlier spike). Capping
        # at the weighted 99th percentile keeps the common error values readable; the
        # few positions past it spill into out_of_range_fraction (reported, not
        # drawn). An explicit `bounds`, or passing bin EDGES as `bins`, overrides it.
        if bounds is None and np.ndim(bins) == 0 and e_f.size and float(w_f.sum()) > 0:
            hi_default = _weighted_quantile(e_f, w_f, 0.99)
            if np.isfinite(hi_default) and hi_default > 0:
                bounds = (0.0, hi_default)
        edges = np.histogram_bin_edges(e_f, bins=bins, range=bounds)
        lo, hi = float(edges[0]), float(edges[-1])
        wtot = float(w_f.sum())
        frac_below = float(w_f[e_f < lo].sum()) / wtot if wtot > 0 else 0.0
        frac_above = float(w_f[e_f > hi].sum()) / wtot if wtot > 0 else 0.0
        out_of_range_fraction = frac_below + frac_above
        counts_raw, edges = np.histogram(e_f, bins=edges, weights=w_f)
        counts = counts_raw / wtot if wtot > 0 else counts_raw

        unit = "°" if component == "rotation" else "%"
        result = {
            "precision": precision,
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
            "min_neighbors": int(min_neighbors),
            "require_full_neighborhood": bool(require_full_neighborhood),
        }

        print("Strain precision  (median local deviation, mask-weighted)")
        print(
            f"  reference={n_neighbors} neighbors (disk, window={window})  "
            f"mask>{mask_threshold:g}  min_neighbors={min_neighbors}  "
            f"rotation_angle={rotation_angle:g} deg"
        )
        for name in ("e_uu", "e_vv", "e_uv"):
            print(f"    {name:<9}: {precision[name]:7.4f} %")
        print(f"    {'rotation':<9}: {precision['rotation']:7.4f} deg")
        print(
            f"    {'combined':<9}: {precision['combined']:7.4f} %   "
            "(strain-only Frobenius norm; rotation excluded)"
        )

        if not (plot or returnfig):
            return result

        fig, ax = plot_strain_precision_histogram(edges, counts, precision, component, unit)
        if returnfig:
            return fig, ax
        return result


# ---- module-level fitting functions ----


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted ``q``-quantile of ``values`` (``q`` in ``[0, 1]``); ``nan`` if no weight.

    Uses cumulative-weight interpolation with weights centered on each sorted sample,
    so with uniform weights it tracks ``np.quantile``'s linear interpolation and is
    robust to a heavy upper tail (the median ignores how far the outliers reach).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if values.size == 0 or total <= 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * total, cw, v))


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
    """Weighted-median reference lattice vectors, else the global median.

    The reference is the per-component **weighted median** of the lattice vectors.
    Weights come from ``strain_mask`` if given, else the continuous ``mask``
    (the ``[0, 1]`` per-position weight from :meth:`create_mask` / ``fit_lattice``):
    strong, well-indexed positions dominate the reference and weak / vacuum / bad-fit
    positions are down-weighted. A boolean ROI (weights in ``{0, 1}``) reduces to the
    plain median over the selected positions, so an explicit ``strain_mask`` behaves
    as before. The weighted median (not ``mask == 1``) is used because a continuous
    weight rarely hits *exactly* 1 -- the old exact-equality test collapsed a min-max
    normalized mask to its single global-max position and made the reference one
    arbitrary pixel.

    Parameters
    ----------
    u_array : np.ndarray
        Per-position first lattice vector, shape ``(scan_row, scan_col, 2)``.
    v_array : np.ndarray
        Per-position second lattice vector, shape ``(scan_row, scan_col, 2)``.
    mask : np.ndarray, optional
        ``(scan_row, scan_col)`` per-position weight in ``[0, 1]``. Used as the median
        weights when ``strain_mask`` is not given.
    strain_mask : np.ndarray, optional
        ``(scan_row, scan_col)`` ROI / weight taking precedence over ``mask``.

    Returns
    -------
    tuple of np.ndarray
        ``(u_ref, v_ref)``, each a length-2 reference vector.
    """
    if strain_mask is not None:
        w = np.asarray(strain_mask, dtype=float).reshape(-1)
    elif mask is not None:
        w = np.asarray(mask, dtype=float).reshape(-1)
    else:
        w = None

    u_flat = u_array.reshape(-1, 2)
    v_flat = v_array.reshape(-1, 2)

    def _wmed(vals: np.ndarray) -> float:
        # weighted median over finite, positively-weighted positions; positions
        # fit_lattice could not fit are NaN and must be dropped, else the reference
        # (and the whole strain map) collapses to NaN. Falls back to the unweighted
        # nan-median when no weight is given or none survives.
        finite = np.isfinite(vals)
        ww = np.ones_like(vals) if w is None else w
        use = finite & (ww > 0)
        if not use.any():
            return float(np.nanmedian(vals)) if finite.any() else float("nan")
        return _weighted_quantile(vals[use], ww[use], 0.5)

    u_ref = np.array((_wmed(u_flat[:, 0]), _wmed(u_flat[:, 1])), dtype=float)
    v_ref = np.array((_wmed(v_flat[:, 0]), _wmed(v_flat[:, 1])), dtype=float)
    return u_ref, v_ref


def _strain_tensor(
    u_array: np.ndarray,
    v_array: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    real_space: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-position strain tensor from lattice vectors relative to a reference.

    Two measurement modalities are supported and are arranged to give *identical*
    strain for the same physical deformation, so correlation (Bragg) and cepstral
    (autocorrelation) maps can be compared directly:

    * ``real_space=False`` -- reciprocal-space lattice vectors (nanobeam Bragg
      disks), which contract under tension. The per-position transform is
      ``strain_trans = U_ref @ inv(U)``.
    * ``real_space=True`` -- real-space lattice vectors (cepstral / Patterson
      autocorrelation peaks, or DPC), which expand under tension. The transform is
      ``strain_trans = (U @ inv(U_ref)).T``.

    Both expressions evaluate to ``F.T`` (the transpose of the real-space deformation
    gradient), so the normal strains, shear, and rotation come out the same
    regardless of modality, and the reciprocal-space sign convention (``const = -1``,
    which sets the shear/rotation handedness) is shared.

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
        ``False`` for reciprocal-space (Bragg/correlation) vectors; ``True`` for
        real-space (cepstral autocorrelation / DPC) vectors. Selects the per-position
        transform above; both yield matching strain.

    Returns
    -------
    tuple of np.ndarray
        ``(e_rr, e_cc, e_rc, phi)``, each of shape ``(scan_row, scan_col)``.
    """
    scan_r, scan_c = u_array.shape[0], u_array.shape[1]
    Uref = np.stack((u_ref, v_ref), axis=1).astype(float)
    strain_trans = np.zeros((scan_r, scan_c, 2, 2))

    # For real-space vectors the reference is inverted once (it is shared by every
    # position); a non-finite or singular reference leaves the whole map undefined.
    Uref_inv = None
    if real_space and np.all(np.isfinite(Uref)) and abs(np.linalg.det(Uref)) >= 1e-12:
        Uref_inv = np.linalg.inv(Uref)

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
            if real_space:
                # real-space vectors expand under tension: (U @ U_ref^-1).T == F.T
                if Uref_inv is None:
                    strain_trans[r, c, :, :] = np.nan
                else:
                    strain_trans[r, c, :, :] = (U @ Uref_inv).T
            else:
                # reciprocal-space vectors contract under tension: U_ref @ U^-1 == F.T
                strain_trans[r, c, :, :] = Uref @ np.linalg.inv(U)

    e_rr = strain_trans[:, :, 0, 0] - 1
    e_cc = strain_trans[:, :, 1, 1] - 1
    e_rc = strain_trans[:, :, 1, 0] * 0.5 + strain_trans[:, :, 0, 1] * 0.5
    phi = strain_trans[:, :, 1, 0] * -0.5 + strain_trans[:, :, 0, 1] * 0.5
    return e_rr, e_cc, e_rc, phi


def _rotate_strain_tensor(
    e_rr: np.ndarray,
    e_cc: np.ndarray,
    e_rc: np.ndarray,
    rotation_angle: float,
    real_space=False
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
    real_space: bool
        Tells whether vector is defined in real or reciprocal space
    Returns
    -------
    tuple of np.ndarray
        ``(e_uu, e_vv, e_uv)`` in the rotated frame.
    """
    angle = np.deg2rad(rotation_angle)
    c = np.cos(angle)
    s = np.sin(angle)
    sign = -1.0 if real_space else 1.0
    e_uu = e_rr * (c * c) + sign * 2.0 * e_rc * (c * s) + e_cc * (s * s)
    e_vv = e_rr * (s * s) - sign * 2.0 * e_rc * (c * s) + e_cc * (c * c)
    e_uv = sign * (e_cc - e_rr) * (c * s) + e_rc * (c * c - s * s)
    return e_uu, e_vv, e_uv

def _raw_vec_to_display(vec_rc: NDArray, *, rotation_ccw_deg: float, transpose: bool) -> NDArray:
    """Map a raw-detector ``(row, col)`` vector into the rotated display frame.

    Applies the optional axis transpose, then a counter-clockwise rotation of
    ``rotation_ccw_deg``. Inverse of :func:`_display_vec_to_raw`.
    """
    v = np.asarray(vec_rc, dtype=float)
    dr, dc = v[..., 0], v[..., 1]

    if transpose:
        dr, dc = dc, dr

    theta = np.deg2rad(rotation_ccw_deg)
    ct = np.cos(theta)
    st = np.sin(theta)

    dr2 = ct * dr - st * dc
    dc2 = st * dr + ct * dc
    return np.stack((dr2, dc2), axis=-1)