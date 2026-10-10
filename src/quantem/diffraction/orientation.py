"""Orientation mapping of crystalline 4D-STEM data.

OrientationMap matches measured Bragg peaks (a quantem Vector) against a
library of simulated kinematical patterns from a Crystal, using sparse polar
correlation (Ophus et al., Microsc. Microanal. 28, 390 (2022)) implemented as
batched torch operations.

The method:

1. Sample zone axes over the symmetry-reduced fundamental wedge (or the
   hemisphere), build a polar-coordinate reference library P(zone, shell,
   gamma) where shells are the reciprocal-lattice radii of the crystal.
2. Convert measured peaks at each probe position into the same sparse polar
   representation X(shell, gamma).
3. Correlate over in-plane angle gamma by FFT, over all zones at once, using
   one batched matrix multiplication per gamma frequency. The mirror channel
   (conjugate FFT) tests inversion-related orientations at no library cost.
4. Optionally refine the best zone axes on a finer local grid.

Orientations are unit quaternions; see quantem.diffraction.rotations.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from tqdm import tqdm

from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.defaults import (
    MIN_NUMBER_PEAKS,
    MIN_PAIRS,
    PAIR_DISTANCE,
    POWER_INTENSITY,
    SIGMA_EXCITATION,
    resolve,
)
from quantem.diffraction.rotations import (
    misorientation_angle_deg,
    qconj,
    qmult,
    qnormalize,
    qrotate,
    quat_from_axis_angle,
    quat_from_zone_axis,
    sample_zone_axes,
)


def position_mask(positions, shape: tuple[int, int]) -> torch.Tensor:
    """Normalize a `positions` argument into an (R, C) boolean mask.

    Used by the staged workflow: run matching or refinement on a handful of
    positions, look at the fits, then run the whole scan with the same
    arguments.

    Parameters
    ----------
    positions : None | list[tuple[int, int]] | np.ndarray
        None for every position, a list of (row, col) scan positions, or an
        (R, C) boolean array.
    shape : tuple[int, int]
        Scan shape (R, C) in probe positions.

    Returns
    -------
    torch.Tensor
        (R, C) boolean mask.

    Raises
    ------
    ValueError
        If a boolean mask has the wrong shape, the input is neither a mask
        nor a list of (row, col), or a position lies outside the scan.
    """
    R, C = shape
    if positions is None:
        return torch.ones((R, C), dtype=torch.bool)
    arr = np.asarray(positions)
    if arr.dtype == bool:
        if arr.shape != (R, C):
            raise ValueError(f"boolean positions mask must have shape {(R, C)}, got {arr.shape}")
        return torch.as_tensor(arr, dtype=torch.bool)
    arr = np.atleast_2d(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("positions must be None, an (R, C) boolean mask, or a list of (row, col)")
    mask = torch.zeros((R, C), dtype=torch.bool)
    for r, c in arr.astype(int):
        if not (0 <= r < R and 0 <= c < C):
            raise ValueError(f"position ({r}, {c}) is outside the scan {(R, C)}")
        mask[r, c] = True
    return mask


def scan_scalebar(metadata: dict) -> dict | None:
    """Scale bar arguments from the scan calibration recorded on the peaks.

    Parameters
    ----------
    metadata : dict
        Peak metadata carrying "scan_sampling" and "scan_units".

    Returns
    -------
    dict | None
        {"sampling": step, "units": units} when the scan was calibrated, or
        None when it is still in pixels, which tells a plot to draw no
        scale bar.
    """
    step = (metadata or {}).get("scan_sampling")
    units = (metadata or {}).get("scan_units")
    if step is None or units is None:
        return None
    step = float(np.mean(np.atleast_1d(np.asarray(step, dtype=float))))
    units = str(units)
    if not np.isfinite(step) or step <= 0 or units.lower() in ("pixels", "px", "pixel"):
        return None
    return {"sampling": step, "units": units}


def smooth_quaternions(
    quats: torch.Tensor,
    active: torch.Tensor,
    sym_quats: torch.Tensor,
    sigma_px: float = 1.0,
    sigma_deg: float = 1.0,
    max_angle_deg: float = 5.0,
) -> torch.Tensor:
    """Bilateral average of an orientation field, (R, C, 4).

    Each position is replaced by the weighted mean of the orientations around
    it, with weight exp(-r^2 / 2 sigma_px^2) * exp(-theta^2 / 2 sigma_deg^2)
    for a neighbour r probe positions away and theta degrees misoriented, and
    with neighbours beyond `max_angle_deg` dropped. The angular term is what
    keeps a grain boundary or a second variant out of the average.

    This is an average, not a fit: it moves each orientation away from the one
    that best explains its own pattern. Use it to display a map, not to
    produce the orientations a later step will measure from.

    Parameters
    ----------
    quats : torch.Tensor
        (R, C, 4) orientation quaternions.
    active : torch.Tensor
        (R, C) boolean mask of positions to smooth and to average over.
    sym_quats : torch.Tensor
        (S, 4) symmetry rotations used to reduce the misorientations.
    sigma_px : float, default=1.0
        Spatial width of the kernel in probe positions.
    sigma_deg : float, default=1.0
        Angular width of the kernel in degrees.
    max_angle_deg : float, default=5.0
        Neighbours misoriented by more than this many degrees are dropped.

    Returns
    -------
    torch.Tensor
        (R, C, 4) smoothed quaternions; inactive positions, and positions
        with no neighbour inside `max_angle_deg`, are returned unchanged.
    """
    q = torch.as_tensor(quats, dtype=torch.float64)
    R, C = q.shape[:2]
    active = torch.as_tensor(active, dtype=torch.bool)
    sym = torch.as_tensor(sym_quats, dtype=torch.float64).reshape(-1, 4)
    rad = max(1, int(np.ceil(3 * sigma_px)))
    cos_max = float(np.cos(np.deg2rad(max_angle_deg) / 2))
    # accumulate the weighted outer products one neighbour offset at a time,
    # over the whole map at once
    M = torch.zeros((R, C, 4, 4), dtype=torch.float64)
    count = torch.zeros((R, C), dtype=torch.long)
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            r0, r1 = max(0, -dr), min(R, R - dr)
            c0, c1 = max(0, -dc), min(C, C - dc)
            if r1 <= r0 or c1 <= c0:
                continue
            q0 = q[r0:r1, c0:c1]
            qn = q[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            ok = active[r0:r1, c0:c1] & active[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            # the symmetry image of the neighbour nearest each centre
            cand = qmult(qn[..., None, :], sym)  # (r, c, S, 4)
            dots = torch.einsum("rcsi,rci->rcs", cand, q0)
            best = dots.abs().argmax(dim=-1)
            qk = torch.gather(cand, 2, best[..., None, None].expand(*best.shape, 1, 4))[..., 0, :]
            dot = (qk * q0).sum(-1)
            qk = qk * torch.sign(dot)[..., None]
            cos_half = dot.abs().clamp(max=1.0)
            ok &= cos_half >= cos_max
            ang = torch.rad2deg(2 * torch.acos(cos_half))
            w = np.exp(-(dr * dr + dc * dc) / (2 * sigma_px**2)) * torch.exp(
                -(ang**2) / (2.0 * sigma_deg**2)
            )
            w = torch.where(ok, w, torch.zeros_like(w))
            M[r0:r1, c0:c1] += w[..., None, None] * qk[..., :, None] * qk[..., None, :]
            count[r0:r1, c0:c1] += ok.to(torch.long)
    out = q.clone()
    # a position needs itself and at least one neighbour inside the angle
    upd = active & (count >= 2)
    if bool(upd.any()):
        _, evecs = torch.linalg.eigh(M[upd])
        out[upd] = qnormalize(evecs[..., -1])
    return out


def fibonacci_hemisphere(n_points: int, dtype=torch.float64) -> torch.Tensor:
    """Spherical Fibonacci sampling of the upper hemisphere, (N, 3)."""
    i = torch.arange(n_points, dtype=dtype) + 0.5
    z = i / n_points  # (0, 1): upper hemisphere
    phi = i * (np.pi * (3 - np.sqrt(5)))
    r = torch.sqrt(1 - z**2)
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi), z), dim=-1)


def _zone_peak_parabolic(
    za: torch.Tensor,
    n_pos: torch.Tensor,
    c_n: torch.Tensor,
    n_ok: torch.Tensor,
    step_rad: float,
) -> torch.Tensor:
    """Sub-grid zone axis from the correlations of a zone and its neighbors.

    A quadratic surface c(x, y) is fit by least squares to the correlation
    over the neighborhood in the tangent plane of the best zone (x, y in
    radians); its vertex is the refined zone axis when it lies within one
    grid step of the node and the surface is concave. Otherwise the
    correlation-weighted centroid of the neighbors above 70 % of the best
    value is used, and the node itself when neither applies.

    Parameters
    ----------
    za : (B, 3) best zone axes; n_pos : (B, K, 3) neighbor directions
    (the best zone included); c_n : (B, K) their correlations; n_ok : (B, K)
    validity; step_rad : zone grid step.
    """
    B, K = c_n.shape
    # tangent frame at the node
    ref = torch.where(
        za[:, 2:3].abs() < 0.9,
        torch.tensor([0.0, 0.0, 1.0], dtype=za.dtype).expand(B, 3),
        torch.tensor([1.0, 0.0, 0.0], dtype=za.dtype).expand(B, 3),
    )
    e1 = torch.cross(za, ref, dim=-1)
    e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp_min(1e-12)
    e2 = torch.cross(za, e1, dim=-1)
    d = n_pos - za[:, None, :]
    x = (d * e1[:, None, :]).sum(-1)
    y = (d * e2[:, None, :]).sum(-1)
    c_best = c_n.amax(dim=1, keepdim=True)
    w = n_ok.to(za.dtype)
    out = za.clone()
    # centroid fallback (the previous estimator)
    wgt = (c_n - 0.7 * c_best).clamp_min(0) * w
    cen = (wgt[:, :, None] * n_pos).sum(1)
    cen_ok = torch.linalg.norm(cen, dim=-1) > 1e-12
    cen = cen / torch.linalg.norm(cen, dim=-1, keepdim=True).clamp_min(1e-12)
    out[cen_ok] = cen[cen_ok]
    # quadratic fit where at least 6 valid neighbors exist
    A = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], dim=-1) * w[:, :, None]
    b = (c_n - c_best) * w
    enough = w.sum(1) >= 6
    if bool(enough.any()):
        At = A.transpose(1, 2)
        AtA = At @ A + 1e-12 * torch.eye(6, dtype=za.dtype)
        coef = torch.linalg.solve(AtA, (At @ b[:, :, None]))[..., 0]  # (B, 6)
        cb, cc, cd, ce, cf = coef[:, 1], coef[:, 2], coef[:, 3], coef[:, 4], coef[:, 5]
        H = torch.stack([torch.stack([2 * cd, ce], -1), torch.stack([ce, 2 * cf], -1)], -2)
        det = 4 * cd * cf - ce * ce
        concave = (cd < 0) & (cf < 0) & (det > 0)
        grad = torch.stack([cb, cc], -1)
        vert = torch.zeros_like(grad)
        ok = enough & concave
        if bool(ok.any()):
            vert[ok] = -torch.linalg.solve(H[ok], grad[ok][..., None])[..., 0]
        inside = ok & (torch.linalg.norm(vert, dim=-1) <= step_rad)
        if bool(inside.any()):
            v = za + vert[:, 0:1] * e1 + vert[:, 1:2] * e2
            v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
            out[inside] = v[inside]
    return out


class OrientationMap(AutoSerialize):
    """Match crystal orientations to Bragg peaks at every probe position.

    Workflow::

        om = OrientationMap.from_vectors(peaks, crystal, energy_ev=200e3)
        om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0)
        om.match_orientations(num_matches=1)
        om.plot_orientation()

    The object is both the engine and the result: after
    `match_orientations()`, `quats` holds (R, C, M, 4) orientation
    quaternions, `corr` the correlation scores, and `mirror` the inversion
    flags.
    """

    _token = object()

    def __init__(
        self,
        peaks: Vector,
        crystal: Crystal,
        energy_ev: float,
        _token: object | None = None,
    ):
        """Private constructor; use :meth:`from_vectors`.

        Parameters
        ----------
        peaks : Vector
            Calibrated Bragg peaks over the scan.
        crystal : Crystal
            Candidate crystal with structure factors already calculated.
        energy_ev : float
            Beam energy in eV.
        _token : object
            Guard against direct construction.

        Raises
        ------
        RuntimeError
            If called without the class token.
        """
        if _token is not self._token:
            raise RuntimeError("Use OrientationMap.from_vectors() to construct.")
        self.peaks = peaks
        self.crystal = crystal
        self.energy_ev = float(energy_ev)
        self.wavelength = electron_wavelength_angstrom(energy_ev)
        # processing hyperparameters of every stage, recorded as they run;
        # later stages inherit from these when an argument is left as None
        self.metadata: dict = {
            "energy_ev": self.energy_ev,
            "peaks": dict(getattr(peaks, "metadata", {}) or {}),
        }

        # plan state
        self.zone_axes: torch.Tensor | None = None
        self.zone_quats: torch.Tensor | None = None
        self.plan_fft: torch.Tensor | None = None
        self.shell_radii: torch.Tensor | None = None

        # results
        self.quats: torch.Tensor | None = None
        self.corr: torch.Tensor | None = None
        self.corr_residual: torch.Tensor | None = None
        self.score: torch.Tensor | None = None
        self.corr_second: torch.Tensor | None = None
        self.reliability: torch.Tensor | None = None
        self.mirror: torch.Tensor | None = None
        # positions carrying a result; a subset after a staged test run
        self.computed: torch.Tensor | None = None

    @classmethod
    def from_vectors(
        cls,
        peaks: Vector,
        crystal: Crystal,
        energy_ev: float = 300e3,
        precession_deg: float = 0.0,
        semiconv_mrad: float = 0.0,
        foil_normal=None,
    ) -> "OrientationMap":
        """Create from detected Bragg peaks.

        Parameters
        ----------
        peaks : Vector
            Ragged peak table over scan positions with fields including
            ('qx', 'qy', 'intensity') in calibrated 1/Angstrom units.
        crystal : Crystal
            Candidate crystal with structure factors already calculated.
        precession_deg, semiconv_mrad : float
            Precession semi-angle and convergence semiangle of the
            experiment, recorded for the dynamical refinements (which
            average the intensities over them) and inherited by them.
        energy_ev : float, default=300e3
            Beam energy in eV.
        foil_normal : sequence of float, optional
            Plate normal of the specimen as a direction in the crystal, [uvw]
            or [UVTW], e.g. (0, 0, 0, 1) for a 2D material lying in its basal
            plane. Reflections are then rods along it, which moves the
            simulated spots of a tilted flake to where the rods meet the
            Ewald sphere (see :meth:`Crystal.generate_pattern`); the library,
            the refinement and every simulated pattern use it. None
            (default) is the usual geometry.
        """
        if crystal.g_vec is None:
            raise RuntimeError("Run crystal.calculate_structure_factors() first.")
        om = cls(peaks, crystal, energy_ev, _token=cls._token)
        om.metadata["precession_deg"] = float(precession_deg)
        om.metadata["semiconv_mrad"] = float(semiconv_mrad)
        om.metadata["foil_normal"] = (
            None if foil_normal is None else [float(v) for v in np.ravel(foil_normal)]
        )
        return om

    def _foil_normal_crystal(self) -> torch.Tensor | None:
        """Unit plate normal in the crystal frame, or None for the usual
        geometry (normal along the beam)."""
        fn = self.metadata.get("foil_normal")
        return None if fn is None else self.crystal.direction_vector(fn)

    # ------------------------------------------------------------------
    # orientation plan
    # ------------------------------------------------------------------

    def build_plan(
        self,
        angle_step_zone_axis_deg: float = 1.0,
        angle_step_in_plane_deg: float = 5.0,
        zone_axis_range="auto",
        fiber_axis=None,
        fiber_angle_deg: float = 0.0,
        corr_kernel_size: float = PAIR_DISTANCE,
        sigma_excitation: float = SIGMA_EXCITATION,
        power_radial: float = 1.0,
        power_intensity: float = POWER_INTENSITY,
        power_intensity_experiment: float | None = None,
        tol_shell_distance: float = 0.01,
        detector_q_max: float | tuple[float, float] | str | None = "auto",
        device: str | torch.device = "cpu",
        verbose: bool = True,
        progress_bar: bool = True,
    ) -> "OrientationMap":
        """Build the polar correlation library over the fundamental wedge.

        Parameters
        ----------
        angle_step_zone_axis_deg : float, default=1.0
            Angular step between sampled zone axes. The zone axis is the
            coordinate the correlation search cannot refine continuously
            (only by the neighbor-weighted centroid), so it is sampled
            finely; the wedge sampling is isotropic at this step.
        angle_step_in_plane_deg : float, default=5.0
            Angular step of the in-plane (gamma) axis; the number of gamma
            samples is round(360 / step). The in-plane angle is refined
            continuously (parabolic sub-bin interpolation, then least
            squares on the paired peaks in refine_orientations), so a
            coarse step costs little accuracy and keeps the library small.
        zone_axis_range : {"auto", "full", "fiber"} | array-like, default="auto"
            Which zone axes the library covers.

            - "auto": the fundamental wedge of the matching point group
              (the pseudo-symmetry group when one was detected), falling
              back to the hemisphere for triclinic and monoclinic cells.
              This is the right choice for an unknown texture.
            - "full": the whole hemisphere, whatever the symmetry. Use when
              the symmetry the cell reports is not the symmetry of its
              diffraction, so the wedge would fold distinct orientations
              onto each other.
            - "fiber": a cap of half angle `fiber_angle_deg` about
              `fiber_axis`, for a known texture (a 2D material or a
              textured film). `fiber_angle_deg=0` samples the fiber axis
              alone, so the match is over the in-plane angle only, which
              makes the library tiny and the match far more robust.
            - an array of 2 or 3 lattice directions [uvw] (or [uvtw] for a
              hexagonal cell): the spherical triangle they span. With two
              rows the wedge runs from [001] through both.

            The in-plane angle is always searched over the full 360 degrees;
            the correlation is circular in it, so restricting it saves
            nothing.
        fiber_axis : array-like | None
            Lattice direction [uvw] (or [uvtw]) of the fiber axis, required
            by zone_axis_range="fiber".
        fiber_angle_deg : float, default=0.0
            Half angle of the fiber cap, degrees.
        corr_kernel_size : float, default=0.05
            Correlation kernel size delta (1/Angstroms): azimuthal extent of
            each reference peak and radial tolerance for shell assignment.
        sigma_excitation : float, default=0.04
            Excitation error envelope of the library (1/Angstroms). Keep this
            about 2x the physical excitation tolerance: orientations halfway
            between sampled zones shift s_g by ~ (step/2) * k, and a wider
            envelope keeps their library intensities from collapsing. With a
            precession angle or convergence semiangle recorded by
            from_vectors, the envelope of every library reflection is its
            exact average over that illumination
            (quantem.diffraction.illumination).
        power_radial, power_intensity : float
            Weighting prefactor q^power_radial * |V_g|^power_intensity for
            library peaks. power_intensity=0 matches on positions only
            (best for strongly dynamical data).
        power_intensity_experiment : float | None
            Exponent applied to the *measured* peak intensities; defaults to
            `power_intensity`. Lower it than the library exponent when the
            measured intensities are less trustworthy than the simulated
            ones (saturation, a beam stop, strong dynamical transfer).
        tol_shell_distance : float, default=0.01
            Reciprocal lattice radii closer than this merge into one shell.
        detector_q_max : float | tuple | "auto" | None, default="auto"
            Half-width of the square detector (1/Angstroms), scalar or
            (row_max, col_max). Library reflections beyond the detector edge
            cannot be measured, and which ones fall off depends on the
            in-plane rotation; the correlation is normalized by the masked
            template norm at every in-plane angle, so orientations with
            strong reflections outside the detector are not penalized.
            "auto" measures the detector footprint from the peaks themselves
            (largest |q| along each detector axis, undoing any
            detector-to-scan rotation recorded on the peaks). None disables
            the correction.
        device : str | torch.device, default="cpu"
            Device for the library and the correlation compute. On Apple
            silicon 'mps' runs the correlation in float32 (about 1.5x faster
            than the CPU); the refinements that follow stay on the CPU.
        verbose : bool, default=True
            Print the plan size and the group used for matching. The crystal
            prints its full symmetry, pseudo-symmetry included, when built.
        progress_bar : bool, default=True
            Show a progress bar while the library is deposited, which is
            most of the time: ~20 s per crystal at k_max = 2 over a trigonal
            wedge, several times that over a full hemisphere.
        """
        crystal = self.crystal
        self.device = torch.device(device)
        # MPS has no float64: the correlation runs in float32 there (the
        # cosine similarities are insensitive to it); results are returned
        # in float64 either way
        self.dtype = torch.float32 if self.device.type == "mps" else torch.float64
        self.cdtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
        self.corr_kernel_size = float(corr_kernel_size)
        self.sigma_excitation = float(sigma_excitation)
        self.power_radial = float(power_radial)
        self.power_intensity = float(power_intensity)
        self.power_intensity_experiment = float(
            power_intensity if power_intensity_experiment is None else power_intensity_experiment
        )

        # zone axis sampling: the symmetry wedge, the hemisphere, a fiber
        # cap, or an explicit spherical triangle of lattice directions
        za = self._sample_zone_axes(
            zone_axis_range, fiber_axis, fiber_angle_deg, angle_step_zone_axis_deg
        )
        self.zone_axis_range = zone_axis_range
        self.zone_axes = za
        self.zone_quats = quat_from_zone_axis(za)
        self.zone_step_deg = float(angle_step_zone_axis_deg)

        # symmetry-complete neighbor sets for sub-grid zone refinement: a
        # zone on the wedge boundary only has in-wedge grid neighbors on one
        # side, and a one-sided correlation centroid would drag it inward;
        # the symmetry images of the grid across the boundary restore the
        # missing side.
        from quantem.diffraction.rotations import quat_to_matrix

        Rs = quat_to_matrix(crystal.sym_quats_matching)
        images = torch.einsum("sij,zj->szi", Rs, za)
        images = torch.cat([images, -images], dim=0).reshape(-1, 3)  # (S2*Z, 3)
        img_zone = torch.arange(za.shape[0]).repeat(2 * crystal.sym_quats_matching.shape[0])
        # deduplicate coincident image positions (keep one per position/zone)
        key = torch.cat(
            [torch.round(images / 1e-6) * 1e-6, img_zone[:, None].to(images.dtype)],
            dim=1,
        )
        _, first = np.unique(key.numpy(), axis=0, return_index=True)
        images = images[torch.as_tensor(np.sort(first))]
        img_zone = img_zone[torch.as_tensor(np.sort(first))]

        cos_lim = np.cos(np.deg2rad(1.6 * self.zone_step_deg))
        nbr_idx_list, nbr_pos_list = [], []
        dots = images @ za.T  # (M, Z)
        for i in range(za.shape[0]):
            sel = torch.nonzero(dots[:, i] > cos_lim).squeeze(1)
            nbr_idx_list.append(img_zone[sel])
            nbr_pos_list.append(images[sel])
        K = max(len(v) for v in nbr_idx_list)
        Z = za.shape[0]
        self.zone_nbr_idx = torch.zeros((Z, K), dtype=torch.long)
        self.zone_nbr_pos = torch.zeros((Z, K, 3), dtype=torch.float64)
        self.zone_nbr_valid = torch.zeros((Z, K), dtype=torch.bool)
        for i, (idx, pos) in enumerate(zip(nbr_idx_list, nbr_pos_list)):
            k = len(idx)
            self.zone_nbr_idx[i, :k] = idx
            self.zone_nbr_pos[i, :k] = pos
            self.zone_nbr_valid[i, :k] = True

        # radial shells from unique reciprocal lattice vector lengths
        g_len = crystal.g_len
        radii = torch.unique(torch.round(g_len / tol_shell_distance) * tol_shell_distance)
        self.shell_radii = radii
        self.num_gamma = int(round(360 / angle_step_in_plane_deg))
        self.gamma = torch.linspace(0, 2 * np.pi, self.num_gamma + 1, dtype=torch.float64)[:-1]

        plan = self._build_reference(self.zone_quats, progress_bar=progress_bar)
        # store conj(fft) along gamma so matching is a single complex matmul
        self.plan_fft = torch.conj(torch.fft.fft(plan, dim=-1)).to(self.cdtype).to(self.device)

        # square-detector aperture correction: the masked template norm at
        # every in-plane shift is the circular correlation of the squared
        # plan with the polar detector mask. The mask lives in the DETECTOR
        # frame: any detector-to-scan rotation recorded on the peaks rotates
        # the square aperture in the calibrated (qx, qy) frame.
        rot_deg = float(self.peaks.metadata.get("rotation_ccw_deg", 0.0) or 0.0)
        if isinstance(detector_q_max, str) and detector_q_max == "auto":
            flat = self.peaks.select_fields("qx", "qy", "intensity").numpy().astype(np.float64)
            if flat.shape[0] == 0:
                detector_q_max = None
            else:
                th_b = np.deg2rad(-rot_deg)
                rb = np.array([[np.cos(th_b), -np.sin(th_b)], [np.sin(th_b), np.cos(th_b)]])
                det_rc = flat[:, :2] @ rb.T
                detector_q_max = (
                    float(np.abs(det_rc[:, 0]).max()) + self.corr_kernel_size,
                    float(np.abs(det_rc[:, 1]).max()) + self.corr_kernel_size,
                )
        if detector_q_max is not None:
            if np.isscalar(detector_q_max):
                qx_max = qy_max = float(detector_q_max)
            else:
                qx_max, qy_max = (float(v) for v in detector_q_max)
            r = self.shell_radii[:, None]
            g = self.gamma[None, :] - np.deg2rad(rot_deg)
            mask = (
                (torch.abs(r * torch.cos(g)) <= qx_max) & (torch.abs(r * torch.sin(g)) <= qy_max)
            ).to(torch.float64)
            self.detector_mask = mask  # (S, G)
            plan_sq_fft = torch.conj(torch.fft.fft(plan**2, dim=-1))
            mask_fft = torch.fft.fft(mask, dim=-1)
            # norm^2 per (zone, shift), direct and mirrored channels
            n2 = torch.fft.ifft(
                torch.einsum("zsg,sg->zg", plan_sq_fft, mask_fft), dim=-1
            ).real.clamp_min(0)
            n2_m = torch.fft.ifft(
                torch.einsum("zsg,sg->zg", plan_sq_fft, torch.conj(mask_fft)), dim=-1
            ).real.clamp_min(0)
            # fraction of template weight on the detector; used to suppress
            # zones that are mostly unmeasurable at a given rotation
            full = (plan**2).sum(dim=(1, 2))[:, None].clamp_min(1e-12)
            self.plan_norm_shift = (
                torch.stack([torch.sqrt(n2), torch.sqrt(n2_m)]).to(self.dtype).to(self.device)
            )  # (2, Z, G)
            self.plan_frac_shift = (
                torch.stack([n2 / full, n2_m / full]).to(self.dtype).to(self.device)
            )
        else:
            self.detector_mask = None
            self.plan_norm_shift = None
            self.plan_frac_shift = None
        self.metadata["plan"] = dict(
            excitation_model="gaussian",
            precession_deg=float(self.metadata.get("precession_deg", 0.0) or 0.0),
            semiconv_mrad=float(self.metadata.get("semiconv_mrad", 0.0) or 0.0),
            angle_step_zone_axis_deg=float(angle_step_zone_axis_deg),
            angle_step_in_plane_deg=float(angle_step_in_plane_deg),
            zone_axis_range=zone_axis_range
            if isinstance(zone_axis_range, str)
            else np.asarray(zone_axis_range).tolist(),
            fiber_axis=None if fiber_axis is None else np.asarray(fiber_axis).tolist(),
            fiber_angle_deg=float(fiber_angle_deg),
            corr_kernel_size=self.corr_kernel_size,
            pair_distance=self.corr_kernel_size,
            sigma_excitation=self.sigma_excitation,
            power_radial=self.power_radial,
            power_intensity=self.power_intensity,
            power_intensity_experiment=self.power_intensity_experiment,
            tol_shell_distance=float(tol_shell_distance),
            detector_q_max=None
            if detector_q_max is None
            else tuple(np.atleast_1d(detector_q_max).tolist()),
        )
        if verbose:
            # the crystal printed its own symmetry when it was built; the plan
            # adds only what it sampled
            print(
                "%s: orientation plan %d zone axes x %d in-plane angles, "
                "%d radial shells, matching %s"
                % (
                    crystal.name,
                    self.zone_axes.shape[0],
                    self.gamma.shape[0],
                    self.shell_radii.shape[0],
                    crystal.pointgroup_matching,
                )
            )
        return self

    def _sample_zone_axes(
        self,
        zone_axis_range,
        fiber_axis,
        fiber_angle_deg: float,
        step_deg: float,
    ) -> torch.Tensor:
        """Zone-axis sampling requested by build_plan, (Z, 3) Cartesian."""
        from quantem.diffraction.rotations import sample_zone_axis_cap

        crystal = self.crystal

        def cartesian(uvw) -> torch.Tensor:
            v = np.asarray(uvw, dtype=float).reshape(-1)
            if v.shape[0] == 4:  # Miller-Bravais [uvtw]
                from quantem.diffraction.crystal import miller_bravais_to_miller

                v = np.asarray(miller_bravais_to_miller(v)).reshape(-1)
            if v.shape[0] != 3:
                raise ValueError("a direction must have 3 indices [uvw] or 4 [uvtw]")
            d = torch.as_tensor(v, dtype=torch.float64) @ crystal.lat_real
            return d / torch.linalg.norm(d).clamp_min(1e-12)

        if isinstance(zone_axis_range, str):
            mode = zone_axis_range.lower()
            if mode == "fiber":
                if fiber_axis is None:
                    raise ValueError('zone_axis_range="fiber" needs a fiber_axis')
                return sample_zone_axis_cap(cartesian(fiber_axis), fiber_angle_deg, step_deg)
            if mode in ("full", "hemisphere"):
                n_zones = int(np.ceil(2 * np.pi / np.deg2rad(step_deg) ** 2))
                return fibonacci_hemisphere(n_zones)
            if mode != "auto":
                raise ValueError(
                    'zone_axis_range must be "auto", "full", "fiber", or an array of directions'
                )
            msg = crystal.matching_symmetry_warning()
            # a crystal built with verbose=True already said this in its summary
            if msg is not None and not getattr(crystal, "_summary_shown", False):
                warnings.warn(msg, stacklevel=3)
            wedge = crystal.zone_axis_wedge()
            if wedge is None:  # triclinic / monoclinic: not a spherical triangle
                n_zones = int(np.ceil(2 * np.pi / np.deg2rad(step_deg) ** 2))
                return fibonacci_hemisphere(n_zones)
            za, _ = sample_zone_axes(wedge, step_deg)
            return za

        rows = np.atleast_2d(np.asarray(zone_axis_range, dtype=float))
        dirs = [cartesian(r) for r in rows]
        if len(dirs) == 2:
            dirs = [cartesian([0, 0, 1]), *dirs]
        if len(dirs) != 3:
            raise ValueError("zone_axis_range as an array needs 2 or 3 directions")
        za, _ = sample_zone_axes(torch.stack(dirs), step_deg)
        return za

    def _deposit_polar(
        self,
        qr: torch.Tensor,
        qphi: torch.Tensor,
        amp: torch.Tensor,
        out: torch.Tensor,
        image: torch.Tensor | None = None,
        progress: str | None = None,
    ) -> torch.Tensor:
        """Deposit peaks into polar images with the shared correlation kernel.

        Every peak spreads as a Gaussian of width delta in both the radial
        direction (across shells) and arc length (along gamma). The library
        and the experimental patterns use this same kernel, so the normalized
        correlation of a pattern with itself is exactly 1.

        Parameters
        ----------
        qr, qphi, amp : torch.Tensor
            Peak radii, azimuths, amplitudes, flat (K,).
        out : torch.Tensor
            (S, G) accumulator, or (N, S, G) when `image` is given; modified
            in place.
        image : torch.Tensor | None
            (K,) image index of every peak, so a whole batch of patterns
            (or a whole library) is deposited in one call.
        progress : str | None
            Description for a progress bar over the chunks; None shows none.
        """
        radii = self.shell_radii.to(qr.dtype)
        delta = self.corr_kernel_size
        S = radii.shape[0]

        dr = qr[:, None] - radii[None, :]  # (K, S)
        k_all, s_all = torch.nonzero(dr.abs() < 3 * delta, as_tuple=True)
        if k_all.numel() == 0:
            return out
        gamma = self.gamma.to(qr.dtype)
        flat = out.view(-1, out.shape[-1])
        # chunked so the (entries, G) weight array stays a few tens of MB
        chunk = max(1, 4_000_000 // gamma.shape[0])
        starts = range(0, k_all.numel(), chunk)
        if progress is not None:
            starts = tqdm(starts, desc=progress)
        for c0 in starts:
            k_idx = k_all[c0 : c0 + chunk]
            s_idx = s_all[c0 : c0 + chunk]
            w_r = torch.exp(-(dr[k_idx, s_idx] ** 2) / (2 * delta**2)) * amp[k_idx]
            dg = qphi[k_idx, None] - gamma[None, :]
            dg = (dg + np.pi) % (2 * np.pi) - np.pi
            arc = dg * qr[k_idx, None]
            w = w_r[:, None] * torch.exp(-(arc**2) / (2 * delta**2))
            rows = s_idx if image is None else image[k_idx] * S + s_idx
            flat.index_add_(0, rows, w)
        return out

    def _build_reference(
        self, zone_quats: torch.Tensor, progress_bar: bool = False
    ) -> torch.Tensor:
        """Polar reference library (Z, S, G) for the given zone-axis quats."""
        crystal = self.crystal
        lam = self.wavelength
        g = crystal.g_vec  # (N, 3)
        delta = self.corr_kernel_size

        gr = qrotate(zone_quats[:, None, :], g[None, :, :])  # (Z, N, 3)
        gz = gr[..., 2]
        g2 = (gr**2).sum(-1)
        s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
        prec = float(self.metadata.get("precession_deg", 0.0) or 0.0)
        conv = float(self.metadata.get("semiconv_mrad", 0.0) or 0.0)
        spot_xy = gr[..., :2]
        n_c = self._foil_normal_crystal()
        f_rod = None
        if n_c is not None:
            # plate geometry: excitation along the rod, spot where the rod
            # meets the sphere; the normal turns with the crystal, so the
            # template still rolls in gamma
            from quantem.diffraction.illumination import relrod_factor

            n_lab = qrotate(zone_quats, n_c[None].expand(zone_quats.shape[0], 3))  # (Z, 3)
            f_rod = relrod_factor(gr, n_lab[:, None, :], self.energy_ev, 0.0)
            s_g = s_g * f_rod
            spot_xy = spot_xy - s_g[..., None] * n_lab[:, None, :2]
        if prec > 0 or conv > 0:
            # excitation envelope averaged over the illumination
            # (quantem.diffraction.illumination); the peak weighting below
            # keeps the established correlation-library semantics
            from quantem.diffraction.illumination import (
                excitation_amplitudes,
                gaussian_envelope,
                gaussian_envelope_ring_torch,
            )

            a_r, b_r = excitation_amplitudes(gr, self.energy_ev, prec, conv)
            if f_rod is not None:
                a_r, b_r = a_r * f_rod.abs(), b_r * f_rod.abs()
            if conv <= 0:
                amp = gaussian_envelope_ring_torch(s_g, a_r, self.sigma_excitation)
            else:
                amp = torch.as_tensor(
                    gaussian_envelope(
                        s_g.numpy(), a_r.numpy(), b_r.numpy(), self.sigma_excitation
                    ),
                    dtype=torch.float64,
                )
            amp = amp * (s_g.abs() < a_r + b_r + delta * 4)
        else:
            amp = torch.exp(-(s_g**2) / (2 * self.sigma_excitation**2))
            amp = amp * (s_g.abs() < delta * 4)

        weight = (
            crystal.g_len**self.power_radial * crystal.struct_factors_int**self.power_intensity
        )
        vals = amp * weight[None, :]  # (Z, N)
        qr = torch.hypot(spot_xy[..., 0], spot_xy[..., 1])
        qphi = torch.atan2(spot_xy[..., 1], spot_xy[..., 0])

        Z = zone_quats.shape[0]
        plan = torch.zeros((Z, self.shell_radii.shape[0], self.num_gamma), dtype=torch.float64)
        z_idx, n_idx = torch.nonzero(vals > 1e-8, as_tuple=True)
        self._deposit_polar(
            qr[z_idx, n_idx],
            qphi[z_idx, n_idx],
            vals[z_idx, n_idx],
            plan,
            image=z_idx,
            progress=f"orientation plan {crystal.name}" if progress_bar else None,
        )

        norm = torch.linalg.norm(plan.reshape(Z, -1), dim=1).clamp_min(1e-12)
        return plan / norm[:, None, None]

    # ------------------------------------------------------------------
    # experimental polar images
    # ------------------------------------------------------------------

    def _grid_quats(self, flat_idx: torch.Tensor, Z: int, G: int, n_ch: int) -> torch.Tensor:
        """Library orientations at flat (channel, zone, gamma) indices.

        No subpixel refinement: these are used only to test whether two
        candidates are the same orientation, where the grid step is far
        finer than the separation being tested.

        Parameters
        ----------
        flat_idx : torch.Tensor
            ``(..., )`` indices into the flattened ``(ch, Z, G)`` correlation.

        Returns
        -------
        torch.Tensor
            ``(..., 4)`` quaternions.
        """
        idx = flat_idx.cpu()
        ch = idx // (Z * G)
        z = (idx // G) % Z
        g = idx % G
        q_zone = self.zone_quats[z]
        gamma = self.gamma[g].clone()
        is_mirror = ch == 1
        if n_ch > 1:
            q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
            q_zone = torch.where(is_mirror[..., None], qmult(q_flip, q_zone), q_zone)
            gamma = torch.where(is_mirror, -gamma - np.pi, gamma)
        half = gamma / 2
        zeros = torch.zeros_like(half)
        q_spin = torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)
        return qmult(q_spin, q_zone)

    def _polar_images(self, arrays: list[np.ndarray], ix: list[int]) -> torch.Tensor:
        """Sparse polar images (B, S, G) of a batch of measured patterns,
        deposited in one call."""
        data = np.concatenate([a[:, ix] for a in arrays], axis=0)
        image = torch.repeat_interleave(
            torch.arange(len(arrays)), torch.tensor([a.shape[0] for a in arrays])
        )
        qx = torch.as_tensor(data[:, 0], dtype=torch.float64)
        qy = torch.as_tensor(data[:, 1], dtype=torch.float64)
        intensity = torch.as_tensor(data[:, 2], dtype=torch.float64)
        qr = torch.hypot(qx, qy)
        qphi = torch.atan2(qy, qx)
        amp = intensity.clamp_min(0) ** self.power_intensity_experiment * qr**self.power_radial
        out = torch.zeros(
            (len(arrays), self.shell_radii.shape[0], self.num_gamma), dtype=torch.float64
        )
        return self._deposit_polar(qr, qphi, amp, out, image=image)

    # ------------------------------------------------------------------
    # matching
    # ------------------------------------------------------------------

    def match_orientations(
        self,
        num_matches: int = 1,
        positions=None,
        include_mirror: bool = True,
        min_number_peaks: int | None = None,
        min_angle_between_matches_deg: float = 15.0,
        suppress_matched: float = 1.0,
        top_k_matches: int = 256,
        subpixel_gamma: bool = True,
        subpixel_zone: bool = True,
        min_detector_fraction: float = 0.3,
        batch_size: int = 128,
        progress_bar: bool = True,
    ) -> "OrientationMap":
        """Match probe positions against the orientation plan.

        Patterns are processed in batches: the polar images are stacked, and
        the correlation over all zones and in-plane angles reduces to one
        complex matrix product per gamma frequency plus a batched inverse FFT.

        Correlation scores are normalized to [0, 1]: the library slices are
        unit vectors and the experimental polar image is divided by its own
        norm, so `corr` is a cosine similarity comparable across patterns,
        crystals, and datasets. After the best match, the highest correlation
        among zone axes at least `min_angle_between_matches_deg` away is
        stored in `corr_second`; `reliability = corr - corr_second` is the
        primary confidence metric.

        Parameters
        ----------
        num_matches : int, default=1
            Number of orientations to return per probe position; matches
            after the first suppress zones within
            `min_angle_between_matches_deg` of earlier matches.
        positions : list[tuple[int, int]] | np.ndarray | None
            Scan positions to match: a list of (row, col), or an (R, C)
            boolean mask. None (default) matches the whole scan. Pass a
            handful of positions to check the plan and these parameters
            with `plot_pattern_matches` before committing to the full
            scan; the positions carrying a result are recorded in
            `computed`, which the refinements and the phase fit follow.
        include_mirror : bool, default=True
            Also correlate against the in-plane mirrored pattern, testing
            inversion-related (opposite hemisphere) zone axes at no library
            cost. Exact in the flat-Ewald / Friedel limit.
        min_number_peaks : int | None
            Skip positions with fewer detected peaks (including the direct
            beam); defaults to MIN_NUMBER_PEAKS (5).
        suppress_matched : float, default=1.0
            Fraction of each accepted match subtracted from the measured
            polar image before the next match is sought, so later matches
            fit the peaks earlier ones leave unexplained. 1.0 removes the
            matched component exactly; 0 disables the deflation and leaves
            `min_angle_between_matches_deg` as the only thing separating the
            matches, which lets two of them index the same peaks. Only
            meaningful with `num_matches` above 1. The deflation steers the
            search only: `corr` scores every match against the pattern as
            measured, so the matches stay comparable, and `corr_residual`
            holds the score against the residual each one actually saw.
        min_angle_between_matches_deg : float, default=15.0
            Minimum separation between matches, applied to the zone axis and
            the in-plane angle together: a candidate is rejected only when it
            is within this angle of an earlier match in both. Two grains
            sharing a zone axis but rotated in plane past this angle are
            therefore kept as separate matches. The same test picks the
            second-best score used in `reliability`.
        top_k_matches : int, default=256
            Number of highest-scoring library entries searched, in order,
            for a candidate that passes the separation test, both for the
            matches after the first and for `corr_second`. Most of the top
            entries are symmetry copies or near neighbours of the best one,
            so too small a value leaves later matches empty.
        subpixel_gamma : bool, default=True
            Parabolic sub-bin refinement of the in-plane angle.
        subpixel_zone : bool, default=True
            Sub-grid zone axis from a quadratic fit of the correlation over
            the best zone and its grid neighbors (centroid fallback).
        min_detector_fraction : float, default=0.3
            With a detector footprint in the plan (`detector_q_max`), library
            orientations that put less than this fraction of their template
            weight on the detector at a given in-plane angle score zero
            there. Ignored when the plan has no detector correction.
        batch_size : int, default=128
            Number of patterns correlated at once.
        progress_bar : bool, default=True
            Show a progress bar over the batches.

        Returns
        -------
        OrientationMap
            Self, with `quats` (R, C, M, 4), `corr` and `corr_residual`
            (R, C, M), `corr_second` and `reliability` (R, C), `mirror`
            (R, C, M) and `computed` (R, C) filled in. Positions not matched
            keep the identity orientation and a correlation of zero.

        Raises
        ------
        RuntimeError
            If the plan has not been built, or no requested position has
            `min_number_peaks` peaks.
        """
        if self.plan_fft is None:
            raise RuntimeError("Run build_plan() first.")
        min_number_peaks = resolve(min_number_peaks, "min_number_peaks", default=MIN_NUMBER_PEAKS)
        self.metadata["match"] = dict(
            num_matches=int(num_matches),
            positions=None if positions is None else "subset",
            include_mirror=bool(include_mirror),
            min_number_peaks=int(min_number_peaks),
            min_angle_between_matches_deg=float(min_angle_between_matches_deg),
            suppress_matched=float(suppress_matched),
            top_k_matches=int(top_k_matches),
            subpixel_gamma=bool(subpixel_gamma),
            subpixel_zone=bool(subpixel_zone),
            min_detector_fraction=float(min_detector_fraction),
        )
        peaks = self.peaks
        shape = peaks.shape
        R, C = shape[0], shape[1]
        M = num_matches
        device = self.device
        G = self.num_gamma
        Z = self.zone_axes.shape[0]

        quats = torch.zeros((R, C, M, 4), dtype=torch.float64)
        quats[..., 0] = 1.0
        corr_out = torch.zeros((R, C, M), dtype=torch.float64)
        corr_res = torch.zeros((R, C, M), dtype=torch.float64)
        corr_second = torch.zeros((R, C), dtype=torch.float64)
        mirror_out = torch.zeros((R, C, M), dtype=torch.bool)

        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

        dtype = getattr(self, "dtype", torch.float64)

        plan_fft = self.plan_fft  # (Z, S, G) complex
        wanted = position_mask(positions, (R, C))
        valid_rc = [
            (rx, ry)
            for rx, ry in np.ndindex(R, C)
            if wanted[rx, ry]
            and peaks[rx, ry].numpy().astype(np.float64).shape[0] >= min_number_peaks
        ]
        if not valid_rc:
            raise RuntimeError(
                "no requested scan position has at least min_number_peaks = %d detected peaks"
                % min_number_peaks
            )
        computed = torch.zeros((R, C), dtype=torch.bool)
        for rx, ry in valid_rc:
            computed[rx, ry] = True
        self.computed = computed
        batches = [valid_rc[i : i + batch_size] for i in range(0, len(valid_rc), batch_size)]
        if progress_bar:
            batches = tqdm(batches, desc=f"matching {self.crystal.name}")

        gamma_grid = self.gamma
        for batch in batches:
            im_stack = (
                self._polar_images(
                    [peaks[rx, ry].numpy().astype(np.float64) for rx, ry in batch], ix
                )
                .to(dtype)
                .to(device)
            )
            B = im_stack.shape[0]
            with warnings.catch_warnings():
                # torch's MPS FFT emits an internal out-tensor resize notice
                warnings.simplefilter("ignore", UserWarning)
                im_fft = torch.fft.fft(im_stack, dim=-1)  # (B, S, G)
            # frequency ramp used to roll a template to an in-plane angle
            k_ramp = torch.fft.fftfreq(G, d=1.0 / G).to(im_fft.dtype).to(device)
            # orientations accepted so far in this batch, one (B, 4) per match
            q_prev: list[torch.Tensor] = []

            for m in range(M):
                norms = torch.linalg.norm(
                    torch.fft.ifft(im_fft, dim=-1).real.reshape(B, -1), dim=1
                ).clamp_min(1e-12)
                # contract shells: (B, Z, G) per channel
                cc = torch.einsum("zsg,bsg->bzg", plan_fft, im_fft)
                channels = [cc]
                if include_mirror:
                    channels.append(torch.einsum("zsg,bsg->bzg", plan_fft, torch.conj(im_fft)))
                corr_raw = torch.fft.ifft(torch.stack(channels, dim=1), dim=-1).real
                # normalize: library slices are unit vectors, so dividing by the
                # experimental norm makes corr a cosine similarity in [0, 1]
                corr = corr_raw / norms[:, None, None, None]
                if self.plan_norm_shift is not None:
                    # square-detector correction: renormalize by the on-detector
                    # template norm at each in-plane shift, and suppress
                    # rotations where most of the template is unmeasurable
                    n_ch = corr.shape[1]
                    corr = corr / self.plan_norm_shift[None, :n_ch].clamp_min(1e-3)
                    corr = corr.masked_fill(
                        self.plan_frac_shift[None, :n_ch] < min_detector_fraction, 0.0
                    )
                # corr: (B, ch, Z, G)
                if m == 0:
                    # the deflation below changes the image every match, so
                    # keep the correlation against the pattern as measured:
                    # selection uses the residual, the reported score does not
                    corr_full = corr
                n_ch = corr.shape[1]
                flat = corr.reshape(B, -1)
                if M > 1:
                    # Rank the candidates and walk down until one is a
                    # genuinely different orientation from every earlier
                    # match. The test is the full misorientation, reduced by
                    # crystal symmetry: neither the zone axis nor the in-plane
                    # angle alone can tell a symmetry copy (same orientation,
                    # different library entry) from two grains sharing a zone
                    # axis but rotated in plane, and those must be treated
                    # oppositely.
                    K = min(flat.shape[1], top_k_matches)
                    top_v, top_i = flat.topk(K, dim=1)
                    q_top = self._grid_quats(top_i, Z, G, n_ch)  # (B, K, 4)
                    keep = torch.zeros((B, K), dtype=torch.bool)
                    if m == 0:
                        keep[:, 0] = True
                    else:
                        ok = torch.ones((B, K), dtype=torch.bool)
                        for q_mm in q_prev:  # (B, 4) each
                            ang = misorientation_angle_deg(
                                q_mm[:, None, :].expand(-1, K, -1).reshape(-1, 4),
                                q_top.reshape(-1, 4),
                                self.crystal.sym_quats_matching,
                            ).reshape(B, K)
                            ok &= ang >= min_angle_between_matches_deg
                        ok &= torch.isfinite(top_v.cpu())
                        first = torch.where(
                            ok.any(dim=1), ok.double().argmax(dim=1), torch.full((B,), -1)
                        )
                        for b in range(B):
                            if first[b] >= 0:
                                keep[b, int(first[b])] = True
                    sel = torch.where(
                        keep.any(dim=1),
                        keep.double().argmax(dim=1),
                        torch.zeros(B, dtype=torch.long),
                    ).to(flat.device)
                    flat_idx = top_i.gather(1, sel[:, None]).squeeze(1)
                    invalid = ~keep.any(dim=1).to(flat.device)
                else:
                    flat_idx = flat.argmax(dim=1)
                    invalid = torch.zeros(B, dtype=torch.bool, device=flat.device)
                ch_i = flat_idx // (Z * G)
                z_i = (flat_idx // G) % Z
                g_i = flat_idx % G
                c_val = flat.gather(1, flat_idx[:, None]).squeeze(1)
                c_val = c_val.masked_fill(invalid, -torch.inf)

                gamma = gamma_grid[g_i.cpu()].clone()
                if subpixel_gamma:
                    b_ar = torch.arange(B, device=corr.device)
                    c1 = c_val
                    c0 = corr[b_ar, ch_i, z_i, (g_i - 1) % G]
                    c2 = corr[b_ar, ch_i, z_i, (g_i + 1) % G]
                    denom = 4 * c1 - 2 * c0 - 2 * c2
                    dg = torch.where(
                        denom.abs() > 1e-12,
                        (c2 - c0) / denom,
                        torch.zeros_like(denom),
                    ) * (2 * np.pi / G)
                    gamma = gamma + dg.cpu().double()

                is_mirror = ch_i.cpu() == 1
                q_zone = self.zone_quats[z_i.cpu()]

                if subpixel_zone:
                    # sub-grid zone axis: correlation-weighted centroid over
                    # the symmetry-complete neighborhood of the best zone
                    # (see build_plan; images across the wedge boundary keep
                    # the centroid unbiased for boundary zones)
                    b_ar = torch.arange(B, device=corr.device)
                    corr_z = corr[b_ar, ch_i].amax(dim=-1).cpu().double()  # (B, Z)
                    zi_cpu = z_i.cpu()
                    n_idx = self.zone_nbr_idx[zi_cpu]  # (B, K)
                    n_pos = self.zone_nbr_pos[zi_cpu]  # (B, K, 3)
                    n_ok = self.zone_nbr_valid[zi_cpu]  # (B, K)
                    c_n = corr_z.gather(1, n_idx)  # (B, K)
                    za_old = self.zone_axes[z_i.cpu()]
                    za_ref = _zone_peak_parabolic(
                        za_old, n_pos, c_n, n_ok, np.deg2rad(self.zone_step_deg)
                    )
                    axis = torch.cross(za_ref, za_old, dim=-1)
                    sin_t = torch.linalg.norm(axis, dim=-1)
                    ang_t = torch.atan2(sin_t, (za_ref * za_old).sum(-1))
                    ok_t = sin_t > 1e-12
                    dq = torch.zeros((B, 4), dtype=torch.float64)
                    dq[:, 0] = 1.0
                    if bool(ok_t.any()):
                        dq[ok_t] = quat_from_axis_angle(
                            axis[ok_t] / sin_t[ok_t, None], ang_t[ok_t]
                        )
                    # rotate za_ref -> za_old in the crystal frame: R' = R S
                    q_zone = qmult(q_zone, dq)

                q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
                q_zone = torch.where(is_mirror[:, None], qmult(q_flip, q_zone), q_zone)
                gamma = torch.where(is_mirror, -gamma - np.pi, gamma)
                half = gamma / 2
                zeros = torch.zeros_like(half)
                q_spin = torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)
                q = qmult(q_spin, q_zone)

                # score every match against the pattern as measured, so the
                # matches are comparable with each other and across positions
                c_report = corr_full.reshape(B, -1).gather(1, flat_idx[:, None]).squeeze(1)
                c_report = c_report.masked_fill(invalid, -torch.inf)
                for b, (rx, ry) in enumerate(batch):
                    if torch.isfinite(c_val[b]):
                        quats[rx, ry, m] = q[b]
                        corr_out[rx, ry, m] = c_report[b].cpu().double()
                        corr_res[rx, ry, m] = c_val[b].cpu().double()
                        mirror_out[rx, ry, m] = bool(is_mirror[b])

                if m == 0:
                    # second-best score at an orientation genuinely different
                    # from the best one -> reliability = corr - corr_second.
                    # Same misorientation test, so a symmetry copy of the
                    # winner never counts as the runner-up, while a real
                    # in-plane degeneracy does.
                    K2 = min(corr.reshape(B, -1).shape[1], top_k_matches)
                    tv, ti = corr.reshape(B, -1).topk(K2, dim=1)
                    q_t2 = self._grid_quats(ti, Z, G, n_ch)
                    q_best = self._grid_quats(flat_idx[:, None], Z, G, n_ch)[:, 0]
                    ang2 = misorientation_angle_deg(
                        q_best[:, None, :].expand(-1, K2, -1).reshape(-1, 4),
                        q_t2.reshape(-1, 4),
                        self.crystal.sym_quats_matching,
                    ).reshape(B, K2)
                    far2 = (ang2 >= min_angle_between_matches_deg) & torch.isfinite(tv.cpu())
                    c2 = torch.where(
                        far2.any(dim=1),
                        tv.cpu().double().masked_fill(~far2, -torch.inf).amax(dim=1),
                        torch.full((B,), -torch.inf, dtype=torch.float64),
                    )
                    for b, (rx, ry) in enumerate(batch):
                        if torch.isfinite(c2[b]):
                            corr_second[rx, ry] = c2[b]

                if M > 1:
                    q_prev.append(q.clone())

                if M > 1 and m < M - 1 and suppress_matched > 0:
                    # Deflate the matched template out of the measured polar
                    # image, so the next match sees only what this one leaves
                    # unexplained. Without this, the exclusion ball keeps the
                    # next zone axis far away in orientation but nothing stops
                    # it from being fitted to the same peaks -- two grains in
                    # one probe then index as one, and the second match is a
                    # different view of the first.
                    #
                    # The templates are unit vectors, so the amount of this
                    # template present in the image is its raw inner product,
                    # read off the un-normalized correlation at the winning
                    # (zone, in-plane angle). Subtracting that multiple is the
                    # matching-pursuit step and removes it exactly.
                    b_ar = torch.arange(B, device=device)
                    alpha = suppress_matched * corr_raw[b_ar, ch_i, z_i, g_i].clamp_min(0).to(
                        im_fft.dtype
                    )
                    # roll the template to the matched in-plane angle: a shift
                    # of g samples is a linear phase on its transform
                    phase = torch.exp(
                        -2j * np.pi * k_ramp[None, :] * g_i[:, None].to(k_ramp.dtype) / G
                    )
                    t_fft = torch.conj(plan_fft[z_i]) * phase[:, None, :]  # (B, S, G)
                    if include_mirror:
                        # the mirrored template is gamma -> -gamma, a conjugate
                        # in the transform, and its own roll
                        t_mir = torch.conj(t_fft)
                        t_fft = torch.where((ch_i == 1)[:, None, None].to(device), t_mir, t_fft)
                    im_fft = im_fft - alpha[:, None, None] * t_fft
                    # measured intensity is non-negative; keep it that way
                    im_real = torch.fft.ifft(im_fft, dim=-1).real.clamp_min(0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        im_fft = torch.fft.fft(im_real.to(dtype), dim=-1)

        self.quats = quats
        self.corr = corr_out
        self.corr_residual = corr_res
        self.corr_second = corr_second
        self.reliability = corr_out[..., 0] - corr_second
        self.mirror = mirror_out
        return self

    # ------------------------------------------------------------------
    # sub-grid refinement
    # ------------------------------------------------------------------

    def smooth_orientations(
        self,
        match: int = 0,
        sigma_px: float = 1.0,
        sigma_deg: float = 1.0,
        max_angle_deg: float = 5.0,
        positions=None,
    ) -> "OrientationMap":
        """Average each orientation with its neighbours, keeping boundaries sharp.

        A bilateral filter on the orientation field: every position is
        replaced by the weighted mean of the orientations around it, with

            w = exp(-r^2 / 2 sigma_px^2) * exp(-theta^2 / 2 sigma_deg^2)

        for a neighbour r probe positions away whose orientation differs by
        theta, and with neighbours beyond `max_angle_deg` excluded outright.
        The angular term is what keeps this from blurring across a grain
        boundary or between two variants: those neighbours are tens of
        degrees away and carry no weight.

        The point is the noise budget. Neighbouring probe positions inside a
        grain measure the same orientation, so their scatter is measurement
        error and averaging it down costs only spatial resolution, at the
        scale of sigma_px probe steps. Running this before
        `refine_orientations` starts the refinement from a cleaner field;
        running it after smooths what the refinement leaves.

        This does not repair the ambiguities that make an orientation map
        jump by tens of degrees, such as two variants with the same
        zero-layer pattern: those differ by far more than `max_angle_deg`
        and are excluded by design. Fold the in-plane angle by
        `Crystal.projected_rotation_order` for those.

        Parameters
        ----------
        match : int, default=0
            Which match index to smooth.
        sigma_px : float, default=1.0
            Spatial width of the kernel in probe positions. The window is
            three sigma wide.
        sigma_deg : float, default=1.0
            Angular width: a neighbour misoriented by this much is weighted
            down by 1/sqrt(e).
        max_angle_deg : float, default=5.0
            Neighbours beyond this misorientation are excluded.
        positions : list[tuple[int, int]] | np.ndarray | None
            Positions to smooth; defaults to those carrying a match.
        """
        assert self.quats is not None, "run match_orientations() first"
        R, C = self.quats.shape[:2]
        active = position_mask(positions, (R, C))
        if self.computed is not None:
            active = active & self.computed
        active = active & (self.corr[..., match] > 0)
        self.quats[..., match, :] = smooth_quaternions(
            self.quats[..., match, :],
            active,
            self.crystal.sym_quats_matching,
            sigma_px=sigma_px,
            sigma_deg=sigma_deg,
            max_angle_deg=max_angle_deg,
        )
        self.metadata["smooth"] = dict(
            match=int(match),
            sigma_px=float(sigma_px),
            sigma_deg=float(sigma_deg),
            max_angle_deg=float(max_angle_deg),
        )
        return self

    def smoothed_quats(self, match: int = 0, **kwargs) -> torch.Tensor:
        """Smoothed copy of the orientations, leaving the stored ones alone.

        Parameters
        ----------
        match : int, default=0
            Which match index to smooth.
        **kwargs
            `sigma_px`, `sigma_deg` and `max_angle_deg` of
            :func:`smooth_quaternions`.

        Returns
        -------
        torch.Tensor
            (R, C, 4) smoothed quaternions.
        """
        assert self.quats is not None
        R, C = self.quats.shape[:2]
        active = torch.ones((R, C), dtype=torch.bool)
        if self.computed is not None:
            active = active & self.computed
        active = active & (self.corr[..., match] > 0)
        return smooth_quaternions(
            self.quats[..., match, :], active, self.crystal.sym_quats_matching, **kwargs
        )

    def refine_orientations(
        self,
        num_iterations: int = 5,
        positions=None,
        pair_distance: float | None = None,
        sigma_excitation: float | None = None,
        min_pairs: int | None = None,
        refine_tilt: bool = False,
        refine_zone: bool = True,
        zone_search_deg: float = 1.5,
        sigma_envelope: float | None = None,
        zone_max_total_deg: float | None = None,
        power_intensity: float | None = None,
        batched: bool = True,
        neighbor_rescue: bool = True,
        rescue_threshold_deg: float = 2.0,
        rescue_passes: int = 3,
        score_tol: float = 0.002,
        consensus_tol: float = 0.01,
        progress_bar: bool = True,
    ) -> "OrientationMap":
        """Refine matched orientations by least squares on paired peak positions.

        For each probe position and match, the simulated pattern is paired to
        the measured peaks (nearest neighbor within `pair_distance`), and the
        small rotation minimizing the weighted in-plane residuals is solved in
        closed form and applied; repeated for `num_iterations` rounds with
        re-pairing. This removes the in-plane quantization of the orientation
        plan (typically to well below 0.1 degrees).

        By default only the in-plane rotation is refined. Zero-layer peak
        positions carry almost no information about out-of-plane tilt (the
        tilt terms in the position residual are proportional to g_z, which is
        near zero for excited reflections), so fitting the full rotation from
        positions is ill-conditioned and amplifies detection noise into
        spurious tilts. Tilt is constrained by the diffracted *intensities*
        (which reflections are excited) and belongs to the dynamical
        refinement pass.

        Parameters
        ----------
        num_iterations : int, default=5
            Pairing + rotation solve rounds.
        positions : list[tuple[int, int]] | np.ndarray | None
            Scan positions to refine, as for `match_orientations`. None
            (default) refines every position that carries a match, so a
            staged test run on a few positions is refined without repeating
            the position list.
        pair_distance : float | None
            Maximum pairing distance (1/Angstroms); defaults to the plan's
            corr_kernel_size.
        sigma_excitation : float | None
            Excitation error envelope used for simulation; defaults to the
            plan's value.
        min_pairs : int | None
            Skip positions with fewer paired peaks; defaults to MIN_PAIRS (4).
        refine_tilt : bool, default=False
            Also solve the two tilt components from peak positions. Only
            meaningful for noise-free simulated data.
        refine_zone : bool, default=True
            Refine the zone-axis tilt from the intensity envelope (the Laue
            circle): the tilt that concentrates the measured intensity on
            the Ewald sphere, searched over +/- zone_search_deg with
            parabolic sub-stepping. Removes the zone-axis quantization of
            the orientation plan.
        zone_search_deg : float, default=1.5
            Half-range of the envelope tilt search, in degrees.
        zone_max_total_deg : float | None
            Trust region: cap on the cumulative envelope tilt applied to
            each orientation, relative to its matched start. The coarse
            match is grid-accurate to about half the zone-axis step, so tilt
            corrections beyond that scale are noise walking the orientation
            out of its basin. Defaults to 0.75 * the plan's zone step.
        power_intensity : float | None
            Power applied to the measured and predicted intensities in the
            tilt envelope fit, inherited from the plan (0.25 by default).
            Linear intensities let the strongest reflections dominate and,
            on dynamical data, drive the fit to the edge of the search range.
        sigma_envelope : float | None
            Excitation-error width of the envelope objective; defaults to
            half the plan's sigma_excitation (the plan value is widened for
            grid robustness).
        batched : bool, default=True
            Refine positions in vectorized chunks instead of one at a time,
            which is several times faster. `refine_tilt=True` disables it,
            since only the per-position path solves the tilt from positions.
        neighbor_rescue : bool, default=True
            Retry every position that disagrees with a matched neighbour by
            more than `rescue_threshold_deg`, from every distinct candidate
            around it: all matches of the eight neighbours, this position's
            own other matches, and its Friedel twin (the orientation rotated
            180 degrees about the beam). The one that best explains the
            measured peaks is kept, judged by the same correlation matching
            maximizes (see below); among candidates within `consensus_tol`
            of the best, the one most neighbours agree with. Repairs wrong
            local optima, near-degenerate variants, and probe positions
            straddling two grains.
        rescue_threshold_deg : float, default=2.0
            Misorientation to a neighbour that triggers a retry.
        rescue_passes : int, default=3
            Rescue passes; each after the first revisits only positions next
            to a change, since a corrected neighbour can offer a better
            candidate.
        score_tol : float, default=0.002
            Correlation margin. A refinement that moves an orientation by
            more than `rescue_threshold_deg` is undone where it lowers the
            correlation below the library match's by more than this, and a
            rescue candidate replaces the current orientation only when it
            beats it by more than this.
        consensus_tol : float, default=0.01
            Correlation within which two candidates count as equally good.
            Sparse patterns often cannot tell a few orientations apart:
            pseudo-symmetric variants whose distinguishing reflections were
            not recorded, and always the Friedel twin, as kinematic spot
            positions are centrosymmetric and only the Ewald curvature
            separates the two. Such ties are broken by agreement with the
            eight neighbours; the Friedel twin is adopted only this way,
            never on its score alone. 0 judges every position by its own
            pattern alone.
        progress_bar : bool, default=True
            Show one progress bar covering refinement and neighbour rescue.

        Returns
        -------
        OrientationMap
            Self, with `quats` refined in place and `score` (R, C) holding
            the correlation of match 0 with the measured peaks.

        Notes
        -----
        Every orientation is judged by the correlation it gives with the
        measured peaks -- the cosine similarity of the two patterns built
        from the library's Gaussian pairing kernel -- and the result is
        stored in :attr:`score`. Refinement itself works on paired peak
        positions, a different objective; on sparse or ambiguous patterns it
        can move an orientation downhill, and scoring every candidate the
        same way is what keeps the stages consistent. The counts of reverted
        refinements and rescued positions are in ``metadata['refine']``.
        """
        assert self.quats is not None
        plan_md = self.metadata.get("plan")
        delta = resolve(pair_distance, "pair_distance", plan_md, default=self.corr_kernel_size)
        sigma = resolve(
            sigma_excitation, "sigma_excitation", plan_md, default=self.sigma_excitation
        )
        min_pairs = resolve(min_pairs, "min_pairs", default=MIN_PAIRS)
        self.metadata["refine"] = dict(
            num_iterations=int(num_iterations),
            positions=None if positions is None else "subset",
            pair_distance=float(delta),
            sigma_excitation=float(sigma),
            min_pairs=int(min_pairs),
            refine_tilt=bool(refine_tilt),
            refine_zone=bool(refine_zone),
            zone_search_deg=float(zone_search_deg),
            zone_max_total_deg=zone_max_total_deg,
            power_intensity=power_intensity,
            sigma_envelope=sigma_envelope,
            neighbor_rescue=bool(neighbor_rescue),
            rescue_threshold_deg=float(rescue_threshold_deg),
            rescue_passes=int(rescue_passes),
            score_tol=float(score_tol),
            consensus_tol=float(consensus_tol),
        )
        peaks = self.peaks
        R, C, M = self.quats.shape[:3]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]
        g_all = self.crystal.g_vec
        lam = self.wavelength
        sigma_env = sigma_envelope if sigma_envelope is not None else sigma / 2
        power_env = resolve(
            power_intensity,
            "power_intensity",
            self.metadata.get("plan", {}),
            default=POWER_INTENSITY,
        )
        if power_env <= 0:
            # a positions-only plan (power 0) carries no intensity weighting;
            # the envelope fit still needs one
            power_env = POWER_INTENSITY
        prec_ill = float(self.metadata.get("precession_deg", 0.0) or 0.0)
        conv_ill = float(self.metadata.get("semiconv_mrad", 0.0) or 0.0)
        n_rod = self._foil_normal_crystal()
        from quantem.diffraction.illumination import relrod_factor

        def envelope(S, g_rows, f_rows=None):
            # Laue-circle envelope of the paired reflections at shifted
            # excitation errors S (P, T, T), averaged over the illumination
            # recorded on this map (ring: Bessel series; disk: transform);
            # f_rows scales the sweep onto the relrod for a foil normal
            if prec_ill <= 0 and conv_ill <= 0:
                return torch.exp(-(S**2) / (2 * sigma_env**2))
            from quantem.diffraction.illumination import (
                excitation_amplitudes,
                gaussian_envelope,
                gaussian_envelope_ring_torch,
            )

            a_r, b_r = excitation_amplitudes(g_rows, self.energy_ev, prec_ill, conv_ill)
            if f_rows is not None:
                a_r, b_r = a_r * f_rows.abs(), b_r * f_rows.abs()
            if conv_ill <= 0:
                return gaussian_envelope_ring_torch(S, a_r[:, None, None], sigma_env)
            return torch.as_tensor(
                gaussian_envelope(
                    S.numpy(), a_r[:, None, None].numpy(), b_r[:, None, None].numpy(), sigma_env
                ),
                dtype=torch.float64,
            )

        f_all = self.crystal.struct_factors_int.to(torch.float64)
        tg = torch.deg2rad(
            torch.linspace(-zone_search_deg, zone_search_deg, 17, dtype=torch.float64)
        )
        eye3 = torch.eye(3, dtype=torch.float64)
        tilt_cap = np.deg2rad(
            zone_max_total_deg if zone_max_total_deg is not None else 0.75 * self.zone_step_deg
        )

        def refine_single(q, q_exp, w_exp):
            """Refine one orientation; return (q, pairing score)."""
            score = 0.0
            tilt_total = torch.zeros(2, dtype=torch.float64)
            for _ in range(num_iterations):
                g = qrotate(q, g_all)
                gz, g2 = g[:, 2], (g**2).sum(dim=1)
                s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
                spot = g[:, :2]
                f_rod = torch.ones_like(s_g)
                if n_rod is not None:
                    # plate geometry: excitation along the rod, the spot
                    # where the rod meets the sphere
                    n_lab = qrotate(q, n_rod[None])[0]
                    f_rod = relrod_factor(g, n_lab, self.energy_ev, 0.0)
                    s_g = s_g * f_rod
                    spot = spot - s_g[:, None] * n_lab[None, :2]
                sel = torch.abs(s_g) < 2 * sigma
                g_sel = g[sel]
                if g_sel.shape[0] == 0:
                    return q, score
                d = torch.cdist(spot[sel], q_exp)
                d_min, j_min = d.min(dim=1)
                pair = d_min < delta
                if int(pair.sum()) < min_pairs:
                    return q, score
                gp = g_sel[pair]
                tgt = q_exp[j_min[pair]]
                w = w_exp[j_min[pair]] * (1 - d_min[pair] / delta)
                score = float(w.sum())
                # solve min sum w | tgt - (g + omega x g)_xy |^2 for omega
                r = tgt - spot[sel][pair]  # (P, 2)
                if refine_tilt:
                    A = torch.zeros((gp.shape[0], 2, 3), dtype=torch.float64)
                    A[:, 0, 1] = gp[:, 2]
                    A[:, 0, 2] = -gp[:, 1]
                    A[:, 1, 0] = -gp[:, 2]
                    A[:, 1, 2] = gp[:, 0]
                    Aw = A * w[:, None, None]
                    AtA = torch.einsum("pki,pkj->ij", Aw, A)
                    Atr = torch.einsum("pki,pk->i", Aw, r)
                    omega = torch.linalg.solve(AtA + 1e-12 * eye3, Atr)
                else:
                    # in-plane only: residual model r = omega_z * (-g_y, g_x)
                    a = torch.stack((-gp[:, 1], gp[:, 0]), dim=1)  # (P, 2)
                    num = (w[:, None] * a * r).sum()
                    den = (w[:, None] * a * a).sum().clamp_min(1e-12)
                    omega = torch.tensor([0.0, 0.0, float(num / den)], dtype=torch.float64)
                angle = torch.linalg.norm(omega)
                if angle > 1e-10:
                    dq = quat_from_axis_angle(omega / angle, angle)
                    q = qmult(dq, q)

                if refine_zone:
                    # continuous zone-axis tilt from the intensity envelope:
                    # a small lab-frame tilt (wx, wy) shifts every excitation
                    # error by s(w) = s0 + wx*gy - wy*gx; maximize the
                    # normalized cosine between the measured intensities and
                    # the predicted |F|^2 * envelope over a grid with
                    # parabolic sub-stepping (the Laue-circle fit -- peak
                    # positions carry no tilt information, the excitation
                    # pattern does)
                    s0 = s_g[sel][pair]
                    fr = f_rod[sel][pair]
                    a1 = gp[:, 1] * fr
                    a2 = -gp[:, 0] * fr
                    f_p = f_all[sel][pair]
                    S = (
                        s0[:, None, None]
                        + tg[None, :, None] * a1[:, None, None]
                        + tg[None, None, :] * a2[:, None, None]
                    )
                    pred = (
                        f_p[:, None, None]
                        * envelope(S, g_sel[pair], None if n_rod is None else fr)
                    ).clamp_min(0) ** power_env
                    w_env = w**power_env
                    E = (w_env[:, None, None] * pred).sum(dim=0) / (
                        (pred**2).sum(dim=0).sqrt().clamp_min(1e-12)
                    )
                    ij = int(E.argmax())
                    i0, j0 = ij // 17, ij % 17
                    wx, wy = float(tg[i0]), float(tg[j0])
                    step = float(tg[1] - tg[0])
                    if 0 < i0 < 16:
                        c0, c1, c2 = (
                            float(E[i0 - 1, j0]),
                            float(E[i0, j0]),
                            float(E[i0 + 1, j0]),
                        )
                        den = 2 * c1 - c0 - c2
                        if abs(den) > 1e-12:
                            wx += 0.5 * (c2 - c0) / den * step
                    if 0 < j0 < 16:
                        c0, c1, c2 = (
                            float(E[i0, j0 - 1]),
                            float(E[i0, j0]),
                            float(E[i0, j0 + 1]),
                        )
                        den = 2 * c1 - c0 - c2
                        if abs(den) > 1e-12:
                            wy += 0.5 * (c2 - c0) / den * step
                    # trust region on the cumulative tilt from the start
                    prop = tilt_total + torch.tensor([wx, wy], dtype=torch.float64)
                    over = float(torch.linalg.norm(prop)) - tilt_cap
                    if over > 0:
                        prop = prop * tilt_cap / float(torch.linalg.norm(prop))
                    step = prop - tilt_total
                    tilt_total = prop
                    tilt = torch.tensor(
                        [float(step[0]), float(step[1]), 0.0],
                        dtype=torch.float64,
                    )
                    t_ang = torch.linalg.norm(tilt)
                    if t_ang > 1e-10:
                        dq = quat_from_axis_angle(tilt / t_ang, t_ang)
                        q = qmult(dq, q)
            return q, score

        def get_exp(rx, ry):
            data = peaks[rx, ry].numpy().astype(np.float64)
            if data.shape[0] < min_pairs:
                return None, None
            q_exp = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
            w_exp = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
            w_exp = w_exp / w_exp.max().clamp_min(1e-12)
            return q_exp, w_exp

        # positions to refine: those requested, or everything matched
        active = position_mask(positions, (R, C))
        if self.computed is not None:
            active = active & self.computed
        # the library matches, kept so refinement can be undone where it
        # made the fit worse
        q_start = self.quats.clone()
        # one bar per crystal covers refinement and neighbour rescue; each
        # stage adds its own work to the total as it starts
        bar = tqdm(total=0, desc=f"refining {self.crystal.name}") if progress_bar else None
        if batched and not refine_tilt:
            self._refine_batched(
                active=active,
                delta=delta,
                sigma=sigma,
                sigma_env=sigma_env,
                tg=tg,
                tilt_cap=tilt_cap,
                num_iterations=num_iterations,
                min_pairs=min_pairs,
                refine_zone=refine_zone,
                power_env=power_env,
                progress_bar=bar if bar is not None else False,
            )
        else:
            iterator = [(rx, ry) for rx, ry in np.ndindex(R, C) if active[rx, ry]]
            if bar is not None:
                bar.total = (bar.total or 0) + len(iterator)
                bar.refresh()
            for rx, ry in iterator:
                if bar is not None:
                    bar.update(1)
                q_exp, w_exp = get_exp(rx, ry)
                if q_exp is None:
                    continue
                for m in range(M):
                    if self.corr[rx, ry, m] <= 0:
                        continue
                    q, _ = refine_single(self.quats[rx, ry, m], q_exp, w_exp)
                    self.quats[rx, ry, m] = q

        # Refinement polishes the orientation on paired peak positions, which
        # is accurate for small corrections but can jump to another basin on
        # sparse or ambiguous patterns. A polish within `rescue_threshold_deg`
        # is trusted: the correlation depends on the excitation envelope,
        # which is only approximately known, so it cannot referee sub-degree
        # moves. A jump beyond that must explain the measured peaks better
        # than the library match did, or it is undone.
        act_list = [(rx, ry) for rx, ry in np.ndindex(R, C) if active[rx, ry]]
        if bar is not None:
            bar.set_description(f"{self.crystal.name} checking against the library match")
            bar.total += len(act_list)
            bar.refresh()
        cscore = torch.zeros((R, C), dtype=torch.float64)
        n_reverted = 0
        for rx, ry in act_list:
            if bar is not None:
                bar.update(1)
            data = peaks[rx, ry].numpy().astype(np.float64)
            meas = self._measured_term(data, ix)
            for m in range(M):
                if self.corr[rx, ry, m] <= 0:
                    continue
                s_new = self._correlation_score(self.quats[rx, ry, m], data, ix, meas)
                moved = float(
                    misorientation_angle_deg(
                        self.quats[rx, ry, m], q_start[rx, ry, m], self.crystal.sym_quats_matching
                    )
                )
                if moved > rescue_threshold_deg:
                    s_old = self._correlation_score(q_start[rx, ry, m], data, ix, meas)
                    if s_new < s_old - score_tol:
                        self.quats[rx, ry, m] = q_start[rx, ry, m]
                        s_new = s_old
                        n_reverted += int(m == 0)
                if m == 0:
                    cscore[rx, ry] = s_new
        self.metadata["refine"]["n_reverted"] = int(n_reverted)

        if neighbor_rescue:
            # A wrong local optimum shows as a position disagreeing with a
            # neighbour. Retry every such position from every distinct
            # candidate around it -- all matches of the eight neighbours,
            # this position's own other matches, and its Friedel twin -- and
            # keep whichever explains the measured peaks best, by the same
            # correlation; candidates within `consensus_tol` of the best are a
            # tie, broken by how many neighbours agree. Only matched
            # neighbours count, and the comparison is in the group the library
            # was built with, where folded variants are one answer. Repeat
            # while anything changes, up to `rescue_passes` times.
            sym_m = self.crystal.sym_quats_matching
            # 180 degrees about the beam: kinematic spot positions are
            # centrosymmetric and the excitation errors nearly so, so only the
            # Ewald curvature tells the two apart
            q_twin = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=self.quats.dtype)
            n_retried = n_rescued = 0
            changed = active.clone()
            for _pass in range(max(int(rescue_passes), 0)):
                q0 = self.quats[..., 0, :]
                miso_max = torch.zeros((R, C), dtype=torch.float64)
                for dr, dc in ((0, 1), (1, 0)):
                    both = active[: R - dr, : C - dc] & active[dr:, dc:]
                    mm = misorientation_angle_deg(
                        q0[: R - dr, : C - dc].reshape(-1, 4), q0[dr:, dc:].reshape(-1, 4), sym_m
                    ).reshape(R - dr, C - dc)
                    mm = torch.where(both, mm, torch.zeros_like(mm))
                    miso_max[: R - dr, : C - dc] = torch.maximum(miso_max[: R - dr, : C - dc], mm)
                    miso_max[dr:, dc:] = torch.maximum(miso_max[dr:, dc:], mm)
                # after the first pass, only where something nearby changed
                near_change = changed.clone()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        near_change |= torch.roll(torch.roll(changed, dr, 0), dc, 1)
                retry = torch.nonzero((miso_max > rescue_threshold_deg) & active & near_change)
                it2 = retry.tolist()
                if not it2:
                    break
                if bar is not None:
                    bar.set_description(
                        f"{self.crystal.name} neighbor rescue {_pass + 1}/{rescue_passes}"
                    )
                    bar.set_postfix_str(f"{len(it2)} positions")
                    bar.total += len(it2)
                    bar.refresh()
                changed = torch.zeros((R, C), dtype=torch.bool)
                for rx, ry in it2:
                    if bar is not None:
                        bar.update(1)
                    n_retried += 1
                    q_exp, w_exp = get_exp(rx, ry)
                    if q_exp is None:
                        continue
                    data = peaks[rx, ry].numpy().astype(np.float64)
                    cur_q = self.quats[rx, ry, 0].clone()
                    cur_s = float(cscore[rx, ry])
                    # every candidate in order: this orientation, its Friedel
                    # twin, its own other matches, the neighbours' matches
                    raw = [cur_q, qmult(q_twin, cur_q)]
                    nbrs = []
                    for m in range(1, M):
                        if self.corr[rx, ry, m] > 0:
                            raw.append(self.quats[rx, ry, m])
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rx + dr, ry + dc
                            if (dr == 0 and dc == 0) or not (0 <= nr < R and 0 <= nc < C):
                                continue
                            if not bool(active[nr, nc]):
                                continue
                            nbrs.append(self.quats[nr, nc, 0])
                            for m in range(M):
                                if self.corr[nr, nc, m] > 0:
                                    raw.append(self.quats[nr, nc, m])
                    # drop repeats within 0.5 degrees, keeping the first, from
                    # one batched misorientation matrix
                    qr = torch.stack(raw)
                    dup = (misorientation_angle_deg(qr[:, None], qr[None], sym_m) <= 0.5).numpy()
                    keep: list[int] = []
                    for i in range(len(raw)):
                        if not any(dup[i, j] for j in keep):
                            keep.append(i)
                    cands = [raw[i] for i in keep]
                    # none where the twin is a symmetry copy of this orientation
                    twin_ix = 1 if 1 in keep else None
                    # score every candidate as it stands -- the neighbours'
                    # were refined on a neighbouring pattern already
                    cq = torch.stack(cands)
                    meas = self._measured_term(data, ix)
                    cs = [cur_s] + [
                        self._correlation_score(qc, data, ix, meas) for qc in cands[1:]
                    ]
                    # polish the best on this pattern, keeping it if it helps
                    k = int(np.argmax(cs))
                    if k > 0:
                        q_ref, _ = refine_single(cq[k].clone(), q_exp, w_exp)
                        s_ref = self._correlation_score(q_ref, data, ix, meas)
                        if s_ref > cs[k]:
                            cq[k], cs[k] = q_ref, s_ref
                    cs = np.asarray(cs)
                    # ties within consensus_tol go to the candidate most
                    # neighbours agree with, then to the higher correlation
                    support = np.zeros(len(cs), dtype=int)
                    if nbrs:
                        nq = torch.stack(nbrs)
                        miso = misorientation_angle_deg(cq[:, None, :], nq[None], sym_m)
                        support = (miso <= rescue_threshold_deg).sum(dim=1).numpy()
                    tied = cs >= cs.max() - consensus_tol
                    pick = max(np.flatnonzero(tied), key=lambda j: (support[j], cs[j]))
                    agreed = (
                        consensus_tol > 0
                        and support[pick] > support[0]
                        and cs[pick] >= cur_s - consensus_tol
                    )
                    if not agreed:
                        # by the pattern alone; the twin never wins here, as
                        # its pattern differs only by the Ewald curvature and
                        # a higher score is the forward model's error
                        own = [j for j in range(len(cs)) if j != twin_ix]
                        pick = max(own, key=lambda j: cs[j])
                    if pick > 0 and (agreed or cs[pick] > cur_s + score_tol):
                        n_rescued += 1
                        changed[rx, ry] = True
                        self.quats[rx, ry, 0] = cq[pick]
                        cscore[rx, ry] = float(cs[pick])
            self.metadata["refine"]["n_retried"] = int(n_retried)
            self.metadata["refine"]["n_rescued"] = int(n_rescued)
        if bar is not None:
            bar.set_description(f"refined {self.crystal.name}")
            bar.set_postfix_str(
                f"kept the library match at {n_reverted}"
                + (
                    f", rescued {self.metadata['refine'].get('n_rescued', 0)}"
                    if neighbor_rescue
                    else ""
                )
            )
        self.score = cscore
        if bar is not None:
            bar.close()
        return self

    # ------------------------------------------------------------------
    # forward simulation of a match
    # ------------------------------------------------------------------

    def _measured_term(self, data: np.ndarray, ix: list[int]) -> tuple:
        """Measured side of :meth:`_correlation_score`: positions, weights
        and self-overlap, the same for every candidate at a position."""
        m_xy = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
        m_i = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
        w_m = (
            m_i**self.power_intensity_experiment
            * torch.linalg.norm(m_xy, dim=1) ** self.power_radial
        )
        inv = 1.0 / (4.0 * self.corr_kernel_size**2)
        mm = float(w_m @ torch.exp(-(torch.cdist(m_xy, m_xy) ** 2) * inv) @ w_m)
        return m_xy, w_m, max(mm, 0.0)

    def _correlation_score(
        self, q: torch.Tensor, data: np.ndarray, ix: list[int], measured: tuple | None = None
    ) -> float:
        """How well orientation `q` explains the measured peaks `data`.

        The cosine similarity between the measured and the simulated
        patterns, each a set of Gaussian spots of the library's pairing
        width: the continuous form of the correlation matching maximizes,
        with the same intensity and radial weights. Unlike a sum of paired
        intensity it charges for predicted spots that were not measured, so
        a denser pattern does not win by pairing more peaks. Simulated
        reflections outside the detector are left out, as in the library.
        """
        pd = float(self.metadata.get("precession_deg", 0.0) or 0.0)
        sc = float(self.metadata.get("semiconv_mrad", 0.0) or 0.0)
        sim = self.crystal.generate_pattern(
            q,
            energy_ev=self.energy_ev,
            sigma_excitation=self.sigma_excitation,
            precession_deg=pd,
            semiconv_mrad=sc,
            foil_normal=self.metadata.get("foil_normal"),
        )
        s_xy = torch.stack((sim["qx"], sim["qy"]), dim=1).to(torch.float64)
        s_i = sim["intensity"].to(torch.float64).clamp_min(0)
        det = (self.metadata.get("plan") or {}).get("detector_q_max")
        if det is not None and s_xy.shape[0]:
            det = np.atleast_1d(det).astype(float)
            qx_max, qy_max = (det[0], det[0]) if det.size == 1 else (det[0], det[1])
            rot = np.deg2rad(-float(self.peaks.metadata.get("rotation_ccw_deg", 0.0) or 0.0))
            c, s_ = np.cos(rot), np.sin(rot)
            r_det = s_xy[:, 0] * c - s_xy[:, 1] * s_
            c_det = s_xy[:, 0] * s_ + s_xy[:, 1] * c
            on = (r_det.abs() <= qx_max) & (c_det.abs() <= qy_max)
            s_xy, s_i = s_xy[on], s_i[on]
        p_rad = float(self.power_radial)
        inv = 1.0 / (4.0 * self.corr_kernel_size**2)

        def overlap(a, wa, b, wb):
            return float(wa @ torch.exp(-(torch.cdist(a, b) ** 2) * inv) @ wb)

        m_xy, w_m, mm = measured if measured is not None else self._measured_term(data, ix)
        if s_xy.shape[0] == 0 or m_xy.shape[0] == 0:
            return 0.0
        w_s = s_i**self.power_intensity * torch.linalg.norm(s_xy, dim=1) ** p_rad
        norm = np.sqrt(mm * max(overlap(s_xy, w_s, s_xy, w_s), 0.0))
        return overlap(m_xy, w_m, s_xy, w_s) / norm if norm > 0 else 0.0

    def _refine_batched(
        self,
        active: torch.Tensor,
        delta: float,
        sigma: float,
        sigma_env: float,
        tg: torch.Tensor,
        tilt_cap: float,
        num_iterations: int,
        min_pairs: int,
        refine_zone: bool,
        progress_bar,
        chunk: int = 64,
        power_env: float = POWER_INTENSITY,
    ) -> None:
        """Chunk-vectorized in-plane + envelope refinement of the active positions."""
        from quantem.diffraction.rotations import quat_to_matrix

        peaks = self.peaks
        R, C, M = self.quats.shape[:3]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]
        g_all = self.crystal.g_vec  # (G, 3)
        f_all = self.crystal.struct_factors_int.to(torch.float64)
        lam = self.wavelength
        n_tg = tg.shape[0]
        prec_ill = float(self.metadata.get("precession_deg", 0.0) or 0.0)
        conv_ill = float(self.metadata.get("semiconv_mrad", 0.0) or 0.0)
        n_rod = self._foil_normal_crystal()
        from quantem.diffraction.illumination import relrod_factor

        def envelope(S, g_rows, f_rows=None):
            # Laue-circle envelope of the paired reflections at shifted
            # excitation errors S (P, T, T), averaged over the illumination
            # recorded on this map (ring: Bessel series; disk: transform);
            # f_rows scales the sweep onto the relrod for a foil normal
            if prec_ill <= 0 and conv_ill <= 0:
                return torch.exp(-(S**2) / (2 * sigma_env**2))
            from quantem.diffraction.illumination import (
                excitation_amplitudes,
                gaussian_envelope,
                gaussian_envelope_ring_torch,
            )

            a_r, b_r = excitation_amplitudes(g_rows, self.energy_ev, prec_ill, conv_ill)
            if f_rows is not None:
                a_r, b_r = a_r * f_rows.abs(), b_r * f_rows.abs()
            if conv_ill <= 0:
                return gaussian_envelope_ring_torch(S, a_r[:, None, None], sigma_env)
            return torch.as_tensor(
                gaussian_envelope(
                    S.numpy(), a_r[:, None, None].numpy(), b_r[:, None, None].numpy(), sigma_env
                ),
                dtype=torch.float64,
            )

        # flatten measured peaks once, padded per position
        cells = [peaks[r, c].numpy().astype(np.float64) for r, c in np.ndindex(R, C)]
        counts = np.array([c.shape[0] for c in cells])
        Pmax = max(1, counts.max())
        N = R * C
        q_exp = torch.full((N, Pmax, 2), 1e6, dtype=torch.float64)
        w_exp = torch.zeros((N, Pmax), dtype=torch.float64)
        for i, arr in enumerate(cells):
            n = arr.shape[0]
            if n == 0:
                continue
            q_exp[i, :n] = torch.as_tensor(arr[:, ix[:2]], dtype=torch.float64)
            wi = torch.as_tensor(arr[:, ix[2]], dtype=torch.float64).clamp_min(0)
            w_exp[i, :n] = wi / wi.max().clamp_min(1e-12)

        quats = self.quats.reshape(N, M, 4)
        corr = self.corr.reshape(N, M)
        valid_pos = torch.as_tensor(counts >= min_pairs) & active.reshape(N)

        chunks = [i for i in range(0, N, chunk) if bool(valid_pos[i : i + chunk].any())]
        # `progress_bar` is either a flag or the bar refine_orientations
        # shares with the neighbour rescue, so one crystal shows one bar
        bar = progress_bar if hasattr(progress_bar, "update") else None
        if bar is not None:
            bar.total = (bar.total or 0) + len(chunks)
            bar.refresh()
        elif progress_bar:
            chunks = tqdm(chunks, desc=f"refining {self.crystal.name}")
        for i0 in chunks:
            if bar is not None:
                bar.update(1)
            i1 = min(i0 + chunk, N)
            B = i1 - i0
            qe = q_exp[i0:i1]  # (B, P, 2)
            we = w_exp[i0:i1]  # (B, P)
            for m in range(M):
                act = valid_pos[i0:i1] & (corr[i0:i1, m] > 0)
                if not bool(act.any()):
                    continue
                q = quats[i0:i1, m].clone()  # (B, 4)
                tilt_total = torch.zeros((B, 2), dtype=torch.float64)
                for _ in range(num_iterations):
                    Rm = quat_to_matrix(q)  # (B, 3, 3)
                    g = torch.einsum("bij,gj->bgi", Rm, g_all)  # (B, G, 3)
                    gz, g2 = g[..., 2], (g**2).sum(dim=-1)
                    s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
                    spot = g[..., :2]
                    f_rod = None
                    if n_rod is not None:
                        # plate geometry: excitation along the rod, the spot
                        # where the rod meets the sphere
                        n_lab = torch.einsum("bij,j->bi", Rm, n_rod)  # (B, 3)
                        f_rod = relrod_factor(g, n_lab[:, None, :], self.energy_ev, 0.0)
                        s_g = s_g * f_rod
                        spot = spot - s_g[..., None] * n_lab[:, None, :2]
                    sel = torch.abs(s_g) < 2 * sigma  # (B, G)
                    d = torch.cdist(spot, qe)  # (B, G, P)
                    d_min, j_min = d.min(dim=-1)  # (B, G)
                    pair = sel & (d_min < delta)
                    w_g = torch.gather(we, 1, j_min) * (1 - d_min / delta).clamp_min(0)
                    w_g = w_g * pair  # (B, G)
                    n_pair = pair.sum(dim=1)
                    ok = act & (n_pair >= min_pairs)
                    if not bool(ok.any()):
                        break
                    tgt = torch.gather(qe, 1, j_min[..., None].expand(-1, -1, 2))  # (B, G, 2)
                    r_vec = tgt - spot
                    # in-plane closed form
                    a_vec = torch.stack((-spot[..., 1], spot[..., 0]), dim=-1)
                    num = (w_g[..., None] * a_vec * r_vec).sum(dim=(1, 2))
                    den = (w_g[..., None] * a_vec * a_vec).sum(dim=(1, 2))
                    wz = torch.where(ok, num / den.clamp_min(1e-12), torch.zeros_like(num))
                    half = wz / 2
                    dq = torch.stack(
                        (
                            torch.cos(half),
                            torch.zeros_like(half),
                            torch.zeros_like(half),
                            torch.sin(half),
                        ),
                        dim=-1,
                    )
                    q = torch.where(ok[:, None], qmult(dq, q), q)

                    if refine_zone:
                        # sparse over paired reflections only
                        idx_b, idx_g = torch.nonzero(pair, as_tuple=True)
                        s0f = s_g[idx_b, idx_g]
                        fr = None if f_rod is None else f_rod[idx_b, idx_g]
                        gyf = g[idx_b, idx_g, 1] * (1.0 if fr is None else fr)
                        gxf = g[idx_b, idx_g, 0] * (1.0 if fr is None else fr)
                        ff = f_all[idx_g]
                        wf = w_g[idx_b, idx_g]
                        S = (
                            s0f[:, None, None]
                            + tg[None, :, None] * gyf[:, None, None]
                            - tg[None, None, :] * gxf[:, None, None]
                        )  # (Np, T, T)
                        pred = (ff[:, None, None] * envelope(S, g[idx_b, idx_g], fr)).clamp_min(
                            0
                        ) ** power_env
                        E_num = torch.zeros((B, n_tg, n_tg), dtype=torch.float64).index_add_(
                            0, idx_b, (wf**power_env)[:, None, None] * pred
                        )
                        E_den = torch.zeros((B, n_tg, n_tg), dtype=torch.float64).index_add_(
                            0, idx_b, pred**2
                        )
                        E = E_num / E_den.sqrt().clamp_min(1e-12)  # (B, T, T)
                        flat_ij = E.reshape(B, -1).argmax(dim=1)
                        i_b, j_b = flat_ij // n_tg, flat_ij % n_tg
                        step = float(tg[1] - tg[0])
                        wx = tg[i_b].clone()
                        wy = tg[j_b].clone()
                        # parabolic sub-stepping where interior
                        b_ar = torch.arange(B)
                        for axis, idx, wv in ((0, i_b, wx), (1, j_b, wy)):
                            interior = (idx > 0) & (idx < n_tg - 1)
                            if not bool(interior.any()):
                                continue
                            if axis == 0:
                                c0 = E[b_ar, (idx - 1).clamp(0), j_b]
                                c1 = E[b_ar, idx, j_b]
                                c2 = E[b_ar, (idx + 1).clamp(max=n_tg - 1), j_b]
                            else:
                                c0 = E[b_ar, i_b, (idx - 1).clamp(0)]
                                c1 = E[b_ar, i_b, idx]
                                c2 = E[b_ar, i_b, (idx + 1).clamp(max=n_tg - 1)]
                            den2 = 2 * c1 - c0 - c2
                            shift = torch.where(
                                interior & (den2.abs() > 1e-12),
                                0.5 * (c2 - c0) / den2 * step,
                                torch.zeros_like(c1),
                            )
                            wv += shift
                        prop = tilt_total + torch.stack((wx, wy), dim=-1)
                        norm = torch.linalg.norm(prop, dim=-1)
                        scale_f = torch.where(
                            norm > tilt_cap,
                            tilt_cap / norm.clamp_min(1e-12),
                            torch.ones_like(norm),
                        )
                        prop = prop * scale_f[:, None]
                        step_t = torch.where(
                            ok[:, None], prop - tilt_total, torch.zeros_like(prop)
                        )
                        tilt_total = torch.where(ok[:, None], prop, tilt_total)
                        t_ang = torch.linalg.norm(step_t, dim=-1)
                        axis_v = torch.zeros((B, 3), dtype=torch.float64)
                        nz = t_ang > 1e-10
                        if bool(nz.any()):
                            axis_v[nz, 0] = step_t[nz, 0] / t_ang[nz]
                            axis_v[nz, 1] = step_t[nz, 1] / t_ang[nz]
                            dq_t = quat_from_axis_angle(axis_v[nz], t_ang[nz])
                            q_nz = q[nz]
                            q[nz] = qmult(dq_t, q_nz)
                quats[i0:i1, m] = torch.where(act[:, None], q, quats[i0:i1, m])
        self.quats = quats.reshape(R, C, M, 4)

    def generate_pattern(self, rx: int, ry: int, match: int = 0, **kwargs):
        """Simulated pattern for the matched orientation at one probe position.

        The illumination (precession, convergence) and foil normal recorded
        on this map are used unless overridden.

        Parameters
        ----------
        rx, ry : int
            Scan row and column.
        match : int, default=0
            Which match index to simulate.
        **kwargs
            Passed to :meth:`Crystal.generate_pattern`, e.g. `k_max`.

        Returns
        -------
        dict[str, torch.Tensor]
            The simulated reflections, as returned by
            :meth:`Crystal.generate_pattern` ("qx", "qy", "intensity", ...).
        """
        assert self.quats is not None
        kwargs.setdefault("precession_deg", self.metadata.get("precession_deg", 0.0))
        kwargs.setdefault("semiconv_mrad", self.metadata.get("semiconv_mrad", 0.0))
        kwargs.setdefault("foil_normal", self.metadata.get("foil_normal"))
        return self.crystal.generate_pattern(
            self.quats[rx, ry, match],
            energy_ev=self.energy_ev,
            sigma_excitation=self.sigma_excitation,
            **kwargs,
        )

    def match_residual(
        self,
        other: "OrientationMap",
        delete_radius: float = 0.04,
        min_number_peaks: int = MIN_NUMBER_PEAKS,
        min_corr_other: float = 0.0,
        progress_bar: bool = True,
    ) -> "OrientationMap":
        """Re-match this crystal on the peaks another crystal cannot explain.

        For overlapping patterns (e.g. a thin lath on a matrix), the direct
        match of the minority phase is poisoned by the majority phase's
        peaks. Here the other crystal's simulated pattern is used to delete
        its measured peaks at each position, and this crystal is matched and
        refined against the remaining peaks only, with this map's plan.
        Where the residual match scores above this map's stored second
        match, it replaces it (match index 1), so the joint phase fit sees
        one clean candidate per phase. A map with a single match is first
        extended to two, the second empty (correlation zero).

        Parameters
        ----------
        other : OrientationMap
            The matched map of the (locally dominant) other crystal.
        delete_radius : float, default=0.04
            Measured peaks within this distance (1/Angstroms) of one of the
            other crystal's simulated peaks are removed.
        min_number_peaks : int, default=MIN_NUMBER_PEAKS (5)
            Positions with fewer measured peaks, or fewer residual peaks,
            are not re-matched.
        min_corr_other : float, default=0.0
            Positions where the other crystal's correlation is at or below
            this are not re-matched (nothing trustworthy to delete).
        progress_bar : bool, default=True
            Show progress bars for the residual matching and refinement.

        Returns
        -------
        OrientationMap
            Self, with `quats`, `corr`, `corr_residual` and `mirror` holding
            at least two matches. Where the residual match was taken, both
            `corr[..., 1]` and `corr_residual[..., 1]` hold its correlation
            with the residual peaks. When no position has enough residual
            peaks, nothing is replaced.
        """
        if self.quats is None or other.quats is None:
            raise RuntimeError("Run match_orientations() on both maps first.")
        if self.plan_fft is None:
            raise RuntimeError("Run build_plan() first.")
        peaks = self.peaks
        R, C = peaks.shape[0], peaks.shape[1]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]
        self.metadata["match_residual"] = dict(
            other=other.crystal.name,
            delete_radius=float(delete_radius),
            min_number_peaks=int(min_number_peaks),
            min_corr_other=float(min_corr_other),
        )

        cells = []
        for rx, ry in np.ndindex(R, C):
            data = peaks[rx, ry].numpy().astype(np.float64)
            if data.shape[0] < min_number_peaks or other.corr[rx, ry, 0] <= min_corr_other:
                cells.append(np.zeros((0, 3)))
                continue
            sim = other.generate_pattern(rx, ry)
            sq = torch.stack((sim["qx"], sim["qy"]), dim=1).to(torch.float64)
            if sq.shape[0] == 0:
                cells.append(data[:, ix])
                continue
            qxy = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
            d_min = torch.cdist(qxy, sq).min(dim=1).values
            keep = (d_min > delete_radius).numpy()
            cells.append(data[keep][:, ix])

        # extend to two matches first, so the result has the same layout
        # whether or not anything is replaced
        if self.quats.shape[2] < 2:
            pad_q = torch.zeros((R, C, 1, 4), dtype=self.quats.dtype)
            pad_q[..., 0] = 1.0
            self.quats = torch.cat([self.quats, pad_q], dim=2)
            pad = torch.zeros((R, C, 1), dtype=self.corr.dtype)
            self.corr = torch.cat([self.corr, pad], dim=2)
            if self.corr_residual is not None:
                self.corr_residual = torch.cat([self.corr_residual, pad.clone()], dim=2)
            self.mirror = torch.cat([self.mirror, torch.zeros((R, C, 1), dtype=torch.bool)], dim=2)
        if self.corr_residual is None:
            self.corr_residual = self.corr.clone()

        # positions with enough residual peaks, among those this map covers
        wanted = torch.as_tensor(
            np.array([c.shape[0] >= min_number_peaks for c in cells]).reshape(R, C)
        )
        if self.computed is not None:
            wanted &= self.computed
        if not bool(wanted.any()):
            return self

        residual = Vector.from_data(
            [cells[r * C : (r + 1) * C] for r in range(R)],
            fields=["qx", "qy", "intensity"],
            units=["A^-1", "A^-1", "counts"],
            name="residual_peaks",
            metadata=dict(peaks.metadata or {}),
            dtype=peaks.dtype,
        )
        om_res = OrientationMap.from_vectors(
            residual,
            self.crystal,
            self.energy_ev,
            precession_deg=self.metadata.get("precession_deg", 0.0) or 0.0,
            semiconv_mrad=self.metadata.get("semiconv_mrad", 0.0) or 0.0,
            foil_normal=self.metadata.get("foil_normal"),
        )
        # share this map's plan rather than rebuilding it
        for attr in (
            "device",
            "dtype",
            "cdtype",
            "corr_kernel_size",
            "sigma_excitation",
            "power_radial",
            "power_intensity",
            "power_intensity_experiment",
            "zone_axis_range",
            "zone_axes",
            "zone_quats",
            "zone_step_deg",
            "zone_nbr_idx",
            "zone_nbr_pos",
            "zone_nbr_valid",
            "plan_fft",
            "shell_radii",
            "num_gamma",
            "gamma",
            "detector_mask",
            "plan_norm_shift",
            "plan_frac_shift",
        ):
            setattr(om_res, attr, getattr(self, attr))
        om_res.metadata["plan"] = dict(self.metadata.get("plan") or {})
        om_res.match_orientations(
            num_matches=1,
            positions=wanted.numpy(),
            min_number_peaks=min_number_peaks,
            progress_bar=progress_bar,
        )
        om_res.refine_orientations(progress_bar=progress_bar)

        # replace the stored second match where the residual match is better
        better = om_res.computed & (om_res.corr[..., 0] > self.corr[..., 1])
        self.quats[..., 1, :] = torch.where(
            better[..., None], om_res.quats[..., 0, :], self.quats[..., 1, :]
        )
        self.corr[..., 1] = torch.where(better, om_res.corr[..., 0], self.corr[..., 1])
        self.corr_residual[..., 1] = torch.where(
            better, om_res.corr[..., 0], self.corr_residual[..., 1]
        )
        self.mirror[..., 1] = torch.where(better, om_res.mirror[..., 0], self.mirror[..., 1])
        return self

    def cluster_orientations(
        self,
        mask: np.ndarray | None = None,
        threshold_deg: float = 5.0,
        min_cluster_size: int = 10,
        match: int = 0,
    ) -> dict:
        """Greedy clustering of the matched orientations into variants.

        Positions are visited in order of decreasing correlation; each seed
        collects every unassigned position within `threshold_deg`
        (symmetry-reduced misorientation) into a cluster. Follows the variant
        analysis of MacLaren et al., J. Microscopy 295, 131 (2024).

        Parameters
        ----------
        mask : np.ndarray | None
            (R, C) boolean or weight mask of positions to include, e.g. a
            phase mask. A position is included where the mask is above 0.5,
            the same rule as :meth:`calculate_strain`.
        threshold_deg : float, default=5.0
            Misorientation radius of a cluster, degrees.
        min_cluster_size : int, default=10
            Smaller clusters are discarded (labels stay -1).
        match : int, default=0
            Which match index to cluster.

        Returns
        -------
        dict
            'labels' (R, C) int tensor, -1 = unassigned; 'mean_quats'
            (K, 4) cluster mean orientations; 'sizes' (K,) member counts.
        """
        assert self.quats is not None
        R, C = self.quats.shape[:2]
        q = self.quats[..., match, :].reshape(-1, 4)
        corr = self.corr[..., match].reshape(-1)
        ok = corr > 0
        if mask is not None:
            ok &= torch.as_tensor(np.asarray(mask, dtype=float).reshape(-1)) > 0.5

        labels = torch.full((R * C,), -1, dtype=torch.long)
        # variants the library folds together belong to one grain
        sym = self.crystal.sym_quats_matching
        unassigned = ok.clone()
        means, sizes = [], []
        k = 0
        while unassigned.any():
            seed = int(torch.where(unassigned, corr, torch.full_like(corr, -1)).argmax())
            miso = misorientation_angle_deg(q[seed][None], q, sym)
            members = unassigned & (miso < threshold_deg)
            unassigned &= ~members
            if int(members.sum()) < min_cluster_size:
                continue
            labels[members] = k
            # symmetry-align members to the seed, then average
            qm = q[members]
            dq = qmult(qconj(q[seed])[None], qm)
            dq_sym = qmult(dq[:, None, :], sym)
            best = dq_sym[..., 0].abs().argmax(dim=1)
            dq_best = dq_sym[torch.arange(qm.shape[0]), best]
            sign = torch.where(dq_best[:, :1] < 0, -1.0, 1.0)
            q_aligned = qmult(q[seed][None], dq_best * sign)
            means.append(qnormalize(q_aligned.mean(dim=0)))
            sizes.append(int(members.sum()))
            k += 1
        # order clusters by size, largest first
        if means:
            order = torch.argsort(torch.tensor(sizes), descending=True)
            relabel = torch.full((len(sizes),), -1, dtype=torch.long)
            relabel[order] = torch.arange(len(sizes))
            labels = torch.where(labels >= 0, relabel[labels.clamp_min(0)], labels)
            means = [means[int(i)] for i in order]
            sizes = [sizes[int(i)] for i in order]
        return {
            "labels": labels.reshape(R, C),
            "mean_quats": torch.stack(means) if means else torch.zeros((0, 4)),
            "sizes": torch.tensor(sizes),
        }

    def calculate_strain(
        self,
        match: int = 0,
        pair_distance: float | None = None,
        min_pairs: int | None = None,
        mask: np.ndarray | None = None,
        ds_sampling: float | None = None,
        ds_units: str | None = None,
        progress_bar: bool = True,
    ):
        """Per-position strain from measured vs simulated peak positions.

        At each probe position the refined orientation's simulated pattern is
        paired to the measured peaks and the in-plane deformation A
        minimizing sum w |A q_sim - q_meas|^2 is solved in closed form. The
        strain is referenced to the crystal's ideal lattice, so unlike
        lattice-vector strain mapping it is absolute, not relative to a
        reference region.

        Parameters
        ----------
        match : int, default=0
            Which match index to measure.
        pair_distance : float | None
            Largest distance (1/Angstroms) between a simulated and a measured
            peak that are paired; inherits the refinement's, then the plan's.
        min_pairs : int | None
            Positions with fewer paired peaks are left as NaN; inherits the
            refinement's value, else MIN_PAIRS (4).
        mask : np.ndarray | None
            (R, C) boolean or weight mask of positions to measure. A position
            is measured where the mask is above 0.5, the same rule as
            :meth:`cluster_orientations`. None measures every matched
            position.
        ds_sampling : float | None
            Scan step, passed to StrainMap for scale bars.
        ds_units : str | None
            Units of `ds_sampling`.
        progress_bar : bool, default=True
            Show a progress bar over the positions.

        Returns
        -------
        StrainMap
            The columns of A enter as per-position reciprocal lattice
            vectors with the identity as the fixed reference, so all
            StrainMap machinery applies: `plot_strain(rotation_angle=...)`
            for user-chosen u/v directions, `rotate_strain`, masking, and
            scale bars. `num_pairs` (R, C) is attached as an attribute.
        """
        from quantem.diffraction.strain import StrainMap

        assert self.quats is not None
        delta = resolve(
            pair_distance,
            "pair_distance",
            self.metadata.get("refine"),
            self.metadata.get("plan"),
            default=self.corr_kernel_size,
        )
        min_pairs = resolve(min_pairs, "min_pairs", self.metadata.get("refine"), default=MIN_PAIRS)
        self.metadata["strain"] = dict(
            match=int(match), pair_distance=float(delta), min_pairs=int(min_pairs)
        )
        peaks = self.peaks
        R, C = peaks.shape[0], peaks.shape[1]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

        A_map = np.full((R, C, 2, 2), np.nan)
        num_pairs = np.zeros((R, C), dtype=int)
        include = None if mask is None else np.asarray(mask, dtype=float) > 0.5

        iterator = list(np.ndindex(R, C))
        if progress_bar:
            iterator = tqdm(iterator, desc=f"strain mapping {self.crystal.name}")
        for rx, ry in iterator:
            if include is not None and not include[rx, ry]:
                continue
            if self.corr[rx, ry, match] <= 0:
                continue
            data = peaks[rx, ry].numpy().astype(np.float64)
            if data.shape[0] < min_pairs:
                continue
            q_exp = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
            w_exp = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
            sim = self.generate_pattern(rx, ry, match=match)
            sq = torch.stack((sim["qx"], sim["qy"]), dim=1)
            if sq.shape[0] == 0:
                continue
            d = torch.cdist(sq, q_exp)
            d_min, j_min = d.min(dim=1)
            pair = d_min < delta
            n = int(pair.sum())
            if n < min_pairs:
                continue
            qs = sq[pair]
            qm = q_exp[j_min[pair]]
            w = w_exp[j_min[pair]] * (1 - d_min[pair] / delta)
            # A = (sum w qm qs^T) (sum w qs qs^T)^-1
            M1 = torch.einsum("p,pi,pj->ij", w, qm, qs)
            M2 = torch.einsum("p,pi,pj->ij", w, qs, qs)
            A = M1 @ torch.linalg.inv(M2 + 1e-12 * torch.eye(2, dtype=torch.float64))
            A_map[rx, ry] = A.numpy()
            num_pairs[rx, ry] = n

        # columns of A are the measured images of the reciprocal unit basis;
        # StrainMap's reciprocal-space branch (U_ref @ inv(U) = F^T) then
        # yields the real-space strain with the shared sign conventions
        sm = StrainMap(
            g1_array=A_map[..., :, 0],
            g2_array=A_map[..., :, 1],
            ds_shape=(R, C),
            real_space=False,
            g1_ref=np.array([1.0, 0.0]),
            g2_ref=np.array([0.0, 1.0]),
            mask=None if mask is None else np.asarray(mask, dtype=float),
            ds_sampling=ds_sampling,
            ds_units=ds_units,
        )
        sm.num_pairs = num_pairs
        return sm

    def in_plane_angle_deg(
        self, match: int = 0, mod_deg: float | str | None = "auto"
    ) -> torch.Tensor:
        """In-plane angle of the crystal a-axis at every position (degrees).

        The angle of the projected crystal [100] Cartesian axis, measured
        from the scan column axis toward the row axis.

        `mod_deg` wraps the angle, which is what makes the map continuous
        where the in-plane orientation is not uniquely indexable. The default
        "auto" wraps each position by 360 / n with n the apparent rotational
        symmetry of its own zero-layer pattern
        (`Crystal.projected_rotation_order`), so the map folds by exactly the
        ambiguity the data carry and no more. For a body-centered cubic
        crystal near <111> that is 60 degrees, where the crystal itself
        repeats only every 120, and folding removes the 60 degree jumps
        between two variants no zero-layer pattern can separate. A float
        wraps everywhere by that value, and None returns the full range.
        """
        from quantem.diffraction.rotations import quat_to_matrix

        assert self.quats is not None
        R = quat_to_matrix(self.quats[..., match, :])
        a_lab = R[..., :, 0]  # crystal x-axis in the lab frame
        ang = torch.rad2deg(torch.atan2(a_lab[..., 0], a_lab[..., 1]))
        if isinstance(mod_deg, str):
            if mod_deg != "auto":
                raise ValueError('mod_deg must be a number, None, or "auto"')
            # beam direction in crystal coordinates, deduplicated on a coarse
            # grid: the projected order is piecewise constant in the zone axis
            zone_c = R[..., 2, :]
            key = torch.round(zone_c.reshape(-1, 3) * 200) / 200
            uniq, inv = torch.unique(key, dim=0, return_inverse=True)
            order = self.crystal.projected_rotation_order(uniq.numpy())
            n = torch.as_tensor(np.asarray(order), dtype=torch.float64)[inv]
            return ang % (360.0 / n.reshape(ang.shape))
        if mod_deg is not None:
            ang = ang % mod_deg
        return ang

    def _default_mask(self, kwargs: dict) -> dict:
        """After a staged run on a subset of positions, plot only those."""
        if kwargs.get("mask") is None and self.computed is not None:
            if not bool(self.computed.all()):
                kwargs["mask"] = self.computed.numpy().astype(float)
        return kwargs

    @property
    def scan_scalebar(self) -> dict | None:
        """Real-space scale bar of the scan, carried from the dataset.

        `BraggVectors` stamps the scan sampling and units of the dataset onto
        the detected peaks, and the calibration keeps them, so every map can
        draw a scale bar without being told the step size. None when the
        dataset was never calibrated, in which case set `dataset.sampling`
        and `dataset.units` before detecting the disks.
        """
        md = self.metadata.get("peaks", {}) or {}
        return scan_scalebar(md)

    def plot_orientation(self, direction: str = "z", match: int = 0, **kwargs):
        """IPF-colored orientation map with the color wedge beside it.

        Parameters
        ----------
        direction : {"z", "r", "c"} | float | array-like, default="z"
            Lab direction whose crystal-frame coordinates are colored; see
            :func:`~quantem.diffraction.orientation_visualization.plot_orientation_map`.
        match : int, default=0
            Which match index to plot.
        **kwargs
            Passed to
            :func:`~quantem.diffraction.orientation_visualization.plot_orientation_map`.
            The scale bar defaults to the scan calibration, and after a
            staged run on a subset of positions the mask defaults to those
            positions.

        Returns
        -------
        tuple
            ``(fig, ax)``.
        """
        from quantem.diffraction.orientation_visualization import plot_orientation_map

        kwargs.setdefault("scalebar", self.scan_scalebar)
        return plot_orientation_map(
            self, direction=direction, match=match, **self._default_mask(kwargs)
        )

    def plot_pole_figure(self, pole=(0, 0, 1), match: int = 0, **kwargs):
        """Stereographic pole figure of a crystal direction family over the map.

        Parameters
        ----------
        pole : array-like, default=(0, 0, 1)
            Crystal direction in Miller indices, [uvw] or [uvtw].
        match : int, default=0
            Which match index to plot.
        **kwargs
            Passed to
            :func:`~quantem.diffraction.orientation_visualization.plot_pole_figure`.
            After a staged run on a subset of positions the mask defaults
            to those positions.

        Returns
        -------
        tuple
            ``(fig, ax)``.
        """
        from quantem.diffraction.orientation_visualization import plot_pole_figure

        return plot_pole_figure(self, pole=pole, match=match, **self._default_mask(kwargs))

    def plot_cluster_map(self, clusters: dict, **kwargs):
        """Map of the orientation clusters, one color per cluster.

        Parameters
        ----------
        clusters : dict
            Output of :meth:`cluster_orientations`.
        **kwargs
            Passed to
            :func:`~quantem.diffraction.orientation_visualization.plot_cluster_map`.
            The scale bar defaults to the scan calibration.

        Returns
        -------
        tuple
            ``(fig, ax)``.
        """
        from quantem.diffraction.orientation_visualization import plot_cluster_map

        kwargs.setdefault("scalebar", self.scan_scalebar)
        return plot_cluster_map(self, clusters, **kwargs)

    def plot_cluster_pole_figure(self, clusters: dict, pole=(0, 0, 1), **kwargs):
        """Pole figure of the cluster mean orientations, one color per cluster.

        Parameters
        ----------
        clusters : dict
            Output of :meth:`cluster_orientations`.
        pole : array-like, default=(0, 0, 1)
            Crystal direction in Miller indices, [uvw] or [uvtw].
        **kwargs
            Passed to
            :func:`~quantem.diffraction.orientation_visualization.plot_cluster_pole_figure`,
            e.g. `pole_label` and `overlay`.

        Returns
        -------
        tuple
            ``(fig, ax)``.
        """
        from quantem.diffraction.orientation_visualization import plot_cluster_pole_figure

        return plot_cluster_pole_figure(self, clusters, pole=pole, **kwargs)

    def misorientation_map(
        self, reference: torch.Tensor | None = None, match: int = 0
    ) -> torch.Tensor:
        """Misorientation angle of every position to a reference orientation.

        Parameters
        ----------
        reference : torch.Tensor | None
            (4,) reference quaternion; None is the identity.
        match : int, default=0
            Which match index to compare.

        Returns
        -------
        torch.Tensor
            (R, C) misorientation angles in degrees, reduced by the matching
            symmetry of the crystal.
        """
        assert self.quats is not None
        q = self.quats[..., match, :]
        if reference is None:
            reference = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
        return misorientation_angle_deg(reference, q, self.crystal.sym_quats_matching)
