"""Dynamical (Bloch wave) electron diffraction.

Simulation and refinement with the Bloch wave formulation of De Graef
(2003), ch. 5. The module has five families of functions:

- Spot patterns: dynamical_pattern() gives the Bloch intensities of one
  orientation at every thickness; refine_thickness() refits thickness and
  phase of a fitted PhaseMap with them.
- Convergent beam patterns: calculate_cbed() and calculate_cbed_library()
  (disk patterns), calculate_lacbed() (one reflection's rocking surface)
  and calculate_kossel() (wide-angle Kossel patterns).
- Kossel reference patterns: calculate_kossel_reference() computes the
  bright field over all beam directions once; kossel_from_reference() and
  kossel_polar_from_reference() look patterns up from it. The line model
  (kossel_lines(), render_kossel_lines(), kossel_line_segments()) describes
  the same patterns as one profile per systematic row, with
  kossel_reference_residual() adding the many-beam correction near zone
  axes.
- Bragg-vector refinement: refine_dynamical() refines orientation,
  thickness, in-plane deformation and phase per position against the
  measured peak intensities; dynamical_maps(), plot_dynamical_maps(),
  strain_crystal_frame() and plot_strain_crystal_frame() present the result.
- Image refinement: fit_disk_shape() and refine_dynamical_image() refine
  against the diffraction pattern pixels.

The structure matrix uses U_g = gamma_rel * F_g / pi with F_g the
kinematical structure factors (scattering amplitude per volume,
1/Angstrom^2), or the absorptive Weickenmeier-Kohl factors when the crystal
carries them (Crystal.calculate_dynamical_structure_factors), off-diagonals
U_(g-h) and diagonal 2 k0 s_g. One eigendecomposition per incident
direction gives the intensities at every thickness:

    psi(t) = C exp(2 pi i gamma t) C^-1 psi_0,   A C = 2 k0 gamma C
"""

from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.defaults import (
    MIN_NUMBER_PEAKS,
    MIN_SIM_INTENSITY_REL,
    PAIR_DISTANCE,
    POWER_INTENSITY,
    SG_MAX,
    resolve,
)
from quantem.diffraction.rotations import qrotate, sample_zone_axes


def relativistic_gamma(energy_ev: float) -> float:
    """Relativistic mass factor 1 + eV / (m0 c^2) at beam energy energy_ev (eV)."""
    return 1.0 + float(energy_ev) / 510998.95


def _coupling_matrix(
    crystal: Crystal, hkl_beams: torch.Tensor, gamma_rel: float
) -> tuple[torch.Tensor, float, bool]:
    """Off-diagonal Bloch coupling matrix U_(g-h) for a beam list.

    Prefers the absorptive Weickenmeier-Kohl factors when the crystal has
    them (calculate_dynamical_structure_factors); they carry the
    relativistic and 1/pi factors already. Falls back to the kinematical
    (Lobato) factors, purely elastic.

    Returns
    -------
    U : torch.Tensor
        (nb, nb) complex coupling matrix with zero diagonal.
    u0_imag : float
        Imaginary part of U_000 (mean absorption), 0 without absorption.
    absorptive : bool
        Whether absorptive factors were used.
    """
    absorptive = getattr(crystal, "U_dyn", None) is not None
    if absorptive:
        hkl_all = crystal.hkl_dyn
        U_all = crystal.U_dyn
    else:
        hkl_all = crystal.hkl
        U_all = crystal.struct_factors * (gamma_rel / np.pi)
    nb = hkl_beams.shape[0]
    diff = hkl_beams[:, None, :] - hkl_beams[None, :, :]  # (nb, nb, 3)
    # integer key that is injective over the union of the stored indices
    # and the queried differences, so a difference outside the stored box
    # can never alias a stored factor (a scalar key over the stored box
    # alone did: (2,0,0) unstored returned the factor of (-1,1,0))
    m = max(int(hkl_all.abs().max()), int(diff.abs().max()))
    span = 2 * m + 1
    key_mult = torch.tensor([1, span, span**2], dtype=torch.long)

    def keys(h):
        return ((h + m) * key_mult).sum(dim=-1)

    # vectorized lookup: binary search of the queried keys in the sorted
    # stored keys
    keys_all = keys(hkl_all)
    order = torch.argsort(keys_all)
    keys_sorted = keys_all[order]

    def lookup(k):
        pos = torch.searchsorted(keys_sorted, k).clamp(max=keys_sorted.shape[0] - 1)
        return torch.where(keys_sorted[pos] == k, order[pos], -1)

    U = torch.zeros((nb, nb), dtype=torch.complex128)
    idx = lookup(keys(diff).reshape(-1)).reshape(nb, nb)
    has = idx >= 0
    U[has] = U_all[idx[has]]
    U.fill_diagonal_(0)

    u0_imag = 0.0
    if absorptive:
        i0 = int(lookup(keys(torch.zeros((1, 3), dtype=torch.long)))[0])
        if i0 >= 0:
            u0_imag = float(U_all[i0].imag)
    return U, u0_imag, absorptive


# guards the per-crystal record of issued coverage warnings, which the
# threads of refine_dynamical() check concurrently
_coverage_lock = threading.Lock()


def _beam_universe(crystal: Crystal) -> tuple[torch.Tensor, torch.Tensor]:
    """Reciprocal lattice points that can carry dynamical intensity.

    With absorptive factors present, every index of the stored factor set
    (calculate_dynamical_structure_factors), including reflections whose
    structure factor is zero: those fill by multiple scattering through
    intermediate beams and would never enter the Bloch state if the beams
    were taken from the kinematical list, which drops them. Without
    absorptive factors, the kinematical list. Returns (hkl (N, 3) long,
    g crystal-frame (N, 3)) without the 000 beam.
    """
    if getattr(crystal, "U_dyn", None) is not None:
        hkl = crystal.hkl_dyn
        g = hkl.to(torch.float64) @ crystal.lat_recip
        keep = torch.linalg.norm(g, dim=1) > 1e-9
        # points of the conventional index box that are not reciprocal
        # lattice points of the primitive cell (centering absences: odd
        # h+k+l in bcc, mixed parity in fcc) are never excited and are
        # dropped; glide and screw absences such as Si 200 and 222 are
        # lattice points and stay
        keep &= _primitive_lattice_mask(crystal, g)
        return hkl[keep], g[keep]
    return crystal.hkl, crystal.g_vec


def _primitive_lattice_mask(crystal: Crystal, g: torch.Tensor) -> torch.Tensor:
    """True where the Cartesian reciprocal vectors g are points of the
    primitive reciprocal lattice of the crystal (integer coordinates
    g . a_i for the primitive real-space vectors a_i)."""
    lat_p = getattr(crystal, "_primitive_lattice", None)
    if lat_p is None:
        import spglib

        from quantem.diffraction.crystal import _spglib_raises

        cell = (
            crystal.lat_real.numpy(),
            crystal.positions_frac.numpy(),
            crystal.numbers.numpy(),
        )
        try:
            with _spglib_raises():
                prim = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True)
        except Exception:
            # no primitive cell found: keep every point of the stored box
            prim = None
        lat_p = np.asarray(crystal.lat_real.numpy() if prim is None else prim[0], dtype=float)
        crystal._primitive_lattice = lat_p
    m = g.to(torch.float64) @ torch.as_tensor(lat_p, dtype=torch.float64).T
    return (m - torch.round(m)).abs().amax(dim=1) < 1e-6


def _check_dynamical_factors(crystal: Crystal, energy_ev: float, g_max_beams: float) -> None:
    """Warn when the absorptive factors are missing, were computed at
    another energy, or stop short of 1.5 times the largest beam (the
    couplings g - h reach twice it, but the factors beyond 1.5 times are
    negligible).

    A warning is issued once per crystal and factor set: the record is
    kept on the crystal, keyed by the energy and extent of its dynamical
    factors and the energy of the calculation, so recomputing the factors
    or running at another energy is checked again.
    """
    key = (
        getattr(crystal, "dyn_energy_ev", None),
        getattr(crystal, "dyn_k_max", None),
        round(float(energy_ev)),
    )
    warned = getattr(crystal, "_bloch_coverage_warned", None)
    if warned is not None and key in warned:
        return
    msgs = []
    if getattr(crystal, "U_dyn", None) is None:
        k_kin = getattr(crystal, "k_max", None)
        msg = (
            "no absorptive structure factors (calculate_dynamical_structure_factors), "
            "so the Bloch calculation uses the elastic kinematical factors"
        )
        if k_kin is not None and 1.5 * g_max_beams > k_kin + 1e-9:
            msg += (
                f", which stop at {k_kin:.2f} 1/A, short of the "
                f"{1.5 * g_max_beams:.2f} 1/A the couplings of this beam list need"
            )
        msgs.append(msg)
    else:
        e_dyn = getattr(crystal, "dyn_energy_ev", None)
        k_dyn = getattr(crystal, "dyn_k_max", None)
        if e_dyn is not None and abs(e_dyn - energy_ev) > 1.0:
            msgs.append(
                f"dynamical structure factors were computed at {e_dyn:.0f} eV, the "
                f"calculation runs at {energy_ev:.0f} eV"
            )
        # couplings g - h reach twice the beam radius, but the factors fall
        # off fast: 1.5 times it keeps every coupling that matters (to 5%,
        # since the fitted in-plane strain stretches the beams a little past
        # k_max)
        if k_dyn is not None and 1.5 * g_max_beams > 1.05 * k_dyn:
            msgs.append(
                f"dynamical structure factors extend to {k_dyn:.2f} 1/A but the beam "
                f"list reaches {g_max_beams:.2f} 1/A, so couplings beyond "
                f"{k_dyn:.2f} 1/A are missing (treated as zero); recompute with "
                f"k_max >= {1.5 * g_max_beams:.2f}"
            )
    if not msgs:
        return
    with _coverage_lock:
        warned = getattr(crystal, "_bloch_coverage_warned", None)
        if warned is None:
            warned = set()
            crystal._bloch_coverage_warned = warned
        if key in warned:
            return
        warned.add(key)
    warnings.warn(f"{crystal.name}: " + "; ".join(msgs), stacklevel=3)


def select_dynamical_beams(
    crystal: Crystal,
    orientation: torch.Tensor,
    energy_ev: float,
    alpha_max_rad: float = 0.0,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
    deform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Beam list for a Bloch calculation over a range of incident directions.

    Selects the reflections with |s_g| < sg_max + alpha_max_rad |g| at the
    given orientation, which covers every incident direction within
    alpha_max_rad of the optic axis (a tilt t shifts s_g by at most
    |t| |g| / k0 to leading order). One list computed with the largest tilt
    of a refinement search keeps every stage of that search in the same
    truncated system.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated.
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    energy_ev : float
        Beam energy in eV.
    alpha_max_rad : float, default=0.0
        Largest incident tilt from the optic axis, in radians.
    sg_max : float, default=SG_MAX
        Excitation error cutoff in 1/Angstroms at zero tilt.
    k_max : float | None
        Largest |g| in 1/Angstroms; None keeps every candidate reflection.
    deform : torch.Tensor | None
        (3, 3) deformation applied to the lab-frame reciprocal vectors.

    Returns
    -------
    torch.Tensor
        (nb, 3) Miller indices, with the 000 beam first.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    hkl_u, g_u = _beam_universe(crystal)
    g_lab = qrotate(orientation, g_u)
    if deform is not None:
        g_lab = g_lab @ deform.to(torch.float64).T
    g_len = torch.linalg.norm(g_lab, dim=1)
    gz, g2 = g_lab[:, 2], (g_lab**2).sum(dim=1)
    s0 = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
    sel = torch.abs(s0) < sg_max + alpha_max_rad * g_len
    if k_max is not None:
        sel &= g_len <= k_max
    return torch.cat([torch.zeros((1, 3), dtype=torch.long), hkl_u[sel]])


def dynamical_pattern(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A: torch.Tensor | np.ndarray | float,
    energy_ev: float = 300e3,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Bloch-wave diffraction intensities for one orientation, all thicknesses.

    The beams are the reflections with |s_g| < sg_max (and |g| <= k_max);
    one eigendecomposition gives the intensities at every thickness.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated. Preferably also with
        calculate_dynamical_structure_factors (absorptive factors, at this
        energy, covering at least 1.5 times k_max so the couplings g - h
        that matter have a factor); a warning is issued otherwise.
    orientation : torch.Tensor
        Unit quaternion (4,) rotating crystal vectors into the lab frame.
    thicknesses_A : array-like or float
        Specimen thicknesses in Angstroms.
    energy_ev : float, default=300e3
        Beam energy in eV.
    sg_max : float, default=SG_MAX
        Excitation error cutoff (1/Angstroms) for including a beam.
    k_max : float | None
        Largest |g| (1/Angstroms) of an included beam; None keeps every
        reflection within sg_max.

    Returns
    -------
    dict
        'qx', 'qy' (N,) lab-frame positions (1/Angstroms), 'hkl' (N, 3),
        's_g' (N,) excitation errors, 'intensity' (T, N) diffracted
        intensities per thickness, 'intensity_000' (T,) the direct beam,
        'thicknesses' (T,) in Angstroms.
    """
    if crystal.g_vec is None:
        raise RuntimeError("Run crystal.calculate_structure_factors() first.")
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)

    t = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    # beam selection in the lab frame, from every lattice point that can
    # carry dynamical intensity
    hkl_u, g_u = _beam_universe(crystal)
    g_lab = qrotate(orientation, g_u)
    gz, g2 = g_lab[:, 2], (g_lab**2).sum(dim=1)
    s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
    sel = torch.abs(s_g) < sg_max
    if k_max is not None:
        sel &= torch.linalg.norm(g_lab, dim=1) <= k_max
    hkl_sel = hkl_u[sel]
    g_sel = g_lab[sel]
    s_sel = s_g[sel]
    _check_dynamical_factors(
        crystal, energy_ev, float(torch.linalg.norm(g_sel, dim=1).max()) if g_sel.shape[0] else 0.0
    )

    # beams list includes the (000) beam at index 0
    hkl_beams = torch.cat([torch.zeros((1, 3), dtype=torch.long), hkl_sel])
    s_beams = torch.cat([torch.zeros(1, dtype=torch.float64), s_sel])

    U, u0_imag, absorptive = _coupling_matrix(crystal, hkl_beams, gamma_rel)
    A = U.clone()
    diag = (2 * k0 * s_beams).to(torch.complex128)
    if absorptive:
        # mean absorption: imaginary part of U_000 damps every beam
        diag = diag + 1j * u0_imag
    A += torch.diag(diag)

    if absorptive:
        # non-Hermitian: general eigendecomposition, complex gamma damps
        evals, C = torch.linalg.eig(A)
        gam = evals / (2 * k0)
    else:
        evals, C = torch.linalg.eigh(A)
        gam = (evals.real / (2 * k0)).to(torch.complex128)
    psi0 = torch.linalg.inv(C)[:, 0]  # C^-1 @ e_0
    phase = torch.exp(2j * np.pi * gam[None, :] * t.to(torch.complex128)[:, None])
    psi = torch.einsum("ij,tj,j->ti", C, phase, psi0)  # (T, nb)
    intensity = torch.abs(psi[:, 1:]) ** 2  # drop the (000) beam

    return {
        "qx": g_sel[:, 0],
        "qy": g_sel[:, 1],
        "hkl": hkl_sel,
        "s_g": s_sel,
        "intensity": intensity,
        "intensity_000": torch.abs(psi[:, 0]) ** 2,
        "thicknesses": t,
    }


def refine_thickness(
    phase_map,
    thicknesses_A: np.ndarray | None = None,
    pair_distance: float | None = None,
    power_intensity: float | None = None,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
    min_number_peaks: int | None = None,
    progress_bar: bool = True,
):
    """Thickness and phase refinement with dynamical intensities.

    For every probe position, the winning candidates of a fitted PhaseMap are
    re-simulated with Bloch waves over a thickness grid at their matched
    orientations. The peak pairing is fixed (positions are kinematic); the
    intensity cost is evaluated for all thicknesses from a single
    eigendecomposition per candidate, and the best (thickness, candidate)
    combination updates the phase decision. refine_dynamical() also refines
    the orientation and the in-plane deformation.

    Parameters left as None inherit the phase fit's values (see
    PhaseMap.fit); the resolved values are recorded in
    phase_map.metadata['thickness'].

    Parameters
    ----------
    phase_map : PhaseMap
        A fitted PhaseMap (fit() has been run).
    thicknesses_A : np.ndarray | None
        Thickness grid in Angstroms; default 50 to 1000 in 25 A steps.
    pair_distance : float | None
        Largest distance (1/Angstroms) at which a simulated and a measured
        peak are paired.
    power_intensity : float | None
        Intensities are compared as I ** power_intensity.
    sg_max : float, default=SG_MAX
        Excitation error cutoff (1/Angstroms) of the Bloch beam list.
    k_max : float | None
        Largest |g| (1/Angstroms) of a beam; None keeps every reflection
        within sg_max.
    min_number_peaks : int | None
        Positions with fewer measured peaks, direct beam included, are
        skipped; None inherits the phase fit's minimum. At least 3.
    progress_bar : bool, default=True
        Show a progress bar over positions.

    Returns
    -------
    dict
        'thickness' (R, C) best-fit thickness of the winning candidate,
        'cost' (R, C, F) dynamical cost per candidate at its best
        thickness, 'phase_index' (R, C) updated phase assignment (-1 where
        no candidate was refined), 'thickness_per_candidate' (R, C, F).
        NaN where a candidate was not refined.
    """
    if thicknesses_A is None:
        thicknesses_A = np.arange(50.0, 1000.0 + 1e-6, 25.0)
    t_grid = torch.as_tensor(thicknesses_A, dtype=torch.float64)

    oms = phase_map.orientation_maps
    fit_md = phase_map.metadata.get("fit") if hasattr(phase_map, "metadata") else None
    pair_distance = resolve(pair_distance, "pair_distance", fit_md, default=PAIR_DISTANCE)
    power_intensity = resolve(power_intensity, "power_intensity", fit_md, default=POWER_INTENSITY)
    min_number_peaks = int(
        resolve(min_number_peaks, "min_number_peaks", fit_md, default=MIN_NUMBER_PEAKS)
    )
    if min_number_peaks < 3:
        raise ValueError(
            f"min_number_peaks={min_number_peaks}: a dynamical fit needs at least the "
            "direct beam and two non-collinear reflections"
        )
    if hasattr(phase_map, "metadata"):
        phase_map.metadata["thickness"] = dict(
            thicknesses_A=np.asarray(thicknesses_A, dtype=float).tolist(),
            pair_distance=float(pair_distance),
            power_intensity=float(power_intensity),
            sg_max=float(sg_max),
            k_max=k_max,
            min_number_peaks=int(min_number_peaks),
        )
    cands = phase_map.candidates
    peaks = oms[0].peaks
    R, C = peaks.shape[0], peaks.shape[1]
    F = len(cands)
    delta = pair_distance

    fields = peaks.fields
    ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

    cost_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)
    thick_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)

    iterator = list(np.ndindex(R, C))
    if progress_bar:
        iterator = tqdm(iterator, desc="dynamical refinement")
    for rx, ry in iterator:
        data = peaks[rx, ry].numpy().astype(np.float64)
        if data.shape[0] < min_number_peaks:
            continue
        qxy = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
        im = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
        im = im**power_intensity
        int_total = float(im.sum())

        for f, (i_om, m) in enumerate(cands):
            om = oms[i_om]
            if om.corr[rx, ry, m] <= 0:
                continue
            # only refine candidates that won weight in the first pass
            if (
                phase_map.phase_weights is not None
                and float(phase_map.phase_weights[rx, ry, f]) <= 0
            ):
                continue
            sim = dynamical_pattern(
                om.crystal,
                om.quats[rx, ry, m],
                t_grid,
                energy_ev=om.energy_ev,
                sg_max=sg_max,
                k_max=k_max,
            )
            sq = torch.stack((sim["qx"], sim["qy"]), dim=1)
            if sq.shape[0] == 0:
                continue
            si = sim["intensity"] ** power_intensity  # (T, N)
            d = torch.cdist(sq, qxy)
            d_min, j_min = d.min(dim=1)
            pair = d_min < delta
            frac = (d_min[pair] / delta).clamp(0, 1)

            a = si[:, pair] * (1 - frac)[None, :]  # (T, P)
            b = im[j_min[pair]][None, :]
            w = (a * b).sum(dim=1) / (a * a).sum(dim=1).clamp_min(1e-12)  # (T,)
            w = w.clamp_min(0)

            c_paired = (
                (b - w[:, None] * a).abs() * (1 - frac)[None, :] + w[:, None] * a * frac[None, :]
            ).sum(dim=1)
            c_unpaired_sim = 0.5 * w * si[:, ~pair].sum(dim=1)
            matched = torch.zeros(im.shape[0], dtype=torch.bool)
            matched[j_min[pair]] = True
            c_unpaired_exp = 0.5 * float(im[~matched].sum())
            cost_t = (c_paired + c_unpaired_sim + c_unpaired_exp) / (int_total + 1e-12)

            t_best = int(cost_t.argmin())
            cost_out[rx, ry, f] = cost_t[t_best]
            thick_out[rx, ry, f] = t_grid[t_best]

    # updated per-crystal phase decision from the dynamical costs
    n_maps = len(oms)
    cost_phase = torch.full((R, C, n_maps), torch.inf, dtype=torch.float64)
    for f, (i_om, _) in enumerate(cands):
        c = torch.nan_to_num(cost_out[..., f], nan=torch.inf)
        cost_phase[..., i_om] = torch.minimum(cost_phase[..., i_om], c)
    done = torch.isfinite(cost_out).any(dim=-1)
    phase_index = torch.where(done, cost_phase.argmin(dim=-1), -1)

    f_best = torch.nan_to_num(cost_out, nan=torch.inf).argmin(dim=-1)
    thickness = torch.gather(thick_out, 2, f_best[..., None]).squeeze(-1)

    return {
        "thickness": thickness,
        "cost": cost_out,
        "phase_index": phase_index,
        "thickness_per_candidate": thick_out,
    }


def _cbed_amplitudes(
    crystal: Crystal,
    orientation: torch.Tensor,
    tilts: torch.Tensor,
    thicknesses_A: torch.Tensor,
    energy_ev: float,
    sg_max: float,
    k_max: float | None,
    tilt_batch: int = 64,
    progress_bar: bool = False,
    fast_absorption: bool = False,
    deform: torch.Tensor | None = None,
    beams: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bloch intensities of every beam at every incident tilt.

    The coupling matrix is built once; only the diagonal (excitation
    errors) changes with tilt, and the eigendecompositions are batched
    over tilt chunks.

    Parameters
    ----------
    tilts : torch.Tensor
        (M, 2) in-plane incident wavevector components (1/Angstroms).
    deform : torch.Tensor | None
        (3, 3) deformation applied to the lab-frame reciprocal lattice
        (g' = deform @ g), for a strained cell; the structure factors are
        those of the ideal cell.
    beams : torch.Tensor | None
        Explicit beam list (nb, 3) hkl with 000 first, e.g. from
        select_dynamical_beams(); when None the list is selected here from
        the tilts given.

    Returns
    -------
    intensity : torch.Tensor
        (M, T, nb) beam intensities per tilt and thickness; beam 0 is the
        direct (000) beam.
    g_xy : torch.Tensor
        (nb, 2) in-plane reciprocal vectors of the beams (000 first).
    hkl_beams : torch.Tensor
        (nb, 3) Miller indices of the beams.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    # beam selection: near the Ewald sphere for ANY tilt in the aperture
    if beams is None:
        alpha_max = float(torch.linalg.norm(tilts, dim=1).max()) / k0
        hkl_beams = select_dynamical_beams(
            crystal, orientation, energy_ev, alpha_max, sg_max, k_max, deform
        )
    else:
        hkl_beams = beams
    g_beams = qrotate(orientation, hkl_beams[1:].to(torch.float64) @ crystal.lat_recip)
    if deform is not None:
        g_beams = g_beams @ deform.to(torch.float64).T
    g_beams = torch.cat([torch.zeros((1, 3), dtype=torch.float64), g_beams])
    nb = hkl_beams.shape[0]
    _check_dynamical_factors(
        crystal, energy_ev, float(torch.linalg.norm(g_beams, dim=1).max()) if nb > 1 else 0.0
    )

    U, u0_imag, absorptive = _coupling_matrix(crystal, hkl_beams, gamma_rel)

    gx, gy, gzb = g_beams[:, 0], g_beams[:, 1], g_beams[:, 2]
    g2b = (g_beams**2).sum(dim=1)
    M = tilts.shape[0]
    out = torch.zeros((M, t_thick.shape[0], nb), dtype=torch.float64)
    chunks = range(0, M, tilt_batch)
    if progress_bar:
        chunks = tqdm(chunks, desc="Bloch tilts")
    for m0 in chunks:
        m1 = min(m0 + tilt_batch, M)
        tt = tilts[m0:m1]  # (B, 2)
        kz = torch.sqrt(k0**2 - (tt**2).sum(dim=1))  # (B,)
        # s_g for incident k = (tx, ty, -kz), surface normal along z
        num = (
            2 * kz[:, None] * gzb[None, :]
            - 2 * (tt[:, 0, None] * gx[None, :] + tt[:, 1, None] * gy[None, :])
            - g2b[None, :]
        )
        den = 2 * (kz[:, None] - gzb[None, :])
        s_t = num / den  # (B, nb)

        out[m0:m1] = _bloch_solve(
            U, u0_imag, absorptive, s_t, k0, t_thick, fast_absorption=fast_absorption
        )
    return out, g_beams[:, :2], hkl_beams


def _bloch_solve(
    U: torch.Tensor,
    u0_imag: float,
    absorptive: bool,
    s_t: torch.Tensor,
    k0: float,
    t_thick: torch.Tensor,
    fast_absorption: bool = False,
) -> torch.Tensor:
    """Batched Bloch solve: intensities (B, T, nb) for excitation errors s_t
    (B, nb) with a shared coupling matrix U (nb, nb).

    With fast_absorption=True the Hermitian part is diagonalized (eigh, much
    faster and better batched than the general complex eig) and the weak
    absorption enters first order: gamma_imag = diag(C^dagger U'' C)/(2 k0).
    Standard for reference (master) pattern computations; the absorptive
    parts of U are a few percent of the elastic parts, so the first-order
    error is small.
    """
    nb = U.shape[0]
    if absorptive and fast_absorption:
        H = 0.5 * (U + U.conj().T)
        W = (U - H) / 1j  # Hermitian absorptive part (off-diagonal)
        A = H[None].expand(s_t.shape[0], nb, nb).clone()
        A += torch.diag_embed((2 * k0 * s_t).to(torch.complex128))
        evals, C = torch.linalg.eigh(A)
        gam_r = evals / (2 * k0)  # (B, nb) real
        CW = torch.einsum("bji,jk,bki->bi", C.conj(), W, C).real
        gam_i = (CW + u0_imag) / (2 * k0)  # (B, nb)
        gam = gam_r.to(torch.complex128) + 1j * gam_i
        psi0 = C.conj().transpose(1, 2)[:, :, 0]  # unitary: C^-1 = C^dagger
    else:
        diag = (2 * k0 * s_t).to(torch.complex128)
        if absorptive:
            diag = diag + 1j * u0_imag
        A = U[None].expand(s_t.shape[0], nb, nb).clone()
        A += torch.diag_embed(diag)
        if absorptive:
            evals, C = torch.linalg.eig(A)
            gam = evals / (2 * k0)
        else:
            evals, C = torch.linalg.eigh(A)
            gam = (evals.real / (2 * k0)).to(torch.complex128)
        psi0 = torch.linalg.inv(C)[:, :, 0]  # (B, nb)
    phase = torch.exp(
        2j * np.pi * gam[:, None, :] * t_thick.to(torch.complex128)[None, :, None]
    )  # (B, T, nb)
    psi = torch.einsum("bij,btj,bj->bti", C, phase, psi0)
    return torch.abs(psi) ** 2


def tilt_grid(semiconv_mrad: float, energy_ev: float, n_rings: int = 8):
    """Concentric-ring sampling of the illumination aperture.

    Parameters
    ----------
    semiconv_mrad : float
        Convergence semiangle in mrad.
    energy_ev : float
        Beam energy in eV.
    n_rings : int, default=8
        Rings outside the center point; ring r has ceil(2 pi r) points.

    Returns
    -------
    torch.Tensor
        (M, 2) in-plane incident wavevectors (1/Angstroms) covering the
        disk with approximately uniform density, the center first.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    pts = [(0.0, 0.0)]
    for r in range(1, n_rings + 1):
        rad = alpha_k * r / n_rings
        n_az = int(np.ceil(2 * np.pi * r))
        th = 2 * np.pi * (np.arange(n_az) + 0.5 * (r % 2)) / n_az
        pts += [(rad * np.cos(a), rad * np.sin(a)) for a in th]
    return torch.tensor(pts, dtype=torch.float64)


def calculate_cbed(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 3.0,
    n_rings: int = 8,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
    pixel_size: float | None = None,
    q_max_plot: float | None = None,
    tilt_batch: int = 64,
) -> dict:
    """Simulate a CBED pattern with Bloch waves.

    Every incident direction inside the aperture is an independent plane
    wave (incoherent illumination): its Bloch intensities are placed at
    g + t in the detector plane, filling each diffraction disk with the
    rocking-curve intensity variation. Uses the absorptive
    Weickenmeier-Kohl structure factors when the crystal carries them.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated, and preferably
        calculate_dynamical_structure_factors at this energy (absorption).
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    thicknesses_A : float | array-like
        One or more specimen thicknesses in Angstroms.
    energy_ev : float, default=300e3
        Beam energy in eV.
    semiconv_mrad : float, default=3.0
        Convergence semiangle in mrad. Disks overlap when it exceeds half
        the smallest g spacing times the wavelength.
    n_rings : int, default=8
        Radial sampling rings across the aperture (~200 tilts at 8).
    sg_max : float, default=SG_MAX
        Excitation error cutoff (1/Angstroms) for beam selection, widened
        automatically by the aperture tilt range.
    k_max : float | None
        Largest |g| (1/Angstroms) of an included beam.
    pixel_size : float | None
        Detector sampling (1/Angstroms per pixel); default disk radius / 12.
    q_max_plot : float | None
        Half-width of the detector (1/Angstroms); default covers all beams
        plus a disk. Disk samples beyond it are dropped.
    tilt_batch : int, default=64
        Incident tilts per batched eigendecomposition (memory versus
        speed).

    Returns
    -------
    dict
        'pattern' ((T, H, H), squeezed to (H, H) for one thickness; rows
        are qx, columns qy, the direct beam at the center pixel; the
        intensity is averaged over the incident tilts, so it sums to the
        transmitted fraction when every disk is on the detector),
        'sampling' (1/Angstroms per pixel), 'disk_radius' (1/Angstroms),
        'thicknesses', 'hkl' (nb, 3) and 'g_xy' (nb, 2) of the beams, 000
        first.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    tilts = tilt_grid(semiconv_mrad, energy_ev, n_rings=n_rings)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    inten, g_xy, hkl_beams = _cbed_amplitudes(
        crystal, orientation, tilts, t_thick, energy_ev, sg_max, k_max, tilt_batch
    )

    if pixel_size is None:
        pixel_size = alpha_k / 12
    if q_max_plot is None:
        q_max_plot = float(torch.linalg.norm(g_xy, dim=1).max()) + 2 * alpha_k
    half = int(np.ceil(q_max_plot / pixel_size))
    H = 2 * half + 1

    # deposit every (beam, tilt) sample with bilinear weights
    qx = (g_xy[:, 0][None, :] + tilts[:, 0][:, None]).numpy()  # (M, nb)
    qy = (g_xy[:, 1][None, :] + tilts[:, 1][:, None]).numpy()
    fx = qx / pixel_size + half
    fy = qy / pixel_size + half
    ix0 = np.floor(fx).astype(int)
    iy0 = np.floor(fy).astype(int)
    wx = fx - ix0
    wy = fy - iy0

    T = t_thick.shape[0]
    pattern = np.zeros((T, H, H))
    inten_np = inten.numpy()  # (M, T, nb)
    for dx in (0, 1):
        for dy in (0, 1):
            w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
            jx = ix0 + dx
            jy = iy0 + dy
            # samples beyond the detector are dropped, not piled on its edge
            ok = (jx >= 0) & (jx < H) & (jy >= 0) & (jy < H)
            for ti in range(T):
                np.add.at(pattern[ti], (jx[ok], jy[ok]), (w * inten_np[:, ti, :])[ok])
    pattern /= tilts.shape[0]

    return {
        "pattern": pattern[0] if T == 1 else pattern,
        "sampling": float(pixel_size),
        "disk_radius": float(alpha_k),
        "thicknesses": t_thick.numpy(),
        "hkl": hkl_beams.numpy(),
        "g_xy": g_xy.numpy(),
    }


def calculate_lacbed(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A,
    hkl,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 10.0,
    n_pixels: int = 48,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
    tilt_batch: int = 64,
) -> dict:
    """Large-angle CBED: one reflection's rocking surface over the aperture.

    The intensity of the chosen reflection is mapped over the incident-tilt
    disk on a square grid (parallax / LACBED view of a single disk, without
    the geometric overlap of neighboring disks).

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated, and preferably
        calculate_dynamical_structure_factors at this energy (absorption).
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    thicknesses_A : float | array-like
        One or more specimen thicknesses in Angstroms.
    hkl : sequence of int
        The reflection to map; (0, 0, 0) gives the bright field disk.
    energy_ev : float, default=300e3
        Beam energy in eV.
    semiconv_mrad : float, default=10.0
        Convergence semiangle in mrad.
    n_pixels : int, default=48
        Pixels across the disk (the incident-tilt sampling).
    sg_max : float, default=SG_MAX
        Excitation error cutoff (1/Angstroms), widened automatically by
        the aperture tilt range.
    k_max : float | None
        Largest |g| (1/Angstroms) of an included beam.
    tilt_batch : int, default=64
        Incident tilts per batched eigendecomposition.

    Returns
    -------
    dict
        'disk' ((T, n, n), squeezed for one thickness; rows are the y
        tilt, columns the x tilt; NaN outside the aperture), 'tilt_max'
        (aperture radius, 1/Angstroms), 'thicknesses'.

    Raises
    ------
    ValueError
        If the reflection is not among the excited beams.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    ax = torch.linspace(-alpha_k, alpha_k, n_pixels, dtype=torch.float64)
    ty, tx = torch.meshgrid(ax, ax, indexing="ij")
    inside = (tx**2 + ty**2) <= alpha_k**2
    tilts = torch.stack([tx[inside], ty[inside]], dim=1)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    inten, _, hkl_beams = _cbed_amplitudes(
        crystal, orientation, tilts, t_thick, energy_ev, sg_max, k_max, tilt_batch
    )
    match = (hkl_beams == torch.as_tensor(hkl, dtype=torch.long)[None, :]).all(dim=1)
    if not bool(match.any()):
        raise ValueError(f"reflection {tuple(hkl)} is not among the excited beams")
    b = int(match.nonzero()[0])

    T = t_thick.shape[0]
    disk = np.full((T, n_pixels, n_pixels), np.nan)
    m = inside.numpy()
    for ti in range(T):
        plane = np.full((n_pixels, n_pixels), np.nan)
        plane[m] = inten[:, ti, b].numpy()
        disk[ti] = plane
    return {
        "disk": disk[0] if T == 1 else disk,
        "tilt_max": float(alpha_k),
        "thicknesses": t_thick.numpy(),
    }


def calculate_cbed_library(
    crystal: Crystal,
    orientations: torch.Tensor,
    thickness_A: float,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 3.0,
    k_max: float | None = None,
    q_max_plot: float | None = None,
    pixel_size: float | None = None,
    progress_bar: bool = True,
    **kwargs,
) -> dict:
    """A stack of simulated CBED patterns on one common detector grid.

    The starting point for CBED orientation matching: all patterns share
    the same sampling and extent, ready for polar transformation and
    correlation. One entry per orientation, all at one thickness.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated.
    orientations : torch.Tensor
        (N, 4) unit quaternions, crystal to lab.
    thickness_A : float
        Specimen thickness in Angstroms.
    energy_ev : float, default=300e3
        Beam energy in eV.
    semiconv_mrad : float, default=3.0
        Convergence semiangle in mrad.
    k_max : float | None
        Largest |g| (1/Angstroms) of an included beam.
    q_max_plot : float | None
        Half-width of the detector (1/Angstroms); default k_max (or half
        the crystal's structure factor range) plus two disk radii.
    pixel_size : float | None
        Detector sampling (1/Angstroms per pixel); default disk radius / 12.
    progress_bar : bool, default=True
        Show a progress bar over orientations.
    **kwargs
        Passed to calculate_cbed() (n_rings, sg_max, tilt_batch).

    Returns
    -------
    dict
        'patterns' (N, H, H), 'quats' (N, 4), 'sampling' (1/Angstroms per
        pixel), 'disk_radius' (1/Angstroms), 'thickness_A'.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    if pixel_size is None:
        pixel_size = alpha_k / 12
    if q_max_plot is None:
        base = k_max if k_max is not None else float(crystal.k_max) / 2
        q_max_plot = base + 2 * alpha_k

    quats = torch.atleast_2d(torch.as_tensor(orientations, dtype=torch.float64))
    pats = []
    it = range(quats.shape[0])
    if progress_bar:
        it = tqdm(it, desc="CBED library")
    for i in it:
        res = calculate_cbed(
            crystal,
            quats[i],
            thickness_A,
            energy_ev=energy_ev,
            semiconv_mrad=semiconv_mrad,
            k_max=k_max,
            pixel_size=pixel_size,
            q_max_plot=q_max_plot,
            **kwargs,
        )
        pats.append(res["pattern"])
    return {
        "patterns": np.stack(pats),
        "quats": quats.numpy(),
        "sampling": float(pixel_size),
        "disk_radius": float(alpha_k),
        "thickness_A": float(thickness_A),
    }


def calculate_kossel(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A,
    energy_ev: float = 300e3,
    semiconv_mrad: float = 40.0,
    n_pixels: int = 192,
    sg_max: float = 0.05,
    k_max: float | None = None,
    tilt_batch: int = 64,
    fast_absorption: bool = False,
    progress_bar: bool = True,
) -> dict:
    """Wide-angle convergent beam (Kossel) pattern with Bloch waves.

    At convergence angles far beyond the Bragg angles the diffraction disks
    overlap completely and the pattern becomes a continuous map of
    deficiency and excess lines (the Kossel regime of CBED; the bright
    field disk alone is the LACBED view). One Bloch computation over the
    incident-tilt grid yields both:

    - 'bright_field': the (000) beam intensity at each incident tilt, the
      deficiency (dark) line system, every line at a Bragg condition.
    - 'pattern': the full detector intensity, the incoherent sum of every
      diffracted cone shifted by its g: deficiency lines from the direct
      beam plus the excess (bright) lines of the diffracted beams.

    Line positions are exact; line profiles carry the many-beam dynamical
    structure, with the deficiency/excess asymmetry from the absorptive
    structure factors when the crystal has them.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated, and preferably
        calculate_dynamical_structure_factors at this energy (absorption).
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    thicknesses_A : float | array-like
        One or more thicknesses in Angstroms.
    energy_ev : float, default=300e3
        Beam energy in eV.
    semiconv_mrad : float, default=40.0
        Convergence semiangle in mrad; the pattern covers this angular
        radius.
    n_pixels : int, default=192
        Detector pixels across the pattern (also the tilt sampling; the
        1-2 mrad dynamical line widths need ~0.5 mrad per pixel).
    sg_max : float, default=0.05
        Excitation error cutoff (1/Angstroms). Smaller than the SG_MAX of
        the spot pattern functions: the beam list is widened by the
        aperture (alpha |g|, already 0.04 1/A for |g| = 1 at 40 mrad), so
        the base cutoff can be tighter without losing lines, and the
        eigensolves over tens of thousands of tilts stay affordable.
    k_max : float | None
        Largest |g| (1/Angstroms) of an included reflection.
    tilt_batch : int, default=64
        Incident tilts per batched eigendecomposition.
    fast_absorption : bool, default=False
        First-order absorption (Hermitian eigensolver, faster); see
        _bloch_solve.
    progress_bar : bool, default=True
        Show a progress bar over tilt batches.

    Returns
    -------
    dict
        'bright_field' and 'pattern' ((T, n, n), squeezed for one
        thickness; rows are theta_y, columns theta_x, as in
        render_kossel_lines; NaN / 0 outside the aperture), 'sampling'
        (1/Angstroms per pixel), 'mrad_per_pixel', 'thicknesses', 'hkl'.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    alpha_k = semiconv_mrad * 1e-3 / lam
    ax = torch.linspace(-alpha_k, alpha_k, n_pixels, dtype=torch.float64)
    px = float(ax[1] - ax[0])
    ty, tx = torch.meshgrid(ax, ax, indexing="ij")
    inside = (tx**2 + ty**2) <= alpha_k**2
    tilts = torch.stack([tx[inside], ty[inside]], dim=1)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))
    T = t_thick.shape[0]

    inten, g_xy, hkl_beams = _cbed_amplitudes(
        crystal,
        orientation,
        tilts,
        t_thick,
        energy_ev,
        sg_max,
        k_max,
        tilt_batch,
        progress_bar=progress_bar,
        fast_absorption=fast_absorption,
    )
    inten_np = inten.numpy()  # (M, T, nb)
    m = inside.numpy()

    # bright field: beam 0 on the tilt grid directly
    bright = np.full((T, n_pixels, n_pixels), np.nan)
    for ti in range(T):
        plane = np.full((n_pixels, n_pixels), np.nan)
        plane[m] = inten_np[:, ti, 0]
        bright[ti] = plane

    # full pattern: every diffracted cone shifted by its g, bilinear deposit
    pattern = np.zeros((T, n_pixels, n_pixels))
    tx_in = tilts[:, 0].numpy()
    ty_in = tilts[:, 1].numpy()
    g_np = g_xy.numpy()
    for b in range(g_np.shape[0]):
        fx = (tx_in + g_np[b, 0] + alpha_k) / px
        fy = (ty_in + g_np[b, 1] + alpha_k) / px
        ix0 = np.floor(fx).astype(int)
        iy0 = np.floor(fy).astype(int)
        wx = fx - ix0
        wy = fy - iy0
        for dx in (0, 1):
            for dy in (0, 1):
                jx = ix0 + dx
                jy = iy0 + dy
                ok = (jx >= 0) & (jx < n_pixels) & (jy >= 0) & (jy < n_pixels)
                w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
                # (row, col) = (theta_y, theta_x), as the bright field
                for ti in range(T):
                    np.add.at(pattern[ti], (jy[ok], jx[ok]), (w * inten_np[:, ti, b])[ok])
    pattern[:, ~m] = 0.0

    return {
        "bright_field": bright[0] if T == 1 else bright,
        "pattern": pattern[0] if T == 1 else pattern,
        "sampling": px,
        "mrad_per_pixel": px * lam * 1e3,
        "thicknesses": t_thick.numpy(),
        "hkl": hkl_beams.numpy(),
    }


def _lambert_raster(
    crystal: Crystal, dirs: torch.Tensor, values: torch.Tensor, step: float
) -> np.ndarray:
    """Expand wedge samples by the crystal symmetry (plus inversion) and
    splat them bilinearly onto a Lambert equal-area grid of the upper
    hemisphere. values is (N, T); returns (T, n, n) with NaN where unhit
    (raster holes inside the disk are filled from their neighbors)."""
    from quantem.diffraction.rotations import quat_to_matrix

    T = values.shape[1]
    Rs = quat_to_matrix(crystal.sym_quats_matching)
    d_all = torch.einsum("sij,nj->sni", Rs, dirs).reshape(-1, 3)
    I_all = values[None, :, :].expand(Rs.shape[0], -1, -1).reshape(-1, T)
    d_all = torch.cat([d_all, -d_all])
    I_all = torch.cat([I_all, I_all])
    up = d_all[:, 2] >= 0
    d_all, I_all = d_all[up], I_all[up]

    # grid always spans the full hemisphere: symmetry expansion moves wedge
    # samples to any polar angle, and clipping them onto a smaller rim
    # corrupts the equatorial region
    rho_max = float(np.sqrt(2.0))
    half = int(np.ceil(rho_max / step))
    n = 2 * half + 1
    rho = torch.sqrt((2 * (1 - d_all[:, 2])).clamp_min(0))
    dxy = torch.linalg.norm(d_all[:, :2], dim=1).clamp_min(1e-12)
    px_x = (d_all[:, 0] / dxy * rho / step + half).numpy()
    px_y = (d_all[:, 1] / dxy * rho / step + half).numpy()

    acc = np.zeros((T, n, n))
    wgt = np.zeros((n, n))
    ix0 = np.floor(px_x).astype(int)
    iy0 = np.floor(px_y).astype(int)
    wx = px_x - ix0
    wy = px_y - iy0
    I_np = I_all.numpy()
    for dx in (0, 1):
        for dy in (0, 1):
            jx = np.clip(ix0 + dx, 0, n - 1)
            jy = np.clip(iy0 + dy, 0, n - 1)
            w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
            np.add.at(wgt, (jx, jy), w)
            for ti in range(T):
                np.add.at(acc[ti], (jx, jy), w * I_np[:, ti])
    lambert = np.where(wgt[None] > 1e-6, acc / np.maximum(wgt[None], 1e-6), np.nan)

    # fill raster holes (unhit pixels between splatted samples) from their
    # neighbors so bilinear lookups never touch NaN inside the disk
    yy, xx = np.mgrid[0:n, 0:n]
    in_disk = ((xx - half) ** 2 + (yy - half) ** 2) <= (rho_max / step) ** 2
    for ti in range(T):
        L = lambert[ti]
        for _ in range(4):
            holes = np.isnan(L) & in_disk
            if not holes.any():
                break
            Lp = np.pad(L, 1, constant_values=np.nan)
            stack = np.stack(
                [
                    Lp[1 + dy : n + 1 + dy, 1 + dx : n + 1 + dx]
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                ]
            )
            with np.errstate(all="ignore"):
                fill = np.nanmean(stack, axis=0)
            L[holes] = fill[holes]
        lambert[ti] = L
    return lambert


def calculate_kossel_reference(
    crystal: Crystal,
    thicknesses_A,
    energy_ev: float = 300e3,
    angle_step_mrad: float = 1.0,
    sg_max: float = 0.05,
    k_max: float | None = None,
    theta_max_deg: float = 90.0,
    chunk: int = 256,
    fast_absorption: bool = True,
    progress_bar: bool = True,
) -> dict:
    """Kossel reference pattern: the dynamical bright field over all directions.

    The bright field intensity depends only on the incident beam direction
    in the CRYSTAL frame (each incident plane wave is independent), so one
    Bloch computation over the symmetry-reduced direction wedge gives the
    pattern for every specimen orientation at once (called a master pattern
    in parts of the EBSD literature). Patterns for arbitrary orientations,
    convergence angles, and all precomputed thicknesses are then
    interpolation lookups via kossel_from_reference(), milliseconds instead
    of a fresh dynamical calculation.

    The wedge samples are expanded by the crystal's proper rotations plus
    inversion and rasterized onto a Lambert azimuthal equal-area grid of
    the upper hemisphere. (The inversion expansion assumes Friedel symmetry
    of the bright field; for non-centrosymmetric crystals with absorption
    this neglects a small polarity contrast.)

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated, and preferably
        calculate_dynamical_structure_factors at this energy (absorption).
    thicknesses_A : float | array-like
        Thickness grid in Angstroms; all thicknesses share the
        eigendecompositions, so a thickness axis is nearly free.
    energy_ev : float, default=300e3
        Beam energy in eV.
    angle_step_mrad : float, default=1.0
        Angular sampling of the wedge, and the pixel size of the Lambert
        grid (in Lambert radius units of 1e-3). The dynamical line widths
        are 1-2 mrad; 0.5 for production references, 1-2 for quick looks.
    sg_max : float, default=0.05
        Excitation error cutoff (1/Angstroms) at the center of each chunk
        of directions, widened by the chunk's angular radius; as in
        calculate_kossel, tighter than SG_MAX because of that widening.
    k_max : float | None
        Largest |g| (1/Angstroms) of an included beam; None keeps every
        reflection of the factor set, which is slow for large sets. The
        cost grows steeply with it.
    theta_max_deg : float, default=90.0
        Polar cutoff of the wedge samples. Keep at 90 unless the wedge's
        far corners are never observed: cutting the wedge leaves coverage
        holes at all their symmetry equivalents.
    chunk : int, default=256
        Directions per batch; each batch shares one beam list and one
        coupling matrix.
    fast_absorption : bool, default=True
        First-order absorption (Hermitian eigensolver, several times
        faster); see _bloch_solve.
    progress_bar : bool, default=True
        Show a progress bar over chunks.

    Returns
    -------
    dict
        'lambert' (T, n, n) bright field on the equal-area grid of the
        upper hemisphere (NaN where unsampled), 'rho_max' (Lambert radius
        of the equator, sqrt(2)), 'step' (Lambert grid spacing),
        'thicknesses', 'energy_ev', 'k_max' (as given, possibly None), and
        the raw wedge samples 'directions' (N, 3) and 'intensity' (N, T).
    """
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))
    T = t_thick.shape[0]

    msg = crystal.matching_symmetry_warning()
    if msg is not None:
        warnings.warn(msg, stacklevel=2)
    wedge = crystal.zone_axis_wedge()
    step_deg = np.rad2deg(angle_step_mrad * 1e-3)
    if wedge is None:
        from quantem.diffraction.orientation import fibonacci_hemisphere

        n_dirs = int(np.ceil(2 * np.pi / np.deg2rad(step_deg) ** 2))
        dirs = fibonacci_hemisphere(n_dirs)
    else:
        dirs, _ = sample_zone_axes(wedge, step_deg)
    keep = dirs[:, 2] >= np.cos(np.deg2rad(theta_max_deg))
    dirs = dirs[keep]
    N = dirs.shape[0]

    hkl_u, g = _beam_universe(crystal)  # crystal frame, orientation is identity
    g2 = (g**2).sum(dim=1)
    g_len = torch.linalg.norm(g, dim=1)

    out = torch.zeros((N, T), dtype=torch.float64)
    chunks = range(0, N, chunk)
    if progress_bar:
        chunks = tqdm(chunks, desc="Kossel reference")
    for c0 in chunks:
        c1 = min(c0 + chunk, N)
        d = dirs[c0:c1]  # (B, 3) beam directions in the crystal frame
        d_c = d.mean(dim=0)
        d_c = d_c / torch.linalg.norm(d_c)
        radius = float(torch.arccos((d @ d_c).clamp(-1, 1)).max())

        # normal-tracking geometry: each sample is computed with the foil
        # normal along the sampled direction (the crystal is conceptually
        # re-tilted per sample). The slab problem is then a function of the
        # crystal-frame beam direction ALONE, which is what makes the
        # symmetry expansion below exact; a pattern lookup only ever probes
        # directions within the convergence semiangle of the true normal,
        # so the approximation error is O(alpha^2).
        u_c = g @ d_c
        s_c = (2 * k0 * u_c - g2) / (2 * (k0 - u_c))
        sel = torch.abs(s_c) < sg_max + (radius + 1e-4) * g_len
        if k_max is not None:
            sel &= g_len <= k_max
        hkl_beams = torch.cat([torch.zeros((1, 3), dtype=torch.long), hkl_u[sel]])
        g_b = torch.cat([torch.zeros((1, 3), dtype=torch.float64), g[sel]])
        U, u0_imag, absorptive = _coupling_matrix(crystal, hkl_beams, gamma_rel)

        g2b = (g_b**2).sum(dim=1)
        u = torch.einsum("bk,nk->bn", d, g_b)  # g . d_hat per sample
        s_t = (2 * k0 * u - g2b[None, :]) / (2 * (k0 - u))
        inten_b = _bloch_solve(
            U, u0_imag, absorptive, s_t, k0, t_thick, fast_absorption=fast_absorption
        )
        out[c0:c1] = inten_b[:, :, 0]

    # symmetry expansion and Lambert raster: with normal-tracking geometry
    # the intensity is a function of the crystal-frame beam direction only,
    # so proper rotations apply directly; the reversed beam (with reversed
    # normal) gives the same bright field by reciprocity.
    step = angle_step_mrad * 1e-3
    lambert = _lambert_raster(crystal, dirs, out, step)
    rho_max = float(np.sqrt(2.0))

    return {
        "lambert": lambert,
        "rho_max": rho_max,
        "step": step,
        "thicknesses": t_thick.numpy(),
        "energy_ev": float(energy_ev),
        "k_max": k_max,
        "directions": dirs.numpy(),
        "intensity": out.numpy(),
    }


def _lambert_lookup(lambert: np.ndarray, step: float, d_c: torch.Tensor) -> np.ndarray:
    """Bilinear lookup of a (T, n, n) Lambert grid at crystal-frame
    directions d_c (..., 3); returns (T, ...). Directions are folded to
    the upper hemisphere (reciprocity)."""
    d_c = torch.where(d_c[..., 2:3] < 0, -d_c, d_c)
    half = (lambert.shape[-1] - 1) // 2
    rho = torch.sqrt((2 * (1 - d_c[..., 2])).clamp_min(0))
    dxy = torch.linalg.norm(d_c[..., :2], dim=-1).clamp_min(1e-12)
    fx = (d_c[..., 0] / dxy * rho / step + half).numpy()
    fy = (d_c[..., 1] / dxy * rho / step + half).numpy()
    n_l = lambert.shape[-1]
    ix0 = np.clip(np.floor(fx).astype(int), 0, n_l - 2)
    iy0 = np.clip(np.floor(fy).astype(int), 0, n_l - 2)
    wx = np.clip(fx - ix0, 0, 1)
    wy = np.clip(fy - iy0, 0, 1)
    out = np.zeros((lambert.shape[0],) + fx.shape)
    for ti in range(lambert.shape[0]):
        L = lambert[ti]
        out[ti] = (
            L[ix0, iy0] * (1 - wx) * (1 - wy)
            + L[ix0 + 1, iy0] * wx * (1 - wy)
            + L[ix0, iy0 + 1] * (1 - wx) * wy
            + L[ix0 + 1, iy0 + 1] * wx * wy
        )
    return out


def kossel_from_reference(
    reference: dict,
    orientation: torch.Tensor,
    semiconv_mrad: float = 40.0,
    n_pixels: int = 192,
) -> dict:
    """Extract a bright field Kossel pattern from a reference pattern.

    Interpolation only, no Bloch calculation. The detector tilt grid is
    mapped into the crystal frame by the orientation and looked up
    bilinearly on the reference's Lambert grid, so the pattern has the
    reference's angular resolution (angle_step_mrad), whatever n_pixels.

    Parameters
    ----------
    reference : dict
        From calculate_kossel_reference().
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    semiconv_mrad : float, default=40.0
        Convergence semiangle in mrad; the pattern covers this radius.
    n_pixels : int, default=192
        Pixels across the pattern.

    Returns
    -------
    dict
        'bright_field' ((T, n, n), squeezed for one thickness; rows are
        theta_y, columns theta_x; NaN outside the aperture),
        'mrad_per_pixel', 'thicknesses'.
    """
    lam = electron_wavelength_angstrom(reference["energy_ev"])
    d_c, inside, _, _ = _detector_directions(
        lam, orientation, semiconv_mrad, False, n_pixels, 1, 1
    )
    bf = _lambert_lookup(reference["lambert"], reference["step"], d_c)
    bf[:, ~inside.numpy()] = np.nan
    T = bf.shape[0]

    return {
        "bright_field": bf[0] if T == 1 else bf,
        "mrad_per_pixel": 2 * semiconv_mrad / (n_pixels - 1),
        "thicknesses": reference["thicknesses"],
    }


def plot_kossel_reference(
    reference: dict,
    crystal: Crystal,
    thickness_index: int = 0,
    max_index: int = 2,
    theta_max_label_deg: float = 75.0,
    min_crossing: float = 1.0,
    min_crossing_rim: float = 0.3,
    sigma_mrad: float = 10.0,
    lines: dict | None = None,
    theta_circles=(),
    label_color=(0.9, 0.0, 0.0),
    label_fontsize: float = 12,
    stroke_color=(1.0, 1.0, 1.0, 0.7),
    stroke_width: float = 6.0,
    upsample: int = 2,
    cmap: str = "gray",
    axsize: tuple[float, float] = (9.0, 9.0),
    filename: str | None = None,
    figax=None,
):
    """The reference pattern with polar angle circles and low index zone labels.

    Every symmetry copy of the zone axes with direction indices up to
    `max_index` is labeled with its own signed indices (4-index for
    hexagonal and trigonal crystals). The circles and labels are vector
    graphics; saving to a PDF via `filename` keeps them sharp at any zoom,
    with the pattern embedded as a smoothly interpolated image.

    Parameters
    ----------
    reference : dict
        From calculate_kossel_reference().
    crystal : Crystal
        The crystal the reference was computed for.
    thickness_index : int, default=0
        Which thickness of the reference to show.
    max_index : int, default=2
        Largest direction index to label.
    theta_max_label_deg : float, default=75.0
        Zones between this polar angle and the equator are left unlabeled
        (rim clutter); the equatorial zones themselves are labeled just
        outside the disk edge.
    min_crossing : float, default=1.0
        Only label a zone whose crossing strength reaches this value. Each
        Kossel band is a pair of lines at +-theta_B about the zone plane,
        so the rows in a zone (zone law g . [uvw] = 0) form a rosette
        around the zone axis rather than lines through it. The crossing
        strength sums, over the rows in the zone, the line depth weighted
        by exp(-theta_B^2 / 2 sigma^2), and subtracts the strongest row:
        a zone on a single band scores zero (such as <221> or <223> in
        diamond, which contain only the 220 row), and a rosette of several
        strong rows with small Bragg angles scores high. In silicon at
        200 kV the default keeps <001>, <011>, <111>, <112> and <013>,
        and drops <113> (0.6, its 422 and 620 rows sit 11-15 mrad out),
        <123> (0.65) and <233> (0.9).
    min_crossing_rim : float, default=0.3
        The same threshold for the equatorial zones labeled outside the
        disk, where there is room for weaker crossings: keeps <120> and
        <130> in silicon and drops <230>, which is a single 400 band.
    sigma_mrad : float, default=10.0
        Rosette scale of the crossing strength: rows with Bragg angles
        beyond this contribute little, since their band edges are too far
        from the zone axis to read as a crossing.
    lines : dict | None
        Line set from kossel_lines() for the crossing strength, on the
        reference's thickness grid; computed from the crystal, the chosen
        thickness and the reference's k_max (1.2 1/A when it has none) if
        omitted.
    theta_circles : sequence, default=()
        Polar angles (degrees) at which to draw dashed circles; off by
        default.
    label_color, label_fontsize, stroke_color, stroke_width :
        Zone label styling: text color, size, and the translucent outline
        drawn behind each label.
    upsample : int, default=2
        Bilinear upsampling factor of the displayed pattern.
    cmap : str, default="gray"
        Colormap of the pattern.
    axsize : tuple[float, float], default=(9.0, 9.0)
        Figure size in inches when a new figure is made.
    filename : str | None
        If given, save the figure (PDF recommended).
    figax : tuple | None
        (fig, ax) to draw into; a new figure if None.

    Returns
    -------
    fig, ax
        The matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    from quantem.diffraction.crystal import miller_to_miller_bravais
    from quantem.diffraction.rotations import quat_to_matrix

    L = reference["lambert"][thickness_index]
    step = reference["step"]
    half = (L.shape[-1] - 1) // 2
    if upsample > 1:
        from scipy.ndimage import zoom

        L = zoom(np.nan_to_num(L, nan=np.nanmax(L)), upsample, order=1)
    scale = upsample if upsample > 1 else 1

    if figax is None:
        fig, ax = plt.subplots(figsize=axsize)
    else:
        fig, ax = figax
    ax.imshow(L, cmap=cmap, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    def to_px(v):
        return (v / step + half) * scale + (scale - 1) / 2

    phi = np.linspace(0, 2 * np.pi, 721)
    for theta_deg in theta_circles:
        r = 2 * np.sin(np.deg2rad(theta_deg) / 2) / step * scale
        c = to_px(0.0)
        ax.plot(c + r * np.cos(phi), c + r * np.sin(phi), ls="--", color="0.45", lw=0.7)
        ax.text(
            c,
            c - r,
            " %d°" % theta_deg,
            color="0.35",
            fontsize=9,
            va="bottom",
        )

    # unique low index zone directions, expanded over the crystal symmetry
    hexagonal = crystal.hexagonal_matching
    A_T = crystal.lat_real.numpy().T  # d_cartesian = A_T @ [u, v, w]
    A_T_inv = np.linalg.inv(A_T)
    Rs = quat_to_matrix(crystal.sym_quats_matching).numpy()

    # crossing strength from the line set: per row, the deepest line
    # weighted by its Bragg angle
    if lines is None:
        lines = kossel_lines(
            crystal,
            reference["thicknesses"][thickness_index],
            energy_ev=reference["energy_ev"],
            # a reference computed without a cutoff stores k_max=None; the
            # line set needs a finite one
            k_max=reference.get("k_max") or 1.2,
        )
        ti = 0
    else:
        ti = thickness_index
    g_hat = lines["g_hat"].numpy()
    n_rows = g_hat.shape[0]
    row_of = lines["line_row"].numpy()
    line_w = lines["line_depth"].numpy()[:, ti] * np.exp(
        -0.5 * (lines["line_u"].numpy() / (sigma_mrad * 1e-3)) ** 2
    )
    row_weight = np.zeros(n_rows)
    np.maximum.at(row_weight, row_of, line_w)

    def crossing_strength(dc):
        w = row_weight[np.abs(g_hat @ dc) < 1e-4]
        return float(w.sum() - w.max()) if w.size else 0.0

    # one label per distinct crystallographic direction, keyed by its
    # canonical index tuple so a direction reached by several symmetry
    # operations (common at the equatorial rim) is drawn only once
    # integer index-space representation of each symmetry rotation, so the
    # zone index of a symmetry copy is computed by exact integer arithmetic
    # rather than by rounding a projected direction (which can alias a
    # high-index direction onto a low-index label)
    M = [np.rint(A_T_inv @ R @ A_T).astype(int) for R in Rs]

    placed: dict[tuple, tuple] = {}
    rng = range(-max_index, max_index + 1)
    for u in rng:
        for v in rng:
            for w in rng:
                uvw = np.array([u, v, w])
                if not uvw.any() or np.gcd.reduce(np.abs(uvw)) != 1:
                    continue
                d = A_T @ uvw
                d = d / np.linalg.norm(d)
                for R, Mi in zip(Rs, M):
                    for sgn in (1, -1):
                        dc = sgn * (R @ d)
                        idx = sgn * (Mi @ uvw)
                        # fold to the upper hemisphere (the reference is
                        # stored there); flip the index to match
                        if dc[2] < 0:
                            dc = -dc
                            idx = -idx
                        rim = dc[2] < np.sin(np.deg2rad(1.0))
                        if not rim and dc[2] < np.cos(np.deg2rad(theta_max_label_deg)):
                            continue
                        if crossing_strength(dc) < (min_crossing_rim if rim else min_crossing):
                            continue
                        key = tuple(int(k) for k in idx)
                        if key in placed:
                            continue
                        ks = np.atleast_2d(miller_to_miller_bravais(idx))[0] if hexagonal else idx
                        txt = (
                            "$["
                            + "".join((r"\bar{%d\!}" % abs(k)) if k < 0 else str(k) for k in ks)
                            + "]$"
                        )
                        rho = np.sqrt(max(2 * (1 - dc[2]), 0))
                        if rim:
                            rho = np.sqrt(2.0) * 1.07  # just outside the disk
                        dxy = max(np.hypot(dc[0], dc[1]), 1e-12)
                        px = to_px(dc[0] / dxy * rho)
                        py = to_px(dc[1] / dxy * rho)
                        placed[key] = (px, py, txt)

    for px, py, txt in placed.values():
        t = ax.text(
            py,
            px,
            txt,
            color=label_color,
            fontsize=label_fontsize,
            ha="center",
            va="center",
        )
        t.set_path_effects(
            [patheffects.withStroke(linewidth=stroke_width, foreground=stroke_color)]
        )
    n_px = L.shape[-1]
    ax.set_xlim(-0.06 * n_px, 1.06 * n_px)
    ax.set_ylim(1.06 * n_px, -0.06 * n_px)
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    return fig, ax


def kossel_polar_from_reference(
    reference: dict,
    orientation: torch.Tensor,
    semiconv_mrad: float = 40.0,
    n_radial: int = 64,
    n_azimuthal: int = 180,
) -> dict:
    """A bright field Kossel pattern sampled directly on a polar grid.

    Dictionary matching correlates over the in-plane rotation, which is a
    cyclic shift of the azimuthal axis in polar coordinates: sampling the
    reference directly at the polar detector positions avoids the intermediate
    Cartesian raster and its interpolation.

    Parameters
    ----------
    reference : dict
        From calculate_kossel_reference().
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    semiconv_mrad : float, default=40.0
        Outer radius of the polar grid in mrad.
    n_radial : int, default=64
        Radial samples, at radii semiconv_mrad * (1 ... n_radial) / n_radial.
    n_azimuthal : int, default=180
        Azimuthal samples, at 2 pi (0 ... n_azimuthal - 1) / n_azimuthal.

    Returns
    -------
    dict
        'polar' ((T, n_azimuthal, n_radial), squeezed for one thickness;
        rows are azimuth, columns radius, matching the quantem polar
        transform convention), 'radii_mrad', 'azimuth_rad', 'thicknesses'.
    """
    lam = electron_wavelength_angstrom(reference["energy_ev"])
    d_c, _, axes, _ = _detector_directions(
        lam, orientation, semiconv_mrad, True, 1, n_radial, n_azimuthal
    )
    out = _lambert_lookup(reference["lambert"], reference["step"], d_c)
    T = out.shape[0]
    return {
        "polar": out[0] if T == 1 else out,
        "radii_mrad": axes["radii_mrad"],
        "azimuth_rad": axes["azimuth_rad"],
        "thicknesses": reference["thicknesses"],
    }


def kossel_lines(
    crystal: Crystal,
    thicknesses_A,
    energy_ev: float = 300e3,
    k_max: float = 1.2,
    u_step_mrad: float = 0.05,
    u_tail_mrad: float = 150.0,
    min_depth: float = 0.005,
    fast_absorption: bool = False,
) -> dict:
    """Vector representation of the Kossel lines: one profile per systematic row.

    The bright field depends on the beam direction d (a unit vector, the
    anti-propagation direction in the crystal frame) only through the
    projections u = d . g_hat onto the row normals. For each systematic row
    {n g} the profile is a Bloch calculation over the row beams alone versus
    the signed projection u, which places the deficiency line of reflection
    +n g at u = +n lambda |g| / 2 and that of -n g at u = -n lambda |g| / 2:
    the two lines of a Kossel band, 2 theta_B apart, and their higher
    orders, all with their dynamical widths and thickness fringes. Rows
    combine multiplicatively as independent attenuation channels; the
    many-beam coupling between different rows at the zone axis crossings is
    the one approximation.

    A row profile is a smooth function of a continuous variable, so patterns
    rendered from the line set (render_kossel_lines) are exact in geometry
    and free of raster interpolation at any pixel size, in Cartesian or
    polar coordinates.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated, and preferably
        calculate_dynamical_structure_factors at this energy (absorption).
    thicknesses_A : float | array-like
        Thickness grid in Angstroms.
    energy_ev : float, default=300e3
        Beam energy in eV.
    k_max : float, default=1.2
        Reflections with |g| up to this (1/Angstroms) are included; a row
        keeps every order |n| |g| <= k_max. Must be a number.
    u_step_mrad : float, default=0.05
        Profile sampling; the line widths are 1-2 mrad.
    u_tail_mrad : float, default=150.0
        Profile extent beyond the outermost line of each row. The rocking
        curve tails fall off as 1 / (s xi)^2 and are still ~1% at 40 mrad
        for the strong reflections, so the window has to be wide for the
        far-from-line background to be the true mean-absorption level.
    min_depth : float, default=0.005
        Lines (and rows) whose deepest deficit at any thickness is below
        this fraction of the background are dropped.
    fast_absorption : bool, default=False
        First-order absorption in the row calculations; see _bloch_solve.

    Returns
    -------
    dict with, per row, 'g_hat' (L, 3) crystal-frame unit normals, 'g_len'
    (L,), 'hkl_row' (L, 3), 'log_trans' (L, T, n_u) log transmission
    versus 'u' (n_u,) (0 far from the lines), 'background' (T,) the
    far-from-line bright field; and per line 'line_row' (K,) row index,
    'line_order' (K,) the order n, 'line_hkl' (K, 3), 'line_u' (K,) the
    cone position u = n lambda |g| / 2, 'line_depth' (K, T) the deepest
    deficit fraction and 'line_width_mrad' (K, T) the equivalent width
    (integrated deficit over depth); also 'energy_ev' and 'thicknesses'.

    Raises
    ------
    ValueError
        If k_max is None, or no line reaches min_depth.
    """
    if crystal.g_vec is None:
        raise RuntimeError("Run crystal.calculate_structure_factors() first.")
    if k_max is None:
        raise ValueError(
            "kossel_lines needs a numeric k_max: every reflection up to it gets a "
            "row profile, so the full factor set would be very slow"
        )
    k_max = float(k_max)
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)
    t_thick = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    # unique rows: group reflections by ray direction (g and -g together),
    # keep the shortest g of each as the row vector
    hkl, g_all = _beam_universe(crystal)
    g_len = torch.linalg.norm(g_all, dim=1)
    sel = (g_len <= k_max) & (g_len > 1e-8)
    idx = torch.nonzero(sel).squeeze(1)
    idx = idx[torch.argsort(g_len[idx])]
    rows: list[int] = []
    dirs: list[torch.Tensor] = []
    for i in idx.tolist():
        d = g_all[i] / g_len[i]
        if any(float(torch.abs(d @ e)) > 0.9999 for e in dirs):
            continue
        rows.append(i)
        dirs.append(d)

    u_max = 0.5 * lam * k_max + u_tail_mrad * 1e-3
    du = u_step_mrad * 1e-3
    n_half = int(np.ceil(u_max / du))
    u = torch.arange(-n_half, n_half + 1, dtype=torch.float64) * du

    g_hat_out, g_len_out, hkl_out, lt_out, bg_rows = [], [], [], [], []
    l_row, l_order, l_hkl, l_u, l_depth, l_width = [], [], [], [], [], []
    for i in rows:
        g1 = float(g_len[i])
        h1 = hkl[i]
        n_ord = int(np.floor(k_max / g1 + 1e-9))
        ns = torch.arange(-n_ord, n_ord + 1, dtype=torch.long)
        ns = ns[torch.argsort((ns != 0).to(torch.long), stable=True)]  # 000 first
        hkl_beams = ns[:, None] * h1[None, :]
        U, u0_imag, absorptive = _coupling_matrix(crystal, hkl_beams, gamma_rel)
        # projection of each row beam on the beam direction: n |g| u, and
        # the same excitation error geometry as the reference pattern
        # (foil normal along the beam)
        ng = ns.to(torch.float64) * g1
        uu = u[:, None] * ng[None, :]
        s_t = (2 * k0 * uu - ng[None, :] ** 2) / (2 * (k0 - uu))
        inten_b = _bloch_solve(
            U, u0_imag, absorptive, s_t, k0, t_thick, fast_absorption=fast_absorption
        )
        bf = inten_b[:, :, 0].transpose(0, 1)  # (T, n_u)
        bg = 0.5 * (bf[:, 0] + bf[:, -1])  # (T,) far-from-line level
        trans = (bf / bg[:, None]).clamp_min(1e-6)
        deficit = 1 - trans

        # per-line depth and width, each order in its own window of half
        # the order spacing on either side of its cone
        lines_here = []
        for n in range(-n_ord, n_ord + 1):
            if n == 0:
                continue
            u_n = n * lam * g1 / 2
            win = torch.abs(u - u_n) <= lam * g1 / 4
            dep = deficit[:, win].amax(dim=1)  # (T,)
            if float(dep.max()) < min_depth:
                continue
            width = deficit[:, win].clamp_min(0).sum(dim=1) * du / dep.clamp_min(1e-9)
            lines_here.append((n, u_n, dep, width * 1e3))
        if not lines_here:
            continue
        row_id = len(g_hat_out)
        g_hat_out.append(g_all[i] / g_len[i])
        g_len_out.append(g1)
        hkl_out.append(h1)
        lt_out.append(torch.log(trans))
        bg_rows.append(bg)
        for n, u_n, dep, width in lines_here:
            l_row.append(row_id)
            l_order.append(n)
            l_hkl.append(n * h1)
            l_u.append(u_n)
            l_depth.append(dep)
            l_width.append(width)
    if not g_hat_out:
        raise ValueError(
            f"no Kossel line reaches min_depth={min_depth} with k_max={k_max} 1/A: "
            "raise k_max or lower min_depth"
        )

    return {
        "g_hat": torch.stack(g_hat_out),
        "g_len": torch.tensor(g_len_out, dtype=torch.float64),
        "hkl_row": torch.stack(hkl_out),
        "u": u,
        "log_trans": torch.stack(lt_out),  # (L, T, n_u)
        "background": torch.stack(bg_rows).mean(dim=0),  # (T,)
        "line_row": torch.tensor(l_row, dtype=torch.long),
        "line_order": torch.tensor(l_order, dtype=torch.long),
        "line_hkl": torch.stack(l_hkl),
        "line_u": torch.tensor(l_u, dtype=torch.float64),
        "line_depth": torch.stack(l_depth),  # (K, T)
        "line_width_mrad": torch.stack(l_width),  # (K, T)
        "energy_ev": float(energy_ev),
        "thicknesses": t_thick.numpy(),
    }


def _detector_directions(
    lam: float,
    orientation: torch.Tensor,
    semiconv_mrad: float,
    polar: bool,
    n_pixels: int,
    n_radial: int,
    n_azimuthal: int,
):
    """Crystal-frame anti-propagation directions of a Cartesian or polar
    detector grid, plus the grid axes. Polar grids follow the quantem
    convention: rows are azimuth, columns radius."""
    from quantem.diffraction.rotations import quat_to_matrix

    k0 = 1.0 / lam
    alpha_k = semiconv_mrad * 1e-3 / lam
    if polar:
        r = torch.linspace(0, alpha_k, n_radial + 1, dtype=torch.float64)[1:]
        phi = torch.arange(n_azimuthal, dtype=torch.float64) * (2 * np.pi / n_azimuthal)
        tx = r[None, :] * torch.cos(phi)[:, None]
        ty = r[None, :] * torch.sin(phi)[:, None]
        inside = torch.ones_like(tx, dtype=torch.bool)
        axes = {"radii_mrad": (r * lam * 1e3).numpy(), "azimuth_rad": phi.numpy()}
    else:
        ax = torch.linspace(-alpha_k, alpha_k, n_pixels, dtype=torch.float64)
        ty, tx = torch.meshgrid(ax, ax, indexing="ij")
        inside = (tx**2 + ty**2) <= alpha_k**2
        axes = {"mrad_per_pixel": 2 * semiconv_mrad / (n_pixels - 1)}
    tz = torch.sqrt((k0**2 - tx**2 - ty**2).clamp_min(0))
    # the beam landing at detector tilt +t propagates along (t, -tz); the
    # line set and the reference parameterize the anti-propagation direction
    d_lab = torch.stack([-tx, -ty, tz], dim=-1) / k0
    R = quat_to_matrix(torch.atleast_2d(torch.as_tensor(orientation, dtype=torch.float64))[0]).to(
        torch.float64
    )
    d_c = torch.einsum("ji,rcj->rci", R, d_lab)  # crystal frame, R^T d
    return d_c, inside, axes, R


def _lines_bright_field(lines: dict, d_c: torch.Tensor) -> torch.Tensor:
    """Line-model bright field at crystal-frame directions d_c (..., 3):
    product of the row transmissions read at u = d . g_hat; returns
    (..., T)."""
    u = torch.einsum("...i,li->...l", d_c, lines["g_hat"])  # (.., L)
    u_ax = lines["u"]
    n_u = u_ax.shape[0]
    du = float(u_ax[1] - u_ax[0])
    lt = lines["log_trans"].permute(0, 2, 1)  # (L, n_u, T)
    f = ((u - float(u_ax[0])) / du).clamp(0, n_u - 1 - 1e-9)
    i0 = f.floor().to(torch.long)
    w = (f - i0)[..., None]
    L_idx = torch.arange(lt.shape[0]).reshape((1,) * (u.dim() - 1) + (-1,))
    v = lt[L_idx, i0] * (1 - w) + lt[L_idx, (i0 + 1).clamp(max=n_u - 1)] * w
    return lines["background"] * torch.exp(v.sum(dim=-2))


def kossel_reference_residual(reference: dict, lines: dict, crystal: Crystal) -> dict:
    """Add the many-beam residual of the line model to a reference pattern.

    The line model is evaluated at the reference's own wedge samples and
    rasterized onto the same Lambert grid, and the difference (reference
    minus line model) is stored as reference['residual']. It is zero away
    from the zone axes, where the rows are independent, and carries the
    many-beam correction of the zone axis rosettes. render_kossel_lines()
    adds it by lookup when given the reference.

    Parameters
    ----------
    reference : dict
        From calculate_kossel_reference(); modified in place.
    lines : dict
        From kossel_lines(), on the same thickness grid and energy.
    crystal : Crystal
        The crystal both were computed for.

    Returns
    -------
    dict
        The reference, with 'residual' (T, n, n) added.

    Raises
    ------
    ValueError
        If the thickness grids differ.
    """
    if not np.allclose(reference["thicknesses"], lines["thicknesses"]):
        raise ValueError("reference and line set must share the thickness grid")
    dirs = torch.as_tensor(reference["directions"], dtype=torch.float64)
    I_lines = _lines_bright_field(lines, dirs)  # (N, T)
    lambert_lines = _lambert_raster(crystal, dirs, I_lines, reference["step"])
    reference["residual"] = np.nan_to_num(reference["lambert"] - lambert_lines, nan=0.0)
    return reference


def render_kossel_lines(
    lines: dict,
    orientation: torch.Tensor,
    semiconv_mrad: float = 40.0,
    n_pixels: int = 256,
    polar: bool = False,
    n_radial: int = 64,
    n_azimuthal: int = 180,
    reference: dict | None = None,
) -> dict:
    """Render the bright field from the Kossel line set, all thicknesses.

    Every detector direction is projected on every row normal and the row
    profiles are read there: one evaluation per pixel and row, no raster
    in between, so the result is smooth at any resolution in Cartesian or
    polar coordinates. A polar pattern is sampled directly at the polar
    detector positions.

    Parameters
    ----------
    lines : dict
        From kossel_lines().
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    semiconv_mrad : float, default=40.0
        Convergence semiangle in mrad; the pattern covers this radius.
    n_pixels : int, default=256
        Pixels across a Cartesian pattern.
    polar : bool, default=False
        Sample on a polar grid instead (see kossel_polar_from_reference
        for the grid).
    n_radial, n_azimuthal : int, default=64, 180
        Polar grid size.
    reference : dict | None
        A reference pattern carrying the many-beam residual from
        kossel_reference_residual(). If given, the residual is added to
        the rendered pattern: the line model then also carries the
        many-beam intensity of the zone axis rosettes (which the
        independent-row product gets too dark), while the lines themselves
        keep their exact analytic geometry.

    Returns
    -------
    dict with 'bright_field' ((T, n, n), squeezed; rows are theta_y,
    columns theta_x; NaN outside the aperture) or, with polar=True, 'polar' ((T, n_azimuthal, n_radial),
    squeezed; rows are azimuth, columns radius), plus the grid axes and
    'thicknesses'.
    """
    lam = electron_wavelength_angstrom(lines["energy_ev"])
    d_c, inside, axes, _ = _detector_directions(
        lam, orientation, semiconv_mrad, polar, n_pixels, n_radial, n_azimuthal
    )
    bf = _lines_bright_field(lines, d_c).permute(2, 0, 1).numpy()  # (T, ..)
    if reference is not None:
        if "residual" not in reference:
            raise ValueError(
                "reference has no many-beam residual: run "
                "kossel_reference_residual(reference, lines, crystal) first."
            )
        bf = bf + _lambert_lookup(reference["residual"], reference["step"], d_c)
    bf[:, ~inside.numpy()] = np.nan
    T = bf.shape[0]

    out = {"polar" if polar else "bright_field": bf[0] if T == 1 else bf}
    out.update(axes)
    out["thicknesses"] = lines["thicknesses"]
    return out


def kossel_line_segments(
    lines: dict,
    orientation: torch.Tensor,
    semiconv_mrad: float = 40.0,
    thickness_index: int = 0,
) -> dict:
    """The Kossel lines crossing the aperture as vector segments.

    Each line is the cone d . g_hat = u of its reflection, which within
    the aperture is a straight line in the detector tilt plane (the
    curvature term is |g_z| alpha^2 / 2, below 0.1 mrad at 40 mrad). The
    end points on the aperture edge are computed exactly from the cone.

    Parameters
    ----------
    lines : dict
        From kossel_lines().
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    semiconv_mrad : float, default=40.0
        Aperture radius in mrad.
    thickness_index : int, default=0
        Thickness of the line set for 'depth' and 'width_mrad'.

    Returns
    -------
    dict of arrays over the K visible lines. Cartesian positions are
    (row, col) tilt angles in mrad, matching the image axes of
    render_kossel_lines: 'start_mrad', 'stop_mrad' (K, 2) the end points
    on the aperture edge; 'normal' (K, 2) the unit normal of the line and
    'distance_mrad' (K,) its signed distance from the optic axis, so the
    line is the set of points with p . normal = distance. Polar positions
    are (azimuth_rad, radius_mrad): 'start_polar', 'stop_polar' (K, 2), the
    end points at radius = semiconv_mrad; in between the line follows
    radius = distance / cos(azimuth - azimuth_normal). Also 'hkl' (K, 3),
    'depth' (K,) the deficit fraction and 'width_mrad' (K,) the equivalent
    width at the chosen thickness.
    """
    from quantem.diffraction.rotations import quat_to_matrix

    alpha = semiconv_mrad * 1e-3
    R = (
        quat_to_matrix(torch.atleast_2d(torch.as_tensor(orientation, dtype=torch.float64))[0])
        .to(torch.float64)
        .numpy()
    )
    g_lab = (R @ lines["g_hat"].numpy().T).T  # d_lab . g_lab = d_c . g_c
    rows = lines["line_row"].numpy()
    u_k = lines["line_u"].numpy()
    g = g_lab[rows]  # (K, 3)
    # cone in tilt angles theta = (theta_x, theta_y), d_lab = (-theta, sqrt(1 - theta^2)):
    #   -g_x theta_x - g_y theta_y + g_z sqrt(1 - theta^2) = u
    gxy = np.hypot(g[:, 0], g[:, 1])
    ok = gxy > 1e-9
    phi_g = np.arctan2(g[:, 1], g[:, 0])
    cz = np.sqrt(1 - alpha**2)
    # on the aperture edge theta = alpha (cos phi, sin phi):
    #   cos(phi - phi_g) = (g_z cz - u) / (|g_xy| alpha)
    c = np.where(ok, (g[:, 2] * cz - u_k) / np.maximum(gxy * alpha, 1e-12), 2.0)
    ok &= np.abs(c) < 1
    dphi = np.arccos(np.clip(c[ok], -1, 1))
    phi_a = phi_g[ok] + dphi
    phi_b = phi_g[ok] - dphi
    # small-angle line: (g_x, g_y) . theta = g_z - u
    p = (g[ok, 2] - u_k[ok]) / gxy[ok]  # signed distance (rad) along -normal
    normal = np.stack([g[ok, 1], g[ok, 0]], axis=1) / gxy[ok, None]  # (row, col)

    def pt(phi):
        # (row, col) = (theta_y, theta_x) in mrad
        return np.stack([alpha * np.sin(phi), alpha * np.cos(phi)], axis=1) * 1e3

    ti = thickness_index
    return {
        "hkl": lines["line_hkl"].numpy()[ok],
        "start_mrad": pt(phi_a),
        "stop_mrad": pt(phi_b),
        "start_polar": np.stack(
            [np.mod(phi_a, 2 * np.pi), np.full(phi_a.shape, semiconv_mrad)], axis=1
        ),
        "stop_polar": np.stack(
            [np.mod(phi_b, 2 * np.pi), np.full(phi_b.shape, semiconv_mrad)], axis=1
        ),
        "normal": normal,
        "distance_mrad": p * 1e3,
        "depth": lines["line_depth"].numpy()[ok, ti],
        "width_mrad": lines["line_width_mrad"].numpy()[ok, ti],
    }


def overlay_kossel_segments(
    ax,
    segments: dict,
    semiconv_mrad: float,
    n_pixels: int | None = None,
    polar: bool = False,
    n_radial: int | None = None,
    n_azimuthal: int | None = None,
    color=(0.9, 0.0, 0.0),
    width_scale: float = 1.0,
    min_depth: float = 0.05,
):
    """Draw the vector line segments over a rendered pattern.

    Line width is the equivalent width of each line in pixels (times
    width_scale) and the opacity is its depth. On a Cartesian axis the
    segments run between their aperture-edge end points; on a polar axis
    (rows azimuth, columns radius) each straight line becomes the curve
    radius = distance / cos(azimuth - azimuth_normal), drawn from end
    point to end point and split at the azimuth wrap. The pixel registration
    is that of render_kossel_lines and the reference lookups: Cartesian
    pixel i at angle -semiconv + 2 semiconv i / (n_pixels - 1), polar
    column j at radius semiconv (j + 1) / n_radial and row i at azimuth
    2 pi i / n_azimuthal.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes showing the rendered pattern (imshow pixel coordinates).
    segments : dict
        From kossel_line_segments().
    semiconv_mrad : float
        Aperture radius of the pattern in mrad.
    n_pixels : int | None
        Pixels across a Cartesian pattern.
    polar : bool, default=False
        Draw on a polar pattern instead.
    n_radial, n_azimuthal : int | None
        Polar grid size.
    color : color, default=(0.9, 0.0, 0.0)
        Line color.
    width_scale : float, default=1.0
        Multiplier of the drawn line width.
    min_depth : float, default=0.05
        Lines shallower than this deficit fraction are not drawn.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    sel = segments["depth"] >= min_depth
    n_lines = int(sel.sum())
    p = segments["distance_mrad"][sel]
    nrm = segments["normal"][sel]
    start = segments["start_mrad"][sel]
    stop = segments["stop_mrad"][sel]
    dep = segments["depth"][sel]
    wid = segments["width_mrad"][sel]
    if polar:
        # radius r_j = semiconv (j + 1) / n_radial, azimuth phi_i = 2 pi i / n_az
        px_r = n_radial / semiconv_mrad
        px_phi = n_azimuthal / (2 * np.pi)
        tang = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1)
        t_edge = np.sqrt(np.maximum(semiconv_mrad**2 - p**2, 0))
        t = np.linspace(-1, 1, 400)
        for k in range(n_lines):
            pts = p[k] * nrm[k][None, :] + (t * t_edge[k])[:, None] * tang[k][None, :]
            r = np.hypot(pts[:, 0], pts[:, 1])
            phi = np.mod(np.arctan2(pts[:, 0], pts[:, 1]), 2 * np.pi)
            jumps = np.abs(np.diff(phi)) > np.pi
            phi = np.ma.array(phi, mask=np.r_[False, jumps])
            ax.plot(
                r * px_r - 1.0,
                phi * px_phi,
                color=color,
                lw=wid[k] * px_r * width_scale,
                alpha=float(dep[k]),
                solid_capstyle="butt",
            )
    else:
        # linspace(-semiconv, semiconv, n_pixels): pixel centers at the ends
        px = (n_pixels - 1) / (2 * semiconv_mrad)
        for k in range(n_lines):
            ax.plot(
                [
                    (start[k, 1] + semiconv_mrad) * px,
                    (stop[k, 1] + semiconv_mrad) * px,
                ],
                [
                    (start[k, 0] + semiconv_mrad) * px,
                    (stop[k, 0] + semiconv_mrad) * px,
                ],
                color=color,
                lw=wid[k] * px * width_scale,
                alpha=float(dep[k]),
                solid_capstyle="butt",
            )
    return ax


def average_bloch_fourier(
    crystal: Crystal,
    orientation: torch.Tensor,
    trial_tilts: torch.Tensor,
    thicknesses_A,
    energy_ev: float,
    precession_deg: float,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
    deform: torch.Tensor | None = None,
    beams: torch.Tensor | None = None,
    n_harmonics: int = 48,
    n_geometry: int = 128,
    n_matrix_harmonics: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precession-averaged Bloch intensities by harmonic propagation.

    On the precession ring the structure matrix is a Fourier series in the
    azimuth, A(phi) = sum_m A_m exp(i m phi); for a ring centered on the
    optic axis only m = 0, +-1 are nonzero and the coefficients are exact,

        A_0 = U + diag(2 k0 c_g + i U0''),
        A_(+1) = diag[-k0 r (g_x - i g_y) / (K - g_z)],   A_(-1) = conj.,

    with r = k0 sin(theta_p), K = sqrt(k0^2 - r^2), c_g the ring-centered
    excitation error. Expanding the wave function in azimuthal modes,
    psi(phi, z) = sum_n x_n(z) exp(i n phi), the Bloch equation couples
    neighboring modes, dx_n/dz = (i pi / k0) sum_m A_m x_(n-m), starting
    from x_0(0) = e_000, and the ring average of the intensity is the
    incoherent sum over modes, I_g = sum_n |x_(n,g)|^2, exactly (Parseval).
    The mode chain is truncated at |n| <= n_harmonics (48 reproduces a
    converged quadrature to 1e-15 for silicon at 600 A and 0.4 degrees; 24
    leaves 1e-7) and a uniformly spaced thickness grid comes from one
    action of the matrix exponential of the block-tridiagonal generator
    at all its time points. For a ring displaced by a trial tilt
    the coefficients are no longer three: the exact excitation errors on
    n_geometry azimuths are Fourier transformed and n_matrix_harmonics
    of them kept (default n_harmonics // 3); both counts and n_harmonics
    must be converged for the result to be exact.

    The absorption is the full complex matrix (there is no first-order
    variant here). Cost: a sparse block matrix of size (2 n_harmonics +
    1) x nb per trial tilt and one Krylov exponential action over the
    thickness grid.

    Status: an alternative to the azimuthal quadrature of
    illumination_nodes(), verified against it in the tests but not used by
    the refinement functions of this module, which average batched
    eigensolves instead (one eigendecomposition serves every thickness).

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated.
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    trial_tilts : torch.Tensor
        (M, 2) ring centers as in-plane incident wavevectors (1/Angstroms).
    thicknesses_A : float | array-like
        Thickness grid in Angstroms; a uniform grid is propagated in one
        pass.
    energy_ev : float
        Beam energy in eV.
    precession_deg : float
        Precession semi-angle in degrees.
    sg_max : float, default=SG_MAX
        Excitation error cutoff (1/Angstroms) of the beam list.
    k_max : float | None
        Largest |g| (1/Angstroms) of a beam.
    deform : torch.Tensor | None
        (3, 3) deformation of the lab-frame reciprocal vectors.
    beams : torch.Tensor | None
        Explicit beam list (nb, 3), 000 first; selected here if None.
    n_harmonics : int, default=48
        Azimuthal modes kept, |n| <= n_harmonics.
    n_geometry : int, default=128
        Azimuths sampled for the coefficients of a displaced ring.
    n_matrix_harmonics : int | None
        Fourier coefficients of the displaced ring kept; default
        n_harmonics // 3.

    Returns
    -------
    intensities : torch.Tensor
        (M, T, nb) ring-averaged intensities, direct beam first.
    g_xy : torch.Tensor
        (nb, 2) in-plane positions of the beams (1/Angstroms).
    """
    from scipy.sparse import csr_matrix, diags, kron
    from scipy.sparse.linalg import expm_multiply

    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)
    t_grid = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))
    r = k0 * np.sin(np.deg2rad(precession_deg))
    K = np.sqrt(k0**2 - r**2)
    trial = torch.atleast_2d(torch.as_tensor(trial_tilts, dtype=torch.float64))
    if beams is None:
        alpha_max = (float(torch.linalg.norm(trial, dim=1).max()) + r) / k0
        beams = select_dynamical_beams(
            crystal, orientation, energy_ev, alpha_max, sg_max, k_max, deform
        )
    g_beams = qrotate(orientation, beams[1:].to(torch.float64) @ crystal.lat_recip)
    if deform is not None:
        g_beams = g_beams @ deform.to(torch.float64).T
    g_beams = torch.cat([torch.zeros((1, 3), dtype=torch.float64), g_beams]).numpy()
    nb = g_beams.shape[0]
    U, u0_imag, _ = _coupling_matrix(crystal, beams, gamma_rel)
    U_np = U.numpy()
    L = int(n_harmonics)
    nm = 2 * L + 1
    H = n_harmonics // 3 if n_matrix_harmonics is None else int(n_matrix_harmonics)
    g2 = (g_beams**2).sum(axis=1)

    out = np.zeros((trial.shape[0], t_grid.shape[0], nb))
    for it, t0 in enumerate(trial.numpy()):
        if np.hypot(*t0) < 1e-12:
            den = K - g_beams[:, 2]
            c = (2 * K * g_beams[:, 2] - g2) / (2 * den)
            coeff = {
                0: U_np + np.diag(2 * k0 * c + 1j * u0_imag),
                1: np.diag(-k0 * r * (g_beams[:, 0] - 1j * g_beams[:, 1]) / den),
                -1: np.diag(-k0 * r * (g_beams[:, 0] + 1j * g_beams[:, 1]) / den),
            }
        else:
            Q = int(n_geometry)
            phi = 2 * np.pi * np.arange(Q) / Q
            t = t0[None, :] + r * np.stack([np.cos(phi), np.sin(phi)], axis=1)
            kz = np.sqrt(k0**2 - (t**2).sum(1))[:, None]
            s_phi = (2 * kz * g_beams[:, 2] - 2 * (t @ g_beams[:, :2].T) - g2) / (
                2 * (kz - g_beams[:, 2])
            )
            d = np.fft.fft(2 * k0 * s_phi, axis=0) / Q
            coeff = {m: np.diag(d[m % Q]) for m in range(-H, H + 1)}
            coeff[0] = coeff[0] + U_np + 1j * u0_imag * np.eye(nb)
        operator = csr_matrix((nm * nb, nm * nb), dtype=complex)
        for m, A in coeff.items():
            if abs(m) > 2 * L:
                continue
            shift = diags(np.ones(nm - abs(m)), -m, shape=(nm, nm), format="csr")
            operator = operator + kron(shift, csr_matrix(A), format="csr")
        generator = (1j * np.pi / k0) * operator
        x0 = np.zeros(nm * nb, dtype=complex)
        x0[L * nb] = 1.0
        zs = t_grid.numpy()
        uniform = zs.shape[0] > 1 and np.allclose(np.diff(zs), zs[1] - zs[0])
        if uniform:
            x = expm_multiply(
                generator,
                x0,
                start=float(zs[0]),
                stop=float(zs[-1]),
                num=zs.shape[0],
                endpoint=True,
            ).reshape(zs.shape[0], nm, nb)
            out[it] = (np.abs(x) ** 2).sum(axis=1)
        else:
            for iz, z in enumerate(zs):
                x = expm_multiply(z * generator, x0).reshape(nm, nb)
                out[it, iz] = (np.abs(x) ** 2).sum(axis=0)
    return torch.as_tensor(out), torch.as_tensor(g_beams[:, :2])


def illumination_nodes(
    energy_ev: float,
    precession_deg: float = 0.0,
    n_precession: int = 32,
    semiconv_mrad: float = 0.0,
    n_disk_radial: int = 4,
    n_disk_azimuthal: int = 16,
    maped_tilts_deg=None,
    maped_weights=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Incident beam tilts and weights that model one measured pattern.

    Precession is a ring of radius k0 sin(theta_p) sampled uniformly in
    azimuth (the Gauss-Chebyshev quadrature of the ring integral); the
    convergence disk of radius k0 sin(alpha) is sampled with Gauss-Legendre
    nodes in (radius / R)^2 and uniform azimuth, which integrates the
    uniform-area measure exactly for polynomials (an equal-weight ring
    grid gives the disk a second moment of 0.70 R^2 instead of 0.50 R^2).
    MAPED is an explicit tilt list with exposure weights. Ring and disk
    combine as a product measure. Zero tilt with unit weight when none
    apply.

    Parameters
    ----------
    energy_ev : float
        Beam energy in eV.
    precession_deg : float, default=0.0
        Precession semi-angle in degrees; 0 for none.
    n_precession : int, default=32
        Azimuthal samples on the precession ring.
    semiconv_mrad : float, default=0.0
        Convergence semiangle in mrad; 0 for a parallel beam.
    n_disk_radial, n_disk_azimuthal : int, default=4, 16
        Gauss-Legendre radii and azimuths of the convergence disk.
    maped_tilts_deg : array-like | None
        (M, 2) explicit beam tilts in degrees (MAPED), replacing the ring.
    maped_weights : array-like | None
        (M,) exposure weights of the MAPED tilts; equal if None.

    Returns
    -------
    tilts : torch.Tensor
        (M, 2) in-plane incident wavevectors (1/Angstroms).
    weights : torch.Tensor
        (M,) weights summing to one.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    if maped_tilts_deg is not None:
        ring = k0 * torch.sin(torch.deg2rad(torch.as_tensor(maped_tilts_deg, dtype=torch.float64)))
        if maped_weights is None:
            w_ring = torch.full((ring.shape[0],), 1.0 / ring.shape[0], dtype=torch.float64)
        else:
            w_ring = torch.as_tensor(maped_weights, dtype=torch.float64)
            w_ring = w_ring / w_ring.sum()
    elif precession_deg > 0:
        phi = torch.arange(n_precession, dtype=torch.float64) * (2 * np.pi / n_precession)
        r = k0 * np.sin(np.deg2rad(precession_deg))
        ring = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=1)
        w_ring = torch.full((n_precession,), 1.0 / n_precession, dtype=torch.float64)
    else:
        ring = torch.zeros((1, 2), dtype=torch.float64)
        w_ring = torch.ones(1, dtype=torch.float64)
    if semiconv_mrad > 0:
        R = k0 * np.sin(semiconv_mrad * 1e-3)
        x, w = np.polynomial.legendre.leggauss(n_disk_radial)
        radius = R * np.sqrt((x + 1) / 2)
        psi = 2 * np.pi * np.arange(n_disk_azimuthal) / n_disk_azimuthal
        disk = torch.as_tensor(
            (radius[:, None, None] * np.stack([np.cos(psi), np.sin(psi)], axis=1)[None]).reshape(
                -1, 2
            )
        )
        w_disk = torch.as_tensor(np.repeat(w / 2 / n_disk_azimuthal, n_disk_azimuthal))
    else:
        disk = torch.zeros((1, 2), dtype=torch.float64)
        w_disk = torch.ones(1, dtype=torch.float64)
    tilts = (ring[:, None, :] + disk[None, :, :]).reshape(-1, 2)
    weights = (w_ring[:, None] * w_disk[None, :]).reshape(-1)
    return tilts, weights


def _dynamical_cost(inten, sq, qxy, im, delta, power: float, min_sim_rel: float = 0.0):
    """Intensity cost (M, T) of simulated raw intensities (M, T, N) at
    positions sq (N, 2) against measured peaks (P, 2) with intensities im
    (P,) already raised to `power`, with a free scale per (tilt,
    thickness). Both sides are compared as I ** power; the power is applied
    here, after any illumination averaging, never before it. Pairing is by
    position (within delta); the position residuals themselves do not
    enter, they belong to the deformation fit. Unpaired simulated beams
    weaker than min_sim_rel times the strongest simulated beam (a
    visibility mask on the RAW intensities, so it is defined for power =
    0 as well) are ignored: they are the beams a detector would not see,
    and with power < 1 they would otherwise dominate the unpaired term."""
    d = torch.cdist(sq, qxy)
    d_min, j_min = d.min(dim=1)
    pair = d_min < delta
    if int(pair.sum()) == 0:
        return None, pair, j_min, d_min
    visible = inten > min_sim_rel * inten.amax(dim=2, keepdim=True)
    si = inten.clamp_min(0) ** power
    a = si[:, :, pair]
    b = im[j_min[pair]][None, None, :]
    w = ((a * b).sum(dim=2) / (a * a).sum(dim=2).clamp_min(1e-12)).clamp_min(0)[:, :, None]
    c_paired = (b - w * a).abs().sum(dim=2)
    c_unpaired_sim = 0.5 * w[:, :, 0] * (si[:, :, ~pair] * visible[:, :, ~pair]).sum(dim=2)
    matched = torch.zeros(im.shape[0], dtype=torch.bool)
    matched[j_min[pair]] = True
    # the measured direct beam is not a diffracted intensity: leave it out
    # of the unexplained-measured term and of the normalization
    direct = torch.linalg.norm(qxy, dim=1) < delta
    matched |= direct
    c_unpaired_exp = 0.5 * float(im[~matched].sum())
    norm = float(im[~direct].sum()) + 1e-12
    cost = (c_paired + c_unpaired_sim + c_unpaired_exp) / norm
    return cost, pair, j_min, d_min


def _fit_deformation(sq, qxy, w_exp, delta):
    """Symmetric in-plane deformation S and in-plane rotation angle wz
    (radians) from the paired positions: A = (sum w qm qs^T)(sum w qs qs^T)^-1
    with measured = A ideal, split by polar decomposition A = S Q."""
    d = torch.cdist(sq, qxy)
    d_min, j_min = d.min(dim=1)
    pair = d_min < delta
    if int(pair.sum()) < 3:
        return None, 0.0, pair
    qs = sq[pair]
    qm = qxy[j_min[pair]]
    w = w_exp[j_min[pair]] * (1 - d_min[pair] / delta).clamp_min(0)
    # the paired positions must span the plane: collinear pairs (one
    # systematic row) leave the deformation across the row undetermined
    sv = torch.linalg.svdvals(torch.sqrt(w)[:, None] * qs)
    if float(sv[-1]) < 0.2 * float(sv[0]):
        return None, 0.0, pair
    M1 = torch.einsum("p,pi,pj->ij", w, qm, qs)
    M2 = torch.einsum("p,pi,pj->ij", w, qs, qs)
    A = M1 @ torch.linalg.inv(M2 + 1e-12 * torch.eye(2, dtype=torch.float64))
    U_, _, Vh_ = torch.linalg.svd(A)
    Q = U_ @ Vh_
    if torch.linalg.det(Q) < 0:
        return None, 0.0, pair
    S = A @ Q.T
    S = 0.5 * (S + S.T)
    wz = float(torch.atan2(Q[1, 0], Q[0, 0]))
    return S, wz, pair


# matched orientations of two positions closer than this (degrees) belong to
# one grain for the neighbor rescue: above the error of kinematical matching
# (a few tenths of a degree), below typical grain boundary angles
_RESCUE_SAME_GRAIN_DEG = 2.0


def _closest_symmetry_variant(q: torch.Tensor, ref: torch.Tensor, sym_quats) -> torch.Tensor:
    """The symmetry equivalent q * s of orientation q closest to ref."""
    if sym_quats is None:
        return q
    from quantem.diffraction.rotations import qmult

    variants = qmult(q[None], torch.as_tensor(sym_quats, dtype=q.dtype))  # (S, 4)
    return variants[int((variants @ ref).abs().argmax())]


def _tilt_twist(dq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a lab-frame rotation into dq = tilt * twist: the twist is a
    rotation about the beam (z), the tilt one about an in-plane axis.

    Returns the tilt as its rotation vector (wx, wy) in radians, and the
    twist quaternion."""
    from quantem.diffraction.rotations import qconj, qmult

    dq = dq / torch.linalg.norm(dq)
    n = float(torch.hypot(dq[0], dq[3]))
    if n < 1e-12:
        twist = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dq.dtype)
    else:
        twist = torch.stack([dq[0], torch.zeros_like(dq[0]), torch.zeros_like(dq[0]), dq[3]]) / n
    swing = qmult(dq, qconj(twist))
    if swing[0] < 0:
        swing = -swing
    sin_half = float(torch.linalg.norm(swing[1:3]))
    if sin_half < 1e-15:
        return torch.zeros(2, dtype=dq.dtype), twist
    angle = 2 * np.arctan2(sin_half, float(swing[0]))
    return swing[1:3] / sin_half * angle, twist


def refine_dynamical(
    phase_map,
    thicknesses_A: np.ndarray | None = None,
    tilt_stages=((0.3, 0.1), (0.1, 0.025), (0.03, 0.006)),
    precession_deg: float | None = None,
    n_precession: int = 32,
    n_precession_search: int | None = None,
    semiconv_mrad: float | None = None,
    n_disk_radial: int = 4,
    n_disk_azimuthal: int = 16,
    maped_tilts_deg=None,
    refine_deformation: bool = True,
    pair_distance: float | None = None,
    power_intensity: float | None = None,
    min_sim_intensity_rel: float | None = None,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
    min_number_peaks: int | None = None,
    mask: np.ndarray | None = None,
    fast_absorption: bool = True,
    update_orientations: bool = True,
    require_phase_weight: bool = True,
    warm_start: bool = True,
    neighbor_rescue: bool = True,
    rescue_thickness_A: float = 100.0,
    rescue_tilt_deg: float = 0.05,
    rescue_max_starts: int = 2,
    num_workers: int | None = None,
    progress_bar: bool = True,
) -> dict:
    """Dynamical refinement on the Bragg vectors: orientation, thickness,
    in-plane deformation and candidate, pixel by pixel.

    Starting from the kinematically matched orientation of each candidate,
    the crystal is re-initialized at every trial orientation of a
    coarse-to-fine tilt grid and its diffracted intensities computed with
    Bloch waves, averaged over the precession ring, the convergence disk or
    the MAPED tilt list, for all thicknesses at once (one batched
    eigendecomposition per stage). The peak pairing is fixed by the
    positions, which the tilt does not move; the intensity cost is
    minimized over (tilt, thickness), and the tilt is interpolated
    parabolically at the finest stage. At the refined orientation the
    symmetric in-plane deformation of the tilted cell is solved in closed
    form from the paired positions (weighted least squares), and its
    antisymmetric part, an in-plane rotation, is folded into the
    orientation. The candidate with the lowest cost decides the phase.

    The intensities are far more tilt-sensitive than the positions: at
    500 A the rocking curve width is ~2e-3 1/A, so a 0.05 degree tilt
    error is already visible in the weak beams. The default stages search
    +-0.3 degrees at 0.1, +-0.1 at 0.025 and +-0.03 at 0.006 degrees, 251
    trial orientations per candidate, and should start from orientations
    refined by refine_orientations(). The compute goes into the Bloch
    eigensolves, one per trial orientation, illumination node and
    candidate; the search stages use fewer precession nodes and the
    first-order absorption, and the reported solution is then evaluated
    once with the full node count and the exact absorption, so the stored
    cost, thickness and orientation are at full accuracy while the search
    costs a fraction of a full-accuracy grid.

    Parameters left as None inherit from the previous stages: the
    pairing distance, intensity power, weak-beam cut and peak minimum from
    the phase fit, and the precession and convergence angles from the
    OrientationMaps (from_vectors). The resolved values are recorded in
    phase_map.metadata['dynamical'] and returned under 'metadata'.

    Parameters
    ----------
    phase_map : PhaseMap
        A fitted PhaseMap (fit() has been run).
    thicknesses_A : np.ndarray | None
        Thickness grid in Angstroms; default 50 to 2000 in 25 A steps (the
        thickness axis is nearly free: all thicknesses come from one
        eigendecomposition).
    tilt_stages : sequence of (half_range_deg, step_deg)
        Successive tilt grids, each centered on the previous optimum.
    precession_deg, n_precession : float | None, int
        Precession semi-angle (inherited from the OrientationMap) and the
        number of azimuthal samples on the ring for the last stage and the
        final evaluation. Uniform sampling is the Gauss-Chebyshev
        quadrature of the ring integral; the rocking curves oscillate at
        pi t rho k0 g along the ring, so 32 or more samples are needed at
        500 A and 0.5 degrees.
    n_precession_search : int | None
        Ring samples for the search stages before the last one; defaults
        to half of n_precession (at least 8). The coarse stages only have
        to find the basin, which the reduced sampling does.
    semiconv_mrad : float | None
        Convergence semiangle (inherited); the intensities are averaged
        over the disk with n_disk_radial Gauss-Legendre radii times
        n_disk_azimuthal azimuths (64 nodes by default, times the ring).
    n_disk_radial, n_disk_azimuthal : int, default=4, 16
        Convergence disk sampling, see semiconv_mrad.
    maped_tilts_deg : array-like | None
        Explicit (M, 2) beam tilt list (degrees) for MAPED, overriding
        precession.
    pair_distance : float | None
        Largest distance (1/Angstroms) at which a simulated and a measured
        peak are paired; inherited from the phase fit.
    power_intensity : float | None
        Intensities are compared as I ** power_intensity; inherited from
        the phase fit.
    min_sim_intensity_rel : float | None
        Unpaired simulated beams weaker than this fraction of the
        strongest simulated beam do not count against a candidate (the
        detector would not have seen them). Inherited from the phase fit,
        else MIN_SIM_INTENSITY_REL (0.02).
    sg_max : float, default=SG_MAX
        Excitation error cutoff (1/Angstroms) of the Bloch beam list,
        widened by the tilt search range and the illumination.
    k_max : float | None
        Largest |g| (1/Angstroms) of a beam; None keeps every reflection
        within the cutoff. Recorded in the metadata, so the image
        refinement uses the same beam set.
    min_number_peaks : int | None
        Positions with fewer measured peaks, direct beam included, are
        skipped. None inherits the minimum of the phase fit, itself the
        matching's (5 by default). At least 3: the direct beam and two
        non-collinear reflections.
    refine_deformation : bool, default=True
        Solve the symmetric in-plane deformation and the in-plane rotation
        from the paired positions before the intensity search; the
        deformation is applied to the tilted cell in the Bloch calculation
        and the rotation folded into the orientation.
    mask : np.ndarray | None
        Positions to refine: an (R, C) boolean mask or a list of
        (row, col), as for OrientationMap.match_orientations. None
        (default) refines every position the orientation maps reached, so
        a staged test run on a few positions carries through.
    fast_absorption : bool, default=True
        First-order treatment of absorption during the search (Hermitian
        eigh, ~4x faster, 0.5% rms intensity error); the final evaluation
        of the reported solution always uses the exact complex absorption.
        The tilt and thickness are never treated perturbatively.
    update_orientations : bool, default=True
        Write the refined quaternions back into the OrientationMaps.
    require_phase_weight : bool, default=True
        Refine only candidates that carried weight in the kinematical
        phase fit; False refines every matched candidate, so the dynamical
        pass can rescue a candidate the kinematical model rejected.
    warm_start : bool, default=True
        Start each position from the refined orientation of an already
        refined neighbor (above or to the left, same candidate) and skip
        the coarsest stage. Neighbors within one grain share their
        orientation to well inside the fine stages, so this removes about
        half of the eigensolves; the in-plane deformation and rotation are
        still fit from the position's own peaks, and the final evaluation
        is unchanged. The reported tilt and zero-tilt cost still refer to
        the position's own matched orientation.
    neighbor_rescue : bool, default=True
        Second pass: positions whose winning solution differs from a
        4-neighbor in the same grain (same crystal, matched orientations
        within 2 degrees; the nearest refined position
        within two steps, so a mask of every second position works too) by
        more than rescue_thickness_A in thickness or rescue_tilt_deg in
        orientation are refined again from that neighbor's solution, and
        the lower cost is kept. Repairs isolated wrong basins (thickness
        aliases, tilt minima at a grid edge).
    rescue_thickness_A : float, default=100.0
        Thickness difference (Angstroms) to a neighbor that triggers a
        rescue.
    rescue_tilt_deg : float, default=0.05
        Misorientation (degrees) between the refined orientations of a
        position and a neighbor that triggers a rescue. Neighbors in one
        grain differ by the true orientation gradient, so keep it above
        that.
    rescue_max_starts : int, default=2
        Neighbor solutions tried per rescued position, lowest cost first,
        skipping neighbors whose solution repeats one already tried.
    num_workers : int | None
        Threads refining positions side by side; None uses every core.
        The Bloch eigensolves are too small to spread over cores on their
        own, so this is where the speed comes from. Positions are handed
        out in contiguous raster-order blocks and warm starts stay inside a
        block, so the result depends on the number of blocks, never on
        which thread finishes first.
    progress_bar : bool, default=True
        Show progress bars over positions and rescues.

    Returns
    -------
    dict
        Per position (R, C) at the winning candidate: 'thickness',
        'thickness_contrast' (range of the final cost over the thickness
        grid; a small value means the thickness is not determined) and
        'tilt_deg' (R, C, 2), the tilt about the lab x and y axes (degrees)
        from 'quats_base' to 'quats'. Per candidate (R, C, F): 'quats'
        (..., 4) refined orientations; 'quats_base' (..., 4) the matched
        orientation with the in-plane rotation of the refinement folded in
        (a symmetry equivalent of it closest to the solution), so quats =
        tilt x quats_base whether or not the position was warm started or
        rescued; 'deformation' (..., 2, 2) symmetric in-plane deformation
        A of the tilted cell in the calibrated frame (measured reciprocal
        positions = A x ideal); 'cost'; 'cost_zero_tilt', the best cost
        over thickness at 'quats_base', evaluated like the final cost (its
        difference to 'cost' is the gain of the tilt search; a small gain
        means the intensities do not constrain the tilt);
        'thickness_per_candidate'; 'warm_started' flags. Also 'rescued'
        (R, C) flags, 'phase_index' (R, C) the winning crystal and
        'candidate' (R, C) the winning candidate (both -1 where nothing
        was refined), and 'metadata', the resolved parameters. Values are
        NaN where a candidate was not refined.
    """
    from quantem.diffraction.rotations import (
        misorientation_angle_deg,
        qconj,
        qmult,
        quat_from_axis_angle,
    )

    if thicknesses_A is None:
        thicknesses_A = np.arange(50.0, 2000.0 + 1e-6, 25.0)
    t_grid = torch.as_tensor(thicknesses_A, dtype=torch.float64)
    T = t_grid.shape[0]

    oms = phase_map.orientation_maps
    fit_md = phase_map.metadata.get("fit") if hasattr(phase_map, "metadata") else None
    om_md = oms[0].metadata if hasattr(oms[0], "metadata") else None
    pair_distance = resolve(pair_distance, "pair_distance", fit_md, default=PAIR_DISTANCE)
    power_intensity = resolve(power_intensity, "power_intensity", fit_md, default=POWER_INTENSITY)
    min_sim_intensity_rel = resolve(
        min_sim_intensity_rel, "min_sim_intensity_rel", fit_md, default=MIN_SIM_INTENSITY_REL
    )
    min_number_peaks = int(
        resolve(min_number_peaks, "min_number_peaks", fit_md, default=MIN_NUMBER_PEAKS)
    )
    if min_number_peaks < 3:
        raise ValueError(
            f"min_number_peaks={min_number_peaks}: a dynamical fit needs at least the "
            "direct beam and two non-collinear reflections"
        )
    precession_deg = float(resolve(precession_deg, "precession_deg", om_md, default=0.0))
    semiconv_mrad = float(resolve(semiconv_mrad, "semiconv_mrad", om_md, default=0.0))
    if n_precession_search is None:
        n_precession_search = max(8, n_precession // 2)
    used = dict(
        thicknesses_A=np.asarray(thicknesses_A, dtype=float).tolist(),
        tilt_stages=[tuple(float(v) for v in st) for st in tilt_stages],
        precession_deg=precession_deg,
        n_precession=int(n_precession),
        n_precession_search=int(n_precession_search),
        semiconv_mrad=semiconv_mrad,
        n_disk_radial=int(n_disk_radial),
        n_disk_azimuthal=int(n_disk_azimuthal),
        maped_tilts_deg=maped_tilts_deg,
        refine_deformation=bool(refine_deformation),
        pair_distance=float(pair_distance),
        power_intensity=float(power_intensity),
        min_sim_intensity_rel=float(min_sim_intensity_rel),
        sg_max=float(sg_max),
        k_max=k_max,
        min_number_peaks=int(min_number_peaks),
        fast_absorption=bool(fast_absorption),
        require_phase_weight=bool(require_phase_weight),
        warm_start=bool(warm_start),
        neighbor_rescue=bool(neighbor_rescue),
        rescue_thickness_A=float(rescue_thickness_A),
        rescue_tilt_deg=float(rescue_tilt_deg),
        rescue_max_starts=int(rescue_max_starts),
        num_workers=None if num_workers is None else int(num_workers),
    )
    if hasattr(phase_map, "metadata"):
        phase_map.metadata["dynamical"] = used
    cands = phase_map.candidates
    peaks = oms[0].peaks
    R, C = peaks.shape[0], peaks.shape[1]
    F = len(cands)
    delta = pair_distance
    energy_ev = oms[0].energy_ev
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    fields = peaks.fields
    ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

    ring, w_ring = illumination_nodes(
        energy_ev,
        precession_deg,
        n_precession,
        semiconv_mrad,
        n_disk_radial,
        n_disk_azimuthal,
        maped_tilts_deg=maped_tilts_deg,
    )  # (Mr, 2), (Mr,)
    Mr = ring.shape[0]
    alpha_ill = float(torch.linalg.norm(ring, dim=1).max()) / k0
    ring_s, w_ring_s = illumination_nodes(
        energy_ev,
        precession_deg,
        n_precession_search,
        semiconv_mrad,
        n_disk_radial,
        n_disk_azimuthal,
        maped_tilts_deg=maped_tilts_deg,
    )
    Mr_s = ring_s.shape[0]

    def stage_grid(center, half, step):
        n = int(round(2 * half / step)) + 1
        tg = torch.linspace(-half, half, n, dtype=torch.float64)
        wx_g, wy_g = torch.meshgrid(tg, tg, indexing="ij")
        w = torch.stack([wx_g.reshape(-1), wy_g.reshape(-1)], dim=1) + center[None, :]
        return w, n, tg

    cost_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)
    cost0_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)
    thick_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)
    tcontrast_out = torch.full((R, C, F), torch.nan, dtype=torch.float64)
    tilt_out = torch.zeros((R, C, F, 2), dtype=torch.float64)
    quat_out = torch.zeros((R, C, F, 4), dtype=torch.float64)
    quat_out[..., 0] = 1.0
    quat_base = quat_out.clone()
    deform_out = torch.zeros((R, C, F, 2, 2), dtype=torch.float64)
    deform_out[..., 0, 0] = 1.0
    deform_out[..., 1, 1] = 1.0
    warm_out = torch.zeros((R, C, F), dtype=torch.bool)
    rescued_out = torch.zeros((R, C), dtype=torch.bool)

    def refine_from(crystal, q_start, q_match, qxy, im, w_exp, stages):
        """Search from q_start: in-plane deformation and rotation from the
        positions, one beam list, the tilt stages, and the exact final
        evaluation. q_match is the kinematically matched orientation of the
        position, the reference of the reported tilt and of the zero-tilt
        cost (q_start differs from it on a warm start or a rescue). Returns
        None or a dict with the solution."""
        q0 = q_start
        S = None
        deform3 = None
        if refine_deformation:
            # in-plane deformation and rotation from the positions first:
            # the rotation is folded into the orientation and the
            # symmetric deformation applied to the tilted cell, so the
            # intensity search below sees the strained lattice and a
            # pairing free of position residuals
            g0 = qrotate(q0, crystal.g_vec)
            near = (
                torch.abs((2 * g0[:, 2] - lam * (g0**2).sum(1)) / (2 - 2 * lam * g0[:, 2]))
                < sg_max
            )
            S, wz, _ = _fit_deformation(g0[near, :2], qxy, w_exp, delta)
            if S is not None:
                half_z = torch.tensor(wz / 2, dtype=torch.float64)
                dqz = torch.stack(
                    [torch.cos(half_z), torch.zeros(()), torch.zeros(()), torch.sin(half_z)]
                ).to(torch.float64)
                q0 = qmult(dqz, q0)
                deform3 = torch.eye(3, dtype=torch.float64)
                deform3[:2, :2] = S
        # the matched orientation with the in-plane rotation of q0: the
        # base the reported tilt is measured from (equal to q0 on a cold
        # start), and its tilt away from q0
        q_m = _closest_symmetry_variant(q_match, q0, crystal.sym_quats)
        base_tilt, twist = _tilt_twist(qmult(q0, qconj(q_m)))
        q_base = qmult(twist, q_m)
        offset = float(torch.linalg.norm(base_tilt))
        # one beam list for the whole search of this candidate: every
        # trial center, the base, the illumination and the deformation are
        # inside its selection, so all stages compare the same truncated
        # system
        beam_list = select_dynamical_beams(
            crystal,
            q0,
            energy_ev,
            np.deg2rad(stages[0][0]) * np.sqrt(2) + alpha_ill + offset,
            sg_max,
            k_max,
            deform3,
        )
        if beam_list.shape[0] < 2:
            return None

        def exact_cost(q):
            # full illumination and exact absorption at one orientation
            inten, g_xy, _ = _cbed_amplitudes(
                crystal,
                q,
                ring,
                t_grid,
                energy_ev,
                sg_max,
                k_max,
                tilt_batch=max(64, Mr * 8),
                progress_bar=False,
                fast_absorption=False,
                deform=deform3,
                beams=beam_list,
            )
            inten = (inten * w_ring[:, None, None]).sum(dim=0, keepdim=True)
            cost, _, _, _ = _dynamical_cost(
                inten[:, :, 1:], g_xy[1:], qxy, im, delta, power_intensity, min_sim_intensity_rel
            )
            return cost

        # untilted reference: the best thickness at the matched orientation,
        # for the gain the tilt search achieves
        cost_base = exact_cost(q_base)
        cost0 = float("nan") if cost_base is None else float(cost_base[0].min())
        center = torch.zeros(2, dtype=torch.float64)
        best = None
        n_stages = len(stages)
        for i_stage, (half, step) in enumerate(stages):
            half = np.deg2rad(half)
            step = np.deg2rad(step)
            w_grid, n, tg = stage_grid(center, half, step)
            Mt = w_grid.shape[0]
            last = i_stage == n_stages - 1
            nodes, w_nodes, M_nodes = (ring, w_ring, Mr) if last else (ring_s, w_ring_s, Mr_s)
            # crystal tilt (wx, wy) about the in-plane axes shifts s_g by
            # wx g_y - wy g_x; the same excitation errors come from a beam
            # tilt k0 (wy, -wx) in the fixed-normal Bloch geometry, so every
            # trial orientation is a full re-solve of the Bloch problem
            # with the coupling matrix shared
            trial = k0 * torch.stack([w_grid[:, 1], -w_grid[:, 0]], dim=1)
            tilts = (trial[:, None, :] + nodes[None, :, :]).reshape(-1, 2)
            inten, g_xy, _ = _cbed_amplitudes(
                crystal,
                q0,
                tilts,
                t_grid,
                energy_ev,
                sg_max,
                k_max,
                tilt_batch=max(64, M_nodes * 8),
                progress_bar=False,
                fast_absorption=fast_absorption,
                deform=deform3,
                beams=beam_list,
            )
            inten = (inten.reshape(Mt, M_nodes, T, -1) * w_nodes[None, :, None, None]).sum(dim=1)
            sq = g_xy[1:]
            if sq.shape[0] == 0:
                break
            cost, pair, j_min, d_min = _dynamical_cost(
                inten[:, :, 1:], sq, qxy, im, delta, power_intensity, min_sim_intensity_rel
            )
            if cost is None:
                break
            flat = int(cost.argmin())
            m_best, t_best = flat // T, flat % T
            i_b, j_b = m_best // n, m_best % n
            cost_t = cost[:, t_best].reshape(n, n)
            wx, wy = float(w_grid[m_best, 0]), float(w_grid[m_best, 1])
            if 0 < i_b < n - 1:
                c0, c1, c2 = cost_t[i_b - 1, j_b], cost_t[i_b, j_b], cost_t[i_b + 1, j_b]
                den = float(c0 - 2 * c1 + c2)
                if den > 1e-12:
                    wx += 0.5 * float(c0 - c2) / den * step
            if 0 < j_b < n - 1:
                c0, c1, c2 = cost_t[i_b, j_b - 1], cost_t[i_b, j_b], cost_t[i_b, j_b + 1]
                den = float(c0 - 2 * c1 + c2)
                if den > 1e-12:
                    wy += 0.5 * float(c0 - c2) / den * step
            center = torch.tensor([wx, wy], dtype=torch.float64)
            best = (float(cost[m_best, t_best]), float(t_grid[t_best]), wx, wy)
        if best is None:
            return None
        c_best, t_fit, wx, wy = best
        q = q0
        ang = float(np.hypot(wx, wy))
        if ang > 1e-12:
            axis = torch.tensor([wx / ang, wy / ang, 0.0], dtype=torch.float64)
            q = qmult(quat_from_axis_angle(axis, torch.tensor(ang, dtype=torch.float64)), q0)
        # the reported solution is the crystal rotated by the interpolated
        # tilt with the beam along the foil normal: evaluate that model once
        # more, with the full illumination and the exact absorption, so the
        # stored cost and thickness belong to the stored orientation at full
        # accuracy (the beam-offset search geometry is paraxially, not
        # exactly, equivalent to it)
        cost = exact_cost(q)
        t_contrast = float("nan")
        if cost is not None:
            t_best = int(cost[0].argmin())
            c_best, t_fit = float(cost[0, t_best]), float(t_grid[t_best])
            # how much the cost varies over the thickness grid at this
            # orientation: a flat curve means the thickness is not
            # determined by these intensities (precession, few beams)
            t_contrast = float(cost[0].max() - cost[0].min())
        # the reported tilt and base split the rotation from the matched
        # orientation exactly, q = tilt x base; on a cold start they are the
        # search's own (wx, wy) and q0, on a warm start the base differs
        # from the zero-tilt one above only at second order in the tilt
        tilt, twist = _tilt_twist(qmult(q, qconj(q_m)))
        q_base = qmult(twist, q_m)
        return dict(
            cost=c_best,
            t=t_fit,
            wx=float(tilt[0]),
            wy=float(tilt[1]),
            q=q,
            q_base=q_base,
            S=S,
            cost0=cost0,
            t_contrast=t_contrast,
        )

    def store(rx, ry, f, sol):
        cost_out[rx, ry, f] = sol["cost"]
        thick_out[rx, ry, f] = sol["t"]
        tcontrast_out[rx, ry, f] = sol.get("t_contrast", float("nan"))
        tilt_out[rx, ry, f, 0] = sol["wx"]
        tilt_out[rx, ry, f, 1] = sol["wy"]
        quat_base[rx, ry, f] = sol["q_base"]
        quat_out[rx, ry, f] = sol["q"]
        if sol["S"] is not None:
            deform_out[rx, ry, f] = sol["S"]
        if np.isfinite(sol["cost0"]):
            cost0_out[rx, ry, f] = sol["cost0"]

    def peaks_at(rx, ry):
        data = peaks[rx, ry].numpy().astype(np.float64)
        if data.shape[0] < min_number_peaks:
            return None
        qxy = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
        im = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0) ** power_intensity
        return qxy, im, im / im.max().clamp_min(1e-12)

    from quantem.diffraction.orientation import position_mask

    mask_rc = position_mask(mask, (R, C))
    for om in oms:
        if om.computed is not None:
            mask_rc = mask_rc & om.computed
    positions = [(r, c) for r, c in np.ndindex(R, C) if mask_rc[r, c]]

    def refine_position(rx, ry, block):
        pk = peaks_at(rx, ry)
        if pk is None:
            return
        qxy, im, w_exp = pk
        for f, (i_om, m) in enumerate(cands):
            om = oms[i_om]
            if om.corr[rx, ry, m] <= 0:
                continue
            if (
                require_phase_weight
                and phase_map.phase_weights is not None
                and float(phase_map.phase_weights[rx, ry, f]) <= 0
            ):
                continue
            q_start = om.quats[rx, ry, m]
            quat_out[rx, ry, f] = q_start
            stages = tilt_stages
            if warm_start and len(tilt_stages) > 1:
                # an already refined neighbor of the same candidate (raster
                # order: above or to the left, in the same block, one or two
                # steps away so a mask of every second position still warm
                # starts) is a start inside the fine stages' reach; its
                # solution costs one coarse stage less
                for nr, nc in ((rx - 1, ry), (rx, ry - 1), (rx - 2, ry), (rx, ry - 2)):
                    if (nr, nc) not in block:
                        continue
                    if not torch.isfinite(cost_out[nr, nc, f]):
                        continue
                    # same grain: the kinematically matched orientations of
                    # the two positions agree within the coarse stage
                    miso = float(
                        misorientation_angle_deg(
                            om.quats[rx, ry, m][None],
                            om.quats[nr, nc, m][None],
                            om.crystal.sym_quats,
                        )[0]
                    )
                    if miso < tilt_stages[0][0]:
                        q_start = quat_out[nr, nc, f]
                        stages = tilt_stages[1:]
                        warm_out[rx, ry, f] = True
                        break
            sol = refine_from(om.crystal, q_start, om.quats[rx, ry, m], qxy, im, w_exp, stages)
            if sol is None:
                continue
            store(rx, ry, f, sol)

    def run_parallel(jobs, work, desc):
        """Run work(job) over jobs on num_workers threads. Each Bloch
        eigensolve is too small to use more than one core, and torch
        releases the GIL inside it, so positions run side by side; every
        job writes only its own positions."""
        bar = tqdm(total=sum(len(j) for j in jobs), desc=desc) if progress_bar else None
        lock = threading.Lock()

        def run(job):
            for item in job:
                work(item)
                if bar is not None:
                    with lock:
                        bar.update(1)

        if n_workers == 1:
            for job in jobs:
                run(job)
        else:
            # one intra-op thread per worker: the worker threads already
            # fill the cores, and torch's own pool on top of them would
            # oversubscribe. The setting is process-global, so the previous
            # value is restored afterwards.
            n_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with ThreadPoolExecutor(n_workers) as pool:
                    for fut in [pool.submit(run, job) for job in jobs]:
                        fut.result()
            finally:
                torch.set_num_threads(n_threads)
        if bar is not None:
            bar.close()

    n_workers = max(1, int(num_workers if num_workers is not None else os.cpu_count() or 1))
    # fill the lazily cached lattice data once, before any thread reads it
    for om in oms:
        _beam_universe(om.crystal)
    # contiguous runs of positions in raster order, several per worker so
    # they balance but long enough that most positions still warm start;
    # warm starts stay inside a run, so the result does not depend on which
    # thread finished first
    n_blocks = 1 if n_workers == 1 else max(1, min(4 * n_workers, len(positions) // 16))
    blocks = []
    for idx in np.array_split(np.arange(len(positions)), max(n_blocks, 1)):
        block = {positions[i] for i in idx}
        blocks.append([(*positions[i], block) for i in idx])
    run_parallel(
        [b for b in blocks if b], lambda item: refine_position(*item), "dynamical refinement"
    )

    if neighbor_rescue and len(tilt_stages) > 1:
        cost_f0 = torch.nan_to_num(cost_out, nan=torch.inf)
        f_win = cost_f0.argmin(dim=-1)
        done = torch.isfinite(cost_out).any(dim=-1)
        rescue_list = []

        def miso(r0, c0, f0, r1, c1, f1):
            # misorientation (degrees) of two refined solutions of the same
            # crystal: the tilt corrections have different bases (the
            # matched orientations of the two positions), the solutions not
            return float(
                misorientation_angle_deg(
                    quat_out[r0, c0, f0][None],
                    quat_out[r1, c1, f1][None],
                    oms[cands[f0][0]].crystal.sym_quats,
                )[0]
            )

        def nearest_done(rx, ry, dr, dc):
            # the refined position one step away, or two on a sparse mask
            for k in (1, 2):
                nr, nc = rx + k * dr, ry + k * dc
                if 0 <= nr < R and 0 <= nc < C and done[nr, nc]:
                    return nr, nc
            return None

        for rx, ry in np.ndindex(R, C):
            if not done[rx, ry]:
                continue
            f = int(f_win[rx, ry])
            i_om = cands[f][0]
            starts = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = nearest_done(rx, ry, dr, dc)
                if nb is None:
                    continue
                nr, nc = nb
                fn = int(f_win[nr, nc])
                if cands[fn][0] != i_om:
                    continue
                # same grain only: a start from another grain is no rescue,
                # and its tilt from the matched orientation would widen the
                # beam list without bound
                if (
                    misorientation_angle_deg(
                        oms[i_om].quats[rx, ry, cands[f][1]][None],
                        oms[i_om].quats[nr, nc, cands[fn][1]][None],
                        oms[i_om].crystal.sym_quats,
                    )[0]
                    >= _RESCUE_SAME_GRAIN_DEG
                ):
                    continue
                dt = abs(float(thick_out[nr, nc, fn]) - float(thick_out[rx, ry, f]))
                if dt > rescue_thickness_A or miso(nr, nc, fn, rx, ry, f) > rescue_tilt_deg:
                    starts.append((float(cost_out[nr, nc, fn]), nr, nc, fn))
            if starts:
                # lowest-cost neighbors first, one start per distinct
                # solution, at most rescue_max_starts (each start is a full
                # fine-stage search)
                starts.sort(key=lambda x: x[0])
                kept: list = []
                for c_n, nr, nc, fn in starts:
                    dup = False
                    for _, kr, kc, kf in kept:
                        if (
                            abs(float(thick_out[nr, nc, fn]) - float(thick_out[kr, kc, kf]))
                            <= rescue_thickness_A
                            and miso(nr, nc, fn, kr, kc, kf) <= rescue_tilt_deg
                        ):
                            dup = True
                            break
                    if not dup:
                        kept.append((c_n, nr, nc, fn))
                    if len(kept) >= max(1, rescue_max_starts):
                        break
                # the neighbors' solutions as they stand now: rescues run
                # in parallel and must not start from each other's updates
                rescue_list.append(
                    (rx, ry, f, [quat_out[nr, nc, fn].clone() for _, nr, nc, fn in kept])
                )

        def rescue(item):
            rx, ry, f, starts = item
            pk = peaks_at(rx, ry)
            if pk is None:
                return
            qxy, im, w_exp = pk
            i_om, m = cands[f]
            crystal = oms[i_om].crystal
            q_match = oms[i_om].quats[rx, ry, m]
            for q_n in starts:
                sol = refine_from(crystal, q_n, q_match, qxy, im, w_exp, tilt_stages[1:])
                if sol is not None and sol["cost"] < float(cost_out[rx, ry, f]) - 1e-9:
                    store(rx, ry, f, sol)
                    rescued_out[rx, ry] = True

        n_jobs = 1 if n_workers == 1 else 4 * n_workers
        run_parallel(
            [rescue_list[k::n_jobs] for k in range(n_jobs) if rescue_list[k::n_jobs]],
            rescue,
            "neighbor rescue",
        )

    n_maps = len(oms)
    cost_f = torch.nan_to_num(cost_out, nan=torch.inf)
    cost_phase = torch.full((R, C, n_maps), torch.inf, dtype=torch.float64)
    for f, (i_om, _) in enumerate(cands):
        cost_phase[..., i_om] = torch.minimum(cost_phase[..., i_om], cost_f[..., f])
    done = torch.isfinite(cost_out).any(dim=-1)
    phase_index = torch.where(done, cost_phase.argmin(dim=-1), -1)
    f_best = cost_f.argmin(dim=-1)
    thickness = torch.gather(thick_out, 2, f_best[..., None]).squeeze(-1)
    thickness_contrast = torch.gather(tcontrast_out, 2, f_best[..., None]).squeeze(-1)
    tilt_deg = torch.rad2deg(
        torch.gather(tilt_out, 2, f_best[..., None, None].expand(R, C, 1, 2)).squeeze(2)
    )
    tilt_deg[~done] = torch.nan

    if update_orientations:
        for f, (i_om, m) in enumerate(cands):
            done = torch.isfinite(cost_out[..., f])
            oms[i_om].quats[..., m, :][done] = quat_out[..., f, :][done]

    return {
        "thickness": thickness,
        "thickness_contrast": thickness_contrast,
        "tilt_deg": tilt_deg,
        "quats": quat_out,
        "deformation": deform_out,
        "cost": cost_out,
        "cost_zero_tilt": cost0_out,
        "quats_base": quat_base,
        "warm_started": warm_out,
        "rescued": rescued_out,
        "phase_index": phase_index,
        "candidate": torch.where(done, f_best, -1),
        "thickness_per_candidate": thick_out,
        "metadata": used,
    }


def dynamical_maps(
    result: dict,
    phase_map,
    crystal_index: int | None = None,
    min_thickness_contrast: float = 0.02,
) -> dict:
    """Maps of the winning candidate of a refine_dynamical() result.

    Parameters
    ----------
    result : dict
        From refine_dynamical().
    phase_map : PhaseMap
        The PhaseMap that was refined (for the candidate list).
    crystal_index : int | None
        Keep only positions won by this crystal; None keeps all.
    min_thickness_contrast : float, default=0.02
        The thickness is NaN where the thickness contrast is below this: a
        flat cost curve, typical of precessed data with few beams, does not
        determine the thickness and the grid minimum there is not a
        measurement.

    Returns
    -------
    dict
        (R, C) maps 'thickness' (Angstroms), 'tilt_deg' (magnitude of the
        tilt from the matched orientation, degrees), 'gain' (cost at the
        matched orientation minus the final cost), 'cost',
        'thickness_contrast', 'phase_index' (-1 outside the mask), 'mask'
        (positions refined, and of the given crystal when crystal_index is
        set), 'quats' (R, C, 4), 'deformation' (R, C, 2, 2) and 'strain',
        the crystal-frame strain components of strain_crystal_frame().
        Positions outside the mask are NaN.
    """
    cand = result["candidate"]
    R, C = cand.shape
    refined = cand >= 0
    cand = cand.clamp_min(0)
    idx4 = cand[..., None, None]
    quats = torch.gather(result["quats"], 2, idx4.expand(R, C, 1, 4)).squeeze(2)
    deform = torch.gather(
        result["deformation"], 2, cand[..., None, None, None].expand(R, C, 1, 2, 2)
    ).squeeze(2)
    cost = torch.gather(result["cost"], 2, cand[..., None]).squeeze(-1)
    cost0 = torch.gather(result["cost_zero_tilt"], 2, cand[..., None]).squeeze(-1)
    tcon = result.get("thickness_contrast")
    if tcon is None:
        tcon = torch.full_like(cost, torch.nan)
    mask = refined & torch.isfinite(cost)
    if crystal_index is not None:
        i_om = torch.tensor([c[0] for c in phase_map.candidates])
        mask &= i_om[cand] == crystal_index
    nan = torch.full((R, C), torch.nan, dtype=torch.float64)
    strain = strain_crystal_frame(deform, quats)
    out = {
        "thickness": torch.where(
            mask & ~(tcon < min_thickness_contrast), result["thickness"], nan
        ),
        "tilt_deg": torch.where(mask, torch.linalg.norm(result["tilt_deg"], dim=-1), nan),
        "gain": torch.where(mask, cost0 - cost, nan),
        "thickness_contrast": torch.where(mask, tcon, nan),
        "cost": torch.where(mask, cost, nan),
        "phase_index": torch.where(mask, result["phase_index"], -1),
        "mask": mask,
        "quats": torch.where(mask[..., None], quats, torch.nan),
        "deformation": torch.where(mask[..., None, None], deform, torch.nan),
        "strain": {k: torch.where(mask, v, nan) for k, v in strain.items() if k != "eps_crystal"},
    }
    return out


def plot_dynamical_maps(
    maps: dict,
    scalebar=None,
    thickness_range_A: tuple[float, float] = (0.0, 2000.0),
    tilt_range_deg: tuple[float, float] = (0.0, 0.3),
    gain_range: tuple[float, float] = (0.0, 0.05),
    axsize: tuple[float, float] = (4.0, 4.0),
):
    """Thickness, tilt correction, gain of the tilt search and final cost
    of a dynamical refinement.

    Parameters
    ----------
    maps : dict
        From dynamical_maps().
    scalebar : dict | None
        Passed to show_2d.
    thickness_range_A : tuple[float, float], default=(0.0, 2000.0)
        Color range of the thickness map (Angstroms).
    tilt_range_deg : tuple[float, float], default=(0.0, 0.3)
        Color range of the tilt map (degrees).
    gain_range : tuple[float, float], default=(0.0, 0.05)
        Color range of the gain map.
    axsize : tuple[float, float], default=(4.0, 4.0)
        Size of each panel in inches.

    Returns
    -------
    fig, axs
        From show_2d. NaN positions are shown as zero.
    """
    from quantem.core.visualization import show_2d

    imgs = [
        [np.nan_to_num(maps["thickness"].numpy()), np.nan_to_num(maps["tilt_deg"].numpy())],
        [np.nan_to_num(maps["gain"].numpy()), np.nan_to_num(maps["cost"].numpy())],
    ]
    cmax = (
        float(np.nanmax(maps["cost"].numpy())) if np.isfinite(maps["cost"].numpy()).any() else 1.0
    )
    return show_2d(
        imgs,
        title=[["thickness (A)", "tilt correction (deg)"], ["gain of the tilt search", "cost"]],
        cmap=[["viridis", "magma"], ["magma", "gray_r"]],
        cbar=True,
        norm=[
            [
                {
                    "interval_type": "manual",
                    "vmin": thickness_range_A[0],
                    "vmax": thickness_range_A[1],
                },
                {"interval_type": "manual", "vmin": tilt_range_deg[0], "vmax": tilt_range_deg[1]},
            ],
            [
                {"interval_type": "manual", "vmin": gain_range[0], "vmax": gain_range[1]},
                {"interval_type": "manual", "vmin": 0.0, "vmax": cmax},
            ],
        ],
        scalebar=scalebar,
        axsize=axsize,
    )


def strain_crystal_frame(deformation: torch.Tensor, quats: torch.Tensor) -> dict:
    """Strain tensor components in the crystal Cartesian frame.

    The measured in-plane reciprocal deformation A (2, 2) of the tilted
    cell (measured = A x ideal) is the reciprocal image of the real-space
    deformation F = A^-T restricted to the beam-normal plane; only that
    in-plane part is observable from one projection, and the components
    along the beam are set to zero before the tensor is rotated into the
    crystal frame with the 3x3 orientation matrix R (v_lab = R v_crystal):
    eps_crystal = R^T eps_lab R. The crystal axes are the Cartesian frame
    of the cell (x along a, z along c; for hexagonal cells 'b' is the
    in-basal-plane direction perpendicular to a).

    Parameters
    ----------
    deformation : torch.Tensor
        (..., 2, 2) symmetric in-plane deformation from refine_dynamical
        (or the columns of OrientationMap.calculate_strain's A).
    quats : torch.Tensor
        (..., 4) orientations.

    Returns
    -------
    dict of (...) tensors 'aa', 'bb', 'cc', 'ab', 'ac', 'bc' (strain
    components) and 'eps_crystal' (..., 3, 3).
    """
    from quantem.diffraction.rotations import quat_to_matrix

    A = deformation.to(torch.float64)
    Fp = torch.linalg.inv(A).transpose(-1, -2)  # real-space in-plane deformation
    eps2 = 0.5 * (Fp + Fp.transpose(-1, -2)) - torch.eye(2, dtype=torch.float64)
    eps_lab = torch.zeros(A.shape[:-2] + (3, 3), dtype=torch.float64)
    eps_lab[..., :2, :2] = eps2
    Rm = quat_to_matrix(quats.to(torch.float64))
    eps_c = torch.einsum("...ji,...jk,...kl->...il", Rm, eps_lab, Rm)
    return {
        "aa": eps_c[..., 0, 0],
        "bb": eps_c[..., 1, 1],
        "cc": eps_c[..., 2, 2],
        "ab": eps_c[..., 0, 1],
        "ac": eps_c[..., 0, 2],
        "bc": eps_c[..., 1, 2],
        "eps_crystal": eps_c,
    }


def plot_strain_crystal_frame(
    strain: dict,
    mask: np.ndarray | None = None,
    strain_range_percent: tuple[float, float] = (-2.0, 2.0),
    scalebar=None,
    axsize: tuple[float, float] = (4.0, 4.0),
    cmap: str = "RdBu_r",
):
    """Six strain components in the crystal frame as maps.

    Normal strains along the crystal a, b and c axes on the top row and
    the ab, ac and bc shears below, in percent, masked where the fit is
    not trusted. Components with a c (beam-direction) index are the
    rotated in-plane measurement only; see strain_crystal_frame().

    Parameters
    ----------
    strain : dict
        (R, C) arrays 'aa', 'bb', 'cc', 'ab', 'ac', 'bc', as returned by
        strain_crystal_frame() or dynamical_maps()['strain'].
    mask : np.ndarray | None
        (R, C) weights (0 hides a position); None shows all.
    strain_range_percent : tuple[float, float], default=(-2.0, 2.0)
        Color range in percent.
    scalebar : dict | None
        Passed to show_2d.
    axsize : tuple[float, float], default=(4.0, 4.0)
        Size of each panel in inches.
    cmap : str, default="RdBu_r"
        Colormap.

    Returns
    -------
    fig, axs
        From show_2d.
    """
    from quantem.core.visualization import show_2d

    keys = [["aa", "bb", "cc"], ["ab", "ac", "bc"]]
    names = [["ε_aa", "ε_bb", "ε_cc"], ["ε_ab", "ε_ac", "ε_bc"]]
    m = 1.0 if mask is None else np.asarray(mask, dtype=float)
    imgs = [[np.asarray(strain[k]) * 100 * m for k in row] for row in keys]
    lo, hi = strain_range_percent
    return show_2d(
        imgs,
        title=[[n + " (%)" for n in row] for row in names],
        cmap=cmap,
        cbar=True,
        norm={"interval_type": "manual", "vmin": lo, "vmax": hi},
        scalebar=scalebar,
        axsize=axsize,
    )


# ----------------------------------------------------------------------
# image-based dynamical refinement (the final step, on the pattern pixels)
# ----------------------------------------------------------------------


def _q_to_pixels(
    q_xy: torch.Tensor, origin_rc, pixel_size: float, rotation_ccw_deg: float, ellipse
):
    """Calibrated (qx, qy) [row, col frame] -> detector pixel (row, col):
    undo the scan rotation and the ellipse correction of
    calibration.peaks_to_calibrated, then scale and shift to the origin."""
    q = q_xy.to(torch.float64)
    if rotation_ccw_deg:
        th = np.deg2rad(-rotation_ccw_deg)
        rot = torch.tensor(
            [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=torch.float64
        )
        q = q @ rot.T
    if ellipse is not None:
        e11, e12 = float(ellipse[0]), float(ellipse[1])
        A = torch.tensor([[1 + e11, e12], [e12, 1 - e11]], dtype=torch.float64)
        q = q @ torch.linalg.inv(A).T
    return q / pixel_size + torch.as_tensor(origin_rc, dtype=torch.float64)[None, :]


def render_disks(
    centers_px: torch.Tensor,
    intensities: torch.Tensor,
    shape: tuple[int, int],
    disk_radius_px: float,
    edge_px: float,
) -> torch.Tensor:
    """Sum of soft-edged disks on a pixel grid.

    The edge is a logistic of width edge_px (the disk profile of a
    defocused or blurred aperture).

    Parameters
    ----------
    centers_px : torch.Tensor
        (N, 2) disk centers in pixels, (row, col).
    intensities : torch.Tensor
        (..., N) disk intensities; leading dimensions give a stack.
    shape : tuple[int, int]
        Image shape (ny, nx).
    disk_radius_px : float
        Disk radius in pixels (the logistic's half point).
    edge_px : float
        Edge width in pixels.

    Returns
    -------
    torch.Tensor
        (..., ny, nx) images.
    """
    ny, nx = shape
    rows = torch.arange(ny, dtype=torch.float64)
    cols = torch.arange(nx, dtype=torch.float64)
    d = torch.sqrt(
        (rows[None, :, None] - centers_px[:, 0, None, None]) ** 2
        + (cols[None, None, :] - centers_px[:, 1, None, None]) ** 2
    )  # (N, ny, nx)
    disks = torch.sigmoid((disk_radius_px - d) / max(edge_px, 1e-3))
    return torch.einsum("...n,nyx->...yx", intensities.to(torch.float64), disks)


def render_pattern_image(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A,
    energy_ev: float,
    shape: tuple[int, int],
    origin_rc,
    pixel_size: float,
    rotation_ccw_deg: float = 0.0,
    ellipse=None,
    deform: torch.Tensor | None = None,
    disk_radius_px: float = 3.0,
    edge_px: float = 1.0,
    tilts: torch.Tensor | None = None,
    trial_tilts: torch.Tensor | None = None,
    sg_max: float = SG_MAX,
    k_max: float | None = None,
    fast_absorption: bool = True,
    tilt_weights: torch.Tensor | None = None,
    beams: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamical diffraction pattern images on the detector grid.

    Bloch intensities, averaged over the precession / convergence tilt set
    `tilts`, rendered as disks for every trial orientation tilt and every
    thickness.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated.
    orientation : torch.Tensor
        Unit quaternion (4,), crystal to lab.
    thicknesses_A : float | array-like
        Thicknesses in Angstroms.
    energy_ev : float
        Beam energy in eV.
    shape : tuple[int, int]
        Detector shape (ny, nx).
    origin_rc : array-like
        Direct beam position (row, col) in pixels.
    pixel_size : float
        Detector sampling (1/Angstroms per pixel).
    rotation_ccw_deg : float, default=0.0
        Scan-to-detector rotation of the calibration, undone here.
    ellipse : sequence | None
        Elliptic distortion (e11, e12) of the calibration, undone here.
    deform : torch.Tensor | None
        (3, 3) deformation of the lab-frame reciprocal vectors.
    disk_radius_px, edge_px : float, default=3.0, 1.0
        Disk shape, see render_disks.
    tilts : torch.Tensor | None
        (Mr, 2) illumination tilts (1/Angstroms), e.g. from
        illumination_nodes(); a single untilted beam if None.
    trial_tilts : torch.Tensor | None
        (M, 2) crystal tilts about the lab x and y axes (radians); none if
        None.
    sg_max : float, default=SG_MAX
        Excitation error cutoff (1/Angstroms) of the beam list.
    k_max : float | None
        Largest |g| (1/Angstroms) of a beam.
    fast_absorption : bool, default=True
        First-order absorption; see _bloch_solve.
    tilt_weights : torch.Tensor | None
        (Mr,) weights of the illumination tilts; equal if None.
    beams : torch.Tensor | None
        Explicit beam list (nb, 3), 000 first.

    Returns
    -------
    images : torch.Tensor
        (M, T, ny, nx), M the number of trial tilts (1 when None).
    centers_px : torch.Tensor
        (nb, 2) disk centers (row, col), direct beam first.
    intensities : torch.Tensor
        (M, T, nb) illumination-averaged intensities.
    """
    t_grid = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    ring = torch.zeros((1, 2), dtype=torch.float64) if tilts is None else tilts
    Mr = ring.shape[0]
    w_ring = (
        torch.full((Mr,), 1.0 / Mr, dtype=torch.float64) if tilt_weights is None else tilt_weights
    )
    if trial_tilts is None:
        trial = torch.zeros((1, 2), dtype=torch.float64)
    else:
        trial = k0 * torch.stack([trial_tilts[:, 1], -trial_tilts[:, 0]], dim=1)
    Mt = trial.shape[0]
    all_tilts = (trial[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    inten, g_xy, _ = _cbed_amplitudes(
        crystal,
        orientation,
        all_tilts,
        t_grid,
        energy_ev,
        sg_max,
        k_max,
        tilt_batch=max(64, Mr * 8),
        progress_bar=False,
        fast_absorption=fast_absorption,
        deform=deform,
        beams=beams,
    )
    inten = (inten.reshape(Mt, Mr, t_grid.shape[0], -1) * w_ring[None, :, None, None]).sum(dim=1)
    centers = _q_to_pixels(g_xy, origin_rc, pixel_size, rotation_ccw_deg, ellipse)
    images = render_disks(centers, inten, shape, disk_radius_px, edge_px)
    return images, centers, inten


def _image_cost(
    meas: torch.Tensor,
    sims: torch.Tensor,
    mask: torch.Tensor,
    power: float,
    background: str = "constant",
    radius: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized residual of the measured image (ny, nx) against each
    simulated one (..., ny, nx). The intensity scale and the background
    are solved by least squares in the raw domain over the mask; the
    residual is then taken between the power-law images so the weak
    diffracted disks weigh as they do in the Bragg-vector cost.

    background : {"constant", "radial"}
        A constant, or a quadratic in the distance from the direct beam
        (b0 + b1 r + b2 r^2, with `radius` in pixels), the smooth
        diffuse-scattering floor of a diffraction pattern.
    """
    m = mask.to(torch.float64)
    lead = sims.shape[:-2]
    sel = m.reshape(-1) > 0
    x = sims.reshape(-1, sims.shape[-2] * sims.shape[-1])[:, sel].to(torch.float64)  # (K, npix)
    y = meas.reshape(-1)[sel].to(torch.float64)
    ones = torch.ones_like(y)
    if background == "radial" and radius is not None:
        r = radius.reshape(-1)[sel].to(torch.float64)
        r = r / r.max().clamp_min(1e-12)
        basis = torch.stack([ones, r, r * r], dim=1)  # (npix, 3)
    else:
        basis = ones[:, None]
    # normal equations for [scale, background coefficients], per image
    nb = basis.shape[1]
    G_bb = basis.T @ basis
    G_xb = x @ basis  # (K, nb)
    G_xx = (x * x).sum(dim=1)
    rhs_x = x @ y
    rhs_b = basis.T @ y
    K = x.shape[0]
    A = torch.zeros((K, nb + 1, nb + 1), dtype=torch.float64)
    A[:, 0, 0] = G_xx
    A[:, 0, 1:] = G_xb
    A[:, 1:, 0] = G_xb
    A[:, 1:, 1:] = G_bb[None]
    rhs = torch.cat([rhs_x[:, None], rhs_b[None].expand(K, -1)], dim=1)
    A = A + 1e-12 * torch.eye(nb + 1, dtype=torch.float64)[None]
    coef = torch.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    a = coef[:, 0].clamp_min(0)
    bg = (basis @ coef[:, 1:].T).T
    model = (a[:, None] * x + bg).clamp_min(0) ** power
    yp = y.clamp_min(0) ** power
    resid = ((yp[None] - model) ** 2).sum(dim=1)
    return (resid / (yp * yp).sum().clamp_min(1e-12)).reshape(lead)


def _image_mask(shape, origin, r_max_px, exclude_direct_px):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.hypot(yy - origin[0], xx - origin[1])
    m = np.ones(shape, dtype=bool) if r_max_px is None else r <= r_max_px
    if exclude_direct_px is not None and exclude_direct_px > 0:
        m &= r > exclude_direct_px
    return torch.as_tensor(m)


def fit_disk_shape(
    dataset,
    phase_map,
    result: dict,
    origins: np.ndarray,
    pixel_size: float,
    rotation_ccw_deg: float = 0.0,
    ellipse=None,
    positions=None,
    n_positions: int = 20,
    radii_px=None,
    edges_px=None,
    power_intensity: float | None = None,
    r_max_px: float | None = None,
    exclude_direct_px: float | None = None,
    background: str = "constant",
    sg_max: float | None = None,
    k_max: float | None = None,
    fast_absorption: bool | None = None,
    progress_bar: bool = True,
) -> dict:
    """Global disk radius and edge width from the best-fit patterns.

    The convergence disk shape is a property of the illumination, not of
    the position, so it is fit once: on the `n_positions` positions with
    the lowest dynamical cost (or the given `positions`), the rendered
    pattern at the refined orientation, thickness and deformation is
    compared with the measured image over a grid of (radius, edge), and
    the pair minimizing the summed image cost is returned for
    refine_dynamical_image() to use.

    The illumination, intensity power and beam set (sg_max, k_max,
    fast_absorption) default to those of the refine_dynamical() result.

    Parameters
    ----------
    dataset : Dataset4dstem
        Measured patterns; anything with `.array` (R, C, ny, nx) and
        `.shape`.
    phase_map : PhaseMap
        The refined PhaseMap.
    result : dict
        From refine_dynamical().
    origins : np.ndarray
        (R, C, 2) direct beam positions (row, col) in pixels.
    pixel_size : float
        Detector sampling (1/Angstroms per pixel).
    rotation_ccw_deg : float, default=0.0
        Scan-to-detector rotation of the calibration.
    ellipse : sequence | None
        Elliptic distortion (e11, e12) of the calibration.
    positions : list of (int, int) | None
        Positions to fit; None takes the n_positions with the lowest
        dynamical cost.
    n_positions : int, default=20
        Number of positions when positions is None.
    radii_px : array-like | None
        Disk radii to try (pixels); default 1.5 to 6 in steps of 0.5.
    edges_px : array-like | None
        Edge widths to try (pixels); default 0.5, 0.75, 1, 1.5, 2.
    power_intensity : float | None
        Images are compared as I ** power_intensity; inherited.
    r_max_px : float | None
        Ignore pixels farther than this from the origin; None uses all.
    exclude_direct_px : float | None
        Ignore pixels within this distance of the origin; default 1.5
        times the largest radius tried.
    background : {"constant", "radial"}, default="constant"
        Background model of the image cost: a constant, or a quadratic in
        the distance from the direct beam.
    sg_max, k_max : float | None
        Beam set; inherited from the result's metadata (SG_MAX if absent).
    fast_absorption : bool | None
        First-order absorption; inherited (True if absent).
    progress_bar : bool, default=True
        Show a progress bar over positions.

    Returns
    -------
    dict
        'disk_radius_px', 'edge_px' the best pair, 'cost' (n_radii,
        n_edges) the summed image cost, 'radii_px', 'edges_px',
        'positions'.
    """
    oms = phase_map.orientation_maps
    cands = phase_map.candidates
    md = result.get("metadata", {})
    energy_ev = oms[0].energy_ev
    power_intensity = float(
        resolve(power_intensity, "power_intensity", md, default=POWER_INTENSITY)
    )
    # the beam set of the Bragg-vector refinement, unless overridden
    sg_max = float(resolve(sg_max, "sg_max", md, default=SG_MAX))
    k_max = resolve(k_max, "k_max", md)
    fast_absorption = bool(resolve(fast_absorption, "fast_absorption", md, default=True))
    tilts, tilt_w = illumination_nodes(
        energy_ev,
        md.get("precession_deg", 0.0),
        md.get("n_precession", 32),
        md.get("semiconv_mrad", 0.0),
        md.get("n_disk_radial", 4),
        md.get("n_disk_azimuthal", 16),
        maped_tilts_deg=md.get("maped_tilts_deg"),
    )
    if radii_px is None:
        radii_px = np.arange(1.5, 6.01, 0.5)
    if edges_px is None:
        edges_px = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    cost = torch.nan_to_num(result["cost"], nan=torch.inf).amin(dim=-1)
    if positions is None:
        flat = torch.argsort(cost.reshape(-1))[:n_positions]
        positions = [
            (int(i) // cost.shape[1], int(i) % cost.shape[1])
            for i in flat
            if torch.isfinite(cost.reshape(-1)[i])
        ]
    shape = tuple(dataset.shape[-2:])
    if exclude_direct_px is None:
        exclude_direct_px = 1.5 * float(np.max(radii_px))
    total = torch.zeros((len(radii_px), len(edges_px)), dtype=torch.float64)
    it = tqdm(positions, desc="disk shape") if progress_bar else positions
    for rx, ry in it:
        f = int(result["candidate"][rx, ry])
        i_om, m = cands[f]
        om = oms[i_om]
        q = result["quats"][rx, ry, f]
        t = float(result["thickness_per_candidate"][rx, ry, f])
        d3 = torch.eye(3, dtype=torch.float64)
        d3[:2, :2] = result["deformation"][rx, ry, f]
        o = origins[rx, ry]
        meas = torch.as_tensor(np.asarray(dataset.array[rx, ry], dtype=float)).clamp_min(0)
        mask = _image_mask(shape, o, r_max_px, exclude_direct_px)
        radius = torch.as_tensor(
            np.hypot(
                *(np.mgrid[0 : shape[0], 0 : shape[1]] - np.asarray(o, dtype=float)[:, None, None])
            )
        )
        _, centers, inten = render_pattern_image(
            om.crystal,
            q,
            [t],
            energy_ev,
            shape,
            o,
            pixel_size,
            rotation_ccw_deg,
            ellipse,
            d3,
            1.0,
            1.0,
            tilts,
            None,
            sg_max,
            k_max,
            fast_absorption,
            tilt_weights=tilt_w,
        )
        for i, r in enumerate(radii_px):
            for j, e in enumerate(edges_px):
                sim = render_disks(centers, inten[0, 0], shape, float(r), float(e))
                total[i, j] += _image_cost(meas, sim, mask, power_intensity, background, radius)
    k = int(total.argmin())
    i, j = k // len(edges_px), k % len(edges_px)
    return {
        "disk_radius_px": float(radii_px[i]),
        "edge_px": float(edges_px[j]),
        "cost": total,
        "radii_px": np.asarray(radii_px),
        "edges_px": np.asarray(edges_px),
        "positions": positions,
    }


def refine_dynamical_image(
    dataset,
    phase_map,
    result: dict,
    origins: np.ndarray,
    pixel_size: float,
    disk_radius_px: float,
    edge_px: float,
    rotation_ccw_deg: float = 0.0,
    ellipse=None,
    thickness_half_range_A: float = 100.0,
    thickness_step_A: float = 10.0,
    tilt_stage=(0.03, 0.01),
    power_intensity: float | None = None,
    r_max_px: float | None = None,
    exclude_direct_px: float | None = None,
    background: str = "constant",
    mask: np.ndarray | None = None,
    sg_max: float | None = None,
    k_max: float | None = None,
    fast_absorption: bool | None = None,
    update_orientations: bool = True,
    progress_bar: bool = True,
) -> dict:
    """Final dynamical refinement against the diffraction images.

    Starting from the Bragg-vector solution of refine_dynamical (winning
    candidate, orientation, thickness, in-plane deformation), every pixel
    of the measured pattern is compared with a rendered pattern: Bloch
    intensities averaged over the precession / convergence tilt set,
    drawn as disks of the global radius and edge width from
    fit_disk_shape(), with a free intensity scale and a constant or
    radial background. The thickness and the orientation tilt are re-searched
    on a local grid (thickness +- thickness_half_range_A, tilt +- the
    stage half-range), the deformation and in-plane rotation are kept
    from the position fit. The image cost is the residual after the
    linear fit, normalized by the image power, so it is comparable
    across positions. The direct beam disk is excluded from the cost
    (exclude_direct_px, default 1.5 disk radii): it carries most of the
    counts, its measured intensity is the least reliable (saturation,
    detector response), and a fraction of a percent of model error on it
    would outweigh every diffracted disk. Run this when the Bragg-vector
    refinement is not accurate enough; it costs one rendered image per
    trial (tilt, thickness) on top of the Bloch solves.

    The illumination, intensity power and beam set (sg_max, k_max,
    fast_absorption) default to those of the refine_dynamical() result.

    Parameters
    ----------
    dataset : Dataset4dstem
        Measured patterns; anything with `.array` (R, C, ny, nx) and
        `.shape`.
    phase_map : PhaseMap
        The refined PhaseMap.
    result : dict
        From refine_dynamical().
    origins : np.ndarray
        (R, C, 2) direct beam positions (row, col) in pixels.
    pixel_size : float
        Detector sampling (1/Angstroms per pixel).
    disk_radius_px, edge_px : float
        Disk shape, from fit_disk_shape().
    rotation_ccw_deg : float, default=0.0
        Scan-to-detector rotation of the calibration.
    ellipse : sequence | None
        Elliptic distortion (e11, e12) of the calibration.
    thickness_half_range_A : float, default=100.0
        Half-width (Angstroms) of the thickness search around the
        Bragg-vector thickness.
    thickness_step_A : float, default=10.0
        Thickness step in Angstroms.
    tilt_stage : (float, float), default=(0.03, 0.01)
        Tilt search half-range and step in degrees.
    power_intensity : float | None
        Images are compared as I ** power_intensity; inherited.
    r_max_px : float | None
        Ignore pixels farther than this from the origin; None uses all.
    exclude_direct_px : float | None
        Ignore pixels within this distance of the origin; default 1.5
        disk radii.
    background : {"constant", "radial"}, default="constant"
        Background model: a constant, or a quadratic in the distance from
        the direct beam (the diffuse scattering floor).
    mask : np.ndarray | None
        Positions to refine, as in refine_dynamical; None refines all.
    sg_max, k_max : float | None
        Beam set; inherited from the result's metadata (SG_MAX if absent).
    fast_absorption : bool | None
        First-order absorption; inherited (True if absent).
    update_orientations : bool, default=True
        Write the refined quaternions back into the OrientationMaps.
    progress_bar : bool, default=True
        Show a progress bar over positions.

    Returns
    -------
    dict
        'thickness' (R, C), 'tilt_deg' (R, C, 2) the additional tilt over
        the Bragg-vector result, 'quats' (R, C, 4), 'cost' (R, C) the
        normalized image residual (NaN where not refined), and 'metadata'.
    """
    from quantem.diffraction.rotations import qmult, quat_from_axis_angle

    oms = phase_map.orientation_maps
    cands = phase_map.candidates
    md = result.get("metadata", {})
    energy_ev = oms[0].energy_ev
    power_intensity = float(
        resolve(power_intensity, "power_intensity", md, default=POWER_INTENSITY)
    )
    # the beam set of the Bragg-vector refinement, unless overridden
    sg_max = float(resolve(sg_max, "sg_max", md, default=SG_MAX))
    k_max = resolve(k_max, "k_max", md)
    fast_absorption = bool(resolve(fast_absorption, "fast_absorption", md, default=True))
    if exclude_direct_px is None:
        exclude_direct_px = 1.5 * disk_radius_px
    tilts, tilt_w = illumination_nodes(
        energy_ev,
        md.get("precession_deg", 0.0),
        md.get("n_precession", 32),
        md.get("semiconv_mrad", 0.0),
        md.get("n_disk_radial", 4),
        md.get("n_disk_azimuthal", 16),
        maped_tilts_deg=md.get("maped_tilts_deg"),
    )
    R, C = result["thickness"].shape
    shape = tuple(dataset.shape[-2:])
    half, step = (np.deg2rad(v) for v in tilt_stage)
    n = int(round(2 * half / step)) + 1
    tg = torch.linspace(-half, half, n, dtype=torch.float64)
    wx_g, wy_g = torch.meshgrid(tg, tg, indexing="ij")
    w_grid = torch.stack([wx_g.reshape(-1), wy_g.reshape(-1)], dim=1)

    thickness = torch.full((R, C), torch.nan, dtype=torch.float64)
    tilt_out = torch.zeros((R, C, 2), dtype=torch.float64)
    quat_out = torch.zeros((R, C, 4), dtype=torch.float64)
    quat_out[..., 0] = 1.0
    cost_out = torch.full((R, C), torch.nan, dtype=torch.float64)

    from quantem.diffraction.orientation import position_mask

    mask_rc = position_mask(mask, (R, C))
    iterator = [(r, c) for r, c in np.ndindex(R, C) if mask_rc[r, c]]
    if progress_bar:
        iterator = tqdm(iterator, desc="image refinement")
    for rx, ry in iterator:
        f = int(result["candidate"][rx, ry])
        if f < 0 or not torch.isfinite(result["cost"][rx, ry, f]):
            continue
        i_om, m = cands[f]
        om = oms[i_om]
        q0 = result["quats"][rx, ry, f]
        t0 = float(result["thickness_per_candidate"][rx, ry, f])
        d3 = torch.eye(3, dtype=torch.float64)
        d3[:2, :2] = result["deformation"][rx, ry, f]
        o = origins[rx, ry]
        meas = torch.as_tensor(np.asarray(dataset.array[rx, ry], dtype=float)).clamp_min(0)
        pmask = _image_mask(shape, o, r_max_px, exclude_direct_px)
        radius = torch.as_tensor(
            np.hypot(
                *(np.mgrid[0 : shape[0], 0 : shape[1]] - np.asarray(o, dtype=float)[:, None, None])
            )
        )
        t_grid = np.arange(
            max(thickness_step_A, t0 - thickness_half_range_A),
            t0 + thickness_half_range_A + 1e-6,
            thickness_step_A,
        )
        images, _, _ = render_pattern_image(
            om.crystal,
            q0,
            t_grid,
            energy_ev,
            shape,
            o,
            pixel_size,
            rotation_ccw_deg,
            ellipse,
            d3,
            disk_radius_px,
            edge_px,
            tilts,
            w_grid,
            sg_max,
            k_max,
            fast_absorption,
            tilt_weights=tilt_w,
        )
        cost = _image_cost(meas, images, pmask, power_intensity, background, radius)  # (M, T)
        T = len(t_grid)
        flat = int(cost.argmin())
        m_best, t_best = flat // T, flat % T
        i_b, j_b = m_best // n, m_best % n
        wx, wy = float(w_grid[m_best, 0]), float(w_grid[m_best, 1])
        cost_t = cost[:, t_best].reshape(n, n)
        if 0 < i_b < n - 1:
            c0, c1, c2 = cost_t[i_b - 1, j_b], cost_t[i_b, j_b], cost_t[i_b + 1, j_b]
            den = float(c0 - 2 * c1 + c2)
            if den > 1e-12:
                wx += 0.5 * float(c0 - c2) / den * step
        if 0 < j_b < n - 1:
            c0, c1, c2 = cost_t[i_b, j_b - 1], cost_t[i_b, j_b], cost_t[i_b, j_b + 1]
            den = float(c0 - 2 * c1 + c2)
            if den > 1e-12:
                wy += 0.5 * float(c0 - c2) / den * step
        q = q0
        ang = float(np.hypot(wx, wy))
        if ang > 1e-12:
            axis = torch.tensor([wx / ang, wy / ang, 0.0], dtype=torch.float64)
            q = qmult(quat_from_axis_angle(axis, torch.tensor(ang, dtype=torch.float64)), q0)
        thickness[rx, ry] = float(t_grid[t_best])
        tilt_out[rx, ry, 0] = wx
        tilt_out[rx, ry, 1] = wy
        quat_out[rx, ry] = q
        cost_out[rx, ry] = cost[m_best, t_best]
        if update_orientations:
            om.quats[rx, ry, m] = q

    used = dict(
        disk_radius_px=float(disk_radius_px),
        edge_px=float(edge_px),
        thickness_half_range_A=float(thickness_half_range_A),
        thickness_step_A=float(thickness_step_A),
        tilt_stage=tuple(float(v) for v in tilt_stage),
        power_intensity=float(power_intensity),
        r_max_px=r_max_px,
        exclude_direct_px=float(exclude_direct_px),
        background=str(background),
        sg_max=float(sg_max),
        k_max=k_max,
        fast_absorption=bool(fast_absorption),
        inherited=dict(md),
    )
    if hasattr(phase_map, "metadata"):
        phase_map.metadata["dynamical_image"] = used
    return {
        "thickness": thickness,
        "tilt_deg": torch.rad2deg(tilt_out),
        "quats": quat_out,
        "cost": cost_out,
        "metadata": used,
    }
