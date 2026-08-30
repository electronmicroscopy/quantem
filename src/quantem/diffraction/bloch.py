"""Dynamical (Bloch wave) diffraction for orientation and phase refinement.

Second-pass refinement: kinematical matching fixes orientations from peak
positions (which dynamical scattering does not move), then this module
recomputes peak intensities with multiple scattering to refine specimen
thickness and phase assignment for the top candidates.

Follows the Bloch wave formulation of De Graef (2003), ch. 5. The structure
matrix uses U_g = gamma_rel * F_g / pi with F_g the kinematical structure
factors (scattering amplitude per volume, 1/Angstrom^2), off-diagonals
U_(g-h) and diagonal 2 k0 s_g. Without absorption the matrix is Hermitian,
so one eigendecomposition per orientation gives the diffracted intensities
at every thickness essentially for free:

    psi(t) = C exp(2 pi i gamma t) C^-1 psi_0,   A C = 2 k0 gamma C
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import qrotate


def relativistic_gamma(energy_ev: float) -> float:
    """Relativistic mass factor 1 + eV / (m0 c^2)."""
    return 1.0 + float(energy_ev) / 510998.95


def dynamical_pattern(
    crystal: Crystal,
    orientation: torch.Tensor,
    thicknesses_A: torch.Tensor | np.ndarray | float,
    energy_ev: float = 300e3,
    sg_max: float = 0.1,
    k_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Bloch-wave diffraction intensities for one orientation, all thicknesses.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated. For accurate couplings,
        calculate_structure_factors should cover 2x the k_max used here so
        every difference vector g - h has a structure factor.
    orientation : torch.Tensor
        Unit quaternion (4,) rotating crystal vectors into the lab frame.
    thicknesses_A : array-like or float
        Specimen thicknesses in Angstroms.
    sg_max : float, default=0.1
        Excitation error cutoff (1/Angstroms) for including a beam.
    k_max : float | None
        In-plane scattering vector cutoff for included beams.

    Returns
    -------
    dict
        'qx', 'qy' (N,), 'hkl' (N, 3), 'intensity' (T, N) diffracted
        intensities per thickness, 's_g' (N,).
    """
    if crystal.g_vec is None:
        raise RuntimeError("Run crystal.calculate_structure_factors() first.")
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    gamma_rel = relativistic_gamma(energy_ev)

    t = torch.atleast_1d(torch.as_tensor(thicknesses_A, dtype=torch.float64))

    # beam selection in the lab frame
    g_lab = qrotate(orientation, crystal.g_vec)
    gz, g2 = g_lab[:, 2], (g_lab**2).sum(dim=1)
    s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
    sel = torch.abs(s_g) < sg_max
    if k_max is not None:
        sel &= crystal.g_len <= k_max
    hkl_sel = crystal.hkl[sel]
    g_sel = g_lab[sel]
    s_sel = s_g[sel]
    n = int(sel.sum())

    # structure factor lookup for all difference vectors (h_i - h_j).
    # Prefer the absorptive Weickenmeier-Kohl factors if the crystal has
    # them (calculate_dynamical_structure_factors); they carry the
    # relativistic and 1/pi factors already. Fall back to the kinematical
    # (Lobato) factors, purely elastic.
    absorptive = getattr(crystal, "U_dyn", None) is not None
    if absorptive:
        hkl_all = crystal.hkl_dyn
        U_all = crystal.U_dyn
    else:
        hkl_all = crystal.hkl
        U_all = crystal.struct_factors * (gamma_rel / np.pi)
    key_mult = torch.tensor(
        [1, 2 * int(hkl_all.abs().max()) + 1, (2 * int(hkl_all.abs().max()) + 1) ** 2],
        dtype=torch.long,
    )

    def keys(h):
        return (h * key_mult[None, :]).sum(dim=1)

    lut = {int(k): i for i, k in enumerate(keys(hkl_all))}

    # beams list includes the (000) beam at index 0
    hkl_beams = torch.cat([torch.zeros((1, 3), dtype=torch.long), hkl_sel])
    s_beams = torch.cat([torch.zeros(1, dtype=torch.float64), s_sel])
    nb = n + 1

    diff = hkl_beams[:, None, :] - hkl_beams[None, :, :]  # (nb, nb, 3)
    diff_keys = (diff * key_mult[None, None, :]).sum(dim=-1)
    U = torch.zeros((nb, nb), dtype=torch.complex128)
    flat = diff_keys.reshape(-1)
    idx = torch.tensor(
        [lut.get(int(k), -1) for k in flat], dtype=torch.long
    ).reshape(nb, nb)
    has = idx >= 0
    U[has] = U_all[idx[has]]

    A = U.clone()
    A.fill_diagonal_(0)
    diag = (2 * k0 * s_beams).to(torch.complex128)
    if absorptive:
        # mean absorption: imaginary part of U_000 damps every beam
        u0 = keys(torch.zeros((1, 3), dtype=torch.long))
        i0 = lut.get(int(u0[0]), -1)
        if i0 >= 0:
            diag = diag + 1j * U_all[i0].imag
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
        "thicknesses": t,
    }


def refine_thickness(
    phase_map,
    thicknesses_A: np.ndarray | None = None,
    pair_distance: float = 0.05,
    power_intensity: float = 0.25,
    sg_max: float = 0.1,
    k_max: float | None = None,
    min_number_peaks: int = 3,
    progress_bar: bool = True,
):
    """Second-pass thickness and phase refinement with dynamical intensities.

    For every probe position, the winning candidates of a fitted PhaseMap are
    re-simulated with Bloch waves over a thickness grid. The peak pairing is
    fixed (positions are kinematic); the intensity cost is evaluated for all
    thicknesses from a single eigendecomposition per candidate, and the best
    (thickness, candidate) combination updates the phase decision.

    Parameters
    ----------
    phase_map : PhaseMap
        A fitted PhaseMap (fit() has been run).
    thicknesses_A : np.ndarray | None
        Thickness grid in Angstroms; default 50 ... 1000 in 25 A steps.

    Returns
    -------
    dict
        'thickness' (R, C) best-fit thickness map, 'cost' (R, C, F) dynamical
        costs per candidate at its best thickness, 'phase_index' (R, C)
        updated phase assignment.
    """
    if thicknesses_A is None:
        thicknesses_A = np.arange(50.0, 1000.0, 25.0)
    t_grid = torch.as_tensor(thicknesses_A, dtype=torch.float64)

    oms = phase_map.orientation_maps
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
        data = peaks[rx, ry].array
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
            if phase_map.phase_weights is not None and float(
                phase_map.phase_weights[rx, ry, f]
            ) <= 0:
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
                (b - w[:, None] * a).abs() * (1 - frac)[None, :]
                + w[:, None] * a * frac[None, :]
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
    phase_index = cost_phase.argmin(dim=-1)

    f_best = torch.nan_to_num(cost_out, nan=torch.inf).argmin(dim=-1)
    thickness = torch.gather(thick_out, 2, f_best[..., None]).squeeze(-1)

    return {
        "thickness": thickness,
        "cost": cost_out,
        "phase_index": phase_index,
        "thickness_per_candidate": thick_out,
    }
