"""Tests for quantem.diffraction.bloch."""

import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction.bloch import dynamical_pattern, refine_thickness
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.phase import PhaseMap
from quantem.diffraction.rotations import quat_from_zone_axis


@pytest.fixture(scope="module")
def ti_beta():
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), name="Ti beta")
    # 2x coverage so all coupling vectors g - h have structure factors
    xtl.calculate_structure_factors(k_max=3.0, tol_structure_factor=1e-6)
    return xtl


def test_flux_conservation(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64))
    p = dynamical_pattern(
        ti_beta, q, np.arange(50, 1500, 50.0), energy_ev=200e3, sg_max=0.08, k_max=1.5
    )
    total = p["intensity"].sum(dim=1)
    assert float(total.max()) <= 1.0 + 1e-6


def test_thin_limit_matches_kinematical(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64))
    p = dynamical_pattern(ti_beta, q, 25.0, energy_ev=200e3, sg_max=0.08, k_max=1.5)
    kin = ti_beta.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    top_dyn = set(map(tuple, p["hkl"][p["intensity"][0].argsort(descending=True)[:4]].tolist()))
    top_kin = set(map(tuple, kin["hkl"][kin["intensity"].argsort(descending=True)[:4]].tolist()))
    assert top_dyn == top_kin


def test_thickness_recovery(ti_beta):
    """Simulate dynamical peaks at a known thickness, recover it."""
    t_true = 600.0
    torch.manual_seed(0)
    zones = torch.tensor(
        [[0.1, 0.9, 1.0], [0.3, 0.5, 1.0], [0.05, 1.0, 1.1]], dtype=torch.float64
    )
    N = zones.shape[0]
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    q_true = quat_from_zone_axis(zones)
    for i in range(N):
        p = dynamical_pattern(
            ti_beta, q_true[i], t_true, energy_ev=200e3, sg_max=0.06, k_max=1.5
        )
        keep = p["intensity"][0] > 1e-4
        peaks[0, i] = np.stack(
            [
                p["qx"][keep].numpy(),
                p["qy"][keep].numpy(),
                p["intensity"][0][keep].numpy(),
            ],
            axis=1,
        )

    om = OrientationMap.from_vectors(peaks, ti_beta, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0)
    om.match_orientations(progress_bar=False)
    # thickness oscillations are sensitive to ~1 degree tilt errors, beyond
    # what kinematical matching provides for dynamical patterns; test the
    # thickness scan itself with the true orientations (dynamical tilt
    # refinement is the future joint pass)
    om.quats[0, :, 0] = q_true

    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(max_patterns=1, progress_bar=False)
    res = refine_thickness(
        pm,
        thicknesses_A=np.arange(100, 1200, 50.0),
        sg_max=0.06,
        progress_bar=False,
    )
    t_fit = res["thickness"][0].numpy()
    assert (np.abs(t_fit - t_true) <= 50.0).all()
