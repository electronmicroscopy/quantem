"""Tests for quantem.diffraction.crystal."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import quat_from_zone_axis


@pytest.fixture
def ti_beta():
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), name="Ti beta")
    xtl.calculate_structure_factors(k_max=1.5)
    return xtl


def test_symmetry_detection(ti_beta):
    assert ti_beta.pointgroup == "m-3m"
    assert ti_beta.laue_group == "m-3m"
    assert ti_beta.sym_quats.shape[0] == 24  # proper rotations of m-3m


def test_hcp_symmetry():
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686))
    assert xtl.pointgroup == "6/mmm"
    assert xtl.sym_quats.shape[0] == 12


def test_bcc_absences(ti_beta):
    # h + k + l odd forbidden in bcc
    parity = ti_beta.hkl.sum(dim=1) % 2
    assert (parity == 0).all()


def test_ring_positions(ti_beta):
    # (110) ring at sqrt(2)/a
    g110 = np.sqrt(2) / 3.31
    assert np.isclose(float(ti_beta.g_len.min()), g110, atol=1e-6)


def test_pseudo_symmetry():
    ortho = Atoms("Au", positions=[[0, 0, 0]], cell=[4.000, 4.001, 4.002], pbc=True)
    exact = Crystal.from_ase(ortho)
    pseudo = Crystal.from_ase(ortho, pseudo_symmetry_tol=0.01)
    assert exact.pointgroup_matching == "mmm"
    assert pseudo.pointgroup_matching == "m-3m"
    assert pseudo.sym_quats_matching.shape[0] == 24
    # exact group is retained for reporting/refinement
    assert pseudo.pointgroup == "mmm"


def test_zone_axis_wedge_anchored_001(ti_beta):
    wedge = ti_beta.zone_axis_wedge()
    assert torch.allclose(
        wedge[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )


def test_generate_pattern(ti_beta):
    q = quat_from_zone_axis(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    p = ti_beta.generate_pattern(q, energy_ev=200e3)
    # [001] zone: peaks on a square grid of 110-type spacings
    assert p["qx"].shape[0] > 4
    qr = torch.hypot(p["qx"], p["qy"])
    assert float(qr.min()) > 0.4  # no direct beam
    # pattern symmetric under 90 degree rotation
    rot = torch.stack((-p["qy"], p["qx"]), dim=1)
    orig = torch.stack((p["qx"], p["qy"]), dim=1)
    d = torch.cdist(rot, orig).min(dim=1).values
    assert float(d.max()) < 1e-6
