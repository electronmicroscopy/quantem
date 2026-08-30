"""End-to-end two-phase mapping on a synthetic alpha/beta titanium scan.

Left region: bcc beta in a fixed orientation. Right region: hcp alpha in a
Burgers-related orientation ((110)beta || (0001)alpha). A two-column band in
the middle contains both patterns superimposed, as at a lath boundary.
"""

import numpy as np
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.phase import PhaseMap
from quantem.diffraction.rotations import (
    misorientation_angle_deg,
    qmult,
    quat_from_axis_angle,
)


def _pattern(xtl, q, rng):
    p = xtl.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    arr = np.column_stack(
        [p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()]
    )
    arr[:, :2] += rng.normal(0, 0.003, (arr.shape[0], 2))
    arr[:, 2] *= rng.lognormal(0, 0.3, arr.shape[0])
    return arr


def test_two_phase_map():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    ti_b = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), name="Ti beta", verbose=False
    ).calculate_structure_factors(k_max=1.5)

    # beta along [111] zone; alpha along [0001]: the Burgers-related pair
    # shares the hexagonal net, the hard case for phase mapping
    q_beta = quat_from_axis_angle(
        torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
        / np.sqrt(2),
        torch.tensor(np.arccos(1 / np.sqrt(3)), dtype=torch.float64),
    )
    q_alpha = qmult(
        quat_from_axis_angle(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            torch.tensor(np.deg2rad(14.0), dtype=torch.float64),
        ),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    )

    R, C = 6, 11
    band = (5, 6)  # columns with both phases
    cells = []
    truth = np.zeros((R, C), dtype=int)  # 0 alpha, 1 beta
    for r in range(R):
        row = []
        for c in range(C):
            if c < band[0]:
                arr = _pattern(ti_b, q_beta, rng)
                truth[r, c] = 1
            elif c > band[1]:
                arr = _pattern(ti_a, q_alpha, rng)
                truth[r, c] = 0
            else:
                a = _pattern(ti_a, q_alpha, rng)
                b = _pattern(ti_b, q_beta, rng)
                a[:, 2] *= 0.5
                b[:, 2] *= 0.5
                arr = np.concatenate([a, b])
                truth[r, c] = 2
            row.append(arr)
        cells.append(row)
    peaks = Vector.from_data(
        cells, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )

    oms = []
    for xtl in (ti_a, ti_b):
        om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
        om.build_plan()
        om.match_orientations(num_matches=2, progress_bar=False)
        om.refine_orientations(progress_bar=False)
        oms.append(om)

    # orientation recovery in the pure regions
    err_a = misorientation_angle_deg(
        q_alpha, oms[0].quats[:, band[1] + 1 :, 0].reshape(-1, 4), ti_a.sym_quats
    ).numpy()
    err_b = misorientation_angle_deg(
        q_beta, oms[1].quats[:, : band[0], 0].reshape(-1, 4), ti_b.sym_quats
    ).numpy()
    assert np.median(err_a) < 1.0
    assert np.median(err_b) < 1.0

    pm = PhaseMap.from_orientation_maps(oms)
    pm.fit(max_patterns=2, progress_bar=False)
    pi = pm.phase_index.numpy()

    pure_a = pi[:, band[1] + 1 :]
    pure_b = pi[:, : band[0]]
    assert (pure_a == 0).mean() > 0.85
    assert (pure_b == 1).mean() > 0.85

    # overlap band: every position must be assigned one of the two true
    # phases with a valid orientation (either is acceptable)
    band_pi = pi[:, band[0] : band[1] + 1]
    assert np.isin(band_pi, [0, 1]).all()
