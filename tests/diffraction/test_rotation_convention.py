"""Self-consistency of the detector-to-scan rotation convention.

Simulates a cubic crystal rotated in-plane by a known angle, records its
peaks in a detector frame that is rotated relative to the scan frame, and
checks that peaks_to_calibrated(rotation_ccw_deg=...) plus orientation
matching recovers the ground-truth in-plane angle in the scan frame.
"""

import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import calibration
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.rotations import quat_from_axis_angle


@pytest.mark.parametrize("rot_scan_deg", [0.0, 34.0, -34.0])
def test_rotation_roundtrip(rot_scan_deg):
    xtl = Crystal.from_ase(bulk("Al", "fcc", a=4.05, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)

    # ground truth: [001] zone, 30 deg in-plane rotation, in the SCAN frame
    angle_true = 30.0
    q_true = quat_from_axis_angle(
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor(np.deg2rad(angle_true), dtype=torch.float64),
    )
    pat = xtl.generate_pattern(q_true, energy_ev=200e3, sigma_excitation=0.02)
    q_scan = np.stack([pat["qx"].numpy(), pat["qy"].numpy()], axis=1)

    # what the detector records: scan-frame vectors rotated by -rot_scan
    # (peaks_to_calibrated undoes this with rotation_ccw_deg=+rot_scan)
    th = np.deg2rad(-rot_scan_deg)
    rot_back = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q_det = q_scan @ rot_back.T

    pixel_size = 0.01
    data = np.column_stack([q_det / pixel_size, pat["intensity"].numpy()])
    peaks_px = Vector.from_data(
        [[data]],
        fields=["q_row", "q_col", "intensity"],
        units=["px", "px", "counts"],
        name="synthetic",
    )
    peaks = calibration.peaks_to_calibrated(
        peaks_px, pixel_size, rotation_ccw_deg=rot_scan_deg
    )

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(power_intensity=0.0)
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False)

    # the recovered orientation must equal the scan-frame ground truth
    # (modulo crystal symmetry) -- independent of the detector rotation
    from quantem.diffraction.rotations import misorientation_angle_deg

    err = float(
        misorientation_angle_deg(q_true, om.quats[0, 0, 0], xtl.sym_quats)
    )
    assert err < 1.0, f"misorientation {err:.2f} deg at rot {rot_scan_deg}"
