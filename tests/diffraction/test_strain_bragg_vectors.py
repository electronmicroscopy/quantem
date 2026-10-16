"""BraggVectors lattice fit and StrainMap on synthetic peaks (no disk detection)."""

import numpy as np
import pytest

from quantem.core.datastructures import Dataset4dstem
from quantem.core.datastructures.vector import Vector
from quantem.diffraction import BraggVectors
from quantem.diffraction.strain import StrainMap

ORIGIN = np.array([32.0, 32.0])
G1 = np.array([9.0, 1.0])
G2 = np.array([-1.5, 11.0])


def _synthetic_bragg_vectors(deform, R=8, C=6, metadata=None):
    """BraggVectors whose peaks sit on origin + a M g1 + b M g2, M = deform(r, c).

    The peaks are written straight into ``bv.peaks``; the dataset is only a
    zero cube that sets the scan and detector shapes. Peaks are stored as
    float32, so fitted vectors match to about 1e-6 pixels.
    """
    nested = []
    for r in range(R):
        row = []
        for c in range(C):
            M = np.asarray(deform(r, c), dtype=float)
            g1, g2 = M @ G1, M @ G2
            pts = []
            for a in range(-2, 3):
                for b in range(-2, 3):
                    q = ORIGIN + a * g1 + b * g2
                    pts.append([q[0], q[1], 10.0 if a == b == 0 else 1.0])
            row.append(np.asarray(pts))
        nested.append(row)

    ds = Dataset4dstem.from_array(np.zeros((R, C, 64, 64), dtype=np.float32))
    if metadata:
        ds.metadata.update(metadata)
    bv = BraggVectors.from_dataset(ds)
    bv.peaks = Vector.from_data(nested, fields=["q_row", "q_col", "intensity"])
    bv.compute_bvm()
    bv.choose_basis_vectors(origin=ORIGIN, g1=G1, g2=G2, plot=False)
    bv.index_peaks(plot=False)
    bv.fit_lattice(min_num_peaks=5, progressbar=False, plot=False)
    return bv


def test_reciprocal_scaling_ramp_gives_compressive_strain():
    # reciprocal vectors grow with row, so the real-space lattice shrinks:
    # e = 1 / s - 1, negative and decreasing down the scan
    s = 1.0 + 0.002 * np.arange(8)
    bv = _synthetic_bragg_vectors(lambda r, c: s[r] * np.eye(2))
    np.testing.assert_allclose(bv.g1_array[:, 0], s[:, None] * G1[None, :], atol=2e-5)
    assert np.all(bv.mask_weight > 0.99)

    with pytest.warns(UserWarning, match="no detector rotation"):
        sm = bv.calculate_strain_map(g1_ref=G1, g2_ref=G2)
    expected = (1.0 / s - 1.0)[:, None] * np.ones((1, 6))
    np.testing.assert_allclose(sm.e_rr.array, expected, atol=2e-5)
    np.testing.assert_allclose(sm.e_cc.array, expected, atol=2e-5)
    np.testing.assert_allclose(sm.e_rc.array, 0.0, atol=2e-5)
    np.testing.assert_allclose(sm.phi.array, 0.0, atol=2e-5)
    assert sm.e_rr.array[-1, 0] < 0
    assert np.all(np.diff(sm.e_rr.array[:, 0]) < 0)

    # automatic reference (weighted median over the scan): same ramp, offset
    with pytest.warns(UserWarning):
        sm_auto = bv.calculate_strain_map()
    ramp = sm_auto.e_rr.array[:, 0]
    assert np.all(np.diff(ramp) < 0)
    assert ramp.min() < 0 < ramp.max()


def test_real_and_reciprocal_vectors_give_same_strain():
    rng = np.random.default_rng(0)
    R, C = 4, 5
    F = np.eye(2)[None, None] + 0.01 * rng.normal(size=(R, C, 2, 2))
    A0 = np.array([[3.0, 0.5], [-0.4, 2.5]])  # real-space basis, columns a1, a2
    G0 = np.linalg.inv(A0).T  # reciprocal basis, columns g1, g2 (G0.T @ A0 = I)
    A = F @ A0
    G = np.linalg.inv(F).transpose(0, 1, 3, 2) @ G0

    common = dict(ds_shape=(R, C), q_to_r_rotation_ccw_deg=0.0, q_transpose=False)
    sm_real = StrainMap(
        g1_array=A[..., :, 0],
        g2_array=A[..., :, 1],
        real_space=True,
        g1_ref=A0[:, 0],
        g2_ref=A0[:, 1],
        **common,
    )
    sm_recip = StrainMap(
        g1_array=G[..., :, 0],
        g2_array=G[..., :, 1],
        real_space=False,
        g1_ref=G0[:, 0],
        g2_ref=G0[:, 1],
        **common,
    )
    expected = {
        "e_rr": F[..., 0, 0] - 1,
        "e_cc": F[..., 1, 1] - 1,
        "e_rc": 0.5 * (F[..., 0, 1] + F[..., 1, 0]),
        "phi": 0.5 * (F[..., 1, 0] - F[..., 0, 1]),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(sm_real, name).array, value, atol=1e-12)
        np.testing.assert_allclose(getattr(sm_recip, name).array, value, atol=1e-12)


def test_counterclockwise_lattice_rotation_gives_positive_phi():
    theta = np.deg2rad(0.5)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    g = rot @ np.stack([G1, G2], axis=1)
    sm = StrainMap(
        g1_array=np.broadcast_to(g[:, 0], (2, 2, 2)).copy(),
        g2_array=np.broadcast_to(g[:, 1], (2, 2, 2)).copy(),
        ds_shape=(2, 2),
        real_space=False,
        g1_ref=G1,
        g2_ref=G2,
    )
    np.testing.assert_allclose(sm.phi.array, np.sin(theta), atol=1e-12)
    np.testing.assert_allclose(sm.e_rc.array, 0.0, atol=1e-12)


def test_q_to_r_rotation_is_applied():
    # uniaxial real-space stretch of 1 % along the detector row axis
    stretch = np.diag([1.0 / 1.01, 1.0])  # reciprocal vectors shrink along rows

    bv = _synthetic_bragg_vectors(lambda r, c: stretch, R=3, C=3)
    with pytest.warns(UserWarning):
        sm0 = bv.calculate_strain_map(g1_ref=G1, g2_ref=G2)
    np.testing.assert_allclose(sm0.e_rr.array, 0.01, atol=2e-5)
    np.testing.assert_allclose(sm0.e_cc.array, 0.0, atol=2e-5)

    # a 90 degree detector-to-scan rotation moves the stretch onto the scan columns
    sm90 = bv.calculate_strain_map(
        g1_ref=G1, g2_ref=G2, q_to_r_rotation_ccw_deg=90.0, q_transpose=False
    )
    np.testing.assert_allclose(sm90.e_rr.array, 0.0, atol=2e-5)
    np.testing.assert_allclose(sm90.e_cc.array, 0.01, atol=2e-5)
    assert bv.metadata["q_to_r_rotation_ccw_deg"] == 90.0

    # the same rotation read from the dataset metadata
    bv_md = _synthetic_bragg_vectors(
        lambda r, c: stretch,
        R=3,
        C=3,
        metadata={"q_to_r_rotation_ccw_deg": 90.0, "q_transpose": False},
    )
    with pytest.warns(UserWarning, match="using Dataset4dstem metadata"):
        sm_md = bv_md.calculate_strain_map(g1_ref=G1, g2_ref=G2)
    np.testing.assert_allclose(sm_md.e_cc.array, 0.01, atol=2e-5)
    np.testing.assert_allclose(sm_md.e_rr.array, 0.0, atol=2e-5)

    # a transpose alone also swaps rows and columns
    sm_t = bv.calculate_strain_map(
        g1_ref=G1, g2_ref=G2, q_to_r_rotation_ccw_deg=0.0, q_transpose=True
    )
    np.testing.assert_allclose(sm_t.e_cc.array, 0.01, atol=2e-5)


def test_estimate_strain_precision_is_quiet_by_default(capsys):
    rng = np.random.default_rng(1)
    g1 = G1[None, None] + 0.01 * rng.normal(size=(12, 12, 2))
    g2 = G2[None, None] + 0.01 * rng.normal(size=(12, 12, 2))
    sm = StrainMap(g1_array=g1, g2_array=g2, ds_shape=(12, 12), real_space=False)
    out = sm.estimate_strain_precision(plot=False)
    assert capsys.readouterr().out == ""
    assert np.isfinite(out["precision"]["combined"])
    sm.estimate_strain_precision(plot=False, verbose=True)
    assert "Strain precision" in capsys.readouterr().out
