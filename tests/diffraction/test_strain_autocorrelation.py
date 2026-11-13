"""StrainMapAutocorrelation on a synthetic lattice with a known strain ramp."""

import warnings

import numpy as np

from quantem.diffraction import StrainMapAutocorrelation

N = 64
Q = np.array([[8.0, -2.5], [2.0, 9.0]])  # columns: diffraction-space basis (row, col)


def _pattern(scale, sigma=1.2):
    rr, cc = np.mgrid[:N, :N]
    dp = np.zeros((N, N))
    for a in range(-4, 5):
        for b in range(-4, 5):
            p = N / 2 + scale * (Q @ (a, b))
            dp += np.exp(-((rr - p[0]) ** 2 + (cc - p[1]) ** 2) / (2 * sigma**2))
    return dp


def test_strain_ramp_is_recovered_by_batched_and_per_pattern_fits():
    # diffraction vectors grow by s down the scan, so the real-space lattice shrinks: e = 1/s - 1
    s = 1.0 + 0.005 * np.arange(5)
    data = np.stack([[_pattern(si)] * 2 for si in s]).astype(np.float32)
    A = N * np.linalg.inv(Q).T  # real-space basis in transform pixels: the seed

    results = {}
    for batch_fit in (True, False):
        work = StrainMapAutocorrelation.from_array(data)
        work.dataset.metadata.update(q_to_r_rotation_ccw_deg=0.0, q_transpose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            work.preprocess(mode="linear", hann_window=True, plot=False)
            work.choose_basis_vectors(g1=A[:, 0], g2=A[:, 1], plot=False)
            work.fit_lattice(
                method="cft", refine_radius_px=3, batch_fit=batch_fit, progressbar=False
            )
        sm = work.calculate_strain_map(g1_ref=work.g1_array[0, 0], g2_ref=work.g2_array[0, 0])
        expected = (1.0 / s - 1.0)[:, None] * np.ones((1, 2))
        for name in ("e_rr", "e_cc"):
            np.testing.assert_allclose(
                getattr(sm, name).array,
                expected,
                atol=5e-4,
                err_msg=f"{name}, batch_fit={batch_fit}",
            )
        np.testing.assert_allclose(
            sm.phi.array, 0.0, atol=5e-4, err_msg=f"phi, batch_fit={batch_fit}"
        )
        results[batch_fit] = work.g1_array, work.g2_array

    # the two paths share the cft refinement, so they agree to rounding error
    np.testing.assert_allclose(results[True][0], results[False][0], atol=1e-6)
    np.testing.assert_allclose(results[True][1], results[False][1], atol=1e-6)
