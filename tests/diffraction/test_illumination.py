"""Illumination-averaged excitation envelopes (precession ring, convergence disk)."""

import numpy as np
import torch
from ase.build import bulk

from quantem.diffraction import bloch
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.illumination import (
    excitation_coefficients,
    gaussian_envelope,
    gaussian_envelope_ring_series,
    ring_disk_quadrature,
    slab_envelope,
)


def _quad_reference(fn, c, a, b, n_phi=256, n_r=12, n_psi=64):
    """Positive angular quadrature of fn(s) over s = c + ring_x + disk_x."""
    t, w = ring_disk_quadrature(a, b, n_phi=n_phi, n_r=n_r, n_psi=n_psi)
    return w @ fn(c + t[:, 0])


def test_gaussian_envelope_limits_and_quadrature():
    sigma = 0.025
    c = np.linspace(-0.1, 0.1, 21)
    # no illumination: the static envelope, exactly
    assert np.allclose(gaussian_envelope(c, 0.0, 0.0, sigma), np.exp(-0.5 * (c / sigma) ** 2))
    # ring, disk, and both, against positive quadrature of the static envelope
    for a, b in ((0.08, 0.0), (0.0, 0.02), (0.08, 0.02), (0.01, 0.005)):
        for ci in (0.0, 0.03, 0.07):
            ref = _quad_reference(lambda s: np.exp(-0.5 * (s / sigma) ** 2), ci, a, b)
            assert abs(gaussian_envelope(ci, a, b, sigma) - ref) < 1e-9
    # the ring series agrees with the transform
    assert np.allclose(
        gaussian_envelope_ring_series(c, 0.009, 0.04),
        gaussian_envelope(c, 0.009, 0.0, 0.04),
        atol=1e-11,
    )
    # torch in, torch out
    out = gaussian_envelope_ring_series(torch.as_tensor(c), torch.full((21,), 0.009), 0.04)
    assert isinstance(out, torch.Tensor) and out.shape == (21,)


def test_slab_envelope_limits_and_quadrature():
    z = 80.0
    c = np.linspace(-0.1, 0.1, 21)
    assert np.allclose(slab_envelope(c, 0.0, 0.0, z), np.sinc(c * z) ** 2, atol=1e-10)
    assert np.isclose(slab_envelope(np.array([0.05]), 0.0, 0.0, 0.0)[0], 1.0)
    for a, b in ((0.08, 0.0), (0.0, 0.02), (0.08, 0.02)):
        for ci in (0.0, 0.03):
            ref = _quad_reference(lambda s: np.sinc(s * z) ** 2, ci, a, b)
            assert abs(slab_envelope(np.array([ci]), a, b, z)[0] - ref) < 1e-9


def test_centered_ring_coefficients_exact():
    # s_g(phi) = c_g + a_g cos(phi - delta_g) exactly for a centered ring
    energy_ev, prec = 200e3, 0.5
    lam = bloch.electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    r = k0 * np.sin(np.deg2rad(prec))
    g = np.array([[0.8, 0.1, 0.02], [-0.3, 0.65, -0.015], [0.2, -0.9, 0.05]])
    c, a, b = excitation_coefficients(g, energy_ev, prec, 0.0)
    assert np.all(b == 0)
    phi = 2 * np.pi * (np.arange(360) + 0.3) / 360
    t = r * np.stack([np.cos(phi), np.sin(phi)], axis=1)
    kz = np.sqrt(k0**2 - (t**2).sum(1))[:, None]
    s = (2 * kz * g[:, 2] - 2 * (t @ g[:, :2].T) - (g**2).sum(1)) / (2 * (kz - g[:, 2]))
    delta = np.arctan2(-g[:, 1], -g[:, 0])
    model = c[None] + a[None] * np.cos(phi[:, None] - delta[None])
    assert np.abs(s - model).max() < 1e-12
    # zero illumination: c is the static excitation error
    c0, a0, b0 = excitation_coefficients(g, energy_ev, 0.0, 0.0)
    s0 = (2 * k0 * g[:, 2] - (g**2).sum(1)) / (2 * (k0 - g[:, 2]))
    assert np.allclose(c0, s0) and np.all(a0 == 0) and np.all(b0 == 0)


def test_slab_pattern_is_thin_bloch_limit():
    """The slab model with elastic couplings equals the Bloch calculation
    for a very thin crystal (first Born), including the 000-relative
    normalization, at zero and at nonzero precession."""
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=2.0)  # elastic Lobato factors only
    torch.manual_seed(1)
    from quantem.diffraction.rotations import qnormalize

    q = qnormalize(torch.randn(4, dtype=torch.float64))
    z = 6.0
    for prec in (0.0, 0.4):
        pat = xtl.generate_pattern(
            q,
            200e3,
            tol_excitation_mult=4.0,
            k_max=1.0,
            precession_deg=prec,
            excitation_model="slab",
            thickness_A=z,
        )
        ring, w = bloch.illumination_nodes(200e3, precession_deg=prec, n_precession=32)
        inten, g_xy, hkl = bloch._cbed_amplitudes(
            xtl, q, ring, torch.tensor([z]), 200e3, 0.3, 1.0, tilt_batch=64
        )
        dyn = (inten[:, 0, :] * w[:, None]).sum(0)
        lut = {tuple(h): i for i, h in enumerate(hkl.tolist())}
        common = [k for k, h in enumerate(pat["hkl"].tolist()) if tuple(h) in lut]
        idx = [lut[tuple(pat["hkl"][k].tolist())] for k in common]
        born = pat["intensity"].numpy()[common]
        strong = born > 0.2 * born.max()
        assert strong.sum() >= 4
        ratio = dyn[idx].numpy()[strong] / born[strong]
        # first Born holds to a few percent for the strong reflections at
        # 6 A (the remainder is the second-order multi-beam term); the
        # illumination average is shared exactly by both sides
        assert np.abs(ratio - 1).max() < 0.05


def test_refine_batched_matches_loop():
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.rotations import misorientation_angle_deg, qnormalize

    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    torch.manual_seed(2)
    N = 6
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        p = xtl.generate_pattern(q_true[i], 200e3, sigma_excitation=0.02, precession_deg=0.5)
        peaks[0, i] = np.stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1)
    out = []
    for batched in (True, False):
        om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3, precession_deg=0.5)
        om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False)
        om.match_orientations(progress_bar=False)
        om.refine_orientations(batched=batched, neighbor_rescue=False, progress_bar=False)
        out.append(om.quats[0, :, 0].clone())
    d = misorientation_angle_deg(out[0], out[1], xtl.sym_quats).numpy()
    assert d.max() < 1e-4


def test_relrod_factor():
    from quantem.diffraction.illumination import relrod_factor

    rng = np.random.default_rng(0)
    g = rng.normal(0, 0.5, (40, 3))
    g[:, 2] *= 0.05
    # normal along the beam: the excitation error is already along the rod
    assert np.allclose(relrod_factor(g, np.array([0.0, 0.0, 1.0]), 200e3), 1.0)
    # tilted normal: g + t n with t = -f s_g lies on the Ewald sphere to
    # first order in s_g
    n = np.array([np.sin(0.3), 0.0, np.cos(0.3)])
    f = relrod_factor(g, n, 200e3, precession_deg=0.0)
    s, _, _ = excitation_coefficients(g, 200e3)
    spot = g - (f * s)[:, None] * n[None]
    s_spot, _, _ = excitation_coefficients(spot, 200e3)
    big = np.abs(s) > 1e-3
    assert np.all(np.abs(s_spot[big]) < 0.05 * np.abs(s[big]))
    # torch in, torch out, same values
    f_t = relrod_factor(torch.as_tensor(g), torch.as_tensor(n), 200e3)
    assert isinstance(f_t, torch.Tensor) and np.allclose(f_t.numpy(), f)
    # an edge-on plate is never excited
    edge = relrod_factor(np.array([[0.0, 0.0, 0.0]]), np.array([1.0, 0.0, 0.0]), 200e3)
    assert edge[0] >= 1e6
