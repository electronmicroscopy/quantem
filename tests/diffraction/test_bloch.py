"""Tests for quantem.diffraction.bloch."""

from functools import lru_cache

import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import bloch
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
    zones = torch.tensor([[0.1, 0.9, 1.0], [0.3, 0.5, 1.0], [0.05, 1.0, 1.1]], dtype=torch.float64)
    N = zones.shape[0]
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    q_true = quat_from_zone_axis(zones)
    for i in range(N):
        p = dynamical_pattern(ti_beta, q_true[i], t_true, energy_ev=200e3, sg_max=0.06, k_max=1.5)
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
    # thickness scan itself with the true orientations (the joint tilt and
    # thickness search is refine_dynamical, tested below)
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


# ----------------------------------------------------------------------
# CBED / LACBED / Kossel / Kossel reference pattern
# ----------------------------------------------------------------------


@lru_cache(maxsize=None)
def _si(absorptive: bool) -> Crystal:
    """Silicon, built once per session; the tests only read it (the Bloch
    code caches lattice data on it, which every test shares safely)."""
    si = Crystal.from_ase(bulk("Si", "diamond", a=5.431, cubic=True), name="Si", verbose=False)
    si.calculate_structure_factors(k_max=3.0)
    if absorptive:
        si.calculate_dynamical_structure_factors(energy_ev=200e3, k_max=3.0)
    return si


def _zone_110() -> torch.Tensor:
    return quat_from_zone_axis(torch.tensor([[1.0, 1.0, 0.0]]) / np.sqrt(2))[0]


def test_zero_tilt_matches_dynamical_pattern():
    si = _si(absorptive=True)
    q = _zone_110()
    t = torch.tensor([800.0])
    tilts = torch.zeros((1, 2), dtype=torch.float64)
    inten, g_xy, hkl = bloch._cbed_amplitudes(si, q, tilts, t, 200e3, sg_max=0.08, k_max=1.3)
    ref = bloch.dynamical_pattern(si, q, t, energy_ev=200e3, sg_max=0.08, k_max=1.3)
    # same beams (000 first in CBED) and identical intensities
    assert hkl.shape[0] == ref["hkl"].shape[0] + 1
    assert torch.allclose(inten[0, 0, 1:], ref["intensity"][0], rtol=1e-10, atol=1e-12)


def test_unitarity_without_absorption():
    si = _si(absorptive=False)
    q = _zone_110()
    tilts = bloch.tilt_grid(2.0, 200e3, n_rings=2)
    inten, _, _ = bloch._cbed_amplitudes(
        si, q, tilts, torch.tensor([500.0, 1500.0]), 200e3, sg_max=0.08, k_max=1.3
    )
    # Hermitian structure matrix: evolution is unitary in the beam space
    total = inten.sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-8)


def test_lacbed_centrosymmetric_disk():
    si = _si(absorptive=True)
    q = _zone_110()
    res = bloch.calculate_lacbed(
        si,
        q,
        800.0,
        hkl=(0, 0, 0),
        energy_ev=200e3,
        semiconv_mrad=6.0,
        n_pixels=24,
        sg_max=0.08,
        k_max=1.3,
    )
    disk = res["disk"]
    # Si is centrosymmetric: the bright field rocking surface at a zone axis
    # is inversion symmetric, I(t) = I(-t). Small residuals come from beam
    # truncation at the s_g cutoff (|s_g| differs slightly for +g and -g),
    # so the tolerance is physical rather than numerical.
    flipped = disk[::-1, ::-1]
    m = np.isfinite(disk) & np.isfinite(flipped)
    assert m.sum() > 100
    assert np.allclose(disk[m], flipped[m], rtol=2e-3, atol=1e-5)


def test_cbed_library_common_grid():
    si = _si(absorptive=True)
    q0 = _zone_110()
    quats = torch.stack([q0, q0])
    lib = bloch.calculate_cbed_library(
        si,
        quats,
        thickness_A=600.0,
        energy_ev=200e3,
        semiconv_mrad=3.0,
        k_max=1.0,
        progress_bar=False,
    )
    assert lib["patterns"].shape[0] == 2
    assert lib["patterns"].shape[1] == lib["patterns"].shape[2]
    assert np.allclose(lib["patterns"][0], lib["patterns"][1])
    assert lib["patterns"][0].max() > 0


def test_cbed_geometry_and_normalization():
    """Without absorption the evolution is unitary, so a detector holding
    every disk carries the full intensity (the pattern is the tilt
    average). In a thin crystal the direct beam disk is uniform and centered
    on the center pixel; diffracted disks sit at their g (rows qx, columns
    qy). Checked on and off a zone axis."""
    si = _si(absorptive=False)
    for q in (_zone_110(), _tilted_110()):
        res = bloch.calculate_cbed(
            si,
            q,
            [10.0, 500.0],
            energy_ev=200e3,
            semiconv_mrad=2.0,
            n_rings=3,
            sg_max=0.06,
            k_max=0.8,
        )
        thin, thick = res["pattern"]
        H = thin.shape[0]
        c = (H - 1) // 2
        assert np.allclose(res["pattern"].sum(axis=(1, 2)), 1.0, rtol=1e-9)
        assert np.allclose(res["g_xy"][0], 0.0)
        r_px = res["disk_radius"] / res["sampling"]
        yy, xx = np.mgrid[0:H, 0:H]
        w = thin * (np.hypot(yy - c, xx - c) <= r_px + 1.5)
        assert w.sum() > 0.95
        assert abs((w * yy).sum() / w.sum() - c) < 0.05
        assert abs((w * xx).sum() / w.sum() - c) < 0.05
        # every disk sits at its g: the pattern is nonzero only inside the
        # disks centered at (row, col) = (qx, qy) / sampling + center
        centers = res["g_xy"] / res["sampling"] + c
        d = np.hypot(yy[..., None] - centers[:, 0], xx[..., None] - centers[:, 1]).min(-1)
        assert np.all(thick[d > r_px + 1.5] == 0)


def test_cbed_detector_crop_drops_outside_samples():
    """A smaller detector is a crop of the larger one: samples beyond its
    edge are dropped, not piled onto the border pixels."""
    si = _si(absorptive=True)
    q = _tilted_110()
    kw = dict(energy_ev=200e3, semiconv_mrad=2.0, n_rings=3, sg_max=0.06, k_max=0.8)
    full = bloch.calculate_cbed(si, q, 500.0, **kw)
    s = full["sampling"]
    small = bloch.calculate_cbed(si, q, 500.0, pixel_size=s, q_max_plot=0.3, **kw)
    h_full = (full["pattern"].shape[0] - 1) // 2
    h_small = (small["pattern"].shape[0] - 1) // 2
    crop = full["pattern"][
        h_full - h_small : h_full + h_small + 1, h_full - h_small : h_full + h_small + 1
    ]
    assert small["pattern"].sum() < 0.99 * full["pattern"].sum()  # disks were cut
    assert np.allclose(small["pattern"], crop, rtol=1e-12, atol=1e-15)


def test_kossel_bright_field_matches_lacbed():
    si = _si(absorptive=True)
    q = _zone_110()
    kw = dict(energy_ev=200e3, semiconv_mrad=15.0, sg_max=0.06, k_max=0.8)
    kos = bloch.calculate_kossel(si, q, 900.0, n_pixels=32, progress_bar=False, **kw)
    lac = bloch.calculate_lacbed(si, q, 900.0, hkl=(0, 0, 0), n_pixels=32, **kw)
    a, b = kos["bright_field"], lac["disk"]
    m = np.isfinite(a) & np.isfinite(b)
    assert m.sum() > 300
    assert np.allclose(a[m], b[m], rtol=1e-10, atol=1e-12)

    # the summed pattern includes the direct beam plus every diffracted
    # cone, so inside the aperture it can only exceed the bright field
    pat = kos["pattern"]
    assert (pat[m] >= a[m] - 1e-9).mean() > 0.99
    assert np.all(pat[~np.isfinite(a)] == 0)


def test_kossel_pattern_orientation_matches_bright_field():
    """The full Kossel pattern is stored on the bright field's axes, (row,
    col) = (theta_y, theta_x): off a zone axis, where no symmetry hides a
    transposition, the deficiency lines of the direct beam make the two
    correlate, and the transposed pattern does not."""
    si = _si(absorptive=True)
    kos = bloch.calculate_kossel(
        si,
        _tilted_110(),
        800.0,
        energy_ev=200e3,
        semiconv_mrad=15.0,
        n_pixels=32,
        sg_max=0.06,
        k_max=0.7,
        progress_bar=False,
    )
    a, p = kos["bright_field"], kos["pattern"]
    m = np.isfinite(a)
    cc = np.corrcoef(a[m], p[m])[0, 1]
    cc_t = np.corrcoef(a[m], p.T[m])[0, 1]
    assert cc > 0.4
    assert cc - cc_t > 0.3
    assert np.isclose(kos["mrad_per_pixel"], 2 * 15.0 / 31)


# The reference-pattern tests share one coarse reference: 3 mrad sampling and
# beams to 0.7 1/A cost ~3 s, against ~30 s for 2 mrad and 1.0 1/A, and keep
# the comparisons meaningful (the lookup and the line model are compared at
# the reference's resolution).
REF_KW = dict(energy_ev=200e3, sg_max=0.06, k_max=0.7)
REF_STEP_MRAD = 3.0


def _tilted_110():
    from quantem.diffraction.rotations import qmult, quat_from_axis_angle

    tilt = quat_from_axis_angle(
        torch.tensor([1.0, 0.3, 0.0], dtype=torch.float64) / np.hypot(1, 0.3),
        torch.tensor(np.deg2rad(5.0), dtype=torch.float64),
    )
    return qmult(tilt, _zone_110())


@pytest.fixture(scope="module")
def si_reference():
    return bloch.calculate_kossel_reference(
        _si(absorptive=True),
        [800.0],
        angle_step_mrad=REF_STEP_MRAD,
        progress_bar=False,
        **REF_KW,
    )


@pytest.fixture(scope="module")
def si_lines():
    return bloch.kossel_lines(_si(absorptive=True), 800.0, energy_ev=200e3, k_max=REF_KW["k_max"])


def _direct_bright_field(q, semiconv_mrad=25.0, n_pixels=48):
    return bloch.calculate_kossel(
        _si(absorptive=True),
        q,
        800.0,
        semiconv_mrad=semiconv_mrad,
        n_pixels=n_pixels,
        progress_bar=False,
        **REF_KW,
    )["bright_field"]


def _blurred_cc(a, b, n_pixels=48, semiconv_mrad=25.0):
    """Correlation of two bright fields inside the aperture after blurring
    both to the reference's resolution. Bilinear splatting onto the Lambert
    grid and bilinear lookup are two triangle kernels of one grid step,
    together a blur of standard deviation step / sqrt(3)."""
    from scipy.ndimage import gaussian_filter

    px_mrad = 2 * semiconv_mrad / (n_pixels - 1)
    sigma = REF_STEP_MRAD / np.sqrt(3) / px_mrad
    m = np.isfinite(a) & np.isfinite(b)
    assert m.sum() > 1000
    af = gaussian_filter(np.nan_to_num(a), sigma)
    bf = gaussian_filter(np.nan_to_num(b), sigma)
    return np.corrcoef(af[m], bf[m])[0, 1]


def test_reference_pattern_lookup(si_reference):
    q = _zone_110()
    assert si_reference["k_max"] == REF_KW["k_max"]
    fast = bloch.kossel_from_reference(si_reference, q, semiconv_mrad=25.0, n_pixels=48)
    assert np.isclose(fast["mrad_per_pixel"], 2 * 25.0 / 47)
    assert _blurred_cc(fast["bright_field"], _direct_bright_field(q)) > 0.9

    # off-zone orientation: catches in-plane sign errors that zone-axis
    # symmetry hides (the reference stores the ANTI-propagation direction)
    q2 = _tilted_110()
    fast2 = bloch.kossel_from_reference(si_reference, q2, semiconv_mrad=25.0, n_pixels=48)
    assert _blurred_cc(fast2["bright_field"], _direct_bright_field(q2)) > 0.9


def test_kossel_polar_from_reference_matches_cartesian(si_reference):
    """The polar lookup samples the same function: on a Cartesian grid with
    pixels at the polar radii, the azimuth 0 and 90 degree rows coincide
    with the center row and column."""
    q = _tilted_110()
    n_r = 8
    pol = bloch.kossel_polar_from_reference(
        si_reference, q, semiconv_mrad=20.0, n_radial=n_r, n_azimuthal=16
    )
    assert pol["polar"].shape == (16, n_r)
    assert np.allclose(pol["radii_mrad"], 20.0 * np.arange(1, n_r + 1) / n_r)
    cart = bloch.kossel_from_reference(si_reference, q, semiconv_mrad=20.0, n_pixels=2 * n_r + 1)
    bf = cart["bright_field"]
    # (row, col) = (theta_y, theta_x): azimuth 0 runs along +col, 90 along +row
    assert np.allclose(pol["polar"][0], bf[n_r, n_r + 1 :])
    assert np.allclose(pol["polar"][4], bf[n_r + 1 :, n_r])


def test_plot_kossel_reference(si_reference, si_lines, monkeypatch):
    import matplotlib.pyplot as plt

    fig, ax = bloch.plot_kossel_reference(si_reference, _si(True), lines=si_lines, upsample=1)
    assert len(ax.images) == 1 and len(ax.texts) > 0
    plt.close(fig)

    # a reference computed without a beam cutoff stores k_max=None; the
    # default line set then falls back to 1.2 1/A instead of failing
    seen = {}

    def fake_lines(crystal, thicknesses_A, energy_ev, k_max):
        seen["k_max"] = k_max
        return si_lines

    monkeypatch.setattr(bloch, "kossel_lines", fake_lines)
    fig, ax = bloch.plot_kossel_reference({**si_reference, "k_max": None}, _si(True), upsample=1)
    assert seen["k_max"] == 1.2
    plt.close(fig)


def test_kossel_lines_rejects_missing_cutoff_and_empty_set():
    si = _si(absorptive=True)
    with pytest.raises(ValueError, match="k_max"):
        bloch.kossel_lines(si, 800.0, energy_ev=200e3, k_max=None)
    with pytest.raises(ValueError, match="min_depth"):
        bloch.kossel_lines(si, 800.0, energy_ev=200e3, k_max=0.4, min_depth=2.0)


def test_kossel_lines_render_matches_direct(si_lines):
    q = _tilted_110()
    lines = si_lines
    direct = _direct_bright_field(q, semiconv_mrad=40.0)
    # every line is a band edge: the +g and -g cones of a row sit at
    # +-theta_B, never on the zone plane
    first = lines["line_order"] == 1
    assert torch.all(lines["line_u"][first] > 0)
    assert torch.all(lines["line_u"][lines["line_order"] == -1] < 0)
    r = bloch.render_kossel_lines(lines, q, semiconv_mrad=40.0, n_pixels=48)
    a = r["bright_field"]
    m = np.isfinite(a) & np.isfinite(direct)
    assert m.sum() > 1000
    assert np.corrcoef(a[m], direct[m])[0, 1] > 0.98

    # polar rendering samples the same function: its first ring must
    # agree with the Cartesian pattern evaluated at those angles
    pol = bloch.render_kossel_lines(
        lines, q, semiconv_mrad=40.0, polar=True, n_radial=10, n_azimuthal=12
    )["polar"]
    assert pol.shape == (12, 10)
    assert np.all(np.isfinite(pol))
    assert pol.min() > 0 and pol.max() < 1.5 * float(lines["background"][0])


def test_kossel_line_segments_on_cones(si_lines):
    from quantem.diffraction.rotations import quat_to_matrix

    lines = si_lines
    q = _tilted_110()
    alpha = 40.0
    seg = bloch.kossel_line_segments(lines, q, semiconv_mrad=alpha)
    n = seg["depth"].shape[0]
    assert n >= 3
    R = quat_to_matrix(q).numpy()
    g_c = lines["g_hat"].numpy()
    hkl_row = lines["hkl_row"].numpy()
    for k in range(n):
        # row of this line from its hkl (an integer multiple of the row vector)
        h = seg["hkl"][k]
        ri = next(i for i in range(g_c.shape[0]) if np.all(np.cross(hkl_row[i], h) == 0))
        n_ord = int(np.round(np.dot(h, hkl_row[ri]) / np.dot(hkl_row[ri], hkl_row[ri])))
        u = n_ord * bloch.electron_wavelength_angstrom(200e3) * float(lines["g_len"][ri]) / 2
        g_lab = R @ g_c[ri]
        for key in ("start_mrad", "stop_mrad"):
            row, col = seg[key][k] * 1e-3
            assert np.isclose(np.hypot(row, col), alpha * 1e-3)
            d_lab = np.array([-col, -row, np.sqrt(1 - row**2 - col**2)])
            assert abs(d_lab @ g_lab - u) < 1e-9
        # polar end points are the same points
        phi, r = seg["start_polar"][k]
        assert np.isclose(r, alpha)
        assert np.allclose([alpha * np.sin(phi), alpha * np.cos(phi)], seg["start_mrad"][k])
        assert seg["width_mrad"][k] > 0 and 0 < seg["depth"][k] <= 1


def test_overlay_kossel_segments_registration(si_lines):
    """Segment end points lie on the aperture edge, which is the pixel
    circle of radius (n - 1) / 2 about the center pixel (Cartesian) and the
    last column (polar)."""
    import matplotlib.pyplot as plt

    q = _tilted_110()
    alpha, n = 40.0, 33
    seg = bloch.kossel_line_segments(si_lines, q, semiconv_mrad=alpha)
    n_draw = int((seg["depth"] >= 0.01).sum())
    assert n_draw >= 3
    fig, ax = plt.subplots()
    bloch.overlay_kossel_segments(ax, seg, alpha, n_pixels=n, min_depth=0.01)
    assert len(ax.lines) == n_draw
    c = (n - 1) / 2
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        assert np.allclose(np.hypot(np.asarray(x) - c, np.asarray(y) - c), c)
    plt.close(fig)

    n_r, n_az = 16, 90
    fig, ax = plt.subplots()
    bloch.overlay_kossel_segments(
        ax, seg, alpha, polar=True, n_radial=n_r, n_azimuthal=n_az, min_depth=0.01
    )
    assert len(ax.lines) == n_draw
    for ln in ax.lines:
        cols = np.asarray(ln.get_xdata())
        # radius semiconv sits in the last column, n_radial - 1
        assert np.isclose(cols.max(), n_r - 1)
        assert np.all(cols <= n_r - 1 + 1e-9)
    plt.close(fig)


def test_reference_residual_hybrid(si_reference, si_lines):
    q = _zone_110()
    si = _si(absorptive=True)
    reference = dict(si_reference)  # the residual is added in place
    bloch.kossel_reference_residual(reference, si_lines, si)
    assert reference["residual"].shape == reference["lambert"].shape
    assert np.all(np.isfinite(reference["residual"]))
    assert "residual" not in si_reference
    direct = _direct_bright_field(q)
    plain = bloch.render_kossel_lines(si_lines, q, semiconv_mrad=25.0, n_pixels=48)
    hybrid = bloch.render_kossel_lines(
        si_lines, q, semiconv_mrad=25.0, n_pixels=48, reference=reference
    )
    cc_plain = _blurred_cc(plain["bright_field"], direct)
    cc_hybrid = _blurred_cc(hybrid["bright_field"], direct)
    # on the zone axis the many-beam residual must improve the line model
    assert cc_hybrid > cc_plain + 0.05
    assert cc_hybrid > 0.9


def test_refine_dynamical_recovery():
    """Bragg-vector dynamical refinement: thickness, tilt, in-plane strain and
    rotation recovered from noise-free patterns of a strained, tilted cell."""
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import (
        misorientation_angle_deg,
        qmult,
        qnormalize,
        quat_from_axis_angle,
    )

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N = 6
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    t_true = torch.tensor([300.0, 450.0, 600.0, 300.0, 450.0, 600.0])
    A_true = torch.tensor([[1.010, 0.003], [0.003, 0.995]], dtype=torch.float64)
    rot = np.deg2rad(0.3)
    qz = torch.tensor([np.cos(rot / 2), 0.0, 0.0, np.sin(rot / 2)], dtype=torch.float64)
    q_expect = torch.stack([qmult(qz, q_true[i]) for i in range(N)])
    deform3 = torch.eye(3, dtype=torch.float64)
    deform3[:2, :2] = A_true
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_expect[i],
            torch.zeros((1, 2), dtype=torch.float64),
            t_true[i : i + 1],
            energy_ev,
            0.06,
            1.0,
            progress_bar=False,
            deform=deform3,
        )
        inten_np = inten[0, 0, 1:].numpy()
        keep = inten_np > 1e-3 * inten_np.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], inten_np[keep]])
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=energy_ev)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=5.0, verbose=False)
    om.match_orientations(progress_bar=False)
    # start 0.15 degrees off the truth about random in-plane axes, unstrained
    phis = rng.uniform(0, 2 * np.pi, N)
    om.quats[0, :, 0] = torch.stack(
        [
            qmult(
                quat_from_axis_angle(
                    torch.tensor([np.cos(p), np.sin(p), 0.0], dtype=torch.float64),
                    torch.tensor(np.deg2rad(0.15), dtype=torch.float64),
                ),
                q_true[i],
            )
            for i, p in enumerate(phis)
        ]
    )
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=np.arange(150, 800, 25.0),
        tilt_stages=((0.25, 0.025), (0.04, 0.005)),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    # positions with too few beams cannot constrain a 2x2 deformation
    valid = np.array([peaks[0, i].numpy().shape[0] >= 6 for i in range(N)])
    assert valid.sum() >= 4
    err = misorientation_angle_deg(q_expect, om.quats[0, :, 0], xtl.sym_quats).numpy()[valid]
    t_err = np.abs(res["thickness"][0].numpy() - t_true.numpy())[valid]
    A_err = (res["deformation"][0, valid, 0] - A_true[None]).abs().max()
    assert float(A_err) < 1e-3
    assert np.median(err) < 0.03
    assert (t_err <= 25).sum() >= valid.sum() - 1
    # crystal-frame strain of an unstrained position is zero, of a strained
    # one has the right magnitude
    sc = bloch.strain_crystal_frame(res["deformation"][0, :, 0], om.quats[0, :, 0])
    eps = sc["eps_crystal"]
    assert torch.allclose(eps, eps.transpose(-1, -2))
    assert float(eps.abs().max()) < 0.02


def test_image_refinement_round_trip():
    """Rendered patterns with known disk shape: fit_disk_shape recovers the
    radius and edge, and the image refinement keeps a correct thickness."""
    from types import SimpleNamespace

    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import qnormalize

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    N = 4
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    t_true = torch.tensor([300.0, 450.0, 600.0, 400.0])
    shape = (64, 64)
    pixel_size, rot, ellipse = 0.04, 15.0, (0.003, -0.002)
    disk_r, edge = 3.0, 0.75
    origins = np.full((1, N, 2), 32.0)
    imgs = np.zeros((1, N) + shape)
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        im, _, _ = bloch.render_pattern_image(
            xtl,
            q_true[i],
            [float(t_true[i])],
            energy_ev,
            shape,
            origins[0, i],
            pixel_size,
            rot,
            ellipse,
            None,
            disk_r,
            edge,
            sg_max=0.06,
            k_max=1.0,
        )
        imgs[0, i] = rng.poisson(im[0, 0].numpy() * 1e5 + 20)
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
            torch.zeros((1, 2), dtype=torch.float64),
            t_true[i : i + 1],
            energy_ev,
            0.06,
            1.0,
            progress_bar=False,
            fast_absorption=True,
        )
        inten_np = inten[0, 0, 1:].numpy()
        keep = inten_np > 1e-3 * inten_np.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], inten_np[keep]])
    peaks.metadata["rotation_ccw_deg"] = rot
    dataset = SimpleNamespace(array=imgs, shape=imgs.shape)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=energy_ev)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=5.0, verbose=False)
    om.match_orientations(progress_bar=False)
    om.quats[0, :, 0] = q_true
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=np.arange(200, 700, 50.0),
        tilt_stages=((0.05, 0.05),),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    valid = [i for i in range(N) if peaks[0, i].numpy().shape[0] >= 6]
    assert len(valid) >= 2
    shape_fit = bloch.fit_disk_shape(
        dataset,
        pm,
        res,
        origins,
        pixel_size,
        rot,
        ellipse,
        positions=[(0, i) for i in valid],
        radii_px=np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
        edges_px=np.array([0.5, 0.75, 1.0, 1.5]),
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    assert shape_fit["disk_radius_px"] == disk_r
    assert shape_fit["edge_px"] == edge
    img = bloch.refine_dynamical_image(
        dataset,
        pm,
        res,
        origins,
        pixel_size,
        disk_r,
        edge,
        rot,
        ellipse,
        thickness_half_range_A=50,
        thickness_step_A=25,
        tilt_stage=(0.02, 0.01),
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    t_err = np.abs(img["thickness"][0].numpy() - t_true.numpy())[valid]
    assert (t_err <= 25).sum() >= len(valid) - 1
    assert np.all(np.isfinite(img["cost"][0].numpy()[valid]))
    assert pm.metadata["dynamical_image"]["disk_radius_px"] == disk_r
    pm.apply_dynamical(res)
    assert pm.metadata["dynamical_applied"]["precession_deg"] == 0.0


def test_coupling_lookup_cannot_alias():
    # a difference vector outside the stored factor box must come back as a
    # missing factor (zero), never as another reflection's factor
    from types import SimpleNamespace

    crystal = SimpleNamespace(
        hkl_dyn=torch.tensor([[1, 0, 0], [-1, 0, 0], [-1, 1, 0]]),
        U_dyn=torch.tensor([1, 1, 7], dtype=torch.complex128),
    )
    U, _, _ = bloch._coupling_matrix(crystal, torch.tensor([[1, 0, 0], [-1, 0, 0]]), 1.0)
    assert U[0, 1] == 0 and U[1, 0] == 0


def test_illumination_nodes_moments():
    # the convergence disk is integrated with the uniform-area measure: the
    # second moment of a disk of radius R is R^2 / 4 per axis; the ring is
    # normalized and its mean vanishes
    lam = bloch.electron_wavelength_angstrom(200e3)
    k0 = 1.0 / lam
    t, w = bloch.illumination_nodes(200e3, semiconv_mrad=5.0, n_disk_radial=3, n_disk_azimuthal=16)
    R = k0 * np.sin(5e-3)
    assert np.isclose(float(w.sum()), 1.0)
    assert np.isclose(float((w * t[:, 0] ** 2).sum()), R**2 / 4, rtol=1e-10)
    t, w = bloch.illumination_nodes(200e3, precession_deg=0.5, n_precession=16)
    assert np.isclose(float(w.sum()), 1.0) and float(t.mean(0).abs().max()) < 1e-12
    assert np.allclose(torch.linalg.norm(t, dim=1).numpy(), k0 * np.sin(np.deg2rad(0.5)))
    t, w = bloch.illumination_nodes(200e3)
    assert t.shape == (1, 2) and float(w[0]) == 1.0


def test_mean_absorption_and_forbidden_beam():
    """Pure mean absorption damps the total intensity as exp(-2 pi u0 z/k0);
    a glide-forbidden reflection (Si 200) acquires intensity through double
    diffraction, which requires it to be in the beam list."""
    si = _si(absorptive=True)
    q = _zone_110()
    z = torch.tensor([400.0, 800.0], dtype=torch.float64)
    inten, g_xy, hkl = bloch._cbed_amplitudes(
        si, q, torch.zeros((1, 2), dtype=torch.float64), z, 200e3, sg_max=0.06, k_max=1.0
    )
    keys = [tuple(h) for h in hkl.tolist()]
    assert (0, 0, 2) in keys or (2, 0, 0) in keys or (0, 2, 0) in keys
    i200 = next(i for i, h in enumerate(keys) if sorted(abs(v) for v in h) == [0, 0, 2])
    assert float(inten[0, 1, i200]) > 1e-4  # populated by multiple scattering
    # mean absorption alone: strip the off-diagonal absorptive part
    U, u0, absorptive = bloch._coupling_matrix(si, hkl, bloch.relativistic_gamma(200e3))
    Uel = 0.5 * (U + U.conj().T)
    lam = bloch.electron_wavelength_angstrom(200e3)
    k0 = 1.0 / lam
    s_t = torch.zeros((1, hkl.shape[0]), dtype=torch.float64)
    gl = bloch.qrotate(q, hkl[1:].to(torch.float64) @ si.lat_recip)
    s_t[0, 1:] = (2 * gl[:, 2] - lam * (gl**2).sum(1)) / (2 - 2 * lam * gl[:, 2])
    inten_np = bloch._bloch_solve(Uel, u0, True, s_t, k0, z, fast_absorption=False)
    total = inten_np[0].sum(dim=1).numpy()
    assert np.allclose(total, np.exp(-2 * np.pi * u0 * z.numpy() / k0), rtol=1e-8)


def test_fourier_ring_matches_quadrature():
    """The harmonic propagation of the centered precession ring reproduces a
    converged azimuthal quadrature, with the full complex coupling."""
    si = _si(absorptive=True)
    q = _zone_110()
    z = torch.tensor([300.0, 600.0])
    trial = torch.zeros((1, 2), dtype=torch.float64)
    beams = bloch.select_dynamical_beams(si, q, 200e3, np.deg2rad(0.4), 0.06, 1.0)
    ring, w = bloch.illumination_nodes(200e3, precession_deg=0.4, n_precession=96)
    inten, g_xy, _ = bloch._cbed_amplitudes(
        si, q, ring, z, 200e3, 0.06, 1.0, tilt_batch=128, beams=beams
    )
    ref = (inten * w[:, None, None]).sum(0)
    got, g2 = bloch.average_bloch_fourier(si, q, trial, z, 200e3, 0.4, 0.06, 1.0, beams=beams)
    assert np.allclose(g2.numpy(), g_xy.numpy())
    assert np.allclose(got[0].numpy(), ref.numpy(), atol=1e-11, rtol=1e-9)
    # a displaced ring through the sampled coefficients
    trial = torch.tensor([[0.15, -0.1]], dtype=torch.float64)
    inten, _, _ = bloch._cbed_amplitudes(
        si, q, ring + trial, z, 200e3, 0.06, 1.0, tilt_batch=128, beams=beams
    )
    ref = (inten * w[:, None, None]).sum(0)
    got, _ = bloch.average_bloch_fourier(
        si, q, trial, z, 200e3, 0.4, 0.06, 1.0, beams=beams, n_geometry=128
    )
    assert np.allclose(got[0].numpy(), ref.numpy(), atol=1e-9, rtol=1e-7)


def test_refine_dynamical_reported_cost_reproducible():
    """The stored cost and thickness belong to the stored orientation."""
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import qnormalize

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(3)
    q_true = qnormalize(torch.randn(3, 4, dtype=torch.float64))
    peaks = Vector.from_shape(
        (1, 3), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(3):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([450.0]),
            energy_ev,
            0.06,
            1.0,
            progress_bar=False,
        )
        inten_np = inten[0, 0, 1:].numpy()
        keep = inten_np > 1e-3 * inten_np.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], inten_np[keep]])
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=energy_ev)
    om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False)
    om.match_orientations(progress_bar=False)
    om.quats[0, :, 0] = q_true
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=np.arange(300, 600, 50.0),
        tilt_stages=((0.1, 0.05),),
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    for i in range(3):
        if not torch.isfinite(res["cost"][0, i, 0]) or peaks[0, i].numpy().shape[0] < 5:
            continue
        q = res["quats"][0, i, 0]
        d3 = torch.eye(3, dtype=torch.float64)
        d3[:2, :2] = res["deformation"][0, i, 0]
        beams = bloch.select_dynamical_beams(
            xtl, res["quats_base"][0, i, 0], energy_ev, np.deg2rad(0.1) * np.sqrt(2), 0.06, 1.0, d3
        )
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q,
            torch.zeros((1, 2), dtype=torch.float64),
            np.arange(300, 600, 50.0),
            energy_ev,
            0.06,
            1.0,
            fast_absorption=False,
            deform=d3,
            beams=beams,
        )
        data = peaks[0, i].numpy().astype(np.float64)
        qxy = torch.as_tensor(data[:, :2])
        im = torch.as_tensor(data[:, 2]).clamp_min(0) ** 0.25
        cost, _, _, _ = bloch._dynamical_cost(inten[:, :, 1:], g_xy[1:], qxy, im, 0.05, 0.25, 0.02)
        t_idx = int(np.argmin(np.abs(np.arange(300, 600, 50.0) - float(res["thickness"][0, i]))))
        assert np.isclose(float(cost[0, t_idx]), float(res["cost"][0, i, 0]), rtol=1e-6, atol=1e-9)


def test_image_cost_radial_background():
    """A quadratic radial floor is removed by the radial background model
    and biases the constant one."""
    torch.manual_seed(0)
    shape = (48, 48)
    yy, xx = np.mgrid[0:48, 0:48]
    radius = torch.as_tensor(np.hypot(yy - 24.0, xx - 24.0))
    centers = torch.tensor([[24.0, 24.0], [30.0, 35.0], [15.0, 20.0], [36.0, 12.0]])
    inten = torch.tensor([0.8, 0.05, 0.02, 0.01], dtype=torch.float64)
    sim = bloch.render_disks(centers, inten, shape, 3.0, 0.7)
    sim_wrong = bloch.render_disks(
        centers, inten * torch.tensor([1.0, 0.5, 2.0, 1.0]), shape, 3.0, 0.7
    )
    floor = 5.0 + 0.2 * radius - 0.004 * radius**2
    meas = 1000 * sim + floor
    mask = bloch._image_mask(shape, (24.0, 24.0), None, 4.5)
    c_const = bloch._image_cost(meas, torch.stack([sim, sim_wrong]), mask, 0.5, "constant")
    c_rad = bloch._image_cost(meas, torch.stack([sim, sim_wrong]), mask, 0.5, "radial", radius)
    assert float(c_rad[0]) < 1e-12  # exact model with the right background
    assert float(c_const[0]) > 1e-4  # the constant background cannot absorb it
    assert float(c_rad[1]) > float(c_rad[0])


def test_refine_dynamical_with_precession_and_convergence():
    """End-to-end recovery with a precession ring and a convergence disk:
    the ground truth is integrated with denser illumination nodes than
    the model uses."""
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import (
        misorientation_angle_deg,
        qmult,
        qnormalize,
        quat_from_axis_angle,
    )

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    N = 3
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    t_true = torch.tensor([300.0, 450.0, 600.0])
    ring, w = bloch.illumination_nodes(
        energy_ev,
        precession_deg=0.4,
        n_precession=32,
        semiconv_mrad=1.5,
        n_disk_radial=3,
        n_disk_azimuthal=12,
    )
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
            ring,
            t_true[i : i + 1],
            energy_ev,
            0.06,
            1.0,
            tilt_batch=256,
            progress_bar=False,
        )
        I_avg = (inten[:, 0, 1:] * w[:, None]).sum(0).numpy()
        keep = I_avg > 1e-3 * I_avg.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], I_avg[keep]])
    om = OrientationMap.from_vectors(
        peaks, xtl, energy_ev=energy_ev, precession_deg=0.4, semiconv_mrad=1.5
    )
    # the matched orientations are replaced below, so a coarse plan will do
    # (the precession-integrated library is the costly part)
    om.build_plan(angle_step_zone_axis_deg=10.0, angle_step_in_plane_deg=10.0, verbose=False)
    om.match_orientations(progress_bar=False)
    phis = rng.uniform(0, 2 * np.pi, N)
    om.quats[0, :, 0] = torch.stack(
        [
            qmult(
                quat_from_axis_angle(
                    torch.tensor([np.cos(p), np.sin(p), 0.0], dtype=torch.float64),
                    torch.tensor(np.deg2rad(0.12), dtype=torch.float64),
                ),
                q_true[i],
            )
            for i, p in enumerate(phis)
        ]
    )
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=np.arange(200, 700, 25.0),
        tilt_stages=((0.15, 0.05), (0.03, 0.01)),
        n_precession=12,
        n_precession_search=12,
        n_disk_radial=2,
        n_disk_azimuthal=6,
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        progress_bar=False,
    )
    assert res["metadata"]["precession_deg"] == 0.4 and res["metadata"]["semiconv_mrad"] == 1.5
    valid = np.array([peaks[0, i].numpy().shape[0] >= 6 for i in range(N)])
    assert valid.sum() >= 2
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()[valid]
    t_err = np.abs(res["thickness"][0].numpy() - t_true.numpy())[valid]
    assert np.median(err) < 0.03
    assert (t_err <= 25).sum() >= valid.sum() - 1


def _smooth_map_setup(n, tilt_start_deg, corrupt=None):
    """A 1 x n 'map' of one grain: orientations a few hundredths of a degree
    apart, thickness varying slowly, strained cell; starts tilted by
    tilt_start_deg (and one position by `corrupt` degrees)."""
    from quantem.core.datastructures.vector import Vector
    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.phase import PhaseMap
    from quantem.diffraction.rotations import qmult, quat_from_axis_angle

    energy_ev = 200e3
    xtl = _si(absorptive=True)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    # a well-populated pattern: 1.5 degrees off the [110] zone axis
    base = qmult(
        quat_from_axis_angle(
            torch.tensor([0.6, 0.8, 0.0], dtype=torch.float64),
            torch.tensor(np.deg2rad(1.5), dtype=torch.float64),
        ),
        _zone_110(),
    )
    q_true = torch.stack(
        [
            qmult(
                quat_from_axis_angle(
                    torch.tensor([1.0, 0.3, 0.0], dtype=torch.float64) / np.hypot(1, 0.3),
                    torch.tensor(np.deg2rad(0.03 * i), dtype=torch.float64),
                ),
                base,
            )
            for i in range(n)
        ]
    )
    t_true = torch.tensor([400.0 + 25.0 * i for i in range(n)])
    A_true = torch.tensor([[1.008, 0.002], [0.002, 0.996]], dtype=torch.float64)
    deform3 = torch.eye(3, dtype=torch.float64)
    deform3[:2, :2] = A_true
    peaks = Vector.from_shape(
        (1, n), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(n):
        inten, g_xy, _ = bloch._cbed_amplitudes(
            xtl,
            q_true[i],
            torch.zeros((1, 2), dtype=torch.float64),
            t_true[i : i + 1],
            energy_ev,
            0.06,
            1.0,
            progress_bar=False,
            deform=deform3,
        )
        inten_np = inten[0, 0, 1:].numpy()
        keep = inten_np > 1e-3 * inten_np.max()
        peaks[0, i] = np.column_stack([g_xy[1:].numpy()[keep], inten_np[keep]])
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=energy_ev)
    om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False)
    om.match_orientations(progress_bar=False)
    phis = rng.uniform(0, 2 * np.pi, n)
    starts = []
    for i, p in enumerate(phis):
        ang = tilt_start_deg if (corrupt is None or i != corrupt[0]) else corrupt[1]
        starts.append(
            qmult(
                quat_from_axis_angle(
                    torch.tensor([np.cos(p), np.sin(p), 0.0], dtype=torch.float64),
                    torch.tensor(np.deg2rad(ang), dtype=torch.float64),
                ),
                q_true[i],
            )
        )
    om.quats[0, :, 0] = torch.stack(starts)
    om.corr[0, :, 0] = 1.0
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    return xtl, om, pm, q_true, t_true, A_true


def test_refine_dynamical_warm_start_matches_cold():
    from quantem.diffraction.rotations import (
        misorientation_angle_deg,
        qconj,
        qmult,
        quat_from_axis_angle,
    )

    n = 5
    kw = dict(
        thicknesses_A=np.arange(300, 700, 25.0),
        tilt_stages=((0.25, 0.05), (0.04, 0.01)),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        neighbor_rescue=False,
        progress_bar=False,
    )
    out = {}
    for warm in (False, True):
        xtl, om, pm, q_true, t_true, A_true = _smooth_map_setup(n, 0.1)
        q_match = om.quats[0, :, 0].clone()
        res = bloch.refine_dynamical(pm, warm_start=warm, **kw)
        out[warm] = (res, om.quats[0, :, 0].clone(), q_true, t_true, xtl)
    res_c, q_c, q_true, t_true, xtl = out[False]
    res_w, q_w, _, _, _ = out[True]
    assert not res_c["warm_started"].any()
    assert res_w["warm_started"][0, 1:, 0].all() and not res_w["warm_started"][0, 0, 0]
    # same solution from both routes, both correct
    d = misorientation_angle_deg(q_c, q_w, xtl.sym_quats).numpy()
    assert d.max() < 0.03
    assert np.allclose(res_c["thickness"][0].numpy(), res_w["thickness"][0].numpy())
    err = misorientation_angle_deg(q_true, q_w, xtl.sym_quats).numpy()
    assert err.max() < 0.03
    assert np.abs(res_w["thickness"][0].numpy() - t_true.numpy()).max() <= 25

    # the reported tilt is measured from the position's own matched
    # orientation on either route: every start was 0.1 degrees off the truth
    for res in (res_c, res_w):
        tilt = res["tilt_deg"][0].numpy()  # (n, 2)
        assert np.all(np.abs(np.hypot(tilt[:, 0], tilt[:, 1]) - 0.1) < 0.03)
        base = res["quats_base"][0, :, 0]
        # the base is the matched orientation turned about the beam only
        # (the in-plane rotation of the deformation fit)
        dq = qmult(base, qconj(q_match))
        assert torch.all(dq[:, 1:3].abs() < 1e-9)
        assert misorientation_angle_deg(base, q_match).max() < 0.05
        # and tilt x base gives the solution
        for i in range(n):
            wx, wy = np.deg2rad(tilt[i])
            ang = np.hypot(wx, wy)
            tq = quat_from_axis_angle(
                torch.tensor([wx / ang, wy / ang, 0.0], dtype=torch.float64),
                torch.tensor(ang, dtype=torch.float64),
            )
            q_rebuilt = qmult(tq, base[i])
            assert torch.allclose(
                q_rebuilt * torch.sign(q_rebuilt @ res["quats"][0, i, 0]),
                res["quats"][0, i, 0],
                atol=1e-9,
            )
        # the zero-tilt cost is at the matched orientation, above the final
        assert torch.all(res["cost_zero_tilt"][0, :, 0] > res["cost"][0, :, 0])
    assert np.allclose(res_c["tilt_deg"].numpy(), res_w["tilt_deg"].numpy(), atol=0.01)
    assert np.allclose(res_c["cost_zero_tilt"].numpy(), res_w["cost_zero_tilt"].numpy(), rtol=0.05)


def test_refine_dynamical_neighbor_rescue():
    from quantem.diffraction.rotations import misorientation_angle_deg

    n = 5
    # position 2 starts 0.45 degrees off: outside the coarse stage's reach,
    # so its cold search settles in a wrong basin; its neighbors are right
    xtl, om, pm, q_true, t_true, A_true = _smooth_map_setup(n, 0.1, corrupt=(2, 0.45))
    kw = dict(
        thicknesses_A=np.arange(300, 700, 25.0),
        tilt_stages=((0.25, 0.05), (0.04, 0.01)),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        warm_start=False,
        progress_bar=False,
    )
    res = bloch.refine_dynamical(pm, neighbor_rescue=False, **kw)
    err0 = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert err0[2] > 0.1  # the cold start fails there
    xtl, om, pm, q_true, t_true, A_true = _smooth_map_setup(n, 0.1, corrupt=(2, 0.45))
    res = bloch.refine_dynamical(pm, neighbor_rescue=True, **kw)
    err1 = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert bool(res["rescued"][0, 2])
    assert err1[2] < 0.03
    assert abs(float(res["thickness"][0, 2]) - float(t_true[2])) <= 25
    # map-level outputs
    maps = bloch.dynamical_maps(res, pm, crystal_index=0)
    assert maps["mask"][0].all()
    assert set(maps["strain"]) == {"aa", "bb", "cc", "ab", "ac", "bc"}
    assert torch.isfinite(maps["gain"][0]).all()


def test_refine_dynamical_threads_match_one_worker():
    from quantem.diffraction.rotations import misorientation_angle_deg

    # 32 positions: two blocks of 16 on two threads, one warm-start chain
    # broken at the block boundary. Both runs start from the same matched
    # orientations (update_orientations=False leaves them in place)
    n = 32
    kw = dict(
        thicknesses_A=np.arange(300, 700, 25.0),
        tilt_stages=((0.25, 0.05), (0.04, 0.01)),
        power_intensity=0.5,
        sg_max=0.06,
        k_max=1.0,
        neighbor_rescue=False,
        update_orientations=False,
        progress_bar=False,
    )
    xtl, om, pm, q_true, _, _ = _smooth_map_setup(n, 0.1)
    res_1 = bloch.refine_dynamical(pm, num_workers=1, **kw)
    res_2 = bloch.refine_dynamical(pm, num_workers=2, **kw)
    q_1, q_2 = res_1["quats"][0, :, 0], res_2["quats"][0, :, 0]
    assert int(res_1["warm_started"].sum()) == n - 1
    assert int(res_2["warm_started"].sum()) == n - 2
    # the first block is the same computation on either route
    assert torch.equal(q_1[:16], q_2[:16])
    assert torch.equal(res_1["thickness"][0, :16], res_2["thickness"][0, :16])
    # breaking the warm-start chain costs nothing in accuracy
    e1 = misorientation_angle_deg(q_true, q_1, xtl.sym_quats).numpy()
    e2 = misorientation_angle_deg(q_true, q_2, xtl.sym_quats).numpy()
    assert e2.mean() <= e1.mean() + 0.01


def _fake_dynamical_result():
    """A 2 x 2 refine_dynamical result with one candidate; position (1, 1)
    was not refined."""
    from types import SimpleNamespace

    from quantem.diffraction.rotations import quat_from_axis_angle

    R, C, F = 2, 2, 1
    q = quat_from_axis_angle(
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64)
    )
    deform = torch.eye(2, dtype=torch.float64).repeat(R, C, F, 1, 1)
    deform[0, 0, 0] = torch.tensor([[1.01, 0.0], [0.0, 0.99]], dtype=torch.float64)
    cost = torch.tensor([[0.10, 0.20], [0.15, torch.nan]], dtype=torch.float64)[..., None]
    done = torch.isfinite(cost[..., 0])
    result = {
        "candidate": torch.where(done, 0, -1),
        "phase_index": torch.where(done, 0, -1),
        "quats": q.repeat(R, C, F, 1),
        "deformation": deform,
        "cost": cost,
        "cost_zero_tilt": cost + 0.01,
        "thickness": torch.where(done, 400.0, torch.nan).to(torch.float64),
        "thickness_contrast": torch.full((R, C), 0.1, dtype=torch.float64),
        "tilt_deg": torch.full((R, C, 2), 0.05, dtype=torch.float64),
    }
    return result, SimpleNamespace(candidates=[(0, 0)])


def test_dynamical_maps_mask_unrefined():
    result, pm = _fake_dynamical_result()
    maps = bloch.dynamical_maps(result, pm)
    assert maps["mask"].tolist() == [[True, True], [True, False]]
    assert int(maps["phase_index"][1, 1]) == -1 and int(maps["phase_index"][0, 0]) == 0
    assert torch.isnan(maps["quats"][1, 1]).all() and torch.isfinite(maps["quats"][0, 0]).all()
    assert torch.isnan(maps["deformation"][1, 1]).all()
    assert torch.isnan(maps["thickness"][1, 1]) and float(maps["thickness"][0, 0]) == 400.0
    assert np.isclose(float(maps["gain"][0, 0]), 0.01)
    assert np.isclose(float(maps["tilt_deg"][0, 0]), 0.05 * np.sqrt(2))
    # the strain of the strained position, in the crystal frame (a pure
    # rotation about the beam keeps the normal strains on the diagonal)
    eps = maps["strain"]
    assert float(eps["aa"][0, 0]) < 0 < float(eps["bb"][0, 0])
    assert torch.isnan(eps["cc"][1, 1])
    # restricted to a crystal that won nowhere: empty mask
    assert not bloch.dynamical_maps(result, pm, crystal_index=1)["mask"].any()


def test_plot_dynamical_and_strain_maps():
    import matplotlib.pyplot as plt

    result, pm = _fake_dynamical_result()
    maps = bloch.dynamical_maps(result, pm)
    fig, axs = bloch.plot_dynamical_maps(maps)
    assert np.asarray(axs).size >= 4
    plt.close(fig)
    strain = {k: v.numpy() for k, v in maps["strain"].items()}
    fig, axs = bloch.plot_strain_crystal_frame(strain, mask=maps["mask"].numpy())
    assert np.asarray(axs).size >= 6
    plt.close(fig)
