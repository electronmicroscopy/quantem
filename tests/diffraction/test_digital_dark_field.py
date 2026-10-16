"""Digital dark field: apertures, polar selection and grain labels on synthetic peaks."""

import numpy as np
import pytest

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import digital_dark_field as ddf


def _lattice_peaks(R=4, C=5, fields=("q_row", "q_col", "intensity")):
    """Square lattice g1=(10,0), g2=(0,10); cells with c >= 3 also carry (5,5)."""
    nested = []
    for r in range(R):
        row = []
        for c in range(C):
            pts = [[10 * i, 10 * j, 1.0 + r] for i in (-1, 0, 1) for j in (-1, 0, 1)]
            if c >= 3:
                pts.append([5.0, 5.0, 2.0])
            row.append(np.asarray(pts, dtype=float))
        nested.append(row)
    return Vector.from_data(nested, fields=list(fields))


def test_aperture_array_modes():
    g1, g2 = (10.0, 0.0), (0.0, 10.0)
    arr = ddf.aperture_array(g1, g2, n1_range=(-1, 1), n2_range=(-1, 1))
    assert arr.shape == (9, 2)
    no_center = ddf.aperture_array(
        g1, g2, n1_range=(-1, 1), n2_range=(-1, 1), radius_range=(1, np.inf)
    )
    assert no_center.shape == (8, 2)
    line = ddf.aperture_array(g1, mode="line", n1_range=(-2, 2), center=(50, 50))
    np.testing.assert_allclose(line[:, 1], 50.0)
    single = ddf.aperture_array(g1, g2, mode="single", shift=(1, 2))
    np.testing.assert_allclose(single, [[10.0, 20.0]])
    clipped = ddf.aperture_array(g1, g2, center=(15, 15), shape=(30, 30), edge=6)
    np.testing.assert_allclose(clipped, [[15.0, 15.0]])  # 5 and 25 lie within the edge
    with pytest.raises(ValueError):
        ddf.aperture_array(g1, g2, mode="bad")


def test_aperture_subtract_and_image():
    fine = ddf.aperture_array((5.0, 0.0), (0.0, 5.0), n1_range=(-2, 2), n2_range=(-2, 2))
    coarse = ddf.aperture_array((10.0, 0.0), (0.0, 10.0), n1_range=(-1, 1), n2_range=(-1, 1))
    super_only = ddf.aperture_array_subtract(fine, coarse, tol=1.0)
    assert super_only.shape == (25 - 9, 2)

    peaks = _lattice_peaks()
    image = ddf.aperture_ddf_image(peaks, super_only, radius=1.0)
    assert image.shape == (4, 5)
    assert np.all(image[:, :3] == 0)
    np.testing.assert_allclose(image[:, 3:], 2.0)

    # overlapping apertures count each peak once
    image_full = ddf.aperture_ddf_image(peaks, np.vstack([coarse, coarse]), radius=1.0)
    np.testing.assert_allclose(image_full[2, 0], 9 * 3.0)


def test_polar_fields_and_mask():
    peaks = _lattice_peaks(fields=("qx", "qy", "intensity"))
    polar = ddf.add_polar_fields(peaks)
    assert polar.fields[-2:] == ["qr", "qphi"]
    flat = polar.select_fields("qx", "qy", "qr", "qphi").numpy()
    np.testing.assert_allclose(flat[:, 2], np.hypot(flat[:, 0], flat[:, 1]), atol=1e-5)
    # (qx, qy) = (-10, 0) is straight up on screen: +90 degrees
    up = (flat[:, 0] == -10) & (flat[:, 1] == 0)
    np.testing.assert_allclose(flat[up, 3], 90.0)

    ring = ddf.polar_mask(peaks, 10.0, tol=0.5)
    assert ring.sum() == 4 * 20
    upper = ddf.polar_mask(peaks, 10.0, tol=0.5, phi_range=(45, 135))
    assert upper.sum() == 20
    wrapped = ddf.polar_mask(peaks, 10.0, tol=0.5, phi_range=(135, -135))  # 180 degrees
    assert wrapped.sum() == 20
    image = ddf.radial_ddf_image(peaks, 10.0, tol=0.5)
    np.testing.assert_allclose(image[1], 4 * 2.0)


def test_assign_grain_labels():
    peaks = _lattice_peaks(R=1, C=1, fields=("qx", "qy", "intensity"))
    labels = np.array([0, 0, 1, 1, -1, 2, 2, 2, -1])
    labeled = peaks.copy()
    labeled.add_fields("cluster", values=labels[:, None])
    out = ddf.assign_grain_labels(labeled, grain_labels=np.array([3, -1, 4]))
    grains = out.select_fields("grain_label").numpy()[:, 0]
    np.testing.assert_array_equal(grains, [3, 3, -1, -1, -2, 4, 4, 4, -2])


def test_refine_lattice_vectors_and_group_images():
    rng = np.random.default_rng(0)
    g1, g2 = np.array([20.0, 3.0]), np.array([4.0, 21.0])
    nested = []
    for r in range(3):
        row = []
        for c in range(3):
            n = np.array([[i, j] for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)], float)
            q = n @ np.stack([g1, g2]) + rng.normal(0, 0.2, (len(n), 2))
            row.append(np.concatenate([q, np.ones((len(n), 1))], axis=1))
        nested.append(row)
    peaks = Vector.from_data(nested, fields=["q_row", "q_col", "intensity"])
    f1, f2 = ddf.refine_lattice_vectors(
        peaks,
        (19.0, 2.0),
        (5.0, 20.0),
        center=(0.0, 0.0),
        radius=4.0,
        n1_range=(-2, 2),
        n2_range=(-2, 2),
    )
    np.testing.assert_allclose(f1, g1, atol=0.1)
    np.testing.assert_allclose(f2, g2, atol=0.1)

    a = np.zeros((4, 4))
    a[:2] = 1
    b = np.zeros((4, 4))
    b[2:] = 1
    images = np.stack([a, 2 * a, a + 0.05 * b, b, 3 * b, np.eye(4)])
    labels = ddf.group_ddf_images(images, min_correlation=0.9)
    assert labels[0] == labels[1] == labels[2] >= 0
    assert labels[3] == labels[4] >= 0 and labels[3] != labels[0]
    assert labels[5] == -1


def test_cluster_centers_and_lattice_distance():
    peaks = _lattice_peaks(R=1, C=2)
    labeled = peaks.copy()
    n = labeled.total_rows
    labels = np.full(n, -1)
    labels[:9] = 0  # the 3x3 lattice in cell (0, 0)
    labeled.add_fields("cluster", values=labels[:, None])
    centers = ddf.cluster_centers(labeled)
    np.testing.assert_allclose(centers, [[0.0, 0.0]], atol=1e-6)

    d = ddf.lattice_distance([[10.0, 10.0], [5.0, 5.0], [11.0, 0.0]], (10.0, 0.0), (0.0, 10.0))
    np.testing.assert_allclose(d, [0.0, np.hypot(5, 5), 1.0])


def _pixel_lattice_peaks(origin=(32.0, 32.0)):
    """3 x 3 scan of a pixel lattice g1=(20, 3), g2=(4, 21) around origin."""
    g = np.array([[20.0, 3.0], [4.0, 21.0]])
    n = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
    q = np.asarray(origin) + n @ g
    pts = np.concatenate([q, np.ones((len(n), 1))], axis=1)
    peaks = Vector.from_data([[pts] * 3 for _ in range(3)], fields=["q_row", "q_col", "intensity"])
    return peaks, g


def test_refine_lattice_vectors_uses_stored_origin_for_pixel_peaks():
    peaks, g = _pixel_lattice_peaks()
    with pytest.raises(ValueError, match="origin_ref"):
        ddf.refine_lattice_vectors(peaks, g[0] + 1, g[1] - 1, radius=4.0)
    peaks.metadata["origin_ref"] = (32.0, 32.0)
    f1, f2 = ddf.refine_lattice_vectors(
        peaks, g[0] + 1, g[1] - 1, radius=4.0, n1_range=(-1, 1), n2_range=(-1, 1)
    )
    np.testing.assert_allclose(f1, g[0], atol=1e-4)
    np.testing.assert_allclose(f2, g[1], atol=1e-4)

    # the polar functions resolve the same origin
    ring = ddf.polar_mask(peaks, np.hypot(20.0, 3.0), tol=0.5)
    assert ring.sum() == 9 * 2
    np.testing.assert_array_equal(
        ring, ddf.polar_mask(peaks, np.hypot(20, 3), 0.5, center=(32, 32))
    )


def test_aperture_array_needs_g2_for_array_mode():
    with pytest.raises(ValueError, match="g2"):
        ddf.aperture_array((10.0, 0.0))
    with pytest.raises(ValueError):
        ddf.aperture_array((10.0, 0.0), mode="2-beam")
    half = ddf.aperture_array((10.0, 0.0), (0.0, 10.0), mode="single", shift=(0.5, 0.5))
    np.testing.assert_allclose(half, [[5.0, 5.0]])


def test_ddf_images_and_cluster_coms():
    # 2 x 3 scan; cluster 0 lives in column 0, cluster 1 in row 1
    nested = []
    for r in range(2):
        row = []
        for c in range(3):
            pts = [[0.0, 0.0, 1.0, -1]]
            if c == 0:
                pts.append([5.0, 0.0, 2.0, 0])
            if r == 1:
                pts.append([0.0, 5.0, 1.0 + c, 1])
            row.append(np.asarray(pts, dtype=float))
        nested.append(row)
    labeled = Vector.from_data(nested, fields=["qx", "qy", "intensity", "cluster"])

    images = ddf.ddf_images(labeled, [0, 1])
    assert images.shape == (2, 2, 3)
    np.testing.assert_allclose(images[0], [[2, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(images[1], [[0, 0, 0], [1, 2, 3]])

    coms, sizes = ddf.cluster_coms(labeled)
    np.testing.assert_array_equal(sizes, [2, 3])
    np.testing.assert_allclose(coms[0], [0.5, 0.0])
    np.testing.assert_allclose(coms[1], [1.0, (0 * 1 + 1 * 2 + 2 * 3) / 6])
    coms_u, _ = ddf.cluster_coms(labeled, weighted=False)
    np.testing.assert_allclose(coms_u[1], [1.0, 1.0])

    centers = ddf.cluster_centers(labeled)
    np.testing.assert_allclose(centers, [[5.0, 0.0], [0.0, 5.0]])


def test_cluster_functions_accept_empty_vector():
    empty = Vector.from_data(
        [[np.empty((0, 4)), np.empty((0, 4))]], fields=["qx", "qy", "intensity", "cluster"]
    )
    coms, sizes = ddf.cluster_coms(empty)
    assert coms.shape == (0, 2) and sizes.shape == (0,)
    assert ddf.cluster_centers(empty).shape == (0, 2)
    assert ddf.ddf_images(empty, [0]).shape == (1, 1, 2)
    fig, ax = ddf.plot_cluster_scatter(empty)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_cluster_scatter_center():
    import matplotlib.pyplot as plt

    peaks, _ = _pixel_lattice_peaks()
    labeled = peaks.copy()
    labeled.add_fields("cluster", values=np.zeros((labeled.total_rows, 1)))
    labeled.metadata["origin_ref"] = (32.0, 32.0)
    fig, ax = ddf.plot_cluster_scatter(labeled)
    assert np.isclose(np.mean(ax.get_xlim()), 32.0)
    plt.close(fig)

    calibrated = Vector.from_data(
        [[np.array([[0.5, 0.1, 1.0, 0], [-0.5, -0.1, 1.0, 0]])]],
        fields=["qx", "qy", "intensity", "cluster"],
    )
    calibrated.metadata["origin_ref"] = (128.0, 128.0)  # detector pixels: ignored
    fig, ax = ddf.plot_cluster_scatter(calibrated)
    assert np.isclose(np.mean(ax.get_xlim()), 0.0)
    assert np.isclose(np.mean(ax.get_ylim()), 0.0)
    plt.close(fig)
