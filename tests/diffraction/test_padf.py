import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from scipy.special import eval_legendre

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.lattices import (
    cluster_patterns,
    gaussian_form_factor,
    make_images,
    polycrystal,
    probe_patterns,
    random_clusters,
)
from quantem.diffraction.multislice import circular_window
from quantem.diffraction.padf import (
    PairAngleDistributionFunction,
    _cluster_power_sphere,
    _solve_sphere,
    order_parameter,
    pair_vector_marks,
)
from quantem.diffraction.wk_scattering_factors import compute_WK_factor

N_PX, PIXEL, SIGMA, BOND = 128, 0.2, 0.4, 2.5


@pytest.fixture(scope="module")
def triangular():
    """FFT intensity of a triangular lattice image, and its atom positions."""
    data, info = make_images(
        im_shape=(N_PX, N_PX),
        pixel_size_ang=PIXEL,
        atom_sigma_ang=SIGMA,
        bond_length_ang=(BOND,),
        lattices=("triangular",),
        plot_images=False,
        seed=0,
    )
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(data[0] * circular_window((N_PX, N_PX))))) ** 2
    return intensity, info[0]["positions_ang"]


def _padf(intensity, origin):
    dq = 1 / (N_PX * PIXEL)
    ds = Dataset2d.from_array(intensity, sampling=(dq, dq), units=["A^-1", "A^-1"])
    return PairAngleDistributionFunction(
        ds,
        origin=origin,
        fq=lambda q: np.maximum(gaussian_form_factor(q, SIGMA), 1e-3),
        k_min=0.1,
        k_max=1.48,
    )


def test_sphere_solve_recovers_invariants_without_folding():
    """B_l up to l_max = 30 is recovered from the ring correlations
    C(psi) = sum_l B_l P_l(cos gamma), on a flat and on the 200 kV Ewald sphere,
    although C also holds an l = 0 background and orders up to 38, which fold
    into the fitted orders of a fit to the harmonics n <= l_max."""
    q = np.linspace(0.3, 2.5, 5)
    ls, l_all = np.arange(2, 31, 2), np.arange(0, 39, 2)
    psi = 2 * np.pi * np.arange(128) / 128
    rng = np.random.default_rng(0)
    for wavelength in (0.0, 0.02508):
        beta = np.pi / 2 + np.arcsin(wavelength * q / 2)
        B = rng.normal(size=(l_all.size, q.size, q.size))
        B = B + B.transpose(0, 2, 1)
        cb, sb = np.cos(beta)[:, None, None], np.sin(beta)[:, None, None]
        cos_gamma = cb * cb.transpose(1, 0, 2) + sb * sb.transpose(1, 0, 2) * np.cos(psi)
        C = sum(B[k][:, :, None] * eval_legendre(ell, cos_gamma) for k, ell in enumerate(l_all))
        S = np.fft.rfft(C, axis=-1).real.transpose(2, 0, 1)[:40] / psi.size
        err = np.abs(_solve_sphere(S, beta, ls) - B[1 : ls.size + 1]).max()
        assert err < 1e-10, f"wavelength {wavelength}: error {err:.1e}"


def test_cluster_power_sphere_matches_addition_theorem():
    """The cluster power, which counts each pair once and evaluates repeated pair
    vectors of a lattice once, equals the double sum over all pair vectors of their
    radial bin weights times (2l + 1) / (4 pi) P_l(cos of the angle between them)."""
    rng = np.random.default_rng(0)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    grid = np.stack(np.meshgrid(*[np.arange(3.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    clusters = [1.7 * grid @ rotation.T, rng.uniform(-3, 3, size=(20, 3))]
    r, ls = np.arange(0, 5.0, 0.1), np.arange(2, 13, 2)
    shell = np.maximum(r, 0.05) ** 2 * 0.1
    expected = np.zeros((ls.size, r.size, r.size))
    for pos in clusters:
        d = (pos[None, :, :] - pos[:, None, :]).reshape(-1, 3)
        dist = np.linalg.norm(d, axis=1)
        d, dist = d[(dist > 0) & (dist < r[-1])], dist[(dist > 0) & (dist < r[-1])]
        i0 = np.floor(dist / 0.1).astype(int)
        f = dist / 0.1 - i0
        W = np.zeros((r.size, dist.size))
        W[i0, np.arange(dist.size)] = (1 - f) / shell[i0]
        W[i0 + 1, np.arange(dist.size)] += f / shell[i0 + 1]
        cos = (d / dist[:, None]) @ (d / dist[:, None]).T
        for k, ell in enumerate(ls):
            expected[k] += (2 * ell + 1) / (4 * np.pi) * W @ eval_legendre(ell, cos) @ W.T
    H = _cluster_power_sphere(clusters, r, ls, memory=0, block=16).numpy()
    assert np.abs(H - expected).max() < 1e-10 * np.abs(expected).max()


def test_beam_correlation_and_patterson_of_lattice(triangular, recwarn):
    """One triangular-lattice pattern: the beam correlation matches the model of the
    atom positions and peaks at the angles pair_vector_marks expects, and the
    Patterson peaks at the bond vector."""
    intensity, positions = triangular
    padf = _padf(intensity, (N_PX // 2, N_PX // 2))
    assert not any("n = 1 harmonic" in str(w.message) for w in recwarn), (
        "centered pattern should not warn"
    )
    theta, T = padf.correlation("beam", r_max=6.0)
    M = padf.model_correlation([np.c_[positions, np.zeros(len(positions))]])
    i = int(np.argmin(np.abs(padf.r - BOND)))
    interior = slice(15, 166)
    assert np.corrcoef(T[i, i, interior], M[i, i, interior])[0, 1] > 0.98
    angles = [a for _, a in pair_vector_marks(positions, BOND, 3.0, model="beam")]
    assert angles == [60.0, 120.0, 240.0, 300.0], angles
    for a in angles:
        near = np.abs(theta - a) <= 20
        assert abs(theta[near][np.argmax(T[i, i, near])] - a) <= 2, f"peak near {a} deg"

    image, px = padf.patterson(r_max=4.0)
    c = image.shape[0] // 2
    rho = (
        np.hypot(
            *np.meshgrid(
                np.arange(image.shape[0]) - c, np.arange(image.shape[1]) - c, indexing="ij"
            )
        )
        * px
    )
    shell = np.abs(rho - BOND) < 0.15
    assert image[shell].max() > 3 * np.abs(image[(rho > 0.8) & (rho < 1.8)]).max(), (
        "Patterson should peak at the bond length"
    )


def test_miscentered_pattern_warns(triangular):
    """A pattern transformed about a point 2 px off its center has an n = 1 mean harmonic."""
    intensity, _ = triangular
    with pytest.warns(UserWarning, match="n = 1 harmonic"):
        _padf(intensity, (N_PX // 2 + 2, N_PX // 2))


def test_probe_patterns_bright_field_and_kinematic_limit():
    """The bright-field disk of multislice patterns of a nanocrystalline film is
    measured at its radius alpha / lambda, and single scattering by this thin
    film matches multislice outside the disk. Under plane-wave illumination,
    single scattering by a Mo-S film equals the sum over atoms with their own
    scattering factors, each at its own depth on the Ewald sphere."""
    film = polycrystal(bulk("Si", "diamond", a=5.431, cubic=True), 8.0, 24.0, 8.0, seed=0)
    ds = probe_patterns(film, 4, 10.0, q_max=1.0)
    padf = PairAngleDistributionFunction(ds, energy=200e3)
    expected = 10e-3 / electron_wavelength_angstrom(200e3)
    dq, n = ds.sampling[-1], ds.shape[-1]
    assert abs(padf.bright_field_radius - expected) < 0.5 * dq

    kinematic = probe_patterns(film, 4, 10.0, q_max=1.0, kinematic=True).array
    q = np.hypot(*np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")) * dq
    outside = q > expected + 2 * dq
    dynamic = ds.array[..., outside]
    error = np.linalg.norm(kinematic[..., outside] - dynamic) / np.linalg.norm(dynamic)
    assert error < 0.06, error

    rng = np.random.default_rng(0)
    width, thickness, energy = 10.0, 6.0, 200e3
    mixed = Atoms(
        numbers=rng.choice([16, 42], 30),
        positions=rng.random((30, 3)) * [width, width, thickness],
        cell=[width, width, thickness],
    )
    pattern = probe_patterns(
        mixed, 1, 1e-3, energy=energy, q_max=2.0, slice_thickness=2.0, kinematic=True
    )
    n = pattern.shape[-1]
    k = (np.arange(n) - n // 2) / width
    q = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    amplitude = 0
    for pos, z in zip(mixed.positions, mixed.numbers):
        depth = 1j * np.pi * electron_wavelength_angstrom(energy) * pos[2] * k**2
        amplitude = amplitude + compute_WK_factor(q.ravel(), int(z), energy).reshape(
            q.shape
        ) * np.outer(
            np.exp(-2j * np.pi * pos[0] * k + depth), np.exp(-2j * np.pi * pos[1] * k + depth)
        )
    ring = (q > 0.05) & (q < 1.9)
    expected, simulated = np.abs(amplitude[ring]) ** 2, pattern.array[0, 0][ring]
    scale = (simulated @ expected) / (simulated @ simulated)
    error = np.linalg.norm(scale * simulated - expected) / np.linalg.norm(expected)
    assert error < 1e-4, error


def test_split_counts_removes_shot_noise_bias():
    """Poisson noise on a flat pattern biases the second moment at k = k'; the
    cross moment of two random halves of the counts does not."""
    counts = np.random.default_rng(0).poisson(20.0, size=(200, 1, 48, 48)).astype(np.float32)
    ds = Dataset4dstem.from_array(
        counts, origin=(0, 0, -24, -24), sampling=(1, 1, 1, 1), units=["pixels"] * 4
    )
    bias = [
        np.diagonal(
            PairAngleDistributionFunction(ds, k_min=4, k_max=20, split_counts=split).second_moment[
                1:
            ],
            axis1=1,
            axis2=2,
        ).real.mean()
        for split in (False, True)
    ]
    assert bias[0] > 0 and abs(bias[1]) < 0.1 * bias[0], bias


def test_order_parameter_removes_noise_bias():
    """Eight groups share a distinct map and a self map, each group with noise
    three times the distinct amplitude. The distinct fraction of their average
    is biased toward 1 by the noise; order_parameter recovers the true value
    inside its confidence interval. Points with a pair vector shorter than 2 A
    hold 100 times more noise and are left out."""
    rng = np.random.default_rng(0)
    r, theta = np.arange(0, 4.05, 0.1), np.arange(0, 181.0)
    shape = (r.size, r.size, theta.size)
    distinct, self_term = rng.normal(size=shape), 0.8 * rng.normal(size=shape)
    noise = 3.0 * rng.normal(size=(8, *shape))
    noise[:, np.minimum.outer(r, r) < 2.0] *= 100
    groups, self_groups = distinct + noise, self_term + 0.1 * rng.normal(size=(8, *shape))
    omega = order_parameter(groups, self_groups, r, theta, r_range=(2.0, 4.0), r_min=2.0)
    assert np.allclose(omega.r_eff, [2.5, 3.5]), omega.r_eff

    r_eff, r_short = np.maximum.outer(r, r), np.minimum.outer(r, r)
    weight = np.sin(np.deg2rad(theta))

    def variance(x):
        """Variance over a bin with each angle weighted by its solid angle."""
        w = np.broadcast_to(weight, x.shape)
        return np.average((x - np.average(x, weights=w)) ** 2, weights=w)

    truth, naive = [], []
    for lo in (2.0, 3.0):
        inside = (r_eff >= lo) & (r_eff < lo + 1) & (r_short >= 2.0)
        d, s = variance(distinct[inside]), variance(self_term[inside])
        truth.append(d / (d + s))
        d, s = variance(groups.mean(0)[inside]), variance(self_groups.mean(0)[inside])
        naive.append(d / (d + s))
    truth, naive = np.mean(truth), np.mean(naive)
    assert omega.interval[0] <= truth <= omega.interval[1], (omega.value, omega.interval, truth)
    assert abs(omega.value - truth) < 0.02, (omega.value, truth)
    assert naive > truth + 0.1, (naive, truth)


def test_self_term_removal(triangular):
    """Theta leaves out the correlation of each pair vector with itself at 0 deg
    on r = r', which Theta_self holds, in both rotation models, and the fitted
    amplitude peaks at the bond length. A single pattern has no fluctuation
    part, so the sphere model uses the total second moment. The projected pair distribution of the mean
    pattern, which holds that two-body information instead, matches the atom
    positions (one projection has no 3D pair distribution), and its first
    neighbor shell holds the bond length and stops short of the next shell."""
    intensity, positions = triangular
    padf = _padf(intensity, (N_PX // 2, N_PX // 2))
    with pytest.raises(ValueError, match="more than one pattern"):
        padf.correlation("sphere", part="fluctuation")
    for model in ("beam", "sphere"):
        _, T = padf.correlation(model, r_max=4.0)
        i = int(np.argmin(np.abs(padf.r - BOND)))
        spike = [abs(V[i, i, 0]) / np.abs(V[i, i, 30:150]).max() for V in (T + padf.Theta_self, T)]
        assert spike[0] > 0.9 and spike[1] < 0.4, (model, spike)
        u, h = padf.self_amplitude
        assert abs(u[np.argmax(h.sum(0))] - BOND) <= 0.1, model
        if model == "beam":
            g2 = padf.pair_distribution()
            g2_model = padf.model_pair_distribution([np.c_[positions, np.zeros(len(positions))]])
            inside = padf.r > 1.0
            assert np.corrcoef(g2[inside], g2_model[inside])[0, 1] > 0.98
            lo, hi = padf.neighbor_shell(g2=g2)
            assert lo < BOND < hi < 0.5 * (1 + np.sqrt(3)) * BOND, (lo, hi)


def test_martin_method():
    """The Martin PADF of bcc Fe cluster patterns matches its model, processed
    the same way, at r = r' = the first shell, with peaks at the bond angles."""
    fe = bulk("Fe", "bcc", a=2.87, cubic=True) * (8, 8, 8)
    ds = cluster_patterns(fe, 200, n_px=96, q_max=3.5, r_cluster=6.0, blur_px=0.0, seed=0)
    padf = PairAngleDistributionFunction(
        ds, fq=lambda q: compute_WK_factor(q, 26, 200e3).real, k_min=0.1
    )
    with pytest.raises(ValueError, match="sphere model only"):
        padf.correlation("beam", method="martin")
    theta, T = padf.correlation("sphere", method="martin", l_max=30, r_max=4.0)
    assert padf.Theta_self is None
    M = padf.model_correlation([pos for pos, _ in random_clusters(fe, 100, 6.0, seed=1)])
    i = int(np.argmin(np.abs(padf.r - 2.485)))
    interior = slice(15, 166)
    assert np.corrcoef(T[i, i, interior], M[i, i, interior])[0, 1] > 0.9
    for a in (70.5, 109.5):
        near = np.abs(theta - a) <= 15
        assert abs(theta[near][np.argmax(T[i, i, near])] - a) <= 3, f"peak near {a} deg"
