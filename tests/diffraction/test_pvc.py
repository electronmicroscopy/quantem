import numpy as np
import pytest
from ase.build import bulk
from scipy.special import eval_legendre

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.lattices import (
    gaussian_form_factor,
    make_images,
    polycrystal,
    probe_patterns,
)
from quantem.diffraction.multislice import circular_window
from quantem.diffraction.pvc import (
    PairVectorCorrelation,
    _cluster_power_sphere,
    _solve_sphere,
    _ylm_azimuth_zero,
    pair_vector_marks,
)

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


def _pvc(intensity, origin):
    dq = 1 / (N_PX * PIXEL)
    ds = Dataset2d.from_array(intensity, sampling=(dq, dq), units=["A^-1", "A^-1"])
    return PairVectorCorrelation(
        ds,
        origin=origin,
        fq=lambda q: np.maximum(gaussian_form_factor(q, SIGMA), 1e-3),
        k_min=0.1,
        k_max=1.48,
    )


def test_sphere_solve_recovers_invariants_on_curved_ewald_sphere():
    """S_n built from known B_l with the 200 kV Ewald geometry is inverted exactly."""
    q = np.linspace(0.3, 2.5, 5)
    beta = np.pi / 2 + np.arcsin(0.02508 * q / 2)
    ls = np.arange(2, 31, 2)
    B = np.random.default_rng(0).normal(size=(ls.size, q.size, q.size))
    B = B + B.transpose(0, 2, 1)
    Y = _ylm_azimuth_zero(30, np.cos(beta)).numpy()
    S = np.zeros((40, q.size, q.size))
    for k, ell in enumerate(ls):
        for n in range(ell + 1):
            S[n] += 4 * np.pi / (2 * ell + 1) * np.outer(Y[ell, n], Y[ell, n]) * B[k]
    assert np.abs(_solve_sphere(S, beta, ls) - B).max() < 1e-10


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
    pvc = _pvc(intensity, (N_PX // 2, N_PX // 2))
    assert not any("n = 1 harmonic" in str(w.message) for w in recwarn), (
        "centered pattern should not warn"
    )
    theta, T = pvc.correlation("beam", r_max=6.0)
    M = pvc.model_correlation([np.c_[positions, np.zeros(len(positions))]])
    i = int(np.argmin(np.abs(pvc.r - BOND)))
    interior = slice(15, 166)
    assert np.corrcoef(T[i, i, interior], M[i, i, interior])[0, 1] > 0.98
    angles = [a for _, a in pair_vector_marks(positions, BOND, 3.0, model="beam")]
    assert angles == [60.0, 120.0, 240.0, 300.0], angles
    for a in angles:
        near = np.abs(theta - a) <= 20
        assert abs(theta[near][np.argmax(T[i, i, near])] - a) <= 2, f"peak near {a} deg"

    image, px = pvc.patterson(r_max=4.0)
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
        _pvc(intensity, (N_PX // 2 + 2, N_PX // 2))


def test_probe_patterns_bright_field_and_kinematic_limit():
    """The bright-field disk of multislice patterns of a nanocrystalline film is
    measured at its radius alpha / lambda, and single scattering by this thin
    film matches multislice outside the disk."""
    film = polycrystal(bulk("Si", "diamond", a=5.431, cubic=True), 8.0, 24.0, 8.0, seed=0)
    ds = probe_patterns(film, 4, 10.0, q_max=1.0)
    pvc = PairVectorCorrelation(ds, energy=200e3)
    expected = 10e-3 / electron_wavelength_angstrom(200e3)
    dq, n = ds.sampling[-1], ds.shape[-1]
    assert abs(pvc.bright_field_radius - expected) < 0.5 * dq

    kinematic = probe_patterns(film, 4, 10.0, q_max=1.0, kinematic=True).array
    q = np.hypot(*np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")) * dq
    outside = q > expected + 2 * dq
    dynamic = ds.array[..., outside]
    error = np.linalg.norm(kinematic[..., outside] - dynamic) / np.linalg.norm(dynamic)
    assert error < 0.06, error


def test_split_counts_removes_shot_noise_bias():
    """Poisson noise on a flat pattern biases the second moment at k = k'; the
    cross moment of two random halves of the counts does not."""
    counts = np.random.default_rng(0).poisson(20.0, size=(200, 1, 48, 48)).astype(np.float32)
    ds = Dataset4dstem.from_array(
        counts, origin=(0, 0, -24, -24), sampling=(1, 1, 1, 1), units=["pixels"] * 4
    )
    bias = [
        np.diagonal(
            PairVectorCorrelation(ds, k_min=4, k_max=20, split_counts=split).second_moment[1:],
            axis1=1,
            axis2=2,
        ).real.mean()
        for split in (False, True)
    ]
    assert bias[0] > 0 and abs(bias[1]) < 0.1 * bias[0], bias


def test_self_term_removal(triangular):
    """Theta leaves out the correlation of each pair vector with itself at 0 deg
    on r = r', which Theta_self holds, in both rotation models, and the fitted
    amplitude peaks at the bond length. The projected pair distribution of the mean
    pattern, which holds that two-body information instead, matches the atom
    positions (one projection has no 3D pair distribution), and its first
    neighbor shell holds the bond length and stops short of the next shell."""
    intensity, positions = triangular
    pvc = _pvc(intensity, (N_PX // 2, N_PX // 2))
    for model in ("beam", "sphere"):
        _, T = pvc.correlation(model, r_max=4.0)
        i = int(np.argmin(np.abs(pvc.r - BOND)))
        spike = [abs(V[i, i, 0]) / np.abs(V[i, i, 30:150]).max() for V in (T + pvc.Theta_self, T)]
        assert spike[0] > 0.9 and spike[1] < 0.3, (model, spike)
        u, h = pvc.self_amplitude
        assert abs(u[np.argmax(h.sum(0))] - BOND) <= 0.1, model
        if model == "beam":
            g2 = pvc.pair_distribution()
            g2_model = pvc.model_pair_distribution([np.c_[positions, np.zeros(len(positions))]])
            inside = pvc.r > 1.0
            assert np.corrcoef(g2[inside], g2_model[inside])[0, 1] > 0.98
            lo, hi = pvc.neighbor_shell(g2=g2)
            assert lo < BOND < hi < 0.5 * (1 + np.sqrt(3)) * BOND, (lo, hi)
