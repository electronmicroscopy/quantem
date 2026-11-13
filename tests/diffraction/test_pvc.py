import numpy as np
import pytest

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.diffraction.lattices import gaussian_form_factor, make_images
from quantem.diffraction.multislice import circular_window
from quantem.diffraction.pvc import PairVectorCorrelation, _solve_sphere, _ylm_azimuth_zero

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


def test_beam_correlation_and_patterson_of_lattice(triangular, recwarn):
    """One triangular-lattice pattern: the beam correlation matches the model of the
    atom positions with peaks at 60 and 120 deg, and the Patterson peaks at the bond vector."""
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
    t = T[i, i, :181]
    assert (
        abs(theta[30 + np.argmax(t[30:90])] - 60) <= 2
        and abs(theta[90 + np.argmax(t[90:150])] - 120) <= 2
    )

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
