import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter

from quantem.core.utils.imaging_utils import (
    cross_correlation_shift,
    cross_correlation_shift_torch,
    dft_upsample,
)

# shifts spanning integer, half-integer, sub-pixel and large displacements
SHIFTS = [(0.0, 0.0), (3.0, -2.0), (1.25, 0.75), (-4.5, 2.5), (0.37, -0.11), (-7.3, 5.9)]


@pytest.fixture(scope="module")
def image():
    rng = np.random.default_rng(0)
    return gaussian_filter(rng.normal(size=(64, 64)), 2)


def _shifted(image, shift):
    return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(image), shift)))


class TestDftUpsample:
    def test_agrees_with_ifft2_at_unit_upsampling(self, image):
        """At `up=1` the fine grid lands on integer pixels, where `ifft2` is the answer."""
        spectrum = np.fft.fft2(image)
        du = int(np.ceil(1.5 * 1))
        row, col = 5, 7

        local = dft_upsample(spectrum, 1, (row, col))
        expected = np.real(np.fft.ifft2(spectrum))[
            row - du : row + du + 1, col - du : col + du + 1
        ]

        # `dft_upsample` omits the 1/(M*N) normalization `ifft2` applies
        assert np.allclose(local, expected * spectrum.size, atol=1e-8)

    @pytest.mark.parametrize("up", [2, 4, 8])
    def test_peak_lands_on_the_center_tap(self, image, up):
        """Centering the fine grid on the true peak must put the maximum at its middle.

        The middle is `ceil(1.5 * up)`, since the grid spans `arange(-du, du+1) / up`.
        """
        cc = np.fft.fft2(image) * np.conj(np.fft.fft2(image))
        local = dft_upsample(cc, up, (0.0, 0.0))

        du = int(np.ceil(1.5 * up))
        assert local.shape == (2 * du + 1, 2 * du + 1)
        assert np.unravel_index(np.argmax(local), local.shape) == (du, du)


class TestCrossCorrelationShift:
    @pytest.mark.parametrize("shift", SHIFTS)
    @pytest.mark.parametrize("up", [2, 4, 8, 16])
    def test_recovers_a_known_shift(self, image, shift, up):
        """The returned shift realigns `im` onto `im_ref`, so it is the negated input."""
        measured = np.asarray(cross_correlation_shift(image, _shifted(image, shift), up))
        assert measured == pytest.approx(-np.asarray(shift), abs=0.02)

    def test_upsampling_beats_no_upsampling(self, image):
        """Sub-pixel refinement has to be an improvement on the integer peak."""
        shift = (1.25, 0.75)
        shifted = _shifted(image, shift)

        coarse = np.asarray(cross_correlation_shift(image, shifted, upsample_factor=1))
        fine = np.asarray(cross_correlation_shift(image, shifted, upsample_factor=8))

        want = -np.asarray(shift)
        assert np.abs(fine - want).max() < np.abs(coarse - want).max()

    def test_identical_images_do_not_shift(self, image):
        """Guards the off-by-`ceil(1.5*up)` that used to bias every result by half a pixel."""
        for up in (2, 4, 8, 16):
            measured = np.asarray(cross_correlation_shift(image, image, upsample_factor=up))
            assert measured == pytest.approx((0.0, 0.0), abs=1e-6)

    @pytest.mark.parametrize("shift", SHIFTS)
    def test_matches_the_torch_implementation(self, image, shift):
        """The two paths are independent ports; they must not disagree."""
        import torch

        shifted = _shifted(image, shift)
        numpy_shift = np.asarray(cross_correlation_shift(image, shifted, upsample_factor=8))
        torch_shift = cross_correlation_shift_torch(
            torch.as_tensor(image), torch.as_tensor(shifted), upsample_factor=8
        ).numpy()

        assert numpy_shift == pytest.approx(torch_shift, abs=0.02)

    def test_shifted_image_is_realigned(self, image):
        """`return_shifted_image` must undo the displacement it just measured."""
        shifted = _shifted(image, (2.5, -1.5))
        _, realigned = cross_correlation_shift(
            image, shifted, upsample_factor=8, return_shifted_image=True
        )

        assert np.corrcoef(realigned.ravel(), image.ravel())[0, 1] > 0.999
