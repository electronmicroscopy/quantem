"""FFT parity: JS fft1d/fft2d/fftshift line-ported to Python, validated against numpy.

Why ports instead of running the JS directly: pytest can't drive a TypeScript
module without a Node bridge or browser harness, both of which add fragility
and slow CI. Instead we mirror js/fft.ts:14-82 line-for-line in Python below
and assert against numpy.fft. If the JS algorithm has a bug, the line-port
inherits it and this test fails — surfacing the bug at unit-test speed.

When js/fft.ts changes, update the ports here in the same commit. The
side-by-side structure makes drift visually obvious during review.
"""
import numpy as np


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _js_fft1d(real: np.ndarray, imag: np.ndarray, inverse: bool = False) -> None:
    """Line-port of js/fft.ts fft1d. In-place. Iterative radix-2 Cooley-Tukey."""
    n = real.size
    if n <= 1:
        return
    # Bit-reversal permutation.
    j = 0
    for i in range(n - 1):
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]
        k = n >> 1
        while k <= j:
            j -= k
            k >>= 1
        j += k
    sign = 1 if inverse else -1
    length = 2
    while length <= n:
        half = length >> 1
        angle = (sign * 2 * np.pi) / length
        w_real = np.cos(angle)
        w_imag = np.sin(angle)
        for i in range(0, n, length):
            cur_real = 1.0
            cur_imag = 0.0
            for k in range(half):
                even = i + k
                odd = i + k + half
                t_real = cur_real * real[odd] - cur_imag * imag[odd]
                t_imag = cur_real * imag[odd] + cur_imag * real[odd]
                real[odd] = real[even] - t_real
                imag[odd] = imag[even] - t_imag
                real[even] += t_real
                imag[even] += t_imag
                new_real = cur_real * w_real - cur_imag * w_imag
                cur_imag = cur_real * w_imag + cur_imag * w_real
                cur_real = new_real
        length <<= 1
    if inverse:
        real /= n
        imag /= n


def _js_fft2d(real: np.ndarray, imag: np.ndarray, width: int, height: int, inverse: bool = False) -> None:
    """Line-port of js/fft.ts fft2d. In-place on (height*width) flattened arrays."""
    padded_w = _next_pow2(width)
    padded_h = _next_pow2(height)
    needs_padding = padded_w != width or padded_h != height
    if needs_padding:
        work_real = np.zeros(padded_w * padded_h, dtype=np.float64)
        work_imag = np.zeros(padded_w * padded_h, dtype=np.float64)
        for y in range(height):
            for x in range(width):
                work_real[y * padded_w + x] = real[y * width + x]
                work_imag[y * padded_w + x] = imag[y * width + x]
    else:
        work_real = real
        work_imag = imag
    row_real = np.empty(padded_w, dtype=np.float64)
    row_imag = np.empty(padded_w, dtype=np.float64)
    for y in range(padded_h):
        offset = y * padded_w
        row_real[:] = work_real[offset:offset + padded_w]
        row_imag[:] = work_imag[offset:offset + padded_w]
        _js_fft1d(row_real, row_imag, inverse)
        work_real[offset:offset + padded_w] = row_real
        work_imag[offset:offset + padded_w] = row_imag
    col_real = np.empty(padded_h, dtype=np.float64)
    col_imag = np.empty(padded_h, dtype=np.float64)
    for x in range(padded_w):
        for y in range(padded_h):
            col_real[y] = work_real[y * padded_w + x]
            col_imag[y] = work_imag[y * padded_w + x]
        _js_fft1d(col_real, col_imag, inverse)
        for y in range(padded_h):
            work_real[y * padded_w + x] = col_real[y]
            work_imag[y * padded_w + x] = col_imag[y]
    if needs_padding:
        for y in range(height):
            for x in range(width):
                real[y * width + x] = work_real[y * padded_w + x]
                imag[y * width + x] = work_imag[y * padded_w + x]


def _js_fftshift(data: np.ndarray, width: int, height: int) -> None:
    """Line-port of js/fft.ts fftshift. In-place."""
    half_w = width >> 1
    half_h = height >> 1
    temp = np.empty(width * height, dtype=data.dtype)
    for y in range(height):
        for x in range(width):
            temp[((y + half_h) % height) * width + ((x + half_w) % width)] = data[y * width + x]
    data[:] = temp


# ---------------------------------------------------------------------------

def test_fft1d_matches_numpy_pow2():
    """1D FFT on power-of-2 input matches numpy.fft.fft."""
    rng = np.random.default_rng(0)
    n = 64
    x = rng.standard_normal(n)
    real = x.astype(np.float64).copy()
    imag = np.zeros(n, dtype=np.float64)
    _js_fft1d(real, imag, inverse=False)
    js = real + 1j * imag
    expected = np.fft.fft(x)
    np.testing.assert_allclose(js, expected, atol=1e-9)


def test_fft1d_inverse_roundtrip():
    """fft1d(fft1d(x), inverse=True) ≈ x."""
    rng = np.random.default_rng(1)
    n = 128
    x = rng.standard_normal(n)
    real = x.astype(np.float64).copy()
    imag = np.zeros(n, dtype=np.float64)
    _js_fft1d(real, imag, inverse=False)
    _js_fft1d(real, imag, inverse=True)
    np.testing.assert_allclose(real, x, atol=1e-9)
    np.testing.assert_allclose(imag, np.zeros(n), atol=1e-9)


def test_fft2d_matches_numpy_pow2():
    """2D FFT on power-of-2 dims matches numpy.fft.fft2."""
    rng = np.random.default_rng(2)
    h, w = 32, 64
    img = rng.standard_normal((h, w))
    real = img.astype(np.float64).flatten()
    imag = np.zeros(h * w, dtype=np.float64)
    _js_fft2d(real, imag, w, h, inverse=False)
    js = (real + 1j * imag).reshape(h, w)
    expected = np.fft.fft2(img)
    np.testing.assert_allclose(js, expected, atol=1e-9)


def test_fft2d_non_pow2_zero_pads():
    """Non-power-of-2 input gets zero-padded; FFT of padded matches numpy of padded."""
    rng = np.random.default_rng(3)
    h, w = 30, 50
    img = rng.standard_normal((h, w))
    real = img.astype(np.float64).flatten()
    imag = np.zeros(h * w, dtype=np.float64)
    _js_fft2d(real, imag, w, h, inverse=False)
    # JS contract: only the (h, w) region of the result is written back to the input arrays.
    js = (real + 1j * imag).reshape(h, w)
    pw, ph = _next_pow2(w), _next_pow2(h)
    padded = np.zeros((ph, pw))
    padded[:h, :w] = img
    expected = np.fft.fft2(padded)[:h, :w]
    np.testing.assert_allclose(js, expected, atol=1e-9)


def test_fftshift_matches_numpy():
    """fftshift matches numpy.fft.fftshift on 2D data."""
    rng = np.random.default_rng(4)
    h, w = 16, 16
    img = rng.standard_normal((h, w))
    flat = img.flatten().copy()
    _js_fftshift(flat, w, h)
    js_shifted = flat.reshape(h, w)
    expected = np.fft.fftshift(img)
    np.testing.assert_array_equal(js_shifted, expected)


def test_fft2d_then_fftshift_matches_numpy():
    """Combined FFT + fftshift matches numpy reference."""
    rng = np.random.default_rng(5)
    h, w = 32, 32
    img = rng.standard_normal((h, w))
    real = img.astype(np.float64).flatten()
    imag = np.zeros(h * w, dtype=np.float64)
    _js_fft2d(real, imag, w, h, inverse=False)
    _js_fftshift(real, w, h)
    _js_fftshift(imag, w, h)
    js = (real + 1j * imag).reshape(h, w)
    expected = np.fft.fftshift(np.fft.fft2(img))
    np.testing.assert_allclose(js, expected, atol=1e-9)
