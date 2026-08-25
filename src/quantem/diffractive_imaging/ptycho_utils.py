from math import ceil
from typing import Literal, Union, overload

import numpy as np
import torch
from scipy.optimize import curve_fit

from quantem.core.utils import array_funcs as af

ArrayLike = Union[np.ndarray, "torch.Tensor"]


# TODO: figure out what here should be put into ptycho base vs kept in a utilities file


class SimpleBatcher:
    def __init__(
        self,
        num: int,
        batch_size: int | None,
        shuffle: bool = True,
        rng: np.random.Generator | int | None = None,
        val_ratio: float = 0.0,
        val_mode: Literal["grid", "random"] = "grid",
        train_indices: np.ndarray | None = None,
        val_indices: np.ndarray | None = None,
    ):
        self.batch_size = batch_size if batch_size is not None else num
        self.shuffle = shuffle
        self.rng = rng

        # Train/validation split (fixed for the lifetime of this batcher)
        if train_indices is not None or val_indices is not None:
            if train_indices is None or val_indices is None:
                raise ValueError("Both train_indices and val_indices must be provided together.")
            self.train_indices = np.asarray(train_indices, dtype=int)
            self.val_indices = np.asarray(val_indices, dtype=int)
        else:
            self.train_indices, self.val_indices = compute_train_val_split(
                num, val_ratio, val_mode, self.rng
            )

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")
        self._rng = rng

    def __iter__(self):
        train_order = (
            self.rng.permutation(self.train_indices) if self.shuffle else self.train_indices
        )
        for i in range(0, len(train_order), self.batch_size):
            yield train_order[i : i + self.batch_size]

    def __len__(self):
        return int(ceil(len(self.train_indices) / self.batch_size))

    def iter_val(self):
        if len(self.val_indices) == 0:
            return iter(())

        # Do not shuffle validation by default
        def _gen():
            for i in range(0, len(self.val_indices), self.batch_size):
                yield self.val_indices[i : i + self.batch_size]

        return _gen()

    @property
    def has_validation(self) -> bool:
        return len(self.val_indices) > 0

    def val_len(self) -> int:
        return int(ceil(len(self.val_indices) / self.batch_size)) if self.has_validation else 0


def compute_train_val_split(
    num: int,
    val_ratio: float,
    val_mode: Literal["grid", "random"],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the train/validation index split.

    Returns ``(train_indices, val_indices)`` as int numpy arrays. ``val_mode="grid"``
    selects every k-th index (with ``k = round(1/val_ratio)``, inverted when
    ``val_ratio > 0.5``); ``"random"`` selects a seeded ``rng.permutation`` slice.
    """
    indices = np.arange(num)
    if val_ratio < 0 or val_ratio >= 1:
        val_ratio = 0.0
    n_val = int(round(len(indices) * val_ratio))
    if n_val <= 0:
        return indices, np.asarray([], dtype=int)

    if val_mode == "random":
        # Random unique selection for validation
        perm = rng.permutation(indices)
        val_indices = perm[:n_val]
        train_indices = np.setdiff1d(indices, val_indices, assume_unique=False)
    else:  # grid/regular selection: every k-th index
        if val_ratio <= 0.5:
            k = max(1, int(round(1.0 / val_ratio)))
            invert = False
        else:
            k = max(1, int(round(1.0 / (1.0 - val_ratio))))
            invert = True

        grid_sel = indices[::k]
        if len(grid_sel) > n_val:
            grid_sel = grid_sel[:n_val]
        if invert:
            train_indices = grid_sel
            val_indices = np.setdiff1d(indices, grid_sel, assume_unique=False)
        else:
            val_indices = grid_sel
            train_indices = np.setdiff1d(indices, val_indices, assume_unique=False)

    return np.asarray(train_indices, dtype=int), np.asarray(val_indices, dtype=int)


def add_input_noise(
    model_input: torch.Tensor,
    noise_std: float,
    dtype: torch.dtype,
    device: "torch.device | str | int",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add gaussian noise to a DIP model input when noise_std > 0."""
    if noise_std > 0.0:
        noise = (
            torch.randn(
                model_input.shape,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * noise_std
        )
        return model_input + noise
    return model_input


@overload
def fourier_shift_expand(
    array: np.ndarray, positions: np.ndarray, expand_dim: bool = True
) -> np.ndarray: ...
@overload
def fourier_shift_expand(
    array: "torch.Tensor", positions: "torch.Tensor", expand_dim: bool = True
) -> "torch.Tensor": ...
def fourier_shift_expand(
    array: ArrayLike, positions: ArrayLike, expand_dim: bool = True
) -> ArrayLike:
    """Fourier-shift array by flat array of positions."""
    dtype = array.dtype if af.is_complex(array) else None
    phase = fourier_translation_operator(positions, array.shape, expand_dim, dtype=dtype)
    fourier_array = af.fft2(array)
    shifted_fourier_array = fourier_array * phase
    shifted_array = af.ifft2(shifted_fourier_array)
    if af.is_complex(array):
        return shifted_array
    else:
        return shifted_array.real  # type:ignore ## will be numeric so this should be safe


@overload
def fourier_translation_operator(
    positions: np.ndarray,
    shape: tuple,
    expand_dim: bool = True,
    dtype: "str|torch.dtype|np.dtype|None" = None,
) -> np.ndarray: ...
@overload
def fourier_translation_operator(
    positions: "torch.Tensor",
    shape: tuple,
    expand_dim: bool = True,
    dtype: "str|torch.dtype|np.dtype|None" = None,
) -> "torch.Tensor": ...
def fourier_translation_operator(
    positions: ArrayLike,
    shape: tuple,
    expand_dim: bool = True,
    dtype: "str|torch.dtype|np.dtype|None" = None,
) -> ArrayLike:
    """Returns phase ramp for fourier-shifting array of shape `shape`."""
    nr, nc = shape[-2:]
    r = positions[..., 0][:, None, None]
    c = positions[..., 1][:, None, None]
    kr = af.match_device(np.fft.fftfreq(nr, d=1.0).astype(np.float32), positions)
    kc = af.match_device(np.fft.fftfreq(nc, d=1.0).astype(np.float32), positions)
    ramp_r = af.exp(-2.0j * np.pi * kr[None, :, None] * r)
    ramp_c = af.exp(-2.0j * np.pi * kc[None, None, :] * c)
    ramp = ramp_r * ramp_c
    if expand_dim:
        for _ in range(len(shape) - 2):
            ramp = ramp[:, None, ...]
    if dtype is not None:
        ramp = af.as_type(ramp, dtype)
    return ramp


def sum_patches_base(
    patches: torch.Tensor, indices: torch.Tensor, obj_shape: tuple
) -> torch.Tensor:
    flat_weights = patches.reshape(-1)
    flat_indices = indices.reshape(-1)
    out = af.match_device(
        torch.zeros(
            int(torch.prod(torch.tensor(obj_shape))), dtype=patches.dtype, device=patches.device
        ),
        patches,
    )
    out.index_add_(0, flat_indices, flat_weights)
    return out.reshape(obj_shape)


def sum_patches(patches: torch.Tensor, indices: torch.Tensor, obj_shape: tuple) -> torch.Tensor:
    if torch.is_complex(patches):
        real = sum_patches_base(patches.real, indices, obj_shape)
        imag = sum_patches_base(patches.imag, indices, obj_shape)
        return real + 1.0j * imag
    else:
        return sum_patches_base(patches, indices, obj_shape)


def shift_array(
    ar: np.ndarray,
    rshift: np.ndarray,
    cshift: np.ndarray,
    periodic: bool = True,
    bilinear: bool = False,
):
    """
        Shifts array ar by the shift vector (rshift, cshift), using the either
    the Fourier shift theorem (i.e. with sinc interpolation), or bilinear
    resampling. Boundary conditions can be periodic or not.

    Args:
            ar (float): input array
            rshift (float): shift along axis 0 (rows) in pixels
            cshift (float): shift along axis 1 (columns) in pixels
            periodic (bool): flag for periodic boundary conditions
            bilinear (bool): flag for bilinear image shifts
            device(str): calculation device will be perfomed on. Must be 'cpu' or 'gpu'
        Returns:
            (array) the shifted array
    """
    xp = af.get_xp_module(ar)

    # Apply image shift
    if bilinear is False:
        nr, nc = xp.shape(ar)
        qr, qc = make_Fourier_coords2D(nr, nc, 1)
        qr = xp.asarray(qr)
        qc = xp.asarray(qc)

        p = xp.exp(-(2j * xp.pi) * ((cshift * qc) + (rshift * qr)))
        shifted_ar = xp.real(xp.fft.ifft2((xp.fft.fft2(ar)) * p))

    else:
        rF = xp.floor(rshift).astype(int).item()
        cF = xp.floor(cshift).astype(int).item()
        wr = rshift - rF
        wc = cshift - cF

        shifted_ar = (
            xp.roll(ar, (rF, cF), axis=(0, 1)) * ((1 - wr) * (1 - wc))
            + xp.roll(ar, (rF + 1, cF), axis=(0, 1)) * ((wr) * (1 - wc))
            + xp.roll(ar, (rF, cF + 1), axis=(0, 1)) * ((1 - wr) * (wc))
            + xp.roll(ar, (rF + 1, cF + 1), axis=(0, 1)) * ((wr) * (wc))
        )

    if periodic is False:
        # Rounded coordinates for boundaries
        rR = (xp.round(rshift)).astype(int)
        cR = (xp.round(cshift)).astype(int)

        if rR > 0:
            shifted_ar[0:rR, :] = 0
        elif rR < 0:
            shifted_ar[rR:, :] = 0
        if cR > 0:
            shifted_ar[:, 0:cR] = 0
        elif cR < 0:
            shifted_ar[:, cR:] = 0

    return shifted_ar


def make_Fourier_coords2D(
    Nr: int, Nc: int, pixelSize: float | tuple[float, float] = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates Fourier coordinates for a (Nr,Nc)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
    """
    if isinstance(pixelSize, (tuple, list)):
        assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
        pixelSize_r = pixelSize[0]
        pixelSize_c = pixelSize[1]
    else:
        pixelSize_r = pixelSize
        pixelSize_c = pixelSize

    qr = np.fft.fftfreq(Nr, pixelSize_r)
    qc = np.fft.fftfreq(Nc, pixelSize_c)
    qc, qr = np.meshgrid(qc, qr)
    return qr, qc


######## Fitting


def _plane(xy, mx, my, b):
    return mx * xy[0] + my * xy[1] + b


def _parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    return (
        c0 + cx1 * xy[0] + cy1 * xy[1] + cx2 * xy[0] ** 2 + cy2 * xy[1] ** 2 + cxy * xy[0] * xy[1]
    )


def _bezier_two(xy, c00, c01, c02, c10, c11, c12, c20, c21, c22):
    return (
        c00 * ((1 - xy[0]) ** 2) * ((1 - xy[1]) ** 2)
        + c10 * 2 * (1 - xy[0]) * xy[0] * ((1 - xy[1]) ** 2)
        + c20 * (xy[0] ** 2) * ((1 - xy[1]) ** 2)
        + c01 * 2 * ((1 - xy[0]) ** 2) * (1 - xy[1]) * xy[1]
        + c11 * 4 * (1 - xy[0]) * xy[0] * (1 - xy[1]) * xy[1]
        + c21 * 2 * (xy[0] ** 2) * (1 - xy[1]) * xy[1]
        + c02 * ((1 - xy[0]) ** 2) * (xy[1] ** 2)
        + c12 * 2 * (1 - xy[0]) * xy[0] * (xy[1] ** 2)
        + c22 * (xy[0] ** 2) * (xy[1] ** 2)
    )


# TODO -- testing this
def fit_origin(
    data: np.ndarray | tuple[np.ndarray, np.ndarray],
    mask: np.ndarray | None = None,
    fit_function: Literal["plane", "parabola", "bezier_two", "constant"] = "plane",
    robust=False,
    robust_steps=3,
    robust_thresh=2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the origin of diffraction space using the specified method."""

    qr0_meas, qc0_meas = data

    if fit_function == "plane":
        f = _plane
    elif fit_function == "parabola":
        f = _parabola
    elif fit_function == "bezier_two":
        f = _bezier_two
    elif fit_function == "constant":
        # only average over the masked-in (and finite) positions; otherwise NaN/masked-out
        # positions (e.g. zeroed diffraction patterns) would poison the mean
        qr0_sel = qr0_meas[mask] if mask is not None else qr0_meas
        qc0_sel = qc0_meas[mask] if mask is not None else qc0_meas
        qr0_fit = np.nanmean(qr0_sel) * np.ones_like(qr0_meas)
        qc0_fit = np.nanmean(qc0_sel) * np.ones_like(qc0_meas)
        qr0_residuals = qr0_meas - qr0_fit
        qc0_residuals = qc0_meas - qc0_fit
        return qr0_fit, qc0_fit, qr0_residuals, qc0_residuals
    else:
        raise ValueError(
            "fit_function must be one of 'plane', 'parabola', 'bezier_two', 'constant'"
        )
    shape = qr0_meas.shape
    r, c = np.indices(shape)
    r1D = r.reshape(1, np.prod(shape))
    c1D = c.reshape(1, np.prod(shape))
    rc = np.vstack((r1D, c1D))

    if mask is not None:
        qr0_meas_masked = qr0_meas[mask]
        qc0_meas_masked = qc0_meas[mask]
        mask1D = mask.reshape(1, np.prod(shape))
        rc_masked = np.vstack((r1D[mask1D], c1D[mask1D]))
        # old failed for zero-valued DPs
        # rc_masked = np.vstack((r1D * mask1D, c1D * mask1D))

        popt_r, _ = curve_fit(f, rc_masked, qr0_meas_masked)
        popt_c, _ = curve_fit(f, rc_masked, qc0_meas_masked)

        if robust:
            popt_r = perform_robust_fitting(
                f, rc_masked, qr0_meas_masked, popt_r, robust_steps, robust_thresh
            )
            popt_c = perform_robust_fitting(
                f, rc_masked, qc0_meas_masked, popt_c, robust_steps, robust_thresh
            )
    else:
        popt_r, _ = curve_fit(f, rc, qr0_meas)
        popt_c, _ = curve_fit(f, rc, qc0_meas)

        if robust:
            popt_r = perform_robust_fitting(f, rc, qr0_meas, popt_r, robust_steps, robust_thresh)
            popt_c = perform_robust_fitting(f, rc, qc0_meas, popt_c, robust_steps, robust_thresh)

    qr0_fit = f(rc, *popt_r).reshape(shape)
    qc0_fit = f(rc, *popt_c).reshape(shape)
    qr0_residuals = qr0_meas - qr0_fit
    qc0_residuals = qc0_meas - qc0_fit

    return qr0_fit, qc0_fit, qr0_residuals, qc0_residuals


def perform_robust_fitting(func, rc, data, initial_guess, robust_steps, robust_thresh):
    """Performs robust fitting by iteratively rejecting outliers."""
    popt = initial_guess
    for k in range(robust_steps):
        fit_vals = func(rc, *popt)
        rmse = np.sqrt(np.mean((fit_vals - data) ** 2))
        mask = np.abs(fit_vals - data) <= robust_thresh * rmse
        rc = np.vstack((rc[0][mask], rc[1][mask]))
        data = data[mask]
        popt, _ = curve_fit(func, rc, data, p0=popt)
    return popt


class AffineTransform:
    """
    Affine Transform Class.

    Simplified version of AffineTransform from tike:
    https://github.com/AdvancedPhotonSource/tike/blob/f9004a32fda5e49fa63b987e9ffe3c8447d59950/src/tike/ptycho/position.py

    AffineTransform() -> Identity

    Parameters
    ----------
    scale0: float
        x-scaling
    scale1: float
        y-scaling
    shear1: float
        \\gamma shear
    angle: float
        \\theta rotation angle
    t0: float
        x-translation
    t1: float
        y-translation
    dilation: float
        Isotropic expansion (multiplies scale0 and scale1)
    """

    def __init__(
        self,
        scale0: float = 1.0,
        scale1: float = 1.0,
        shear1: float = 0.0,
        angle: float = 0.0,
        t0: float = 0.0,
        t1: float = 0.0,
        dilation: float = 1.0,
    ):
        self.scale0 = scale0 * dilation
        self.scale1 = scale1 * dilation
        self.shear1 = shear1
        self.angle = angle
        self.t0 = t0
        self.t1 = t1

    @classmethod
    def from_array(cls, T: np.ndarray):
        """
        Return an Affine Transfrom from a 2x2 matrix.
        Use decomposition method from Graphics Gems 2 Section 7.1
        """
        R = T[:2, :2].copy()
        scale0 = np.linalg.norm(R[0])
        if scale0 <= 0:
            return cls()
        R[0] /= scale0
        shear1 = R[0] @ R[1]
        R[1] -= shear1 * R[0]
        scale1 = np.linalg.norm(R[1])
        if scale1 <= 0:
            return cls()
        R[1] /= scale1
        shear1 /= scale1
        angle = np.arctan2(-R[0, 1], R[0, 0])

        if T.shape[0] > 2:
            t0, t1 = T[2]
        else:
            t0 = t1 = 0.0

        return cls(
            scale0=float(scale0),
            scale1=float(scale1),
            shear1=float(shear1),
            angle=float(angle),
            t0=t0,
            t1=t1,
        )

    def asarray(self):
        """
        Return an 2x2 matrix of scale, shear, rotation.
        This matrix is scale @ shear @ rotate from left to right.
        """
        cosx = np.cos(self.angle)
        sinx = np.sin(self.angle)
        return (
            np.array(
                [
                    [self.scale0, 0.0],
                    [0.0, self.scale1],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [1.0, 0.0],
                    [self.shear1, 1.0],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [+cosx, -sinx],
                    [+sinx, +cosx],
                ],
                dtype="float32",
            )
        )

    def asarray3(self):
        """
        Return an 3x2 matrix of scale, shear, rotation, translation.
        This matrix is scale @ shear @ rotate from left to right.
        Expects a homogenous (z) coordinate of 1.
        """
        T = np.empty((3, 2), dtype="float32")
        T[2] = (self.t0, self.t1)
        T[:2, :2] = self.asarray()
        return T

    def astuple(self):
        """Return the constructor parameters in a tuple."""
        return (
            self.scale0,
            self.scale1,
            self.shear1,
            self.angle,
            self.t0,
            self.t1,
        )

    def __call__(self, x: np.ndarray, origin=(0, 0), xp=np) -> np.ndarray:
        origin = xp.asarray(origin, dtype=xp.float32)
        tf_matrix = self.asarray()
        tf_matrix = xp.asarray(tf_matrix, dtype=xp.float32)
        tf_translation = xp.array((self.t0, self.t1)) + origin
        return ((x - origin) @ tf_matrix) + tf_translation

    def __str__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )

    def __repr__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )


def center_crop_arr(
    arr: np.ndarray, shape: tuple[int, ...], pad_if_needed: bool = False
) -> np.ndarray:
    """
    Crop an array to a given shape, centered along all axes.

    Parameters
    ----------
    arr : np.ndarray
        The input n-dimensional array to be cropped.
    shape : tuple[int, ...]
        The desired output shape. Must have the same number of dimensions as arr,
        and each dimension must be less than or equal to the corresponding dimension of arr.
    """
    if len(shape) != arr.ndim:
        raise ValueError(
            f"Shape must have the same number of dimensions as arr. "
            f"Got shape with {len(shape)} dimensions and arr with {arr.ndim} dimensions."
        )

    pad = [[0, 0]] * len(shape)
    for i, (s, a) in enumerate(zip(shape, arr.shape)):
        if s > a:
            if not pad_if_needed:
                raise ValueError(
                    f"Dimension {i} of shape ({s}) is larger than dimension {i} of arr ({a})."
                )
            pad[i] = [(s - a) // 2, ceil((s - a) / 2)]

    if any(p != [0, 0] for p in pad):
        arr = np.pad(arr, pad_width=pad, mode="constant")

    slices = []
    for i, (s, a) in enumerate(zip(shape, arr.shape)):
        start = (a - s) // 2
        end = start + s
        slices.append(slice(start, end))

    # Return the cropped array
    return arr[tuple(slices)]
