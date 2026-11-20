from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d

# image units converted to Angstroms, so the reciprocal axes match Crystal
_LENGTH_TO_ANGSTROM = {"a": 1.0, "å": 1.0, "angstrom": 1.0, "nm": 10.0, "pm": 0.01, "um": 1e4}


def window_mask(
    window_size: int,
    edge_width: float | None = None,
) -> NDArray:
    """Smooth circular window for local Fourier transforms.

    The window is 1 inside a disk and falls to 0 at radius ``window_size / 2``
    as ``sin(u * pi / 2) ** 2``, where ``u`` runs from 1 at the inner edge of
    the taper to 0 at the outer radius. With the default ``edge_width`` the
    taper spans the whole radius, so ``u = 1 - r / R`` and no part of the
    window is flat.

    Parameters
    ----------
    window_size : int
        Width of the square window in pixels; the disk diameter.
    edge_width : float, optional
        Radial width of the taper in pixels. None (default) uses the full
        radius ``window_size / 2``.

    Returns
    -------
    np.ndarray
        ``(window_size, window_size)`` float32 window.
    """
    radius = window_size / 2.0
    edge = radius if edge_width is None else float(edge_width)
    if not 0.0 < edge <= radius:
        raise ValueError(f"edge_width must be in (0, {radius}], got {edge_width}.")
    x = np.arange(window_size) - (window_size - 1) / 2.0
    r = np.hypot(x[:, None], x[None, :])
    u = np.clip((radius - r) / edge, 0.0, 1.0)
    return (np.sin(u * np.pi / 2.0) ** 2).astype(np.float32)


class LocalFFT(AutoSerialize):
    """Pseudo-4DSTEM dataset from the local Fourier transforms of an image.

    A circular window with a smooth edge (:func:`window_mask`) is stepped over
    the image on a regular grid, and the Fourier transform of each windowed
    region becomes the diffraction pattern at that grid position. Lattice
    fringes in the window produce pairs of spots at plus and minus their
    reciprocal lattice vectors, so the standard 4DSTEM workflow (Bragg spot
    detection, calibration, orientation and phase matching) applies to the
    result. Compared with a 4DSTEM experiment, the pattern center is known
    exactly, the patterns have no descan or elliptic distortion, and the
    reciprocal pixel size follows from the image calibration and the window
    size. The intensities are set by the contrast transfer of the microscope
    and the spot shape by the window, so they are not kinematic diffraction
    intensities.

    Each window has its weighted mean subtracted before the transform, which
    removes most of the zero-frequency peak. The image is divided by its
    global mean, so the patterns are in units of fractional contrast, and the
    power is divided by the sum of the squared window: white noise of variance
    ``sigma ** 2`` (in fractional contrast) then gives a flat background of
    ``sigma ** 2``. The background of an HRTEM image falls steeply with
    spatial frequency (amorphous contrast, the envelope of the contrast
    transfer and the detector response), so by default every pattern is
    normalized by the azimuthal median of the scan-mean pattern
    (:attr:`background`) as ``I / background - 1``. The background is then 0
    and the noise close to 1 at every frequency, the spot intensities are
    excess over the local background, and one detection threshold applies
    across the whole pattern.

    Use :meth:`from_image` to construct an instance. The pseudo-4DSTEM data is
    held in :attr:`dataset`, and :attr:`origins` holds the exact pattern
    centers in the form :meth:`BraggVectors.correct_peak_origins` expects.

    Parameters
    ----------
    image : Dataset2d
        The image, with its sampling converted to Angstroms when the units
        are known.
    window_size, step, edge_width, k_max, fft_size, intensity, normalize, center_radius
        See :meth:`from_image`.
    """

    _token = object()

    def __init__(
        self,
        image: Dataset2d,
        window_size: int,
        step: int,
        edge_width: float | None,
        k_max: float | None,
        fft_size: int,
        intensity: str,
        normalize: bool,
        center_radius: float,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use LocalFFT.from_image() to instantiate this class.")
        super().__init__()
        self.image = image
        self.window_size = int(window_size)
        self.step = int(step)
        self.edge_width = edge_width
        self.k_max = k_max
        self.fft_size = int(fft_size)
        self.intensity = intensity
        self.normalize = bool(normalize)
        self.center_radius = float(center_radius)
        self.window = window_mask(self.window_size, edge_width)
        self.dataset: Dataset4dstem | None = None
        self.background: NDArray | None = None
        self.corners: NDArray | None = None
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_image(
        cls,
        image: Dataset2d | NDArray,
        window_size: int = 128,
        step: int | None = None,
        edge_width: float | None = None,
        k_max: float | None = None,
        fft_size: int | None = None,
        intensity: Literal["power", "amplitude"] = "power",
        normalize: bool = True,
        center_radius: float = 1.5,
        sampling: float | None = None,
        device: str = "cpu",
        batch_size: int = 1024,
    ) -> "LocalFFT":
        """Compute the local Fourier transforms of an image.

        The window grid is centered in the image, with ``step`` pixels between
        neighboring windows and every window fully inside the image.

        Parameters
        ----------
        image : Dataset2d | np.ndarray
            The image. A Dataset2d in nm, pm, um or Angstroms is converted to
            Angstroms; an array needs ``sampling``.
        window_size : int, default=128
            Width of the square window in pixels, and the diameter of the
            circular mask. The reciprocal pixel size is
            ``1 / (fft_size * sampling)``, and a spot from a crystal filling
            the window is about two pixels wide.
        step : int, optional
            Step between windows in pixels. None (default) uses
            ``window_size // 4``.
        edge_width : float, optional
            Radial width of the ``sin ** 2`` taper in pixels. None (default)
            tapers over the full radius; a narrower edge keeps more of the
            window at full weight, which sharpens the spots and raises the
            side lobes of strong fringes.
        k_max : float, optional
            Crop each pattern to ``|k_x|, |k_y| <= k_max`` (1/Angstroms) around
            the center. None (default) keeps every frequency up to Nyquist.
            Spots beyond the information limit only add noise to the
            detection and the matching.
        fft_size : int, optional
            Fourier transform size in pixels; the window is zero padded to it,
            which samples the pattern more finely without adding information.
            None (default) uses ``window_size``.
        intensity : {"power", "amplitude"}, default="power"
            Store the power spectrum ``|F| ** 2`` or the amplitude ``|F|``.
            The amplitude compresses the range between strong and weak
            fringes.
        normalize : bool, default=True
            Remove :attr:`background`, the azimuthal median of the scan-mean
            pattern, as ``I / background - 1``. The background is then 0 and
            the noise close to 1 at every frequency, and a spot intensity is
            its excess over the local background. The median ignores the
            strong arcs of a textured sample.
        center_radius : float, default=1.5
            With ``normalize``, pixels closer than this to zero frequency are
            set to 0. The orientation matching weights every spot by its
            radius, so the center carries no weight there and nothing is
            lost by removing it.
        sampling : float, optional
            Image pixel size in Angstroms. Overrides the calibration of a
            Dataset2d, and is required for an array.
        device : str, default="cpu"
            Torch device for the Fourier transforms.
        batch_size : int, default=1024
            Number of windows transformed at once.

        Returns
        -------
        LocalFFT
            The instance, with :attr:`dataset` holding the pseudo-4DSTEM data
            in shape ``(scan_row, scan_col, k_row, k_col)``.
        """
        image = _as_angstrom_dataset(image, sampling)
        window_size = int(window_size)
        step = window_size // 4 if step is None else int(step)
        fft_size = window_size if fft_size is None else int(fft_size)
        if fft_size < window_size:
            raise ValueError("fft_size must be at least window_size.")
        if intensity not in ("power", "amplitude"):
            raise ValueError("intensity must be 'power' or 'amplitude'.")
        if min(image.shape) < window_size:
            raise ValueError("The window is larger than the image.")

        lfft = cls(
            image=image,
            window_size=window_size,
            step=step,
            edge_width=edge_width,
            k_max=k_max,
            fft_size=fft_size,
            intensity=intensity,
            normalize=normalize,
            center_radius=center_radius,
            _token=cls._token,
        )
        lfft._compute(device=device, batch_size=batch_size)
        return lfft

    @property
    def sampling(self) -> float:
        """Image pixel size in Angstroms."""
        return float(self.image.sampling[0])

    @property
    def pixel_size(self) -> float:
        """Reciprocal pixel size of the patterns in 1/Angstroms."""
        return 1.0 / (self.fft_size * self.sampling)

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Number of windows along the image rows and columns."""
        return tuple(int(v) for v in self.dataset.shape[:2])

    @property
    def center(self) -> tuple[float, float]:
        """Pattern pixel (row, col) of zero spatial frequency."""
        c = float(self.metadata["center_px"])
        return (c, c)

    @property
    def origins(self) -> NDArray:
        """``(scan_row, scan_col, 2)`` pattern centers for every window.

        Pass to :meth:`BraggVectors.correct_peak_origins` with
        ``origin_ref=(0, 0)`` to put the detected spots in a frame centered on
        zero frequency, in place of measuring and fitting the origins.
        """
        return np.broadcast_to(np.asarray(self.center), self.scan_shape + (2,)).copy()

    @property
    def window_fwhm(self) -> float:
        """Diameter of the window where its weight is 1/2, in Angstroms.

        This is the effective spatial resolution of the maps: ``R`` for the
        full-radius taper, ``2 R - edge_width`` for a narrower edge, with ``R``
        the window radius.
        """
        radius = self.window_size / 2.0
        edge = radius if self.edge_width is None else float(self.edge_width)
        return (2.0 * radius - edge) * self.sampling

    @property
    def window_centers(self) -> NDArray:
        """``(scan_row, scan_col, 2)`` window centers in image pixels."""
        return self.corners + (self.window_size - 1) / 2.0

    def _compute(self, device: str, batch_size: int) -> None:
        """Fill :attr:`dataset` with the windowed Fourier transforms."""
        img = np.asarray(self.image.array, dtype=np.float32)
        mean = float(img.mean())
        if mean <= 0:
            raise ValueError("The image mean must be positive to normalize the contrast.")
        img = img / mean

        n, s, nf = self.window_size, self.step, self.fft_size
        num = [(size - n) // s + 1 for size in img.shape]
        offset = [(size - n - (m - 1) * s) // 2 for size, m in zip(img.shape, num)]
        rows = offset[0] + s * np.arange(num[0])
        cols = offset[1] + s * np.arange(num[1])
        self.corners = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1)

        # crop symmetric about zero frequency to an odd size, so the center is
        # a pixel and every spot keeps its Friedel mate
        c = nf // 2
        if self.k_max is None:
            h = c - 1 if nf % 2 == 0 else c
        else:
            h = min(int(np.floor(self.k_max / self.pixel_size)), c - 1 if nf % 2 == 0 else c)
        crop = slice(c - h, c + h + 1)
        size = 2 * h + 1

        dev = torch.device(device)
        w = torch.as_tensor(self.window, device=dev)
        w_sum = w.sum()
        norm = (w * w).sum()
        image_t = torch.as_tensor(img, device=dev)
        flat = self.corners.reshape(-1, 2)
        out = np.empty((flat.shape[0], size, size), dtype=np.float32)
        ar = torch.arange(n, device=dev)
        for start in range(0, flat.shape[0], batch_size):
            corner = torch.as_tensor(flat[start : start + batch_size], device=dev)
            ri = (corner[:, 0, None] + ar[None, :])[:, :, None]
            ci = (corner[:, 1, None] + ar[None, :])[:, None, :]
            x = image_t[ri, ci]
            x_mean = (x * w).sum(dim=(1, 2), keepdim=True) / w_sum
            x = (x - x_mean) * w
            F = torch.fft.fftshift(torch.fft.fft2(x, s=(nf, nf)), dim=(-2, -1))[:, crop, crop]
            p = F.real**2 + F.imag**2
            p = p / norm
            if self.intensity == "amplitude":
                p = torch.sqrt(p)
            out[start : start + batch_size] = p.cpu().numpy()

        # azimuthal median of the mean pattern in 1 pixel shells, interpolated
        # linearly at the radius of every pixel so the background is smooth
        x = np.arange(size) - h
        r = np.hypot(x[:, None], x[None, :])
        r_int = np.round(r).astype(int)
        mean_dp = out.mean(axis=0)
        profile = np.array([np.median(mean_dp[r_int == i]) for i in range(r_int.max() + 1)])
        self.background = np.interp(r, np.arange(profile.size), profile).astype(np.float32)
        if self.normalize:
            # inside the center disk the background is the zero-frequency
            # remainder of the mean subtraction, close to 0, so the ratio is
            # meaningless there; the disk is set to 0 instead
            inside = r < self.center_radius
            bg = np.where(inside, 1.0, self.background)
            out /= bg
            out -= 1.0
            out[:, inside] = 0.0

        dx = self.sampling
        dk = self.pixel_size
        origin_image = np.asarray(self.image.origin, dtype=float)
        self.dataset = Dataset4dstem.from_array(
            out.reshape(num[0], num[1], size, size),
            name=f"{self.image.name} local FFT",
            origin=[
                origin_image[0] + (offset[0] + (n - 1) / 2) * dx,
                origin_image[1] + (offset[1] + (n - 1) / 2) * dx,
                -h * dk,
                -h * dk,
            ],
            sampling=[s * dx, s * dx, dk, dk],
            units=["A", "A", "1/A", "1/A"],
            signal_units=(
                "ratio to background"
                if self.normalize
                else ("contrast^2" if self.intensity == "power" else "contrast")
            ),
        )
        self.metadata.update(
            center_px=float(h),
            image_mean=mean,
            window_size=n,
            step=s,
            edge_width=self.edge_width,
            fft_size=nf,
            k_max=self.k_max,
            intensity=self.intensity,
            normalize=self.normalize,
            center_radius=self.center_radius,
        )

    def show(
        self,
        positions: list[tuple[int, int]] | None = None,
        norm: dict | str | None = None,
        axsize: tuple[float, float] = (4, 4),
        **kwargs,
    ):
        """Show the window grid on the image, and some windows with their patterns.

        Parameters
        ----------
        positions : list of (row, col), optional
            Scan positions to show. None (default) shows the center of the
            scan.
        norm : dict | str, optional
            Normalization of the patterns, passed to
            :func:`~quantem.core.visualization.show_2d`. None (default) uses
            ``{"power": 0.5, "upper_quantile": 0.999}``.
        axsize : tuple, default=(4, 4)
            Size of each panel in inches.
        **kwargs
            Further arguments of :func:`~quantem.core.visualization.show_2d`
            for the pattern panels.

        Returns
        -------
        fig, axs
            The matplotlib figure and axes.
        """
        if positions is None:
            positions = [tuple(v // 2 for v in self.scan_shape)]
        norm = {"power": 0.5, "upper_quantile": 0.999} if norm is None else norm
        num = len(positions)
        fig, axs = plt.subplots(num, 3, figsize=(3 * axsize[0], num * axsize[1]), squeeze=False)

        img = np.asarray(self.image.array, dtype=np.float32)
        lo, hi = np.quantile(img, (0.01, 0.99))
        n = self.window_size
        for a, (r, c) in enumerate(positions):
            ax = axs[a, 0]
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            # the area covered by the window centers
            (r_lo, c_lo), (r_hi, c_hi) = self.window_centers[0, 0], self.window_centers[-1, -1]
            ax.plot(
                [c_lo, c_hi, c_hi, c_lo, c_lo],
                [r_lo, r_lo, r_hi, r_hi, r_lo],
                color=(1.0, 0.3, 0.3),
                lw=1,
            )
            r0, c0 = self.corners[r, c]
            t = np.linspace(0, 2 * np.pi, 181)
            ax.plot(
                c0 + (n - 1) / 2 + n / 2 * np.cos(t),
                r0 + (n - 1) / 2 + n / 2 * np.sin(t),
                color=(1.0, 0.8, 0.0),
            )
            ax.set_title(f"window ({r}, {c})")
            ax.set_axis_off()

            crop = img[r0 : r0 + n, c0 : c0 + n] * self.window
            axs[a, 1].imshow(crop, cmap="gray")
            axs[a, 1].set_title("masked window")
            axs[a, 1].set_axis_off()

            show_2d(
                np.asarray(self.dataset.array[r, c]),
                norm=norm,
                scalebar={"sampling": self.pixel_size, "units": "1/Å"},
                title="local FFT",
                figax=(fig, axs[a, 2]),
                **kwargs,
            )
        fig.tight_layout()
        return fig, axs

    def __repr__(self) -> str:
        if self.dataset is None:
            return "LocalFFT()"
        dx = self.sampling
        return (
            f"LocalFFT\n"
            f"  image      {self.image.shape[0]} x {self.image.shape[1]} px at {dx:.4f} A\n"
            f"  window     {self.window_size} px = {self.window_size * dx:.1f} A diameter, "
            f"{self.window_fwhm:.1f} A FWHM (effective resolution)\n"
            f"  step       {self.step} px = {self.step * dx:.1f} A, "
            f"scan {self.scan_shape[0]} x {self.scan_shape[1]}\n"
            f"  patterns   {self.dataset.shape[2]} x {self.dataset.shape[3]} px at "
            f"{self.pixel_size:.5f} 1/A per px, |k| <= {self.metadata['center_px'] * self.pixel_size:.3f} 1/A"
        )


def _as_angstrom_dataset(image: Dataset2d | NDArray, sampling: float | None) -> Dataset2d:
    """A float32 Dataset2d calibrated in Angstroms."""
    if isinstance(image, Dataset2d):
        units = [str(u).strip().lower() for u in image.units]
        scale = _LENGTH_TO_ANGSTROM.get(units[0])
        array = np.asarray(image.array, dtype=np.float32)
        if sampling is not None:
            sampling_A = float(sampling)
            origin_A = np.zeros(2)
        elif scale is not None:
            sampling_A = float(image.sampling[0]) * scale
            origin_A = np.asarray(image.origin, dtype=float) * scale
        else:
            raise ValueError(
                f"Image units {image.units} are not a length; pass sampling in Angstroms."
            )
        if not np.isclose(image.sampling[0], image.sampling[1], rtol=1e-6):
            raise ValueError("LocalFFT needs square image pixels.")
        name = image.name
    else:
        if sampling is None:
            raise ValueError("Pass sampling (Angstroms per pixel) with an array image.")
        array = np.asarray(image, dtype=np.float32)
        sampling_A = float(sampling)
        origin_A = np.zeros(2)
        name = "image"
    return Dataset2d.from_array(
        array,
        name=name,
        origin=origin_A,
        sampling=[sampling_A, sampling_A],
        units=["A", "A"],
    )
