"""
Radon transform and filtered back-projection in Torch, after scikit-image's radon and iradon.

Reference: van der Walt, S., et al. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.

Angles follow quantem's tomography convention (see `quantem.tomography.utils`): theta is the
tilt in degrees with the sign of the INR rays, and slices rotate about the array centre
(N - 1) / 2. scikit-image's theta is minus this tilt, and it rotates about N // 2, so for even N
the two differ by half a pixel.
"""

import math

import torch
import torch.nn.functional as F


def radon_torch(images, theta=None, device=None):
    if images.ndim == 2:
        images = images.unsqueeze(0)
    B, H, W = images.shape
    device = device or images.device
    theta = theta if theta is not None else torch.arange(180, device=device)
    A = len(theta)

    shape_min = min(H, W)
    radius = shape_min // 2
    center_h, center_w = (H - 1) / 2, (W - 1) / 2

    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    mask = (X - center_w) ** 2 + (Y - center_h) ** 2 <= radius**2
    images = images.clone() * mask

    excess = torch.tensor([H, W], device=device) - shape_min
    slices = tuple(
        slice(int((e.item() + 1) // 2), int((e.item() + 1) // 2 + shape_min))
        if e > 0
        else slice(None)
        for e in excess
    )
    images = images[:, slices[0], slices[1]]  # [B, N, N]
    N = images.shape[-1]
    center = (N - 1) / 2

    # Build all rotation matrices at once: [A, 2, 2]
    angles_rad = torch.deg2rad(theta.float())
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    rot = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=1),
            torch.stack([sin_a, cos_a], dim=1),
        ],
        dim=1,
    )  # [A, 2, 2]

    # Pixel coords: [N*N, 2]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(N, dtype=torch.float32, device=device),
        torch.arange(N, dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.stack((grid_x - center, grid_y - center), dim=-1).view(-1, 2)  # [N*N, 2]

    # Rotate all angles at once and normalize to [-1, 1] for grid_sample: [A, N, N, 2]
    grid = (coords @ rot.transpose(1, 2) / center).view(A, N, N, 2)

    # Expand images: [B, 1, N, N] -> sample with [B*A, 1, N, N] style
    # Tile images over angles, tile grid over batch
    images_exp = images.unsqueeze(1).expand(-1, A, -1, -1).reshape(B * A, 1, N, N)
    grid_exp = grid.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * A, N, N, 2)

    sampled = F.grid_sample(
        images_exp, grid_exp, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    # sampled: [B*A, 1, N, N] -> sum over rows -> [B, A, N]
    radon_images = sampled.squeeze(1).sum(dim=1).view(B, A, N)

    return radon_images.squeeze(0) if B == 1 else radon_images


def iradon_torch(
    sinograms,
    theta=None,
    output_size=None,
    filter_name: str | None = "ramp",
    circle=True,
    device=None,
):
    if sinograms.ndim == 2:
        sinograms = sinograms.unsqueeze(0)
    B, A, N = sinograms.shape
    device = device or sinograms.device
    # Match radon_torch / scikit-image: A angles evenly spanning [0, 180) (endpoint excluded).
    theta = theta if theta is not None else torch.linspace(0, 180, steps=A + 1, device=device)[:-1]

    if output_size is None:
        output_size = N if circle else int((N**2 / 2) ** 0.5)

    # Pad the detector to the diagonal of the reconstruction circle, as scikit-image does, so
    # pixels near the circle edge read filtered values instead of extrapolating past the
    # detector.
    D = math.ceil(math.sqrt(2) * N) if circle else N
    pad = D // 2 - N // 2
    sinograms = F.pad(sinograms, (pad, D - N - pad))

    # Filter
    padded_size = max(64, int(2 ** math.ceil(math.log2(2 * D))))
    sinograms_padded = F.pad(sinograms, (0, padded_size - D))
    f_filter = get_fourier_filter_torch(padded_size, filter_name, device=device)
    filtered = torch.real(
        torch.fft.ifft(torch.fft.fft(sinograms_padded, dim=2) * f_filter, dim=2)
    )[:, :, :D]  # [B, A, D]

    center = (output_size - 1) / 2
    y, x = torch.meshgrid(
        torch.arange(output_size, device=device) - center,
        torch.arange(output_size, device=device) - center,
        indexing="ij",
    )
    x = x.float().flatten()  # [H*W]
    y = y.float().flatten()

    # Detector coordinate of every pixel for all angles at once: [A, H*W]
    angles_rad = torch.deg2rad(theta.float())
    t = x.unsqueeze(0) * torch.cos(angles_rad).unsqueeze(1) + y.unsqueeze(0) * torch.sin(
        angles_rad
    ).unsqueeze(1)  # [A, H*W]
    t_idx = t + (N - 1) / 2 + pad  # [A, H*W]

    t0 = t_idx.floor().long().clamp(0, D - 2)  # [A, H*W]
    t1 = t0 + 1
    w = t_idx - t0.float()  # [A, H*W]
    inside = (t_idx >= 0) & (t_idx <= D - 1)  # zero outside the detector

    # Gather from filtered: [B, A, D] -> gather at [A, H*W] indices
    # Expand: [B, A, H*W]
    t0_exp = t0.unsqueeze(0).expand(B, -1, -1)  # [B, A, H*W]
    t1_exp = t1.unsqueeze(0).expand(B, -1, -1)
    w_exp = w.unsqueeze(0).expand(B, -1, -1)

    val0 = torch.gather(filtered, 2, t0_exp)  # [B, A, H*W]
    val1 = torch.gather(filtered, 2, t1_exp)
    proj = (((1 - w_exp) * val0 + w_exp * val1) * inside).sum(dim=1)  # [B, H*W]

    recon = proj.view(B, output_size, output_size)

    if circle:
        radius = output_size // 2
        mask = (
            x.view(output_size, output_size) ** 2 + y.view(output_size, output_size) ** 2
            > radius**2
        )
        recon[:, mask] = 0.0

    recon *= torch.pi / (2 * A)
    return recon.squeeze(0) if B == 1 else recon


def get_fourier_filter_torch(
    size, filter_name: str | None = "ramp", device=None, dtype=torch.float32
):
    """
    Construct the Fourier filter in PyTorch.
    """
    if size % 2 != 0:
        raise ValueError("Filter size must be even")

    n = torch.cat(
        [
            torch.arange(1, size // 2 + 1, 2, device=device),
            torch.arange(size // 2 - 1, 0, -2, device=device),
        ]
    )
    f = torch.zeros(size, device=device, dtype=dtype)
    f[0] = 0.25
    f[1::2] = -1.0 / (torch.pi * n.float()) ** 2

    fourier_filter = 2 * torch.real(torch.fft.fft(f))

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = torch.pi * torch.fft.fftfreq(size, device=device)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega
    elif filter_name == "cosine":
        freq = torch.linspace(0, torch.pi, steps=size, device=device)
        fourier_filter *= torch.fft.fftshift(torch.sin(freq))
    elif filter_name == "hamming":
        hamming = torch.hamming_window(size, periodic=False, dtype=dtype, device=device)
        fourier_filter *= torch.fft.fftshift(hamming)
    elif filter_name == "hann":
        hann = torch.hann_window(size, periodic=False, dtype=dtype, device=device)
        fourier_filter *= torch.fft.fftshift(hann)
    elif filter_name is None:
        fourier_filter[:] = 1.0
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    # Reshape filter for broadcasting with sinogram
    return fourier_filter.unsqueeze(0)  # Shape: [1, size] for broadcasting with [num_angles, size]
