"""
scikit-image's radon and iradon functions fully implemented in Torch.

Reference: van der Walt, S., et al. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.
"""

import torch
import torch.nn.functional as F


def radon_torch(images, theta=None, device=None):
    """
    Batched Radon transform implemented in PyTorch.
    images: torch.Tensor of shape [B, H, W]
    Returns: torch.Tensor of shape [B, N_angles, N_pixels]
    """
    if images.ndim == 2:
        images = images.unsqueeze(0)  # [1, H, W]
    B, H, W = images.shape

    if device is None:
        device = images.device

    if theta is None:
        theta = torch.arange(180, device=device)

    N_angles = len(theta)
    shape_min = min(H, W)
    radius = shape_min // 2
    center = torch.tensor([H // 2, W // 2], device=device)

    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    dist2 = (X - center[1]) ** 2 + (Y - center[0]) ** 2
    mask = dist2 <= radius**2
    images = images.clone()
    images *= mask  # broadcasting over batch

    # Crop to square
    excess = torch.tensor([H, W], device=device) - shape_min
    slices = tuple(
        slice(int((e.item() + 1) // 2), int((e.item() + 1) // 2 + shape_min))
        if e > 0
        else slice(None)
        for e in excess
    )
    images = images[:, slices[0], slices[1]]  # [B, N, N]
    N = images.shape[-1]
    center = N // 2

    radon_images = torch.zeros((B, N_angles, N), dtype=images.dtype, device=device)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(N, dtype=torch.float32, device=device),
        torch.arange(N, dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.stack((grid_x - center, grid_y - center), dim=-1)  # (N, N, 2)
    coords = coords.view(1, N, N, 2).expand(B, -1, -1, -1)  # [B, N, N, 2]

    for i, angle in enumerate(theta):
        angle_rad = torch.deg2rad(angle)
        rot = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [-torch.sin(angle_rad), -torch.cos(angle_rad)],
            ],
            device=device,
            dtype=torch.float32,
        )

        rot = rot.unsqueeze(0).expand(B, -1, -1)  # [B, 2, 2]
        coords_rot = torch.matmul(coords.view(B, -1, 2), rot.transpose(1, 2)).view(B, N, N, 2)
        coords_rot += center

        # Normalize to [-1, 1]
        grid = 2 * coords_rot / (N - 1) - 1  # [B, N, N, 2]

        # grid = grid.unsqueeze(1)  # [B, 1, N, N, 2]
        imgs = images.unsqueeze(1)  # [B, 1, N, N]

        sampled = F.grid_sample(
            imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        projection = sampled.squeeze(1).sum(dim=1)  # [B, N]
        radon_images[:, i, :] = projection

    return radon_images.squeeze(0) if radon_images.shape[0] == 1 else radon_images


def iradon_torch(
    sinograms,
    theta=None,
    output_size=None,
    filter_name="ramp",
    circle=True,
    device=None,
):
    """
    Batched inverse Radon transform (filtered backprojection).
    sinograms: [B, N_angles, N_pixels] or [N_angles, N_pixels] (automatically batched)
    Returns: [B, output_size, output_size] or [output_size, output_size]
    """
    if sinograms.ndim == 2:
        sinograms = sinograms.unsqueeze(0)  # [1, A, P]
    B, A, N = sinograms.shape

    device = sinograms.device if device is None else device
    theta = theta if theta is not None else torch.linspace(0, 180, steps=A, device=device)

    if theta.shape[0] != A:
        raise ValueError("theta does not match number of projections")

    if output_size is None:
        output_size = N if circle else int(torch.floor(torch.sqrt(torch.tensor(N**2 / 2.0))))

    # Padding for FFT
    padded_size = max(
        64, int(2 ** torch.ceil(torch.log2(torch.tensor(2 * N, dtype=torch.float32))))
    )
    pad_y = padded_size - N
    sinograms_padded = F.pad(sinograms, (0, pad_y))  # [B, A, padded]

    f_filter = get_fourier_filter_torch(padded_size, filter_name, device=device)  # [1, padded]
    spectrum = torch.fft.fft(sinograms_padded, dim=2)
    filtered = torch.real(torch.fft.ifft(spectrum * f_filter, dim=2))[:, :, :N]

    # Backprojection
    recon = torch.zeros((B, output_size, output_size), device=device)
    radius = output_size // 2

    y, x = torch.meshgrid(
        torch.arange(output_size, device=device) - radius,
        torch.arange(output_size, device=device) - radius,
        indexing="ij",
    )
    x = x.flatten()
    y = y.flatten()

    for i, angle in enumerate(torch.deg2rad(theta)):
        t = (x * torch.cos(angle) - y * torch.sin(angle)).reshape(1, output_size, output_size)
        t_idx = t + (N // 2)

        t0 = torch.floor(t_idx).long().clamp(0, N - 2)  # [1, H, W]
        t1 = t0 + 1
        w = t_idx - t0.float()

        t0 = t0.expand(B, -1, -1)  # [B, H, W]
        t1 = t1.expand(B, -1, -1)

        filtered_i = filtered[:, i, :]  # [B, N]
        val0 = torch.gather(filtered_i, 1, t0.view(B, -1)).view(B, output_size, output_size)
        val1 = torch.gather(filtered_i, 1, t1.view(B, -1)).view(B, output_size, output_size)

        proj = (1 - w) * val0 + w * val1
        recon += proj

    if circle:
        mask = (
            x.view(output_size, output_size) ** 2 + y.view(output_size, output_size) ** 2
            > radius**2
        )
        recon[:, mask] = 0.0

    recon *= torch.pi / (2 * A)
    return recon.squeeze(0) if recon.shape[0] == 1 else recon


def get_fourier_filter_torch(size, filter_name="ramp", device=None, dtype=torch.float32):
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
