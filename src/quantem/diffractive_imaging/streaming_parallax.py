import math
from typing import TYPE_CHECKING, Tuple

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.core.utils.validators import (
    validate_aberration_coefficients,
    validate_tensor,
)
from quantem.diffractive_imaging.complex_probe import (
    aberration_surface,
    aberration_surface_cartesian_gradients,
    polar_coordinates,
    spatial_frequencies,
)

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class StreamingParallax(AutoSerialize):
    _token = object()

    def __init__(
        self,
        gpts: Tuple[int, int],
        reciprocal_sampling: Tuple[float, float],
        scan_gpts: Tuple[int, int],
        scan_sampling: Tuple[float, float],
        wavelength: float,
        bf_mask: torch.Tensor,
        aberration_coefs: dict | None = None,
        rotation_angle: float | None = None,
        upsampling_factor: int = 1,
        enable_phase_flipping: bool = True,
        device: str | None = None,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use StreamingParallax.from_parameters() to instantiate this class."
            )

        self.reciprocal_sampling = reciprocal_sampling
        self.gpts = gpts
        self.scan_gpts = scan_gpts
        self.scan_sampling = scan_sampling
        self.wavelength = wavelength
        self.bf_mask = bf_mask
        self.aberration_coefs = aberration_coefs or {}
        self.rotation_angle = rotation_angle
        self.upsampling_factor = upsampling_factor
        self.enable_phase_flipping = enable_phase_flipping
        self.device = device

        self._preprocess()

    @classmethod
    def from_parameters(
        cls,
        gpts: Tuple[int, int],
        scan_gpts: Tuple[int, int],
        scan_sampling: Tuple[float, float],
        energy: float,
        bf_mask: torch.Tensor,
        reciprocal_sampling: Tuple[float, float] | None = None,
        angular_sampling: Tuple[float, float] | None = None,
        aberration_coefs: dict | None = None,
        rotation_angle: float | None = None,
        upsampling_factor: int = 1,
        enable_phase_flipping: bool = True,
        device: str | None = None,
    ):
        wavelength = electron_wavelength_angstrom(energy)
        if angular_sampling is not None:
            if reciprocal_sampling is not None:
                raise ValueError(
                    "Only one of reciprocal_sampling / angular_sampling can be specified"
                )
            reciprocal_sampling = tuple(a / wavelength / 1e3 for a in angular_sampling)

        return cls(
            gpts,
            reciprocal_sampling,
            scan_gpts,
            scan_sampling,
            wavelength,
            bf_mask,
            aberration_coefs=aberration_coefs,
            rotation_angle=rotation_angle,
            upsampling_factor=upsampling_factor,
            enable_phase_flipping=enable_phase_flipping,
            device=device,
            _token=cls._token,
        )

    @property
    def sampling(self):
        return tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))

    @property
    def upsampled_gpts(self):
        return tuple(g * self.upsampling_factor for g in self.scan_gpts)

    @property
    def upsampled_sampling(self):
        return tuple(s / self.upsampling_factor for s in self.scan_sampling)

    @property
    def aberration_coefs(self) -> dict:
        return self._aberration_coefs

    @aberration_coefs.setter
    def aberration_coefs(self, value: dict):
        value = validate_aberration_coefficients(value)
        self._aberration_coefs = value

    @property
    def bf_mask(self) -> torch.Tensor:
        return self._bf_mask

    @bf_mask.setter
    def bf_mask(self, value: torch.Tensor):
        self._bf_mask = validate_tensor(value, "bf_mask", dtype=torch.bool).to(device=self.device)

    @property
    def device(self) -> str:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        if hasattr(self, "_device"):
            return self._device
        else:
            return config.get("device")

    @device.setter
    def device(self, device: str | int | None):
        # allow setting gpu/cpu, but not changing the device from the config gpu device
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev

    def _return_upsampled_qgrid(
        self,
    ):
        qxa, qya = spatial_frequencies(
            self.upsampled_gpts,
            self.upsampled_sampling,
            device=self.device,
        )

        return qxa, qya

    def _return_rotated_kgrid(
        self,
    ):
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=self.rotation_angle,
            device=self.device,
        )

        return kxa, kya

    def _return_scan_positions(
        self,
    ):
        upsampled_gpts = self.upsampled_gpts
        upsampling_factor = self.upsampling_factor
        r_up = torch.dstack(
            torch.meshgrid(
                torch.arange(0, upsampled_gpts[0], upsampling_factor),
                torch.arange(0, upsampled_gpts[1], upsampling_factor),
                indexing="ij",
            ),
        ).reshape((-1, 2))

        return r_up

    def _prepare_phase_flipping(self, H, s_m_up):
        device = self.device
        Ny, Nx = self.upsampled_gpts

        h, w = H.shape
        M = s_m_up.shape[0]
        L0 = h * w

        # kernel grid
        dy = torch.arange(h, device=device)
        dx = torch.arange(w, device=device)
        dy_grid = dy.repeat_interleave(w)
        dx_grid = dx.repeat(h)

        dy_rep = dy_grid.repeat(M)
        dx_rep = dx_grid.repeat(M)

        s_my = s_m_up[:, 0].repeat_interleave(L0)
        s_mx = s_m_up[:, 1].repeat_interleave(L0)

        offsets = (dy_rep + s_my) * Nx + (dx_rep + s_mx)
        offsets = offsets.long()

        unique_offsets, inv = torch.unique(offsets, sorted=True, return_inverse=True)
        U = unique_offsets.numel()

        # build grouped kernel
        H_flat = H.reshape(-1)
        H_all = H_flat.repeat(M)
        m_idx = torch.arange(M, device=device).repeat_interleave(L0)

        K = torch.zeros((U, M), dtype=H.dtype, device=device)
        K.index_put_(
            (inv, m_idx),
            H_all,
            accumulate=True,
        )

        return unique_offsets, K

    def _preprocess(self):
        # ---- BF indices ----
        self.inds_i, self.inds_j = torch.where(self.bf_mask)
        ny, nx = self.gpts
        self.bf_flat_inds = self.inds_i * nx + self.inds_j
        self._bf_lut = torch.full(
            (ny * nx,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self._bf_lut[self.bf_flat_inds] = torch.arange(len(self.bf_flat_inds), device=self.device)

        # ---- parallax shifts ----
        kxa, kya = self._return_rotated_kgrid()
        k, phi = polar_coordinates(kxa, kya)
        dx, dy = aberration_surface_cartesian_gradients(
            k * self.wavelength,
            phi,
            aberration_coefs=self.aberration_coefs,
        )
        grad_k = torch.stack(
            (
                dx[self.inds_i, self.inds_j],
                dy[self.inds_i, self.inds_j],
            ),
            -1,
        )

        upsampled_sampling = torch.tensor(self.upsampled_sampling)
        self.s_m_up = torch.round(grad_k / 2 / math.pi / upsampled_sampling).int()

        # ---- phase flipping prep ----
        if self.enable_phase_flipping:
            qxa, qya = self._return_upsampled_qgrid()
            q, theta = polar_coordinates(qxa, qya)
            chi_q = aberration_surface(
                q * self.wavelength,
                theta,
                self.wavelength,
                aberration_coefs=self.aberration_coefs,
            )
            sign_sin_chi_q = torch.sign(torch.sin(chi_q))

            nx, ny = sign_sin_chi_q.shape
            sign_sin_chi_q[nx // 2, :] = 0.0
            sign_sin_chi_q[:, ny // 2] = 0.0
            H = torch.fft.ifft2(sign_sin_chi_q).real.contiguous()

            self.unique_offsets, self.K = self._prepare_phase_flipping(H, self.s_m_up)

        self._preprocessed = True
        return self

    def reconstruct_parallax(self, psi, I_batch, r_batch):
        """
        psi     : (Ny, Nx) accumulator
        I_batch : (B, ny, nx)
        r_batch : (B, 2) upsampled scan positions (y, x)
        """

        if not self._preprocessed:
            raise RuntimeError("Call preprocess() first.")

        Ny, Nx = self.upsampled_gpts
        I_bf = torch.fft.ifftshift(I_batch.to(torch.float32).to(self.device), dim=(-1, -2))[
            ..., self.inds_i, self.inds_j
        ]  # (B, M)

        I_bf -= I_bf.mean((-1, -2), keepdim=True)

        xs = (r_batch[:, None, 0] + self.s_m_up[None, :, 0]) % Ny
        ys = (r_batch[:, None, 1] + self.s_m_up[None, :, 1]) % Nx

        idx = (xs * Nx + ys).reshape(-1)
        vals = I_bf.reshape(-1)

        psi.view(-1).scatter_add_(0, idx, vals)

        return self

    def reconstruct_phase_flipping(self, psi_flipped, I_batch, r_batch):
        """
        psi_flipped : (Ny, Nx) accumulator
        I_batch     : (B, ny, nx)
        r_batch     : (B, 2)
        """
        if not self.enable_phase_flipping:
            raise RuntimeError("Phase flipping not enabled.")

        if not self._preprocessed:
            raise RuntimeError("Call preprocess() first.")

        Ny, Nx = self.upsampled_gpts
        I_bf = torch.fft.ifftshift(I_batch.to(torch.float32).to(self.device), dim=(-1, -2))[
            ..., self.inds_i, self.inds_j
        ]  # (B, M)

        I_bf -= I_bf.mean((-1, -2), keepdim=True)

        # vals[b, u] = sum_m K[u, m] * I_bf[b, m]
        vals = I_bf @ self.K.T  # (B, U)

        r_off = r_batch[:, 0] * Nx + r_batch[:, 1]
        idx = (r_off[:, None] + self.unique_offsets[None, :]) % (Ny * Nx)

        psi_flipped.view(-1).scatter_add_(
            0,
            idx.reshape(-1),
            vals.reshape(-1).to(psi_flipped.dtype),
        )

        return self

    def reconstruct(self, psi, psi_flipped, I_batch, r_batch):
        """
        psi         : (Ny, Nx) prlx accumulator
        psi_flipped : (Ny, Nx) phase-flipped accumulator
        I_batch     : (B, ny, nx)
        r_batch     : (B, 2)
        """
        self.reconstruct_parallax(psi, I_batch, r_batch)
        if self.enable_phase_flipping:
            self.reconstruct_phase_flipping(psi_flipped, I_batch, r_batch)
        return self

    def reconstruct_parallax_sparse(
        self,
        psi,
        streamed_hits,  # list[list[int]] or list[Tensor]
        r_batch,  # (B, 2)
    ):
        if not self._preprocessed:
            raise RuntimeError("Call preprocess() first.")

        device = self.device
        Ny, Nx = self.upsampled_gpts
        ny, nx = self.bf_mask.shape
        M = self.s_m_up.shape[0]

        # ---- concatenate hits ----
        lengths = torch.tensor(
            [len(h) for h in streamed_hits],
            device=device,
            dtype=torch.float32,
        )
        if lengths.sum() == 0:
            return self

        hits = torch.cat([torch.as_tensor(h, device=device) for h in streamed_hits])

        frame_idx = torch.repeat_interleave(
            torch.arange(len(streamed_hits), device=device),
            lengths.to(torch.int64),
        )

        # ---- ifftshift detector coordinates ----
        y = hits // nx
        x = hits % nx
        y = (y + ny // 2) % ny
        x = (x + nx // 2) % nx
        hits_shifted = y * nx + x

        # ---- BF lookup ----
        m = self._bf_lut[hits_shifted]
        valid = m >= 0
        if not valid.any():
            return self

        m = m[valid]
        frame_idx = frame_idx[valid]

        r = r_batch[frame_idx]
        s = self.s_m_up[m]

        ys = (r[:, 0] + s[:, 0]) % Ny
        xs = (r[:, 1] + s[:, 1]) % Nx
        idx_pos = ys * Nx + xs

        psi.view(-1).scatter_add_(
            0,
            idx_pos,
            torch.ones_like(idx_pos, dtype=psi.dtype),
        )

        # bg_weight_j = -N_j / M
        bg_weights = -lengths / M  # (B,)

        # Expand over BF pixels
        r_bg = r_batch[:, None, :]  # (B, 1, 2)
        s_bg = self.s_m_up[None, :, :]  # (1, M, 2)

        ys_bg = (r_bg[..., 0] + s_bg[..., 0]) % Ny
        xs_bg = (r_bg[..., 1] + s_bg[..., 1]) % Nx
        idx_bg = (ys_bg * Nx + xs_bg).reshape(-1)

        vals_bg = bg_weights[:, None].expand(-1, M).reshape(-1)

        psi.view(-1).scatter_add_(
            0,
            idx_bg,
            vals_bg.to(psi.dtype),
        )

        return self

    def reconstruct_phase_flipping_sparse(
        self,
        psi_flipped,
        streamed_hits,  # list[list[int]] or list[Tensor]
        r_batch,  # (B, 2)
    ):
        if not self.enable_phase_flipping:
            raise RuntimeError("Phase flipping not enabled.")

        if not self._preprocessed:
            raise RuntimeError("Call preprocess() first.")

        device = self.device
        Ny, Nx = self.upsampled_gpts
        ny, nx = self.bf_mask.shape
        M = self.s_m_up.shape[0]

        # ---- gather hits ----
        lengths = torch.tensor(
            [len(h) for h in streamed_hits],
            device=device,
            dtype=torch.float32,
        )
        if lengths.sum() == 0:
            return self

        hits = torch.cat([torch.as_tensor(h, device=device) for h in streamed_hits])

        frame_idx = torch.repeat_interleave(
            torch.arange(len(streamed_hits), device=device),
            lengths.to(torch.int64),
        )

        # ---- ifftshift detector coords ----
        y = hits // nx
        x = hits % nx
        y = (y + ny // 2) % ny
        x = (x + nx // 2) % nx
        hits_shifted = y * nx + x

        # ---- BF pixel lookup ----
        m = self._bf_lut[hits_shifted]
        valid = m >= 0
        if not valid.any():
            return self

        m = m[valid]
        frame_idx = frame_idx[valid]

        # Build sparse BF intensity matrix: I_pos[b, m] = hit count
        I_pos = torch.zeros(
            (len(streamed_hits), M),
            device=device,
            dtype=self.K.dtype,
        )
        I_pos.index_put_(
            (frame_idx, m),
            torch.ones_like(m, dtype=I_pos.dtype),
            accumulate=True,
        )

        # vals_pos[b, u] = sum_m K[u, m] * I_pos[b, m]
        vals_pos = I_pos @ self.K.T  # (B, U)

        r_off = r_batch[:, 0] * Nx + r_batch[:, 1]
        idx_pos = (r_off[:, None] + self.unique_offsets[None, :]) % (Ny * Nx)

        psi_flipped.view(-1).scatter_add_(
            0,
            idx_pos.reshape(-1),
            vals_pos.reshape(-1).to(psi_flipped.dtype),
        )

        # bg_weight_j = -N_j / M
        bg_weights = -lengths / M  # (B,)

        # Uniform BF vector per frame
        I_bg = bg_weights[:, None].expand(-1, M)

        vals_bg = I_bg @ self.K.T  # (B, U)

        idx_bg = idx_pos  # same indices

        psi_flipped.view(-1).scatter_add_(
            0,
            idx_bg.reshape(-1),
            vals_bg.reshape(-1).to(psi_flipped.dtype),
        )

        return self

    def reconstruct_sparse(self, psi, psi_flipped, hits_batch, r_batch):
        """
        psi         : (Ny, Nx) prlx accumulator
        psi_flipped : (Ny, Nx) phase-flipped accumulator
        hits_batch  : list[list[int]] or list[Tensor]
        r_batch     : (B, 2)
        """
        self.reconstruct_parallax_sparse(psi, hits_batch, r_batch)
        if self.enable_phase_flipping:
            self.reconstruct_phase_flipping_sparse(psi_flipped, hits_batch, r_batch)
        return self
