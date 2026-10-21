import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Tilted KPlanes ---

# ---------------------------------------------------------------------------
# SO(3) quaternion parameter module
# ---------------------------------------------------------------------------


class SO3ParamQuat(nn.Module):
    """
    Stores T unit quaternions as learnable parameters in R^4 and normalises
    them on every call to `as_matrix()`.

    Quaternion convention: [x, y, z, w]  (scalar-last, same as scipy).

    Initialisation
    --------------
    "random"  – uniform sampling over SO(3) via Shoemake's method.
    "identity" – all rotations start as the identity (good for fine-tuning).
    """

    def __init__(self, T: int, init: str = "random"):
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        quats = self._init_quaternions(T, init)  # (T, 4)
        self.quats = nn.Parameter(quats)

    @staticmethod
    def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Unit quaternion (..., 4) [x, y, z, w] -> rotation matrix (..., 3, 3).
        Assumes q is already normalized."""
        x, y, z, w = q.unbind(dim=-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = torch.stack(
            [
                1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
                2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
                2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
            ],
            dim=-1,
        ).reshape(*q.shape[:-1], 3, 3)
        return R

    @staticmethod
    def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
        """Rotation matrix (..., 3, 3) -> unit quaternion (..., 4) [x, y, z, w].

        Shepperd's method: build the four candidate quaternions, each dividing
        by a different diagonal combination, then per-element pick the branch
        with the largest denominator so we never divide by a near-zero number.
        The naive trace-only formula blows up when trace ~ -1 (180deg rotations).
        """
        m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

        # 4 * (component^2) for w, x, y, z respectively; these sum to 4.
        t = torch.stack(
            [
                1.0 + m00 + m11 + m22,  # 4 w^2
                1.0 + m00 - m11 - m22,  # 4 x^2
                1.0 - m00 + m11 - m22,  # 4 y^2
                1.0 - m00 - m11 + m22,  # 4 z^2
            ],
            dim=-1,
        )  # (..., 4)

        eps = torch.finfo(R.dtype).eps
        S = 2.0 * torch.sqrt(t.clamp_min(eps))  # S[k] = 4 * |component_k|
        S0, S1, S2, S3 = S.unbind(-1)

        # each candidate in [x, y, z, w] order
        cand_w = torch.stack([(m21 - m12) / S0, (m02 - m20) / S0, (m10 - m01) / S0, 0.25 * S0], dim=-1)
        cand_x = torch.stack([0.25 * S1, (m01 + m10) / S1, (m02 + m20) / S1, (m21 - m12) / S1], dim=-1)
        cand_y = torch.stack([(m01 + m10) / S2, 0.25 * S2, (m12 + m21) / S2, (m02 - m20) / S2], dim=-1)
        cand_z = torch.stack([(m02 + m20) / S3, (m12 + m21) / S3, 0.25 * S3, (m10 - m01) / S3], dim=-1)

        cands = torch.stack([cand_w, cand_x, cand_y, cand_z], dim=-2)  # (..., 4, 4)
        idx = t.argmax(dim=-1)  # (...,)
        idx = idx[..., None, None].expand(*idx.shape, 1, 4)  # (..., 1, 4)
        q = cands.gather(-2, idx).squeeze(-2)  # (..., 4)
        return F.normalize(q, p=2, dim=-1)

    def as_matrix(self) -> torch.Tensor:
        return self.quat_to_rotmat(self.normalized())

    @classmethod
    def from_matrix(cls, R: torch.Tensor) -> "SO3ParamQuat":
        """Initialize a bank close to the given rotations R (T, 3, 3)."""
        obj = cls(R.shape[0], init="identity")
        with torch.no_grad():
            obj.quats.copy_(cls.rotmat_to_quat(R))
        return obj

    def extra_repr(self) -> str:
        return f"T={self.quats.shape[0]}"

    # ------------------------------------------------------------------
    # Initialisers
    # ------------------------------------------------------------------

    @staticmethod
    def _shoemake_sample(T: int) -> torch.Tensor:
        """Uniform SO(3) sampling via Shoemake (1992). Returns (T, 4) [x,y,z,w]."""
        u = torch.rand(T, 3)
        sqrt1_u0 = torch.sqrt(1.0 - u[:, 0])
        sqrt_u0 = torch.sqrt(u[:, 0])
        two_pi = 2.0 * math.pi
        x = sqrt1_u0 * torch.sin(two_pi * u[:, 1])
        y = sqrt1_u0 * torch.cos(two_pi * u[:, 1])
        z = sqrt_u0 * torch.sin(two_pi * u[:, 2])
        w = sqrt_u0 * torch.cos(two_pi * u[:, 2])
        return torch.stack([x, y, z, w], dim=-1)  # (T, 4)

    @staticmethod
    def _identity(T: int) -> torch.Tensor:
        """All-identity rotations: [0,0,0,1] * T."""
        q = torch.zeros(T, 4)
        q[:, 3] = 1.0
        return q

    @classmethod
    def _init_quaternions(cls, T: int, init: str) -> torch.Tensor:
        if init == "random":
            return cls._shoemake_sample(T)
        elif init == "identity":
            return cls._identity(T)
        else:
            raise ValueError(f"Unknown init '{init}'; choose 'random' or 'identity'.")

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def normalized(self) -> torch.Tensor:
        """Returns (T, 4) unit quaternions."""
        return F.normalize(self.quats, p=2, dim=-1)


class SO3ParamR9SVD(nn.Module):
    """
    SO(3) rotation bank using R9+SVD parameterization.
    Each rotation is stored as an unconstrained 3x3 matrix M,
    projected to SO(3) via SVD+(M) = U diag(1,1,det(UVt)) Vt.

    Based on Rene Geist et al., 2024: https://arxiv.org/abs/2404.11735v1
    """

    def __init__(self, T: int, init: Literal["random", "identity"] = "random"):
        super().__init__()
        if init == "random":
            M = torch.eye(3).unsqueeze(0).repeat(T, 1, 1) + 0.1 * torch.randn(T, 3, 3)
        elif init == "identity":
            M = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        else:
            raise ValueError(f"Unknown init '{init}'")
        self.M = nn.Parameter(M)

    @staticmethod
    def rotmat_to_r9(R: torch.Tensor) -> torch.Tensor:
        """Rotation matrix (..., 3, 3) -> R9. Identity embedding: any R in SO(3)
        is a fixed point of the SVD projection, so this just stores R directly."""
        return R

    @staticmethod
    def r9_to_rotmat(M: torch.Tensor) -> torch.Tensor:
        """R9 (..., 3, 3) -> nearest SO(3) matrix via SVD+."""
        with torch.autocast(device_type=M.device.type, enabled=False):
            M_work = M if M.dtype in (torch.float32, torch.float64) else M.float()
            U, _, Vh = torch.linalg.svd(M_work)
            d = torch.det(U @ Vh)
            diag = torch.ones(*M.shape[:-2], 3, device=M.device, dtype=M_work.dtype)
            diag[..., 2] = d
            return U @ (diag.unsqueeze(-1) * Vh)

    def as_matrix(self) -> torch.Tensor:
        return self.r9_to_rotmat(self.M)

    @classmethod
    def from_matrix(cls, R: torch.Tensor) -> "SO3ParamR9SVD":
        """Initialize a bank close to given rotations R (T, 3, 3)."""
        obj = cls(R.shape[0], init="identity")
        with torch.no_grad():
            obj.M.copy_(cls.rotmat_to_r9(R))
        return obj
