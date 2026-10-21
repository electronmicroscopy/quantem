from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantem.core.utils.rotations import quat_from_matrix, quat_random_uniform, quat_to_matrix

# ---------------------------------------------------------------------------
# SO(3) quaternion parameter module
# ---------------------------------------------------------------------------


class SO3ParamQuat(nn.Module):
    """
    Stores T unit quaternions as learnable parameters in R^4 and normalizes
    them on every call to `as_matrix()`.

    Quaternions are scalar-first (w, x, y, z), the convention of
    :mod:`quantem.core.utils.rotations`, and `as_matrix()` is the active
    rotation matrix of :func:`~quantem.core.utils.rotations.quat_to_matrix`.

    Models saved before 2026-10 stored the quaternions scalar-last
    (x, y, z, w). They are recognized by the missing `scalar_first` buffer
    and reordered when loaded, from a state_dict or a pickled module, so
    they keep their rotations.

    Initialization
    --------------
    "random"  – uniform sampling over SO(3) via Shoemake's method.
    "identity" – all rotations start as the identity (good for fine-tuning).
    """

    def __init__(self, T: int, init: str = "random"):
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if init == "random":
            quats = quat_random_uniform((T,), dtype=torch.float32)
        elif init == "identity":
            quats = torch.zeros(T, 4)
            quats[:, 0] = 1.0
        else:
            raise ValueError(f"Unknown init '{init}'; choose 'random' or 'identity'.")
        self.quats = nn.Parameter(quats)
        # marks the scalar-first order; absent in models saved scalar-last
        self.register_buffer("scalar_first", torch.tensor(True))

    def as_matrix(self) -> torch.Tensor:
        return quat_to_matrix(self.normalized())

    @classmethod
    def from_matrix(cls, R: torch.Tensor) -> "SO3ParamQuat":
        """Initialize a bank close to the given rotations R (T, 3, 3)."""
        obj = cls(R.shape[0], init="identity")
        obj.load_rotation_state(R)
        return obj

    def rotation_state(self) -> torch.Tensor:
        """Detached (T, 3, 3) rotation matrices, for `load_rotation_state`."""
        return self.as_matrix().detach().clone()

    def load_rotation_state(self, R: torch.Tensor) -> None:
        """Set the rotations from (T, 3, 3) matrices, projected onto SO(3)."""
        U, _, Vh = torch.linalg.svd(R.to(torch.float64))
        d = torch.det(U @ Vh)
        U = torch.cat((U[..., :2], U[..., 2:] * d[..., None, None]), dim=-1)
        with torch.no_grad():
            self.quats.copy_(quat_from_matrix(U @ Vh).to(self.quats))

    def extra_repr(self) -> str:
        return f"T={self.quats.shape[0]}"

    # ------------------------------------------------------------------
    # Loading models saved with scalar-last quaternions
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        flag = prefix + "scalar_first"
        quats = prefix + "quats"
        if flag not in state_dict and quats in state_dict:
            state_dict[quats] = state_dict[quats][..., [3, 0, 1, 2]]
            state_dict[flag] = torch.tensor(True)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __setstate__(self, state):
        super().__setstate__(state)
        if "scalar_first" not in self._buffers:
            with torch.no_grad():
                self.quats.copy_(self.quats[..., [3, 0, 1, 2]].clone())
            self.register_buffer("scalar_first", torch.tensor(True, device=self.quats.device))

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

    def rotation_state(self) -> torch.Tensor:
        """Detached raw (T, 3, 3) matrices M, for `load_rotation_state`."""
        return self.M.detach().clone()

    def load_rotation_state(self, M: torch.Tensor) -> None:
        """Set the raw (T, 3, 3) matrices M; `as_matrix()` projects them onto SO(3)."""
        with torch.no_grad():
            self.M.copy_(M.to(self.M))

    @classmethod
    def from_matrix(cls, R: torch.Tensor) -> "SO3ParamR9SVD":
        """Initialize a bank close to given rotations R (T, 3, 3)."""
        obj = cls(R.shape[0], init="identity")
        obj.load_rotation_state(cls.rotmat_to_r9(R))
        return obj
