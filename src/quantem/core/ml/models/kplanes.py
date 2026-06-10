"""
Tensor Decomposition Methods for INR-based reconstructions
"""

import itertools
from typing import Callable, Optional, Sequence

# import tinycudann as tcnn
import torch
import torch.nn.functional as F
from torch import nn

from .model_base import PPLR, TensorDecompositionModel
from .so3params import SO3ParamQuat, SO3ParamR9SVD

"""
K-planes utility functions
"""


def grid_sample_wrapper(
    grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True
) -> torch.Tensor:
    """
    Performs bilinear interpolation on a grid at given coordinates.

    Args:
        grid: Grid tensor of shape [B, C, H, W] or [C, H, W]
        coords: Coordinate tensor of shape [B, N, 2] or [N, 2]
        align_corners: Whether to align corners

    Returns:
        Interpolated values of shape [B, N, C] or [N, C]
    """
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(
            f"Grid-sample was called with {grid_dim}D data but is only "
            f"implemented for 2 and 3D data."
        )

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode="bilinear",
        padding_mode="border",
    )
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


def init_planes(
    in_dim: int,
    out_dim: int,
    resolution: Sequence[int],
    init_range: tuple = (0.1, 0.5),
) -> nn.ParameterList:
    """Create the set of 2D planes for a k-plane decomposition.

    For in_dim=3 (spatial), this creates 3 planes: XY, XZ, YZ.
    For in_dim=4 (spatial + time), this creates 6 planes: XY, XZ, XT, YZ, YT, ZT.
    Time planes (those involving axis 3) are initialized to 1 so they start
    as identity multipliers.

    Args:
        in_dim: Dimensionality of the input coordinates (3 or 4).
        out_dim: Number of feature channels per plane.
        resolution: Resolution along each axis, e.g. [128, 128, 128].
        init_range: (a, b) for uniform initialization of spatial planes.

    Returns:
        nn.ParameterList of plane parameters, each of shape [1, out_dim, res_j, res_i].
    """
    assert len(resolution) == in_dim
    # All pairs of axes
    axis_pairs = list(itertools.combinations(range(in_dim), 2))
    planes = nn.ParameterList()
    a, b = init_range
    for pair in axis_pairs:
        # grid_sample expects (N, C, H, W) — so resolution is reversed
        shape = [1, out_dim] + [resolution[ax] for ax in reversed(pair)]
        param = nn.Parameter(torch.empty(*shape))
        # Time planes init to 1; spatial planes init uniform
        if in_dim == 4 and 3 in pair:
            nn.init.ones_(param)
        else:
            nn.init.uniform_(param, a=a, b=b)
        planes.append(param)
    return planes


def query_planes(
    pts: torch.Tensor,
    planes: nn.ParameterList,
    in_dim: int,
) -> float:
    """Query the k-plane representation at a batch of points.

    Projects each point onto every axis-pair plane, bilinearly interpolates,
    and returns the element-wise product across all planes.

    Args:
        pts: (B, in_dim) coordinates in [-1, 1].
        planes: The ParameterList from init_planes.
        in_dim: 3 or 4.

    Returns:
        (B, out_dim) features.
    """
    axis_pairs = list(itertools.combinations(range(in_dim), 2))
    result = 1.0
    for plane_param, pair in zip(planes, axis_pairs):
        # Extract the 2D coords for this plane
        coords_2d = pts[..., list(pair)]  # (B, 2)
        coords_2d = coords_2d.view(1, -1, 1, 2)  # (1, B, 1, 2) for grid_sample
        # grid_sample: input (N,C,H,W), grid (N, H_out, W_out, 2)
        sampled = F.grid_sample(
            plane_param,  # (1, C, H, W)
            coords_2d,  # (1, B, 1, 2)
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )  # -> (1, C, B, 1)
        sampled = sampled.squeeze(0).squeeze(-1).T  # (B, C)
        result = result * sampled
    return result  # pyright: ignore[reportReturnType]


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: nn.ParameterList,
) -> torch.Tensor:
    mat_mode = [[0, 1], [0, 2], [1, 2]]
    coord_plane = torch.stack(
        [
            pts[:, mat_mode[0]],
            pts[:, mat_mode[1]],
            pts[:, mat_mode[2]],
        ]
    ).view(3, -1, 1, 2)

    per_scale = []
    for plane_coef in ms_grids:
        C = plane_coef.shape[1]
        feats = F.grid_sample(
            plane_coef, coord_plane, align_corners=True, mode="bilinear", padding_mode="border"
        ).reshape(3, C, -1)
        fused = feats[0] * feats[1] * feats[2]
        per_scale.append(fused.T)

    return torch.cat(per_scale, dim=-1)


class KPlanes(PPLR, TensorDecompositionModel):
    """
    K-Planes model adapted from Fridovich-Keil et al., https://arxiv.org/abs/2301.10241
    """

    def __init__(
        self,
        # Grid parameters
        grid_dimensions: int = 2,
        input_coords_dims: int = 3,
        M_features: int = 32,
        resolution: Sequence[int] = (200, 200, 200),
        multiscale_res_multipliers: Optional[Sequence[int]] = None,
        concat_features: bool = True,
        density_activation: Callable = lambda x: F.softplus(
            x - 1
        ),  # Keep playing around with this and trunc_exp
        # Hybrid MLP parameters
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
    ):
        """
        Assume coords are [-1, 1] in each dimension.
        """
        super().__init__()
        self._td_type = "kplanes"
        self.grid_dimensions = grid_dimensions
        self.input_coords_dims = input_coords_dims
        self.M_features = M_features
        self.resolution = resolution
        self.multiscale_res_multipliers = multiscale_res_multipliers or [1]
        self.concat_features = concat_features
        self.density_activation = density_activation

        # All three planes share one (3, C, res[1], res[0]) tensor, which ignores
        # res[2]: an anisotropic resolution would silently give the XZ/YZ planes the
        # wrong grid along z. Refuse it rather than misallocate.
        if len(set(self.resolution)) != 1:
            raise ValueError(
                f"KPlanes currently requires an isotropic resolution, got {self.resolution}; "
                "the plane grids are allocated as (res[1], res[0]) for all three axis pairs."
            )

        self.grids = nn.ParameterList()
        self.feature_dim = 0
        for res_mult in self.multiscale_res_multipliers:
            scaled_res = [int(r * res_mult) for r in self.resolution]
            plane = nn.Parameter(torch.empty(3, self.M_features, scaled_res[1], scaled_res[0]))
            nn.init.uniform_(plane, 0.1, 0.5)
            self.grids.append(plane)
            self.feature_dim += self.M_features

        # Network head
        if use_hybrid_mlp:
            hybrid_hidden_dim = int(hybrid_hidden_dim)
            hybrid_num_layers = int(hybrid_num_layers)
            if hybrid_hidden_dim <= 0:
                raise ValueError(f"hybrid_hidden_dim must be >= 1, got {hybrid_hidden_dim}")
            if hybrid_num_layers <= 0:
                raise ValueError(f"hybrid_num_layers must be >= 1, got {hybrid_num_layers}")

            factory = {}  # add dtype/device kwargs here if needed
            layers = []
            in_dim = self.feature_dim
            for _ in range(hybrid_num_layers):
                lin = nn.Linear(in_dim, hybrid_hidden_dim, **factory)
                nn.init.kaiming_uniform_(lin.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.append(lin)
                layers.append(nn.ReLU(inplace=True))
                in_dim = hybrid_hidden_dim

            out = nn.Linear(in_dim, 1, bias=True, **factory)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            layers.append(out)
            self.sigma_net = nn.Sequential(*layers)

    def get_densities(self, coords: torch.Tensor):
        """Computes and returns densities"""
        pts = coords.reshape(-1, 3)
        features = interpolate_ms_features(
            pts=pts,
            ms_grids=self.grids,
        )
        density_before_activation = self.sigma_net(features)
        density = self.density_activation(density_before_activation)
        return density

    def forward(self, pts: torch.Tensor):
        return self.get_densities(pts)

    def get_params(self) -> dict[str, list[torch.nn.Parameter]]:
        return {
            "grids": list(self.grids.parameters()),
            "sigma_net": list(self.sigma_net.parameters()),
        }

    @property
    def param_keys(self) -> list[str]:
        return ["grids", "sigma_net"]

    @property
    def td_type(self) -> str:
        return self._td_type

    @td_type.setter
    def td_type(self, td_type: str):
        if not isinstance(td_type, str):
            raise TypeError("td_type must be a string")
        self._td_type = td_type

    @property
    def tilted(self) -> bool:
        return False

    @tilted.setter
    def tilted(self, tilted: bool):
        if not isinstance(tilted, bool):
            raise TypeError("tilted must be a boolean")
        self._tilted = tilted

    @property
    def grids(self) -> torch.nn.ParameterList:
        return self._grids

    @grids.setter
    def grids(self, grids: torch.nn.ParameterList):
        if not isinstance(grids, torch.nn.ParameterList):
            raise TypeError("Grids must be a ParameterList")
        self._grids = grids

    @property
    def resolution(self) -> list[int]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: Sequence[int]):
        if not isinstance(resolution, Sequence):
            raise TypeError("Resolution must be a sequence")
        self._resolution = list(resolution)


# ---------------------------------------------------------------------------
# KPlanesTILTED
# ---------------------------------------------------------------------------


def interpolate_ms_features_tilted(
    pts: torch.Tensor,  # (B, 3)
    ms_grids: nn.ParameterList,  # each grid: (3*T, C, H, W)
    rotation_matrices: torch.Tensor,  # (T, 3, 3)
) -> torch.Tensor:
    """
    Fully-vectorized multi-scale, multi-rotation K-Planes feature interpolation.
    Returns features of shape (B, C * T * num_scales).
    """
    T = rotation_matrices.shape[0]
    B = pts.shape[0]

    # (T, B, 3)  — rotate all points by all rotations at once
    rotated = torch.einsum("tij,bj->tbi", rotation_matrices, pts)

    # Build (T, 3, B, 2) coords for planes XY, ZX, YZ in one shot.
    # index_select is faster and cleaner than advanced indexing with python lists.
    # Plane axis layout: XY=(0,1), ZX=(2,0), YZ=(1,2)
    idx = torch.tensor([[0, 1], [2, 0], [1, 2]], device=pts.device)  # (3, 2)
    # rotated: (T, B, 3) -> gather along last dim with idx (3, 2)
    # Result: (T, 3, B, 2)
    coords = (
        rotated.unsqueeze(1).expand(T, 3, B, 3).gather(-1, idx.view(1, 3, 1, 2).expand(T, 3, B, 2))
    )

    # Flatten (T, 3) -> 3*T so it matches grid's first dim, and add the H_out=1 axis
    coord_tensor = coords.reshape(3 * T, B, 1, 2)  # (3T, B, 1, 2)

    per_scale_features = []
    for plane_coef in ms_grids:
        # plane_coef: (3T, C, H, W)
        C = plane_coef.shape[1]

        sampled = F.grid_sample(
            plane_coef,
            coord_tensor,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )  # (3T, C, B, 1)

        # (3T, C, B) -> (T, 3, C, B) -> Hadamard across the "3" dim -> (T, C, B)
        sampled = sampled.squeeze(-1).view(T, 3, C, B).prod(dim=1)

        # (T, C, B) -> (B, T, C) -> (B, T*C) to concatenate rotations along feature dim
        per_scale_features.append(sampled.permute(2, 0, 1).reshape(B, T * C))

    # Concatenate across scales -> (B, T * C * num_scales)
    return torch.cat(per_scale_features, dim=-1)


class KPlanesTILTED(KPlanes):
    """
    K-Planes with T learned SO(3) rotations (TILTED). Adapted from Yi et al., https://arxiv.org/abs/2308.15461

    Inherits KPlanes for the sigma_net, density_activation, and get_params
    interface.  Overrides:
      * __init__       – replaces the axis-aligned grids with (3*T)-plane
                         grids and adds SO3Param.
      * get_densities  – calls the TILTED interpolation instead.
      * get_params     – adds "so3" key so callers can set a separate lr.
      * param_keys     – updated list.

    Two-phase optimization
    ----------------------
    Phase 1: instantiate with small `M_features` (and optionally smaller
             `resolution` / fewer scales) — the bottleneck model. Train it
             until τ converges, then call `extract_tau_state()`.
    Phase 2: instantiate at full capacity, call `load_tau_state(M_bneck)`
             to seed the rotations, then train normally.

    Parameters
    ----------
    M_features : int
        Feature channels *per transform per scale*.  Total feature_dim will
        be M_features * T * len(multiscale_res_multipliers).
    T : int
        Number of learned rotations (TILTED-T in the paper; 4 or 8 recommended).
        Must match between phase 1 and phase 2 when doing two-phase transfer.
    tau_init : str
        "random" (paper default) or "identity".
        Irrelevant if you're calling load_tau_state() right after __init__.
    All other args are forwarded to KPlanes.
    """

    def __init__(
        self,
        # Grid parameters
        input_coords_dims: int = 3,
        M_features: int = 32,
        resolution: Sequence[int] = (200, 200, 200),
        multiscale_res_multipliers: Optional[Sequence[float]] = None,
        density_activation: Callable = lambda x: F.softplus(x - 1),
        # TILTED parameters
        T: int = 4,
        tau_init: str = "random",
        # Hybrid MLP parameters
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
        so3_param_type: str = "r9svd",
    ):
        self._td_type = "tilted"
        if input_coords_dims != 3:
            raise NotImplementedError("KPlanesTILTED is implemented for 3D only.")
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")

        multiscale_res_multipliers = list(multiscale_res_multipliers or [1])
        num_scales = len(multiscale_res_multipliers)

        # Total feature dim seen by the MLP head.
        # Each scale contributes M_features * T channels.
        feature_dim = M_features * T * num_scales

        # Call KPlanes.__init__ with grid_dimensions=2 so it builds sigma_net
        # correctly; we immediately replace self.grids below.
        super().__init__(
            grid_dimensions=2,
            input_coords_dims=3,
            M_features=M_features,
            resolution=resolution,
            multiscale_res_multipliers=multiscale_res_multipliers,
            concat_features=True,
            density_activation=density_activation,
            use_hybrid_mlp=use_hybrid_mlp,
            hybrid_hidden_dim=hybrid_hidden_dim,
            hybrid_num_layers=hybrid_num_layers,
        )

        self.T = T

        # ---- Rebuild grids: (3*T, M_features, H, W) per scale ----
        self.grids = nn.ParameterList()
        for res_mult in multiscale_res_multipliers:
            scaled_res = [int(r * res_mult) for r in resolution]
            plane = nn.Parameter(torch.empty(3 * T, M_features, scaled_res[1], scaled_res[0]))
            nn.init.uniform_(plane, 0.1, 0.5)
            self.grids.append(plane)

        # ---- Rebuild sigma_net with the correct feature_dim ----
        # KPlanes built sigma_net with self.feature_dim (= M * num_scales),
        # which is wrong for T > 1.  Rebuild here.
        self.feature_dim = feature_dim
        self._build_sigma_net(use_hybrid_mlp, hybrid_hidden_dim, hybrid_num_layers)

        # ---- Learnable rotations ----
        self.set_so3_param_type(so3_param_type, init=tau_init)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sigma_net(
        self,
        use_hybrid_mlp: bool,
        hybrid_hidden_dim: int,
        hybrid_num_layers: int,
    ) -> None:
        """Rebuild sigma_net for self.feature_dim (called after grids are set)."""
        if use_hybrid_mlp:
            layers = []
            in_dim = self.feature_dim
            for _ in range(hybrid_num_layers):
                lin = nn.Linear(in_dim, hybrid_hidden_dim)
                nn.init.kaiming_uniform_(lin.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.append(lin)
                layers.append(nn.ReLU(inplace=True))
                in_dim = hybrid_hidden_dim
            out = nn.Linear(in_dim, 1, bias=True)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            layers.append(out)
            self.sigma_net = nn.Sequential(*layers)
        else:
            # Single-linear "explicit" decoder. Small init -> density ~ 0 initially.
            self.sigma_net = nn.Linear(self.feature_dim, 1, bias=True)
            nn.init.normal_(self.sigma_net.weight, std=0.01)
            nn.init.zeros_(self.sigma_net.bias)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def get_densities(self, coords: torch.Tensor) -> torch.Tensor:
        pts = coords.reshape(-1, 3)
        R = self.so3.as_matrix()  # (T, 3, 3)
        features = interpolate_ms_features_tilted(
            pts=pts,
            ms_grids=self.grids,
            rotation_matrices=R,
        )
        density_before_activation = self.sigma_net(features)
        return self.density_activation(density_before_activation)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return self.get_densities(pts)

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, list[nn.Parameter]]:
        return {
            "grids": list(self.grids.parameters()),
            "sigma_net": list(self.sigma_net.parameters()),
            "so3": list(self.so3.parameters()),
        }

    @property
    def param_keys(self) -> list[str]:
        return ["grids", "sigma_net", "so3"]

    # ------------------------------------------------------------------
    # Two-phase transfer: extract / load learned rotations
    # ------------------------------------------------------------------

    def extract_tau_state(self) -> torch.Tensor:
        """
        Returns the current raw R^9 matrices (detached copy) so they can be
        used to initialise a phase-2 model via `load_tau_state`.

        Returns
        -------
        torch.Tensor of shape (T, 3, 3)
        """
        return self.so3.M.detach().cpu().clone()

    def load_tau_state(self, M: torch.Tensor) -> None:
        """
        Load pre-trained rotation matrices (e.g. from a bottleneck phase-1 model).

        No orthogonalization is needed — SO3Param.as_matrix() projects to SO(3)
        via SVD on every forward pass.

        Parameters
        ----------
        M : torch.Tensor of shape (T, 3, 3)
            Raw unconstrained matrices from `extract_tau_state()`.
        """
        if M.shape != self.so3.M.shape:
            raise ValueError(
                f"Shape mismatch: got {M.shape}, expected {self.so3.M.shape}. "
                f"Make sure T matches between phase 1 and phase 2."
            )
        with torch.no_grad():
            self.so3.M.copy_(M.to(self.so3.M.device))

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, "
            f"M_features={self.M_features}, "
            f"feature_dim={self.feature_dim}, "
            f"num_scales={len(self.multiscale_res_multipliers)}"
        )

    def set_so3_param_type(self, so3_param_type: str, init: str = "rand") -> None:
        """
        Set the SO3 parameterization type.

        Parameters
        ----------
        so3_param_type : str
            SO3 parameterization type ("quat" or "r9svd").
        """
        if so3_param_type == "r9svd":
            self.so3 = SO3ParamR9SVD(self.T, init=init)
        elif so3_param_type == "quat":
            self.so3 = SO3ParamQuat(self.T, init=init)
        else:
            raise ValueError(f"Invalid SO3 parameterization type: {so3_param_type}")

    @property
    def tilted(self) -> bool:
        return True


# CP Decomp for Warmup SO3 rotations


def interpolate_ms_features_cp_tilted(
    pts: torch.Tensor,  # (B, 3)
    ms_grids: nn.ParameterList,  # each grid: (3*T, C, L) — 1D lines
    rotation_matrices: torch.Tensor,  # (T, 3, 3)
) -> torch.Tensor:
    """
    CP (vector outer product) version of TILTED interpolation.
    Returns features of shape (B, C * T * num_scales).
    """
    T = rotation_matrices.shape[0]
    B = pts.shape[0]

    # Rotate all points by all rotations: (T, B, 3)
    rotated = torch.einsum("tij,bj->tbi", rotation_matrices, pts)

    per_scale_features = []
    for line_coef in ms_grids:
        # line_coef: (3T, C, L)  — three 1D feature lines per transform (x, y, z)
        C, _ = line_coef.shape[1], line_coef.shape[2]

        # For each transform t, we need three 1D samples: at x_t, y_t, z_t.
        # Lay them out as (3T, B) coords, matching line_coef's first dim.
        # Axis order per transform: x, y, z.
        coords_1d = rotated.reshape(T, B, 3).permute(0, 2, 1).reshape(3 * T, B)
        # coords_1d: (3T, B), each row is samples along one axis for one transform

        # grid_sample wants 4D input for 2D sampling, or we can use 1D via a
        # (3T, C, 1, L) reshape and pass 2D coords with y fixed at 0.
        # Simpler: use F.grid_sample with a 4D trick, or just do manual linear interp.
        # Here's the grid_sample way:
        line_coef_4d = line_coef.unsqueeze(2)  # (3T, C, 1, L)
        # grid: need (3T, Hout=1, Wout=B, 2), with x = coord, y = 0
        grid = torch.stack(
            [
                coords_1d,  # x
                torch.zeros_like(coords_1d),  # y
            ],
            dim=-1,
        ).unsqueeze(1)  # (3T, 1, B, 2)

        sampled = F.grid_sample(
            line_coef_4d,
            grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        ).squeeze(2)  # (3T, C, B)

        # Hadamard across the 3 axes per transform: (T, 3, C, B) -> (T, C, B)
        sampled = sampled.view(T, 3, C, B).prod(dim=1)

        # (T, C, B) -> (B, T*C)
        per_scale_features.append(sampled.permute(2, 0, 1).reshape(B, T * C))

    return torch.cat(per_scale_features, dim=-1)


class CPTilted(PPLR, TensorDecompositionModel):
    """
    CP decomposition with TILTED rotations — the true bottleneck model for
    phase 1. Rank-1-per-channel feature representation. Adapted from Yi et al., https://arxiv.org/abs/2308.15461

    Shares the SO3Param and sigma_net design with KPlanesTILTED so you can
    lift τ directly across: cp_model.extract_tau_state() ->
    kplanes_model.load_tau_state().
    """

    def __init__(
        self,
        C: int = 4,  # channels per transform per scale
        resolution: Sequence[int] = (128, 128, 128),
        multiscale_res_multipliers: Optional[Sequence[int]] = None,
        T: int = 4,
        tau_init: str = "random",
        density_activation: Callable = lambda x: F.softplus(x - 1),
        so3_param_type: str = "r9svd",
    ):
        super().__init__()
        self._td_type = "cp_tilted"
        self.T = T
        self.C = C
        self.multiscale_res_multipliers = list(multiscale_res_multipliers or [1])
        self.density_activation = density_activation

        # 1D feature lines, one per axis per transform per scale.
        # Shape per scale: (3*T, C, L).  We use max(resolution) for L; if your
        # scene is strongly anisotropic use 3 separate lines per axis.
        self.grids = nn.ParameterList()
        for mult in self.multiscale_res_multipliers:
            L = int(max(resolution) * mult)
            line = nn.Parameter(torch.empty(3 * T, C, L))
            nn.init.uniform_(line, 0.1, 0.5)
            self.grids.append(line)

        self.feature_dim = C * T * len(self.multiscale_res_multipliers)

        # Same minimal single-linear decoder as your KPlanesTILTED default.
        self.sigma_net = nn.Linear(self.feature_dim, 1, bias=True)
        nn.init.normal_(self.sigma_net.weight, std=0.01)
        nn.init.zeros_(self.sigma_net.bias)

        if so3_param_type == "r9svd":
            self.so3 = SO3ParamR9SVD(T, init=tau_init)
        elif so3_param_type == "quat":
            self.so3 = SO3ParamQuat(T, init=tau_init)
        else:
            raise ValueError(f"Unknown SO3 param type: {so3_param_type}")

    def get_densities(self, coords: torch.Tensor) -> torch.Tensor:
        pts = coords.reshape(-1, 3)
        R = self.so3.as_matrix()
        features = interpolate_ms_features_cp_tilted(pts, self.grids, R)
        return self.density_activation(self.sigma_net(features))

    def forward(self, pts):
        return self.get_densities(pts)

    def get_params(self):
        return {
            "grids": list(self.grids.parameters()),
            "sigma_net": list(self.sigma_net.parameters()),
            "so3": list(self.so3.parameters()),
        }

    @property
    def param_keys(self):
        return ["grids", "sigma_net", "so3"]

    @property
    def td_type(self) -> str:
        return self._td_type

    def extract_tau_state(self) -> torch.Tensor:
        return self.so3.M.detach().clone()

    @property
    def tilted(self) -> bool:
        return True


KPlanesType = KPlanes | KPlanesTILTED | CPTilted
