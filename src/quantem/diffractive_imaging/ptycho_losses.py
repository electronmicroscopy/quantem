"""Data-fidelity criteria for iterative ptychography.

A *criterion* decouples the two concerns that used to be tangled in ``error_estimate``:

1. the **measurement space** it compares in (``target_space``: ``"amplitude"`` or
   ``"intensity"``) — this is what ``PtychographyDatasetBase._set_targets`` builds targets in
   and what predictions are mapped to; and
2. the **comparison** itself (``__call__``) — L2, L1, smooth-L1, Poisson, S3IM, ....

``error_estimate`` masks the predictions/targets, calls the criterion, and divides the result by
the mean diffraction intensity. To add a new loss, write a ``DataCriterion`` subclass and register
it in ``_REGISTRY`` (or pass an instance straight to ``reconstruct(loss_type=...)``).
"""

from typing import Callable, Literal

import torch
import torch.nn.functional as F

TargetSpace = Literal["amplitude", "intensity"]


def _global_scale(preds: torch.Tensor, n: int) -> float:
    """Batch -> full-scan normalization: a batch sum is rescaled to a full-dataset-equivalent
    sum so the loss magnitude is independent of batch size (``B / n``)."""
    return preds.shape[0] / n


class DataCriterion:
    """Base class for ptychography data-fidelity criteria.

    Subclasses set ``target_space`` and implement ``__call__(preds, targets, n)``, returning a
    scalar error (before the mean-intensity normalization applied by ``error_estimate``). ``preds``
    and ``targets`` are already in ``target_space`` and detector-masked; ``n`` is the global scan
    count (for batch-size-independent scaling).
    """

    target_space: TargetSpace = "amplitude"

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError


class L2(DataCriterion):
    """Sum of squared residuals in ``target_space`` (amplitude or intensity)."""

    def __init__(self, target_space: TargetSpace = "amplitude"):
        self.target_space = target_space

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
        return torch.sum((preds - targets) ** 2) / _global_scale(preds, n)


class L1(DataCriterion):
    """Sum of absolute residuals in ``target_space`` (amplitude or intensity)."""

    def __init__(self, target_space: TargetSpace = "amplitude"):
        self.target_space = target_space

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
        return torch.sum((preds - targets).abs()) / _global_scale(preds, n)


class Poisson(DataCriterion):
    """Poisson negative log-likelihood in intensity space (up to a pred-independent constant)."""

    target_space: TargetSpace = "intensity"

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
        nll = torch.sum(preds - targets * torch.log(preds + 1e-6))
        return nll / _global_scale(preds, n)


class AmplitudeSmoothL1(DataCriterion):
    """Smooth-L1 (Huber) on the amplitude residual: ~L2 for ``|r| < beta``, ~L1 beyond."""

    target_space: TargetSpace = "amplitude"

    def __init__(self, beta: float = 1.0):
        self.beta = float(beta)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
        return F.smooth_l1_loss(preds, targets, beta=self.beta, reduction="sum") / _global_scale(
            preds, n
        )


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d[None, None]  # (1, 1, size, size)


def _ssim(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Mean SSIM between two single-channel images ``x``, ``y`` of shape ``(1, 1, H, W)``."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.conv2d(x, window)
    mu_y = F.conv2d(y, window)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x = F.conv2d(x * x, window) - mu_x2
    sigma_y = F.conv2d(y * y, window) - mu_y2
    sigma_xy = F.conv2d(x * y, window) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


class AmplitudeS3IM(DataCriterion):
    """Stochastic Structural SIMilarity loss (Xie et al. 2023), as ``MSE + lambda * (1 - S3IM)``.

    S3IM applies SSIM to *non-local* groups of pixels: the flattened predictions/targets are
    randomly permuted ``repeats`` times, tiled into a 2D patch, and compared with windowed SSIM;
    this captures structural relationships a per-pixel loss misses. It is used as an auxiliary
    term on top of an MSE term (both mean-reduced here, so ``lambda`` ~ O(1) balances them). The
    SSIM passes make this notably more expensive than L2 — keep ``repeats`` modest.

    Note: both terms are **mean**-reduced. A mean is already batch-size independent (no
    ``_global_scale`` rescale needed), so this criterion is well-behaved across batch sizes and
    multi-GPU. It does, however, sit on a different absolute scale than the sum-based criteria
    (``L2``/``L1``/Poisson, which rescale to a full-dataset sum), so learning rates do **not**
    transfer between ``s3im_amplitude`` and those losses — retune the LR when switching.

    Pass ``generator`` (a ``torch.Generator`` on the compute device) to make the random
    permutations — and hence the loss values — reproducible run-to-run.
    """

    target_space: TargetSpace = "amplitude"

    def __init__(
        self,
        lambda_s3im: float = 1.0,
        repeats: int = 5,
        patch_height: int = 32,
        window_size: int = 11,
        sigma: float = 1.5,
        generator: torch.Generator | None = None,
    ):
        self.lambda_s3im = float(lambda_s3im)
        self.repeats = int(repeats)
        self.patch_height = int(patch_height)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.generator = generator

    def _s3im(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        num = src.numel()
        idx_list = [torch.arange(num, device=src.device)]
        for _ in range(self.repeats - 1):
            idx_list.append(torch.randperm(num, device=src.device, generator=self.generator))
        idx = torch.cat(idx_list)
        ph = self.patch_height
        usable = (idx.numel() // ph) * ph  # trim so it reshapes to (ph, -1)
        idx = idx[:usable]
        src_img = src[idx].reshape(1, 1, ph, -1)
        tar_img = tar[idx].reshape(1, 1, ph, -1)
        window = _gaussian_window(self.window_size, self.sigma, src.device, src.dtype)
        return 1.0 - _ssim(src_img, tar_img, window)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
        mse = torch.mean((preds - targets) ** 2)
        s3im = self._s3im(preds.reshape(-1), targets.reshape(-1))
        return mse + self.lambda_s3im * s3im


_REGISTRY: dict[str, Callable[[], DataCriterion]] = {
    "l2_amplitude": lambda: L2("amplitude"),
    "l1_amplitude": lambda: L1("amplitude"),
    "l2_intensity": lambda: L2("intensity"),
    "l1_intensity": lambda: L1("intensity"),
    "poisson": Poisson,
    "smooth_l1_amplitude": AmplitudeSmoothL1,
    "s3im_amplitude": AmplitudeS3IM,
}


def get_data_criterion(loss_type: "str | DataCriterion") -> DataCriterion:
    """Resolve a ``loss_type`` to a :class:`DataCriterion`.

    Accepts a registered name (e.g. ``"l2_amplitude"``, ``"smooth_l1_amplitude"``,
    ``"s3im_amplitude"``) for the default-configured criterion, or a ``DataCriterion`` instance
    (use this to tune parameters, e.g. ``AmplitudeS3IM(lambda_s3im=0.5, repeats=10)``).
    """
    if isinstance(loss_type, DataCriterion):
        return loss_type
    if isinstance(loss_type, str):
        key = loss_type.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown loss_type {loss_type!r}; expected one of {sorted(_REGISTRY)} "
                "or a DataCriterion instance."
            )
        return _REGISTRY[key]()
    raise TypeError(f"loss_type must be a str or DataCriterion, got {type(loss_type)}")
