"""Piecewise-rigid correction for slow-scan residual drift.

After affine correction, scan deceleration can leave displacement that changes
smoothly with slow-scan position. Masked NCC measures one rigid shift per strip,
then interpolation maps those measurements back to the scanline knots.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

import quantem.imaging.drift.core.warping as warping
import quantem.imaging.drift.plot as drift_plot


@dataclass(frozen=True, slots=True)
class StripPass:
    """Describe one coarse-to-fine strip correction pass.

    Strip passes remove slow-scan-dependent residual displacement left after
    affine correction. Wider searches recover large residuals; later passes
    use more strips and narrower bounds to refine local alignment without
    introducing scanline jitter.

    Parameters
    ----------
    num_strips : int, optional
        Number of slow-scan regions measured independently. At least two are
        required to measure variation along the slow-scan direction. Default
        is 24.
    max_row_shift, max_column_shift : int, optional
        Search bounds in scan pixels. Defaults are 8 and 80.
    correction_start_fraction : float or None, optional
        Fraction of the scan held fixed before residual correction begins.
        ``None`` corrects the full field. Default is ``None``.
    ramp_fraction : float, optional
        Fraction of the scan used to blend into the corrected region.
        Default is 0.08.
    smoothing_sigma : float, optional
        Gaussian smoothing along the slow-scan direction in scanlines.
        Default is 12.
    update_fraction : float, optional
        Fraction of the measured residual applied in this pass. Default is 1.
    """

    num_strips: int = 24
    max_row_shift: int = 8
    max_column_shift: int = 80
    correction_start_fraction: float | None = None
    ramp_fraction: float = 0.08
    smoothing_sigma: float = 12.0
    update_fraction: float = 1.0

    def __post_init__(self):
        if self.num_strips < 2:
            raise ValueError(
                f"num_strips must be at least 2, got {self.num_strips}"
            )


# ---------------------------------------------------------------------------
# pure numerics (no DriftCorrection dependency)
# ---------------------------------------------------------------------------


def free_weight(
    n_rows: int,
    *,
    free_from_frac: float | None = 0.55,
    ramp_frac: float = 0.08,
) -> np.ndarray:
    """Per-scanline weight in ``[0, 1]``: 0 = freeze, 1 = apply residual.

    ``free_from_frac=None`` → all ones (full FOV residual).
    """
    if free_from_frac is None:
        return np.ones(n_rows, dtype=np.float32)
    free0 = int(np.clip(free_from_frac, 0.0, 1.0) * n_rows)
    ramp = max(8, int(ramp_frac * n_rows))
    w = np.zeros(n_rows, dtype=np.float32)
    w[free0:] = 1.0
    r0 = max(0, free0 - ramp)
    if free0 > r0:
        t = np.linspace(0.0, 1.0, free0 - r0, dtype=np.float32)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        w[r0:free0] = t
    return w


def _as_torch2d(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().to(device=device, dtype=dtype)
    else:
        t = torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
    if t.ndim != 2:
        raise ValueError(f"expected 2-D image, got shape {tuple(t.shape)}")
    return t


def _masked_ncc_batch(
    ref_s: torch.Tensor,
    mov_s: torch.Tensor,
    mask_s: torch.Tensor,
) -> torch.Tensor:
    """Masked NCC for a batch of strips. Shapes ``(S, Hs, W)`` → ``(S,)``."""
    m = mask_s.to(dtype=ref_s.dtype)
    n = m.sum(dim=(-2, -1)).clamp_min(1.0)
    ref_mu = (ref_s * m).sum(dim=(-2, -1), keepdim=True) / n.view(-1, 1, 1)
    mov_mu = (mov_s * m).sum(dim=(-2, -1), keepdim=True) / n.view(-1, 1, 1)
    r = (ref_s - ref_mu) * m
    v = (mov_s - mov_mu) * m
    num = (r * v).sum(dim=(-2, -1))
    den = r.norm(dim=(-2, -1)) * v.norm(dim=(-2, -1))
    return num / den.clamp_min(1e-12)


def _ncc_at_shift(
    ref_s: torch.Tensor,
    mov_s: torch.Tensor,
    msk_s: torch.Tensor,
    dr: int,
    dc: int,
) -> torch.Tensor:
    """Masked NCC after rolling all strips by the same ``(dr, dc)``.

    Mask semantics match the historical brute search:
    ``roll(mask, dr, row) * roll(mask, dc, col)`` (not a single 2-D roll).
    """
    mov_rc = torch.roll(torch.roll(mov_s, shifts=dr, dims=-2), shifts=dc, dims=-1)
    msk_rc = torch.roll(msk_s, shifts=dr, dims=-2) * torch.roll(msk_s, shifts=dc, dims=-1)
    n_pix = msk_rc.sum(dim=(-2, -1))
    ncc = _masked_ncc_batch(ref_s, mov_rc, msk_rc)
    return torch.where(n_pix >= 64.0, ncc, torch.full_like(ncc, -1.0e9))


def _roll2d_per_strip(x: torch.Tensor, dr: torch.Tensor, dc: torch.Tensor) -> torch.Tensor:
    """Circular 2-D roll with a different ``(dr, dc)`` per strip. ``x`` is ``(S, H, W)``."""
    S, H, W = x.shape
    # torch.roll(x, +k) → x[(i - k) % n]; same via advanced indexing, one gather kernel.
    rows = (torch.arange(H, device=x.device)[None, :] - dr.to(dtype=torch.long)[:, None]) % H
    cols = (torch.arange(W, device=x.device)[None, :] - dc.to(dtype=torch.long)[:, None]) % W
    s_idx = torch.arange(S, device=x.device)[:, None, None]
    r_idx = rows[:, :, None].expand(S, H, W)
    c_idx = cols[:, None, :].expand(S, H, W)
    return x[s_idx, r_idx, c_idx]


def _ncc_at_per_strip_shifts(
    ref_s: torch.Tensor,
    mov_s: torch.Tensor,
    msk_s: torch.Tensor,
    dr: torch.Tensor,
    dc: torch.Tensor,
) -> torch.Tensor:
    """Masked NCC with a different integer shift per strip (``dr``, ``dc`` shape ``(S,)``)."""
    dr = dr.to(dtype=torch.long, device=mov_s.device)
    dc = dc.to(dtype=torch.long, device=mov_s.device)
    mov_rc = _roll2d_per_strip(mov_s, dr, dc)
    # Historical mask: roll(m, dr, row) * roll(m, dc, col) - product of 1-D rolls.
    msk_r = _roll2d_per_strip(msk_s, dr, torch.zeros_like(dr))
    msk_c = _roll2d_per_strip(msk_s, torch.zeros_like(dc), dc)
    msk_rc = msk_r * msk_c
    n_pix = msk_rc.sum(dim=(-2, -1))
    ncc = _masked_ncc_batch(ref_s, mov_rc, msk_rc)
    return torch.where(n_pix >= 64.0, ncc, torch.full_like(ncc, -1.0e9))


def _search_shifts_brute(
    ref_s: torch.Tensor,
    mov_s: torch.Tensor,
    msk_s: torch.Tensor,
    max_shift_row: int,
    max_shift_col: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exhaustive integer search; strips batched, shifts in Python (small caps only)."""
    S = ref_s.shape[0]
    device, dtype = ref_s.device, ref_s.dtype
    best_ncc = torch.full((S,), -1.0e9, device=device, dtype=dtype)
    best_dr = torch.zeros(S, device=device, dtype=torch.long)
    best_dc = torch.zeros(S, device=device, dtype=torch.long)
    for dr in range(-int(max_shift_row), int(max_shift_row) + 1):
        for dc in range(-int(max_shift_col), int(max_shift_col) + 1):
            ncc = _ncc_at_shift(ref_s, mov_s, msk_s, dr, dc)
            improved = ncc > best_ncc
            best_ncc = torch.where(improved, ncc, best_ncc)
            best_dr = torch.where(improved, torch.full_like(best_dr, dr), best_dr)
            best_dc = torch.where(improved, torch.full_like(best_dc, dc), best_dc)
    return best_dr, best_dc, best_ncc


def _search_shifts_fft(
    ref_s: torch.Tensor,
    mov_s: torch.Tensor,
    msk_s: torch.Tensor,
    max_shift_row: int,
    max_shift_col: int,
    *,
    refine_radius: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU FFT phase-corr candidate + local true masked-NCC refine.

    Cost is ~one ``rfft2`` pair over all strips plus ``(2R+1)²`` gather-NCC
    evals - not ``(2·max_row+1)·(2·max_col+1)`` full-image rolls.
    ``refine_radius=5`` matches exhaustive brute on EDS residual tests.
    """
    S, hs, W = ref_s.shape
    device, dtype = ref_s.device, ref_s.dtype
    max_r = int(max_shift_row)
    max_c = int(max_shift_col)
    R = int(refine_radius)

    # Zero-mean within mask → circular cross-correlation peak ≈ best shift.
    n = msk_s.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    ref_mu = (ref_s * msk_s).sum(dim=(-2, -1), keepdim=True) / n
    mov_mu = (mov_s * msk_s).sum(dim=(-2, -1), keepdim=True) / n
    r = (ref_s - ref_mu) * msk_s
    v = (mov_s - mov_mu) * msk_s

    # sum_i ref[i] * mov[i - lag]  peaks at lag = roll amount of mov toward ref.
    # ifft(fft(r) * conj(fft(v)))[lag] ≈ that sum → lag is our (dr, dc).
    Fr = torch.fft.rfft2(r)
    Fm = torch.fft.rfft2(v)
    corr = torch.fft.irfft2(Fr * Fm.conj(), s=(hs, W))  # (S, hs, W)

    # Restrict peak search to the allowed window (positive + wrapped negative).
    # Build a boolean window once, gather scores.
    rr = torch.arange(hs, device=device)
    cc = torch.arange(W, device=device)
    # lag dr maps to index dr % hs; allow dr in [-max_r, max_r]
    if max_r == 0:
        row_ok = rr == 0
    else:
        row_ok = (rr <= max_r) | (rr >= hs - max_r)
    if max_c == 0:
        col_ok = cc == 0
    else:
        col_ok = (cc <= max_c) | (cc >= W - max_c)
    window = row_ok[:, None] & col_ok[None, :]  # (hs, W)
    neg_inf = torch.finfo(dtype).min
    corr_win = torch.where(window[None], corr, torch.full_like(corr, neg_inf))
    flat = corr_win.reshape(S, -1)
    peak = flat.argmax(dim=-1)
    peak_dr = peak // W  # 0..hs-1
    peak_dc = peak % W

    # Convert FFT indices → signed roll amounts in [-max, max]
    cand_dr = torch.where(peak_dr <= max_r, peak_dr, peak_dr - hs)
    cand_dc = torch.where(peak_dc <= max_c, peak_dc, peak_dc - W)
    cand_dr = cand_dr.clamp(-max_r, max_r)
    cand_dc = cand_dc.clamp(-max_c, max_c)

    # Local true masked-NCC refine around the FFT peak (handles mask / sign edge cases).
    best_ncc = torch.full((S,), -1.0e9, device=device, dtype=dtype)
    best_dr = cand_dr.clone()
    best_dc = cand_dc.clone()
    for ddr in range(-R, R + 1):
        for ddc in range(-R, R + 1):
            dr = (cand_dr + ddr).clamp(-max_r, max_r)
            dc = (cand_dc + ddc).clamp(-max_c, max_c)
            ncc = _ncc_at_per_strip_shifts(ref_s, mov_s, msk_s, dr, dc)
            improved = ncc > best_ncc
            best_ncc = torch.where(improved, ncc, best_ncc)
            best_dr = torch.where(improved, dr, best_dr)
            best_dc = torch.where(improved, dc, best_dc)
    return best_dr, best_dc, best_ncc


@torch.no_grad()
def measure_strip_residual_torch(
    reference: np.ndarray | torch.Tensor,
    moving: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    n_strips: int = 12,
    max_shift_col: int = 80,
    max_shift_row: int = 8,
    device: str | torch.device | None = None,
    min_mask_frac: float = 0.25,
    method: str = "auto",
) -> dict[str, object]:
    """Batched strip residual of ``moving`` vs fixed ``reference`` (GPU torch).

    For each horizontal strip, find integer shifts
    ``(dr, dc) ∈ [-max_shift_row, max_shift_row] × [-max_shift_col, max_shift_col]``
    maximizing masked NCC. All strips share the same frozen images.

    Parameters
    ----------
    reference, moving
        2-D images in the same frame (typically reference HAADF and
        affine-corrected EDS HAADF).
    mask
        Boolean coverage / common FOV. False pixels are excluded from NCC.
    n_strips
        Number of horizontal bands (slow-scan blocks).
    max_shift_col, max_shift_row
        Integer search half-width in pixels. Defaults are large enough that
        post-affine EDS residual is not clipped (was ±40/±4 historically).
    device
        Torch device; default CUDA if available else CPU.
    min_mask_frac
        Skip strips with valid fraction below this (no residual written).
    method
        ``"auto"`` (default) - FFT phase-corr + local masked-NCC refine for
        large windows (fast on GPU); exact brute for tiny windows.
        ``"fft"`` - always FFT+refine.
        ``"brute"`` - exhaustive (slow for large caps; exact).

    Returns
    -------
    dict
        ``centers`` (S,), ``drow`` (S,), ``dcol`` (S,), ``ncc`` (S,),
        ``valid`` (S,) bool, ``n_strips``, ``strip_height``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    dtype = torch.float32

    ref = _as_torch2d(reference, device, dtype)
    mov = _as_torch2d(moving, device, dtype)
    if isinstance(mask, torch.Tensor):
        msk = mask.to(device=device, dtype=dtype)
        if msk.ndim != 2:
            raise ValueError(f"expected 2-D mask, got shape {tuple(msk.shape)}")
    else:
        msk = _as_torch2d(np.asarray(mask, dtype=np.float32), device, dtype)
    if ref.shape != mov.shape or ref.shape != msk.shape:
        raise ValueError(
            f"shape mismatch ref{tuple(ref.shape)} mov{tuple(mov.shape)} mask{tuple(msk.shape)}"
        )

    H, W = ref.shape
    if n_strips < 2:
        raise ValueError(f"n_strips must be >= 2, got {n_strips}")
    hs = H // n_strips
    if hs < 1:
        raise ValueError(f"strip height {hs} too small for H={H}, n_strips={n_strips}")

    S = n_strips
    H_use = S * hs
    ref_s = ref[:H_use].reshape(S, hs, W)
    mov_s = mov[:H_use].reshape(S, hs, W)
    msk_s = msk[:H_use].reshape(S, hs, W)
    mask_frac = msk_s.mean(dim=(-2, -1))  # (S,)

    n_shifts = (2 * int(max_shift_row) + 1) * (2 * int(max_shift_col) + 1)
    method = (method or "auto").lower()
    if method == "auto":
        # Prefer FFT on CUDA for anything beyond a tiny window - brute launches
        # one full-strip roll kernel per (dr, dc) and wastes the GPU on large caps.
        if device.type == "cuda" and n_shifts > 81:
            method = "fft"
        elif n_shifts > 200:
            method = "fft"
        else:
            method = "brute"

    if method == "brute":
        best_dr, best_dc, best_ncc = _search_shifts_brute(
            ref_s, mov_s, msk_s, max_shift_row, max_shift_col
        )
    elif method == "fft":
        best_dr, best_dc, best_ncc = _search_shifts_fft(
            ref_s, mov_s, msk_s, max_shift_row, max_shift_col
        )
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'auto', 'fft', or 'brute'"
        )

    valid = (mask_frac >= min_mask_frac) & (best_ncc > -1.0e8)
    centers = (torch.arange(S, device=device, dtype=dtype) + 0.5) * hs

    return {
        "centers": centers.detach().cpu().numpy(),
        "drow": best_dr.detach().cpu().numpy().astype(np.float64),
        "dcol": best_dc.detach().cpu().numpy().astype(np.float64),
        "ncc": best_ncc.detach().cpu().numpy().astype(np.float64),
        "valid": valid.detach().cpu().numpy().astype(bool),
        "mask_frac": mask_frac.detach().cpu().numpy().astype(np.float64),
        "n_strips": S,
        "strip_height": hs,
        "image_shape": (H, W),
        "method": method,
    }


def interpolate_residual_to_rows(
    centers: np.ndarray,
    drow: np.ndarray,
    dcol: np.ndarray,
    valid: np.ndarray,
    n_rows: int,
    *,
    smooth_sigma_rows: float = 48.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate strip residuals to every scanline; Gaussian-smooth along slow axis.

    Always returns length ``n_rows``. (``np.convolve(..., mode="same")`` can grow
    to the kernel length when the kernel is longer than the signal - e.g. σ=48
    on a 64-px test FOV - so we pad + ``mode="valid"`` instead.)
    """
    if not np.any(valid):
        return np.zeros(n_rows, dtype=np.float32), np.zeros(n_rows, dtype=np.float32)
    c = centers[valid]
    dr = drow[valid]
    dc = dcol[valid]
    rows = np.arange(n_rows, dtype=np.float64)
    dr_i = np.interp(rows, c, dr, left=dr[0], right=dr[-1])
    dc_i = np.interp(rows, c, dc, left=dc[0], right=dc[-1])
    if smooth_sigma_rows and smooth_sigma_rows > 0:
        rad = int(max(1, round(3 * smooth_sigma_rows)))
        x = np.arange(-rad, rad + 1, dtype=np.float64)
        ker = np.exp(-(x**2) / (2 * smooth_sigma_rows**2))
        ker /= ker.sum()
        # edge-pad so valid convolution is exactly n_rows (not max(n, ker))
        dr_i = np.convolve(np.pad(dr_i, rad, mode="edge"), ker, mode="valid")
        dc_i = np.convolve(np.pad(dc_i, rad, mode="edge"), ker, mode="valid")
    if dr_i.shape[0] != n_rows or dc_i.shape[0] != n_rows:
        raise RuntimeError(
            f"interpolate_residual_to_rows length bug: n_rows={n_rows} "
            f"drow={dr_i.shape[0]} dcol={dc_i.shape[0]}"
        )
    return dr_i.astype(np.float32), dc_i.astype(np.float32)


def region_ncc(
    reference: np.ndarray | torch.Tensor,
    moving: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Common + top/middle/bottom masked NCC.

    Uses torch on CUDA when available (or when ``device`` is set); falls back
    to numpy on CPU-only hosts. Same numeric definition either way.
    """
    if device is None:
        use_torch = torch.cuda.is_available()
        device = torch.device("cuda") if use_torch else torch.device("cpu")
    else:
        device = torch.device(device)
        use_torch = True

    if use_torch:
        ref = _as_torch2d(reference, device, torch.float32)
        mov = _as_torch2d(moving, device, torch.float32)
        if isinstance(mask, torch.Tensor):
            m = mask.to(device=device)
        else:
            m = torch.as_tensor(np.asarray(mask), device=device)
        m = (m > 0.5).to(dtype=torch.float32)  # 0/1 multiply-mask (no bool gather)
        h = ref.shape[0]

        def _ncc_tensor(a, b, mm):
            n = mm.sum().clamp_min(1.0)
            xa = (a - (a * mm).sum() / n) * mm
            yb = (b - (b * mm).sum() / n) * mm
            num = (xa * yb).sum()
            den = (xa.norm() * yb.norm()).clamp_min(1e-12)
            return torch.where(n >= 64.0, num / den, torch.full((), float("nan"), device=a.device))

        scores = [
            _ncc_tensor(ref, mov, m),
            _ncc_tensor(ref, mov, m * torch.cat([
                torch.ones(h // 3, device=device),
                torch.zeros(h - h // 3, device=device),
            ])[:, None]),
            _ncc_tensor(ref, mov, m * torch.cat([
                torch.zeros(h // 3, device=device),
                torch.ones(h // 3, device=device),
                torch.zeros(h - 2 * (h // 3), device=device),
            ])[:, None]),
            _ncc_tensor(ref, mov, m * torch.cat([
                torch.zeros(2 * (h // 3), device=device),
                torch.ones(h - 2 * (h // 3), device=device),
            ])[:, None]),
            m.mean(),
        ]
        # single host sync for all diagnostics
        vals = torch.stack([s.reshape(()) for s in scores]).detach().cpu().tolist()
        return {
            "common": float(vals[0]),
            "top": float(vals[1]),
            "middle": float(vals[2]),
            "bottom": float(vals[3]),
            "mask_frac": float(vals[4]),
        }

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    h = ref.shape[0]

    def _ncc(a, b, mm):
        if mm.sum() < 64:
            return float("nan")
        x, y = a[mm], b[mm]
        x = x - x.mean()
        y = y - y.mean()
        return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))

    out = {"common": _ncc(ref, mov, m)}
    for name, (r0, r1) in zip(
        ("top", "middle", "bottom"),
        ((0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)),
    ):
        band = np.zeros_like(m)
        band[r0:r1] = True
        out[name] = _ncc(ref, mov, m & band)
    out["mask_frac"] = float(m.mean())
    return out


def apply_row_residual_to_knots(
    dc,
    drow: np.ndarray,
    dcol: np.ndarray,
    weight: np.ndarray,
    *,
    moving_index: int = 1,
) -> None:
    """Add per-scanline residual to knot coordinates in-place (topology unchanged).

    ``weight[r]`` scales the residual on scanline ``r`` (0 = leave knot as-is).
    """
    k = dc.knots[moving_index]
    if k.shape[1] != len(drow) or k.shape[1] != len(weight):
        raise ValueError(
            f"row residual length mismatch: knots H={k.shape[1]} "
            f"drow={len(drow)} weight={len(weight)}"
        )
    with torch.no_grad():
        dev, dtype = k.device, k.dtype
        w = torch.as_tensor(weight, device=dev, dtype=dtype)[:, None]  # (H, 1)
        # K trailing dim: broadcast over knots along the scanline
        k[0].add_(torch.as_tensor(drow, device=dev, dtype=dtype)[:, None] * w)
        k[1].add_(torch.as_tensor(dcol, device=dev, dtype=dtype)[:, None] * w)


def correct_strip(
    self,
    *,
    num_strips: int = 24,
    max_row_shift: int = 8,
    max_column_shift: int = 80,
    smoothing_sigma: float = 12.0,
    update_fraction: float = 1.0,
    num_refine_cycles: int = 1,
    passes: Sequence[StripPass] | None = None,
    fixed_scans: list[int] | None = None,
    correction_start_fraction: float | None = None,
    ramp_fraction: float = 0.08,
    min_overlap_fraction: float = 0.25,
    show_combined: bool = True,
    show_scans: bool = False,
    show_knots: bool = True,
    show_knot_plot: bool = False,
    show_report: bool = False,
    verbose: bool = True,
):
    """Remove slow-scan-dependent residual drift after affine correction.

    Horizontal strips provide stable local displacement measurements when a
    single affine rate cannot describe scan acceleration or bending. Each strip
    contributes one masked-NCC ``(row, col)`` shift; interpolation and Gaussian
    smoothing turn those measurements into a continuous scanline correction.

    Reference mode fixes image 0 automatically. Mutual multi-scan mode measures
    each free scan against the leave-one-out mean of the other corrected scans.
    Use :meth:`correct_nonrigid` when independent per-scanline motion is needed.

    Parameters
    ----------
    num_strips : int, default 24
        Number of horizontal bands for independent rigid shifts.
    max_column_shift, max_row_shift : int
        Integer NCC search range (px) per band. Defaults are wide so
        post-affine residual is not clipped; search is FFT-accelerated.
    correction_start_fraction : float or None, default None
        If set, only apply residual from this fraction of the FOV
        downward (slow-scan) with a smooth ramp - useful when only the
        bottom of a long scan still bends. ``None`` = full FOV.
    smoothing_sigma : float, default 12.0
        Gaussian sigma (rows) when expanding strip shifts to scanlines.
    num_refine_cycles : int, default 1
        Outer rebuilds of the corrected stack (re-measure residual after
        applying knot deltas). Multipass helps large decelerating residuals.
    update_fraction : float, default 1.0
        Multiplier applied to each measured row/column shift before it is
        written into the knots. Values below one damp a pass, trading
        convergence speed for stability. Used only by the scalar API.
    passes : sequence of StripPass, optional
        Heterogeneous coarse-to-fine recipe. Scalar pass settings must remain
        at their defaults when this is supplied.
    fixed_scans : list[int] or None
        Images held fixed. Default ``None`` → auto ``[0]`` in reference
        mode; otherwise leave-one-out mutual mode.
    min_overlap_fraction : float, default 0.25
        Minimum valid-mask fraction required to measure a strip.
    show_combined, show_scans, show_knots, show_knot_plot
        Display knobs (same spirit as :meth:`correct_nonrigid`).
    show_report : bool, default False
        Print the before/affine/strip regional NCC table after alignment.
    verbose : bool, default True
        Show strip-pass progress and print per-pass region NCC deltas when
        available.

    Returns
    -------
    Self
        For method chaining.

    Examples
    --------
    >>> drift.correct_affine(show_combined=False)
    >>> drift.correct_strip(num_strips=24, show_combined=True)

    """
    if not hasattr(self, "knots") or not hasattr(self, "_knots_after_affine"):
        raise RuntimeError("correct_strip requires correct_affine() first.")
    scalar_values = {
        "num_strips": num_strips,
        "max_row_shift": max_row_shift,
        "max_column_shift": max_column_shift,
        "smoothing_sigma": smoothing_sigma,
        "update_fraction": update_fraction,
        "num_refine_cycles": num_refine_cycles,
        "correction_start_fraction": correction_start_fraction,
        "ramp_fraction": ramp_fraction,
    }
    scalar_defaults = {
        "num_strips": 24,
        "max_row_shift": 8,
        "max_column_shift": 80,
        "smoothing_sigma": 12.0,
        "update_fraction": 1.0,
        "num_refine_cycles": 1,
        "correction_start_fraction": None,
        "ramp_fraction": 0.08,
    }
    if passes is None:
        if num_refine_cycles < 1:
            raise ValueError(f"num_refine_cycles must be >= 1, got {num_refine_cycles!r}.")
        pass_configs = [
            StripPass(
                num_strips=num_strips,
                max_row_shift=max_row_shift,
                max_column_shift=max_column_shift,
                correction_start_fraction=correction_start_fraction,
                ramp_fraction=ramp_fraction,
                smoothing_sigma=smoothing_sigma,
                update_fraction=update_fraction,
            )
            for _ in range(num_refine_cycles)
        ]
    else:
        mixed = [
            name for name, value in scalar_values.items() if value != scalar_defaults[name]
        ]
        if mixed:
            raise ValueError(
                "passes= cannot be combined with scalar pass settings: "
                + ", ".join(mixed)
                + ". Put those values on each StripPass instead."
            )
        pass_configs = list(passes)
        if not pass_configs:
            raise ValueError("passes must contain at least one StripPass.")
        if not all(isinstance(config, StripPass) for config in pass_configs):
            raise TypeError(
                "passes must contain only StripPass objects. Import with "
                "`from quantem.imaging.drift import StripPass`."
            )

    # Same anchoring rule as correct_affine / correct_nonrigid.
    if fixed_scans is None and self._reference_mode:
        fixed_scans = [0]
    fixed_set = frozenset(fixed_scans) if fixed_scans is not None else frozenset()
    num_images = self.shape[0]
    moving_indices = [i for i in range(num_images) if i not in fixed_set]
    if not moving_indices:
        raise ValueError(
            "All images are fixed - nothing to strip-align. "
            "fixed_scans must leave at least one moving image."
        )
    if fixed_set and any(i < 0 or i >= num_images for i in fixed_set):
        raise ValueError(
            f"fixed_scans out of range for {num_images} images: {sorted(fixed_set)}"
        )
    device = self._device
    method = "auto"

    # Keep solve tensors on the active device (reload can leave CPU knots).
    self.imgs_t = [t.to(self._device) for t in self.imgs_t]
    self.knots = [k.to(self._device) for k in self.knots]
    for attr in ("_knots_after_affine", "_initial_knots"):
        snapshot = getattr(self, attr, None)
        if snapshot is not None:
            setattr(self, attr, [k.to(self._device) for k in snapshot])

    direct_reference_mode = self._reference_mode and fixed_set == frozenset({0})
    reference_mask = (
        np.asarray(self.coverage_mask(), dtype=bool) if direct_reference_mode else None
    )

    strip_progress = tqdm(
        enumerate(pass_configs),
        total=len(pass_configs),
        desc="Solving strip drift",
        unit="pass",
        disable=not verbose,
    )
    for pass_idx, config in strip_progress:
        # In reference mode, keep the external reference frame fixed and
        # warp the moving alignment image directly, exactly as downstream
        # spectrum/map channels are warped. Mutual mode still needs the
        # co-registered leave-one-out canvas.
        corrected = (
            warping.reference_scan_stack(self)
            if direct_reference_mode
            else warping.co_registered_scan_stack(self, fixed_set=fixed_set)
        )
        shapes = {im.shape for im in corrected}
        if len(shapes) != 1:
            raise ValueError(f"correct_strip: corrected scans have mixed shapes {shapes}")
        mask = (
            reference_mask
            if reference_mask is not None
            else np.asarray(self.coverage_mask(), dtype=bool)
        )
        if mask.shape != corrected[0].shape:
            raise ValueError(
                f"coverage_mask shape {mask.shape} != scan shape {corrected[0].shape}"
            )

        # Measure all free scans against frozen refs, then apply together.
        pending = []
        for i in moving_indices:
            strip_progress.set_postfix_str(
                f"strips={config.num_strips}, scan={i}",
                refresh=verbose,
            )
            if fixed_set:
                ref = np.mean([corrected[j] for j in sorted(fixed_set)], axis=0).astype(
                    np.float32
                )
            else:
                others = [corrected[j] for j in range(num_images) if j != i]
                ref = np.mean(others, axis=0).astype(np.float32)
            mov = corrected[i]
            ncc_before = region_ncc(ref, mov, mask, device=device)
            measured = measure_strip_residual_torch(
                ref,
                mov,
                mask,
                n_strips=config.num_strips,
                max_shift_col=config.max_column_shift,
                max_shift_row=config.max_row_shift,
                device=device,
                min_mask_frac=min_overlap_fraction,
                method=method,
            )
            H = ref.shape[0]
            drow, dcol = interpolate_residual_to_rows(
                measured["centers"],
                measured["drow"],
                measured["dcol"],
                measured["valid"],
                H,
                smooth_sigma_rows=config.smoothing_sigma,
            )
            weight = free_weight(
                H,
                free_from_frac=config.correction_start_fraction,
                ramp_frac=config.ramp_fraction,
            )
            drow = drow * config.update_fraction
            dcol = dcol * config.update_fraction
            pending.append(
                (
                    i,
                    drow,
                    dcol,
                    weight,
                    ncc_before,
                )
            )

        for i, drow, dcol, weight, _ in pending:
            apply_row_residual_to_knots(self, drow, dcol, weight, moving_index=i)

        corrected_after = (
            warping.reference_scan_stack(self)
            if direct_reference_mode
            else warping.co_registered_scan_stack(self, fixed_set=fixed_set)
        )
        for i, _, _, _, ncc_before in pending:
            if fixed_set:
                ref = np.mean(
                    [corrected[j] for j in sorted(fixed_set)], axis=0
                ).astype(np.float32)
            else:
                others = [corrected[j] for j in range(num_images) if j != i]
                ref = np.mean(others, axis=0).astype(np.float32)
            ncc_after = region_ncc(
                ref, corrected_after[i], mask, device=device
            )
            if verbose:
                common_b = ncc_before.get("common", float("nan"))
                common_a = ncc_after.get("common", float("nan"))
                strip_progress.write(
                    f"correct_strip pass {pass_idx + 1}/{len(pass_configs)} "
                    f"image {i}: NCC common {common_b:.4f} → {common_a:.4f} "
                    f"(Δ={common_a - common_b:+.4f})"
                )

    self._images_warped_stale = True
    # Snapshot knots after strip for optional stage plots (affine / strip / NR).
    self._knots_after_strip = [k.detach().clone() for k in self.knots]

    drift_plot.show_after_step(
        self,
        "strip",
        show_combined=show_combined,
        show_scans=show_scans,
        show_knots=show_knots,
    )
    if show_knot_plot:
        self.plot_knots()
    if show_report:
        print(self.report().to_string())
    return self
