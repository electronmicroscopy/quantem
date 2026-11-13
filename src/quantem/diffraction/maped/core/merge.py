"""Merging the aligned tilts, one bounded scan region at a time.

A complete float32 merge of a 512 x 512 x 192 x 192 series is 38 GB.
:class:`MergedRegions` computes it a few scan rows at a time from native or
encoded tilts. IO stores each region as exact float32 or calibrated scaled
uint16 before the next region is computed.

Every merged pattern is the weighted mean of the tilts, each moved by its
(row, col) scan shift ``s_t`` and its (row, col) detector shift ``d_t``::

    merged(x, k) = sum_t w_t(x) tilt_t(x - s_t, k - d_t) / (sum_t w_t(x) v_t(k) + e(k))

for scan position ``x`` and detector pixel ``k``. ``w_t`` and ``v_t`` are the
tilt's scan and detector coverage after shifting, so positions some tilts do
not reach average the others; a tilt reads zero off its detector. ``e(k) =
max(0, 1 - sum_t v_t(k))`` lets a detector pixel that the tilts cover less than
once in total fade toward zero instead of being divided by a small coverage.
A shift by a fraction of a pixel on all four axes is a quadrilinear
interpolation: each value mixes the 16 corners of the 4D cell around it.
"""

import json
import time
from collections.abc import Iterator, Sequence
from itertools import product
from math import floor

import numpy as np
import torch.nn.functional as F
from torch import (
    Tensor,
    arange,
    cuda,
    einsum,
    float32,
    int16,
    int32,
    mps,
    ones,
    stack,
    tensor,
    uint8,
    uint16,
    zeros,
)
from torch import compile as compile_kernel

from quantem.diffraction.maped.core.align import translation_grid
from quantem.diffraction.maped.io import read_region


class MergedRegions:
    """The merge of the aligned tilts, computed region by region as it is read.

    IO reads :meth:`blocks` once, storing each float32 region before the next
    is computed. Native CPU examples use the same calculation.
    """

    dtype = np.dtype("float32")
    report_context = {
        "scope": "all merged values",
        "range_scope": "complete merged output",
        "measurement": "comparison against merged float32 regions",
    }

    def __init__(
        self,
        tilts: Sequence,
        real_space_shifts: Tensor,
        diffraction_shifts: Tensor,
        *,
        release_tilts: bool = False,
        compile_merge: bool | None = None,
        max_region_frames: int | None = None,
        crop: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        tilts : Sequence
            The loaded tilts, all of shape ``(scan_row, scan_col, k_row, k_col)``.
        real_space_shifts, diffraction_shifts : Tensor
            ``(num_tilts, 2)`` (row, col) shifts, on the tilts' device.
        release_tilts : bool
            Free each tilt's scan rows once the complete merge has passed them,
            so the tilts shrink while the merge grows. The tilts cannot be read
            again afterwards.
        compile_merge : bool or None
            ``False`` runs the merge kernel uncompiled, for machines where
            ``torch.compile`` fails; the result agrees to float32 rounding.
        max_region_frames : int or None
            Upper limit on scan positions per region; saving uses 2048 so the
            file writer has room alongside.
        crop : bool
            Keep only the scan positions every tilt covers.
        """
        self.tilts = list(tilts)
        self.source_shape = tuple(int(size) for size in self.tilts[0].shape)
        num_rows, num_cols, detector_rows, detector_cols = self.source_shape
        num_tilts = len(self.tilts)
        self.device = real_space_shifts.device
        self.real_space_shifts, self.diffraction_shifts = real_space_shifts, diffraction_shifts
        self.release_tilts = release_tilts
        if compile_merge is None:
            compile_merge = self.device.type in ("cuda", "mps")
        self.accumulate = _accumulate_compiled if compile_merge else _accumulate
        # plain floats: reading a device scalar inside the region loop would wait
        # for all queued work on every tilt
        self.scan_shifts = real_space_shifts.tolist()
        self.detector_shifts = diffraction_shifts.tolist()

        with self.device:
            # coverage: each tilt's scan window and detector, moved by its shifts;
            # the window is zero on the scan's outermost ring, so a tilt's outermost
            # scan rows and columns do not count
            window = zeros((num_tilts, 1, num_rows, num_cols))
            window[..., 1:-1, 1:-1] = 1.0
            scan_grid = translation_grid(real_space_shifts, (num_rows, num_cols))
            self.real_weights = F.grid_sample(window, scan_grid, align_corners=True)[:, 0]
            detector = ones((num_tilts, 1, detector_rows, detector_cols))
            detector_grid = translation_grid(diffraction_shifts, (detector_rows, detector_cols))
            self.detector_weights = F.grid_sample(detector, detector_grid, align_corners=True)[
                :, 0
            ].clamp(0, 1)
            self.detector_edge = 1 - self.detector_weights.sum(0).clamp(0, 1)

            # interpolation taps of every shift on every axis, gathered by the kernel
            self.taps = [
                (
                    _bilinear_taps(row_shift, num_rows),
                    _bilinear_taps(col_shift, num_cols),
                    _bilinear_taps(detector_row_shift, detector_rows),
                    _bilinear_taps(detector_col_shift, detector_cols),
                )
                for (row_shift, col_shift), (detector_row_shift, detector_col_shift) in zip(
                    self.scan_shifts, self.detector_shifts
                )
            ]

        # crop to the positions every tilt covers: border positions average fewer
        # tilts; an interior weight is 1 up to float32 rounding
        self.scan_box = (0, num_rows, 0, num_cols)
        if crop:
            covered = (self.real_weights >= 1 - 1e-5).all(dim=0)
            rows = covered.any(dim=1).nonzero().flatten().tolist()
            cols = covered.any(dim=0).nonzero().flatten().tolist()
            if not rows or not cols:
                raise ValueError("The aligned scans have no common interior; use crop=False.")
            self.scan_box = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)
        row0, row1, col0, col1 = self.scan_box
        self.shape = (row1 - row0, col1 - col0, detector_rows, detector_cols)

        # region size from free memory, budgeting 24 bytes per detector pixel per
        # position for a region's numerator, denominator, decoded counts and
        # encoded output; 512 MB stays free for everything else
        if self.device.type == "cuda":
            free = cuda.mem_get_info(self.device)[0]
        elif self.device.type == "mps":
            free = mps.recommended_max_memory() - mps.current_allocated_memory()
        else:
            free = 2 * 1024**3
        frames = max(
            1,
            min(
                max_region_frames or 4096,
                (free - 512 * 1024**2) // (detector_rows * detector_cols * 24),
            ),
        )
        self.region_frames = frames // num_cols * num_cols if frames >= num_cols else frames
        self.rows_per_region = max(1, self.region_frames // num_cols)

        # the merge summed over the detector and over the scan, for the pictures
        self.scan_sum = self.real_weights.new_zeros(self.shape[:2])
        self.detector_sum = self.real_weights.new_zeros((detector_rows, detector_cols))
        self.merge_seconds = 0.0
        summary = {
            "version": 1,
            "source_count": num_tilts,
            "hot_pixel_correction": [
                tilt.metadata.get("hot_pixel_correction") for tilt in self.tilts
            ],
            "mean_bright_field": "mean over the whole detector, every pixel in the divisor",
            "mean_diffraction_pattern": "mean over the whole scan",
        }
        self.save_metadata = {"quantem_maped_summary_v1": json.dumps(summary)}
        self._record_merge()

    def blocks(self, scan_region: tuple[int, int, int, int] | None = None) -> Iterator[Tensor]:
        """The merge, one row-aligned region at a time, as ``(frames, k_row, k_col)``.

        A region is computed only when the reader reaches it. With
        ``scan_region``, only that part of the scan; otherwise the whole scan
        box, which also adds up the sums behind the summary pictures.
        """
        row_start, row_stop, col_start, col_stop = scan_region or self.scan_box
        for row0 in range(row_start, row_stop, self.rows_per_region):
            yield self._merge_region(
                (row0, min(row_stop, row0 + self.rows_per_region)),
                (col_start, col_stop),
                scan_region is None,
            )
        if self.release_tilts and scan_region is None:
            # Release consumed inputs before a storage reader reopens the output.
            for tilt in self.tilts:
                tilt.close()
            if self.device.type == "cuda":
                cuda.empty_cache()
            elif self.device.type == "mps":
                mps.empty_cache()

    def _merge_region(
        self, rows: tuple[int, int], cols: tuple[int, int], whole_box: bool
    ) -> Tensor:
        """Why: :meth:`blocks` hands out the merge one region per call, computed only then.

        Adds every tilt in turn, each decoding only the scan rows its shift
        reaches (one beyond the last, for the interpolation), then
        divides by the summed coverage; positions no tilt covers are zero.
        """
        started = time.perf_counter()
        num_rows, num_cols, detector_rows, detector_cols = self.source_shape
        row0, row1 = rows
        col_start, col_stop = cols
        numerator = self.real_weights.new_zeros(
            (row1 - row0, col_stop - col_start, detector_rows, detector_cols)
        )
        for index, tilt in enumerate(self.tilts):
            (
                (row_taps, row_weights),
                (col_taps, col_weights),
                detector_row_taps,
                detector_col_taps,
            ) = self.taps[index]
            # whole scan rows: they are one run of frames, decoded in one call and
            # about three times faster than part of each row, decoded row by row
            decoded_row0 = max(0, row0 + floor(-self.scan_shifts[index][0]))
            decoded_row1 = min(num_rows, row1 + floor(-self.scan_shifts[index][0]) + 1)
            if decoded_row0 >= decoded_row1:
                continue
            counts = read_region(tilt, (decoded_row0, decoded_row1, 0, num_cols))
            # Preserve uint16 bits for Torch indexing; native float values stay float.
            if counts.dtype == uint16:
                counts = counts.view(int16)
            elif counts.dtype not in (uint8, float32):
                counts = counts.to(float32)
            # the row taps index the whole scan; move them onto the decoded rows,
            # where taps outside them get zero weight
            region_rows = row_taps[:, row0:row1] - decoded_row0
            rows_inside = (region_rows >= 0) & (region_rows < counts.shape[0])
            self.accumulate(
                # torch.compile on Apple GPUs has no uint16 (torch 2.13); an int16
                # view keeps the same bits, and the kernel reads them back exactly
                counts,
                (
                    region_rows.clamp(0, counts.shape[0] - 1),
                    rows_inside * row_weights[:, row0:row1],
                ),
                (col_taps[:, col_start:col_stop], col_weights[:, col_start:col_stop]),
                detector_row_taps,
                detector_col_taps,
                self.real_weights[index, row0:row1, col_start:col_stop],
                numerator,
            )
        if self.release_tilts and whole_box:
            # regions come in scan order, so no later region reads a tilt's rows
            # above the first one the next region needs
            for tilt, (row_shift, _) in zip(self.tilts, self.scan_shifts):
                tilt.release_rows_before(max(0, row1 + floor(-row_shift)))
        scan_weights = self.real_weights[:, row0:row1, col_start:col_stop]
        denominator = einsum("nrc,nhw->rchw", scan_weights, self.detector_weights)
        denominator += self.detector_edge
        # positions no tilt covers are 0 / 0; they become zero
        numerator.div_(denominator).nan_to_num_(0)
        if whole_box:
            box_row0 = self.scan_box[0]
            self.scan_sum[row0 - box_row0 : row1 - box_row0] = numerator.sum(dim=(2, 3))
            self.detector_sum += numerator.sum(dim=(0, 1))
        self.merge_seconds += time.perf_counter() - started
        self._record_merge()
        return numerator.reshape(-1, detector_rows, detector_cols)

    def _record_merge(self) -> None:
        """Why: a saved merge must say which shifts produced it, so it can be checked or redone."""
        record = {
            "version": 1,
            "backend": self.device.type,
            "region_frames": self.region_frames,
            "merge_seconds": self.merge_seconds,
            "real_space_shifts_row_column": self.scan_shifts,
            "diffraction_shifts_row_column": self.detector_shifts,
            "scan_box": list(self.scan_box),
            "scan_region": list(self.scan_box),
        }
        self.save_metadata["quantem_maped_merge_v1"] = json.dumps(record)


def _bilinear_taps(shift: float, size: int) -> tuple[Tensor, Tensor]:
    """Why: all four axes move by a bilinear shift, so each needs the same two-tap table.

    Output pixel ``j`` reads pixels ``floor(j - shift)`` and ``floor(j - shift) + 1``
    with weights ``1 - f`` and ``f``, where ``f = (j - shift) - floor(j - shift)``
    is the same for every pixel. Returns the indices ``(2, size)``, clamped
    onto the axis, and the weights ``(2, size)``, zero where a tap is off it.
    """
    lower = floor(-shift)
    fraction = -shift - lower
    indices = arange(size) + lower
    indices = stack((indices, indices + 1))
    weights = tensor([[1 - fraction], [fraction]]) * ((indices >= 0) & (indices < size))
    return indices.clamp(0, size - 1), weights


def _accumulate(
    counts: Tensor,
    scan_rows: tuple[Tensor, Tensor],
    scan_cols: tuple[Tensor, Tensor],
    detector_rows: tuple[Tensor, Tensor],
    detector_cols: tuple[Tensor, Tensor],
    weight: Tensor,
    numerator: Tensor,
) -> None:
    """Why: the merge is bound by memory traffic, so a tilt is added in a single fused pass.

    ``numerator += weight * sum over the 16 corners of (corner weight * counts at the corner)``,
    where each axis contributes two taps ``(indices, weights)``. Done in steps,
    the scan-shifted and the detector-shifted blocks would each be written out
    and read back. ``counts`` is float32, uint8, or uint16 viewed as int16:
    ``& 0xFFFF`` turns the int16 bits back into the count 0 ... 65535.
    """
    for row, col, detector_row, detector_col in product((0, 1), repeat=4):
        corner = counts[
            scan_rows[0][row][:, None, None, None],
            scan_cols[0][col][None, :, None, None],
            detector_rows[0][detector_row][None, None, :, None],
            detector_cols[0][detector_col][None, None, None, :],
        ]
        if counts.dtype == int16:
            corner = (corner.to(int32) & 0xFFFF).to(numerator.dtype)
        else:
            corner = corner.to(numerator.dtype)
        corner_weight = (
            scan_rows[1][row][:, None, None, None]
            * scan_cols[1][col][None, :, None, None]
            * detector_rows[1][detector_row][None, None, :, None]
            * detector_cols[1][detector_col][None, None, None, :]
        )
        numerator += weight[..., None, None] * corner_weight * corner


_accumulate_compiled = compile_kernel(_accumulate, fullgraph=True)
