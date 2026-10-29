"""MAPED on tilts loaded by quantem.gpu: one sample, three beam tilts, known displacements."""

import os

import pytest
from torch import arange, backends, cuda, equal, exp, float32, meshgrid, tensor, uint16, zeros
from torch.testing import assert_close

from quantem.diffraction import MAPED

io = pytest.importorskip("quantem.gpu.io", reason="Encoded-input tests require quantem.gpu")

# Each tilt is the same sample seen with the beam moved on the detector and the
# sample moved in the scan, by whole pixels: (beam_row, beam_col), (scan_row, scan_col).
DISPLACEMENTS = [((0, 0), (0, 0)), ((2, -1), (3, 1)), ((-1, 2), (-2, -3))]


@pytest.mark.parametrize("custom_images", [False, True])
def test_run_matches_the_step_by_step_stages(tmp_path, custom_images):
    """Automatic and manual stages recover the known shifts and the same merge."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    one, steps = MAPED.from_datasets(tilts), MAPED.from_datasets(tilts)
    images = [image.square() for image in one.bright_field] if custom_images else None
    run_merged_t = one.run(images=images, show=False, verbose=False).read()
    for shifts_t, expected in zip(
        (one.diffraction_shifts, one.real_space_shifts), zip(*DISPLACEMENTS)
    ):
        # A shift undoes a displacement; the measured recovery error is below 0.07 px.
        expected_t = -tensor(expected, dtype=float32, device=one.device)
        assert_close(shifts_t - shifts_t[0], expected_t, rtol=0, atol=0.07)
    steps.find_beam_center(blur=1, show=False)
    steps.align_diffraction(border_taper=2, show=False)
    steps.align_images(images=images, iterations=20, hann_window=True, max_shift=5, show=False)
    # run crops to the positions every tilt covers; the stage call must say so too
    steps_merged_t = steps.merge(dtype="float32", crop=True, show=False).read()
    assert equal(one.diffraction_shifts, steps.diffraction_shifts)
    assert equal(one.real_space_shifts, steps.real_space_shifts)
    assert run_merged_t.dtype == float32
    assert equal(run_merged_t, steps_merged_t)


def test_float32_full_crop_matches_region_reads(tmp_path):
    """A cropped full merge agrees exactly with a separately merged scan region."""
    tilts = _load_tilts(
        tmp_path, scan_shape=(128, 128), detector_shape=(8, 8), backend="cpu"
    )
    maped = MAPED.from_datasets(tilts)
    merged = maped.run(iterations=20, dtype="float32", show=False, verbose=False)
    region = merged.metadata["maped_merge"]["scan_region"]
    row0, _, col0, _ = region
    patch = (row0 + 24, row0 + 40, col0 + 8, col0 + 24)
    exact = maped.merge(scan_region=patch, dtype="float32", show=False).read()
    assert equal(merged.read()[24:40, 8:24], exact)


def test_whole_pixel_scan_copies_merge_back_to_the_sample(tmp_path):
    """Tilts that are whole-pixel scan copies of one sample merge to that sample."""
    scan_only = [((0, 0), scan) for _, scan in DISPLACEMENTS]
    tilts = _load_tilts(
        tmp_path, scan_shape=(32, 32), detector_shape=(16, 16), displacements=scan_only
    )
    maped = MAPED(tilts)
    maped.diffraction_shifts = zeros((3, 2), device=maped.device)
    maped.real_space_shifts = -tensor(
        [scan for _, scan in scan_only], dtype=float32, device=maped.device
    )
    merged_t = maped.merge(dtype="float32", show=False).read()
    # away from the scan border every tilt contributes the same counts, so their mean is
    # the sample up to float32 rounding of the three-tilt average (1.2e-4 at 400 counts)
    interior = (slice(8, 24), slice(8, 24))
    assert_close(merged_t[interior], tilts[0].read()[interior].to(float32), rtol=0, atol=2e-4)


def test_scaled_merge_and_saved_file_match_float32_regions(tmp_path):
    """Scaled storage stays within its reported error, across regions and after saving."""
    # 72 x 64 scan positions exceed one 4096-frame merge region, so two regions are merged
    tilts = _load_tilts(tmp_path, scan_shape=(72, 64), detector_shape=(8, 8))
    maped = MAPED(tilts)
    maped.run(show=False, verbose=False)
    # the second region starts at scan row 64; this patch straddles it
    regions = [(0, 4, 0, 4), (62, 66, 0, 64), (68, 72, 60, 64)]
    exact = [
        maped.merge(scan_region=region, show=False).read() for region in regions
    ]
    merged = maped.merge(dtype="scaled_uint16", show=False, verbose=False)
    max_error = merged.metadata["precision"]["max_abs_error"]
    for region, exact_t in zip(regions, exact):
        assert_close(merged.read(scan_region=region), exact_t, rtol=0, atol=max_error)
    io.save(tmp_path / "merged_master.h5", merged)
    saved = io.load(tmp_path / "merged_master.h5", verbose=False)
    for region in regions:
        assert equal(saved.read(scan_region=region), merged.read(scan_region=region))


def test_float_merge_saves_exactly_and_releases_input_memory(tmp_path):
    """A complete float merge can be written in blocks and reopened losslessly."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    maped = MAPED(tilts)
    reference = maped.run(dtype="float32", show=False, verbose=False)
    reference_t = reference.read()
    assert all(not tilt.data.is_released for tilt in tilts)
    saved = maped.merge(
        save_to=tmp_path / "merged_master.h5", crop=True,
        release_tilts=True, show=False, verbose=False,
    )
    assert equal(saved.read(), reference_t)
    assert saved.lossless
    assert saved.metadata["maped_merge"]["scan_region"] == reference.metadata["maped_merge"]["scan_region"]
    assert all(tilt.data.is_released for tilt in tilts)

    # The saved merge reopens encoded, so it can also be archived directly as QEM.
    io.save(tmp_path / "merged.qem", saved)
    with io.load(tmp_path / "merged.qem", device=maped.device, verbose=False) as reopened:
        assert reopened.lossless
        assert equal(reopened.read(), reference_t)


def test_summaries_are_the_plain_means(tmp_path):
    """Mean diffraction pattern and bright-field image equal the means of the decoded tilt."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    maped = MAPED(tilts)
    for tilt, dp_mean_t, im_bf_t in zip(tilts, maped.dp_mean, maped.im_bf):
        decoded_t = tilt.read()
        # the reference mean is taken in float64, which Apple GPUs do not have
        counts_t = decoded_t.cpu().double()
        assert equal(dp_mean_t.cpu(), counts_t.mean(dim=(0, 1)).to(float32))
        assert equal(im_bf_t.cpu(), counts_t.mean(dim=(2, 3)).to(float32))


def _load_tilts(tmp_path, scan_shape, detector_shape, displacements=DISPLACEMENTS, backend=None) -> list:
    """Three tilts of one sample, saved as master files and loaded encoded on the GPU.

    counts = 200 * sample * beam. The sample is two blobs in the scan, so the scan
    alignment has structure to lock onto; the beam is one disk on the detector. Each
    tilt moves both by the whole pixels in ``displacements``. CUDA cases run only in an
    owned GPU0 window, because the tests share that card with interactive work.
    """
    if backend == "cpu":
        pass
    elif cuda.is_available():
        if os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1":
            pytest.skip("Set QUANTEM_CUDA_ANS_TEST=1 in an owned GPU0 test window.")
        backend = "cuda"
    elif backends.mps.is_available():
        backend = "mps"
    else:
        pytest.skip("Run this case on a CUDA or MPS accelerator.")
    scan_row, scan_col = meshgrid(
        arange(scan_shape[0], dtype=float32, device=backend),
        arange(scan_shape[1], dtype=float32, device=backend),
        indexing="ij",
    )
    k_row, k_col = meshgrid(
        arange(detector_shape[0], dtype=float32, device=backend),
        arange(detector_shape[1], dtype=float32, device=backend),
        indexing="ij",
    )
    center_row, center_col = scan_shape[0] // 2, scan_shape[1] // 2
    files = []
    for index, (beam, scan) in enumerate(displacements):
        row, col = scan_row - scan[0], scan_col - scan[1]
        sample = (
            1
            + exp(-((row - center_row) ** 2 + (col - center_col) ** 2) / 18)
            + 0.5 * exp(-((row - center_row + 8) ** 2 + (col - center_col - 6) ** 2) / 8)
        )
        disk = exp(
            -(
                (k_row - detector_shape[0] // 2 - beam[0]) ** 2
                + (k_col - detector_shape[1] // 2 - beam[1]) ** 2
            )
            / (detector_shape[0] / 2)
        )
        # detector counts are integers and the file format stores uint16
        counts_t = (200 * sample[:, :, None, None] * disk).round().to(uint16)
        path = tmp_path / f"tilt_{index}_master.h5"
        io.save(path, counts_t, dtype="uint16", backend=backend, verbose=False, wait=True)
        files.append(path)
    return io.load(files, device=backend, stack=False, verbose=False)
