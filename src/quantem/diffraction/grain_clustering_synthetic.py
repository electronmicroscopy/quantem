"""Synthetic signal-level datasets for grain-clustering benchmarks.

The generator operates at the :class:`~quantem.diffraction.grain_clustering.SignalTable`
layer.  Every generated signal has an authoritative grain label, so two or more grains
can occupy the same probe without forcing the truth into a single raster label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label as connected_components,
)

try:  # supports the direct-module import style used by the lightweight tests
    from .grain_clustering import SignalTable, orientation_to_rgb
except ImportError:  # pragma: no cover - exercised by direct test imports
    from grain_clustering import SignalTable, orientation_to_rgb

__all__ = [
    "CANONICAL_SCENARIOS",
    "SyntheticGrainSample",
    "generate_canonical_sample",
    "generate_canonical_suite",
    "generate_random_sample",
    "generate_dataset",
    "save_sample",
    "load_sample",
    "save_quicklook",
]


SCHEMA_VERSION = 1
CANONICAL_SCENARIOS = (
    "uniform_large",
    "bicrystal",
    "trijunction",
    "disconnected_equal_orientation",
    "continuous_bend",
    "focal_flow",
    "orientation_cusp",
    "small_inclusion",
    "mixed_scale_voronoi",
    "dispersed_islands",
    "partial_overlap",
    "crossing_interpenetration",
    "triple_overlap",
    "equal_orientation_distinct_radius",
    "continuous_radius_drift",
    "dropout",
    "dense_interior_noisy_edges",
    "dense_interior_with_holes",
    "sparse_smooth_percolation",
    "radius2_false_bridge",
    "broad_continuous_gradient",
    "false_positive_outliers",
    "multi_window_mixture",
)


def _as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclass
class SyntheticGrainSample:
    """One synthetic clustering sample with exact signal-level ground truth."""

    signals: SignalTable
    ground_truth: np.ndarray
    scenario: str
    seed: int
    metadata: dict = field(default_factory=dict)
    grain_mask_ids: np.ndarray | None = None
    grain_masks_truth: np.ndarray | None = None
    grain_windows: np.ndarray | None = None

    def __post_init__(self):
        self.ground_truth = np.asarray(self.ground_truth, dtype=np.int64).reshape(-1)
        if len(self.ground_truth) != len(self.signals):
            raise ValueError("ground_truth length must equal the number of signals")
        if self.grain_mask_ids is None or self.grain_masks_truth is None:
            ids = np.unique(self.ground_truth[self.ground_truth >= 0])
            masks = []
            windows = []
            for grain_id in ids:
                keep = self.ground_truth == grain_id
                mask = np.zeros(self.signals.map_shape, dtype=bool)
                mask[tuple(self.signals.pos[keep].T)] = True
                masks.append(mask)
                windows.append(int(self.signals.window[keep][0]))
            self.grain_mask_ids = ids.astype(np.int64)
            self.grain_masks_truth = np.stack(masks) if masks else np.zeros((0, *self.signals.map_shape), bool)
            self.grain_windows = np.asarray(windows, dtype=np.int64)
        else:
            self.grain_mask_ids = np.asarray(self.grain_mask_ids, dtype=np.int64).reshape(-1)
            self.grain_masks_truth = np.asarray(self.grain_masks_truth, dtype=bool)
            self.grain_windows = np.asarray(self.grain_windows, dtype=np.int64).reshape(-1)
        self.validate()

    @property
    def grain_ids(self) -> np.ndarray:
        return self.grain_mask_ids.copy()

    def validate(self) -> None:
        s = self.signals
        if np.any(s.pos < 0) or np.any(s.pos[:, 0] >= s.map_shape[0]) or np.any(
            s.pos[:, 1] >= s.map_shape[1]
        ):
            raise ValueError("signal position lies outside map_shape")
        if not all(np.all(np.isfinite(x)) for x in (s.theta, s.r, s.intensity)):
            raise ValueError("synthetic signal fields must be finite")
        if np.any((s.theta < 0) | (s.theta >= 180)):
            raise ValueError("orientations must be folded to [0, 180)")
        if np.any(s.r <= 0) or np.any(s.intensity <= 0) or np.any(s.window < 0):
            raise ValueError("radius/intensity must be positive and windows non-negative")
        expected_shape = (len(self.grain_mask_ids), *s.map_shape)
        if self.grain_masks_truth.shape != expected_shape:
            raise ValueError(f"grain_masks_truth shape {self.grain_masks_truth.shape} != {expected_shape}")
        if len(self.grain_windows) != len(self.grain_mask_ids):
            raise ValueError("grain window metadata length is inconsistent")
        if len(np.unique(self.grain_mask_ids)) != len(self.grain_mask_ids):
            raise ValueError("grain_mask_ids must be unique")
        # A grain may coexist with other grains at a probe, but cannot contribute twice.
        width = int(s.map_shape[1])
        valid = self.ground_truth >= 0
        keys = np.column_stack(
            [self.ground_truth[valid], s.pos[valid, 0] * width + s.pos[valid, 1]]
        )
        if len(keys) != len(np.unique(keys, axis=0)):
            raise ValueError("a ground-truth grain contributes more than one signal at a probe")
        for grain_index, grain_id in enumerate(self.grain_ids):
            windows = np.unique(s.window[self.ground_truth == grain_id])
            if len(windows) > 1:
                raise ValueError("a ground-truth grain cannot span immutable radial windows")
            if len(windows) == 1 and windows[0] != self.grain_windows[grain_index]:
                raise ValueError("observed signal window disagrees with latent grain metadata")

    def membership_count_map(self, window: int | None = None) -> np.ndarray:
        """Number of latent ground-truth grain extents covering each probe."""
        keep = np.ones(len(self.grain_mask_ids), dtype=bool)
        if window is not None:
            keep = self.grain_windows == int(window)
        return self.grain_masks_truth[keep].sum(axis=0, dtype=np.int16)

    def observed_membership_count_map(self, window: int | None = None) -> np.ndarray:
        """Number of grains with an actually observed signal at each probe."""
        counts = np.zeros(self.signals.map_shape, dtype=np.int16)
        seen = set()
        for index, (rx, ry) in enumerate(self.signals.pos):
            grain_id = int(self.ground_truth[index])
            if grain_id < 0 or (window is not None and self.signals.window[index] != window):
                continue
            key = (int(rx), int(ry), grain_id)
            if key not in seen:
                seen.add(key)
                counts[rx, ry] += 1
        return counts

    def primary_label_map(self, window: int | None = None) -> np.ndarray:
        """Highest-intensity grain per probe; overlap truth remains in ``ground_truth``."""
        labels = np.full(self.signals.map_shape, -1, dtype=np.int64)
        best = np.full(self.signals.map_shape, -np.inf)
        for index, (rx, ry) in enumerate(self.signals.pos):
            grain_id = int(self.ground_truth[index])
            if grain_id < 0 or (window is not None and self.signals.window[index] != window):
                continue
            if self.signals.intensity[index] >= best[rx, ry]:
                best[rx, ry] = self.signals.intensity[index]
                labels[rx, ry] = grain_id
        return labels

    def grain_mask(self, grain_id: int) -> np.ndarray:
        match = np.flatnonzero(self.grain_mask_ids == int(grain_id))
        if not len(match):
            return np.zeros(self.signals.map_shape, dtype=bool)
        return self.grain_masks_truth[int(match[0])].copy()

    def grain_table(self) -> list[dict]:
        rows = []
        for grain_index, grain_id in enumerate(self.grain_ids):
            keep = self.ground_truth == grain_id
            mask = self.grain_mask(int(grain_id))
            n_observed = int(keep.sum())
            area = int(mask.sum())
            observed_mask = np.zeros(self.signals.map_shape, dtype=bool)
            if n_observed:
                observed_mask[tuple(self.signals.pos[keep].T)] = True
            distance = distance_transform_edt(mask)
            core = mask & (distance > 2)
            edge = mask & ~core
            component_labels, n_components = connected_components(
                observed_mask, structure=np.ones((3, 3), int)
            )
            component_sizes = np.bincount(component_labels.ravel())[1:]
            largest_component_fraction = (
                float(component_sizes.max() / n_observed) if n_observed and component_sizes.size else 0.0
            )
            extent_positions = np.argwhere(mask)
            if area:
                span = extent_positions.max(axis=0) - extent_positions.min(axis=0) + 1
                bbox_area = int(np.prod(span))
            else:
                bbox_area = 0
            if n_observed:
                theta = np.deg2rad(2.0 * self.signals.theta[keep])
                theta_mean = (np.rad2deg(np.arctan2(np.sin(theta).sum(), np.cos(theta).sum())) / 2) % 180
                radius_mean = float(self.signals.r[keep].mean())
                intensity_median = float(np.median(self.signals.intensity[keep]))
            else:
                theta_mean = radius_mean = intensity_median = None
            rows.append(
                {
                    "grain_id": int(grain_id),
                    "window": int(self.grain_windows[grain_index]),
                    "n_signals": n_observed,
                    "area_pixels": area,
                    "observed_fraction": float(n_observed / area) if area else 0.0,
                    "core_observed_fraction": float(np.count_nonzero(observed_mask & core) / np.count_nonzero(core))
                    if np.any(core)
                    else 0.0,
                    "edge_observed_fraction": float(np.count_nonzero(observed_mask & edge) / np.count_nonzero(edge))
                    if np.any(edge)
                    else 0.0,
                    "observed_bbox_occupancy": float(n_observed / bbox_area) if bbox_area else 0.0,
                    "observed_components_8": int(n_components),
                    "largest_observed_component_fraction": largest_component_fraction,
                    "theta_mean_deg": float(theta_mean) if theta_mean is not None else None,
                    "r_mean": radius_mean,
                    "intensity_median": intensity_median,
                    "centroid": np.argwhere(mask).mean(axis=0).tolist() if area else None,
                }
            )
        return rows


class _SignalBuilder:
    def __init__(self, map_shape: tuple[int, int], rng: np.random.Generator):
        self.map_shape = tuple(int(x) for x in map_shape)
        self.rng = rng
        self.rows = []
        self.next_grain_id = 0
        self.grain_extents = {}
        self.grain_windows = {}

    def add_grain(
        self,
        mask: np.ndarray,
        *,
        theta=30.0,
        radius=0.13,
        intensity=1.0,
        window: int = 0,
        dropout: float | np.ndarray = 0.0,
        theta_noise: float = 0.0,
        radius_noise_rel: float = 0.0,
        intensity_noise: float = 0.0,
        grain_id: int | None = None,
    ) -> int:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.map_shape:
            raise ValueError(f"mask shape {mask.shape} != map shape {self.map_shape}")
        grain_id = self.next_grain_id if grain_id is None else int(grain_id)
        self.next_grain_id = max(self.next_grain_id, grain_id + 1)
        self.grain_extents[grain_id] = self.grain_extents.get(grain_id, np.zeros(self.map_shape, bool)) | mask
        if grain_id in self.grain_windows and self.grain_windows[grain_id] != int(window):
            raise ValueError("a grain cannot be added to more than one radial window")
        self.grain_windows[grain_id] = int(window)
        positions = np.argwhere(mask)
        if len(positions):
            if np.ndim(dropout) == 0:
                probability = np.full(len(positions), float(dropout))
            else:
                dropout_map = np.asarray(dropout, dtype=float)
                if dropout_map.shape != self.map_shape:
                    raise ValueError("spatial dropout map must match map_shape")
                probability = dropout_map[tuple(positions.T)]
            positions = positions[self.rng.random(len(positions)) >= np.clip(probability, 0, 1)]
        for rx, ry in positions:
            th = theta(rx, ry) if callable(theta) else theta
            rr = radius(rx, ry) if callable(radius) else radius
            ii = intensity(rx, ry) if callable(intensity) else intensity
            th = (float(th) + self.rng.normal(0, theta_noise)) % 180.0
            rr = float(rr) * (1.0 + self.rng.normal(0, radius_noise_rel))
            ii = float(ii) * np.exp(self.rng.normal(0, intensity_noise))
            self.rows.append((int(rx), int(ry), th, max(rr, 1e-6), max(ii, 1e-6), window, grain_id))
        return grain_id

    def add_outliers(self, n: int, *, windows: int = 1) -> None:
        for _ in range(int(n)):
            rx = int(self.rng.integers(0, self.map_shape[0]))
            ry = int(self.rng.integers(0, self.map_shape[1]))
            window = int(self.rng.integers(0, max(windows, 1)))
            base_r = (0.03, 0.13, 0.26)[window % 3]
            self.rows.append(
                (rx, ry, float(self.rng.uniform(0, 180)), base_r, float(self.rng.uniform(0.05, 0.5)), window, -1)
            )

    def build(self, scenario: str, seed: int, metadata: dict | None = None) -> SyntheticGrainSample:
        if not self.rows:
            raise ValueError("cannot build an empty sample")
        rows = self.rows
        signals = SignalTable(
            np.asarray([(x[0], x[1]) for x in rows], dtype=np.int64),
            np.asarray([x[2] for x in rows], dtype=float),
            np.asarray([x[3] for x in rows], dtype=float),
            np.asarray([x[4] for x in rows], dtype=float),
            np.asarray([x[5] for x in rows], dtype=np.int64),
            self.map_shape,
        )
        truth = np.asarray([x[6] for x in rows], dtype=np.int64)
        grain_ids = np.asarray(sorted(self.grain_extents), dtype=np.int64)
        grain_masks = np.stack([self.grain_extents[g] for g in grain_ids])
        grain_windows = np.asarray([self.grain_windows[g] for g in grain_ids], dtype=np.int64)
        return SyntheticGrainSample(
            signals,
            truth,
            scenario,
            seed,
            metadata or {},
            grain_mask_ids=grain_ids,
            grain_masks_truth=grain_masks,
            grain_windows=grain_windows,
        )


def _grid(shape):
    return np.mgrid[0 : shape[0], 0 : shape[1]]


def _circle(shape, center, radius):
    xx, yy = _grid(shape)
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2


def _ellipse(shape, center, radii, angle=0.0):
    xx, yy = _grid(shape)
    c, s = np.cos(angle), np.sin(angle)
    x = c * (xx - center[0]) + s * (yy - center[1])
    y = -s * (xx - center[0]) + c * (yy - center[1])
    return (x / max(radii[0], 1)) ** 2 + (y / max(radii[1], 1)) ** 2 <= 1


def _voronoi_masks(shape, centers):
    xx, yy = _grid(shape)
    dist = np.stack([(xx - x) ** 2 + (yy - y) ** 2 for x, y in centers])
    labels = np.argmin(dist, axis=0)
    return [labels == i for i in range(len(centers))]


def _irregular_blob(shape, rng, center, scale):
    noise = gaussian_filter(rng.normal(size=shape), sigma=max(scale / 4, 1))
    envelope = _ellipse(shape, center, (scale, scale * rng.uniform(0.6, 1.2)), rng.uniform(0, np.pi))
    threshold = np.quantile(noise[envelope], 0.35) if np.any(envelope) else 0
    return envelope & (noise >= threshold)


def _edge_dropout_map(mask, *, interior=0.02, edge=0.55, edge_depth=4.0):
    """Detection-dropout probability: dense core, increasingly incomplete edge."""
    distance = distance_transform_edt(mask)
    transition = np.clip(distance / max(float(edge_depth), 1e-6), 0, 1)
    probability = edge * (1 - transition) + interior * transition
    probability[~mask] = 0
    return probability


def _hole_dropout_map(mask, rng, *, n_holes=3, radius=3.0, background=0.03):
    probability = np.full(mask.shape, float(background))
    h, w = mask.shape
    for _ in range(int(n_holes)):
        center = (rng.uniform(0, h), rng.uniform(0, w))
        probability[_circle(mask.shape, center, radius * rng.uniform(0.6, 1.4))] = 0.95
    probability[~mask] = 0
    return probability


def _orientation_function(kind, base, shape, rng):
    if kind == "uniform":
        return float(base)
    if kind == "gradient":
        gx, gy = rng.uniform(-0.7, 0.7, size=2)
        return lambda x, y: base + gx * (x - shape[0] / 2) + gy * (y - shape[1] / 2)
    focal = np.array([rng.uniform(-shape[0], 2 * shape[0]), rng.uniform(-shape[1], 2 * shape[1])])
    return lambda x, y: np.degrees(np.arctan2(focal[0] - x, focal[1] - y)) % 180


def generate_canonical_sample(
    name: str, *, map_shape: tuple[int, int] = (64, 64), seed: int = 42
) -> SyntheticGrainSample:
    """Generate one deterministic named acceptance/visualization scenario."""
    if name not in CANONICAL_SCENARIOS:
        raise ValueError(f"unknown canonical scenario {name!r}")
    rng = np.random.default_rng(seed)
    b = _SignalBuilder(map_shape, rng)
    h, w = map_shape
    xx, yy = _grid(map_shape)
    full = np.ones(map_shape, bool)

    if name == "uniform_large":
        b.add_grain(full, theta=30)
    elif name == "bicrystal":
        b.add_grain(xx < h // 2, theta=20)
        b.add_grain(xx >= h // 2, theta=70)
    elif name == "trijunction":
        for mask, theta in zip(_voronoi_masks(map_shape, [(8, 8), (h - 8, 8), (h // 2, w - 8)]), (15, 70, 125)):
            b.add_grain(mask, theta=theta)
    elif name == "disconnected_equal_orientation":
        b.add_grain((xx < h // 3) & (yy < w // 3), theta=45)
        b.add_grain((xx >= 2 * h // 3) & (yy >= 2 * w // 3), theta=45)
    elif name == "continuous_bend":
        b.add_grain(full, theta=lambda x, y: 10 + 35 * x / max(h - 1, 1))
    elif name == "focal_flow":
        b.add_grain(full, theta=lambda x, y: np.degrees(np.arctan2(-h / 2 - x, w / 2 - y)) % 180)
    elif name == "orientation_cusp":
        flow = lambda x, y: np.degrees(np.arctan2(-h / 2 - x, w / 2 - y)) % 180
        b.add_grain(yy < w // 2, theta=flow)
        b.add_grain(yy >= w // 2, theta=lambda x, y: flow(x, y) + 45)
    elif name == "small_inclusion":
        inclusion = _circle(map_shape, (h // 2, w // 2), max(3, min(h, w) // 10))
        b.add_grain(~inclusion, theta=25)
        b.add_grain(inclusion, theta=95)
    elif name == "mixed_scale_voronoi":
        centers = [(8, 8), (h - 9, 10), (h // 2, w // 2), (10, w - 9), (h - 8, w - 8), (h // 2, 8)]
        for i, mask in enumerate(_voronoi_masks(map_shape, centers)):
            b.add_grain(mask, theta=15 + 25 * i)
    elif name == "dispersed_islands":
        for i, center in enumerate([(12, 12), (20, 44), (43, 18), (48, 48), (31, 31)]):
            center = (min(center[0], h - 2), min(center[1], w - 2))
            b.add_grain(_circle(map_shape, center, max(2, min(h, w) // 12)), theta=20 + 29 * i)
    elif name == "partial_overlap":
        b.add_grain((xx < 2 * h // 3) & (yy > w // 8), theta=15)
        b.add_grain((xx > h // 3) & (yy < 7 * w // 8), theta=85)
    elif name == "crossing_interpenetration":
        b.add_grain(np.abs(xx - h // 2) < max(2, h // 7), theta=0)
        b.add_grain(np.abs(yy - w // 2) < max(2, w // 7), theta=90)
    elif name == "triple_overlap":
        radius = max(4, min(h, w) // 4)
        for center, theta in [((h * 2 // 5, w * 2 // 5), 10), ((h * 3 // 5, w * 2 // 5), 70), ((h // 2, w * 3 // 5), 130)]:
            b.add_grain(_circle(map_shape, center, radius), theta=theta)
    elif name == "equal_orientation_distinct_radius":
        b.add_grain(full, theta=30, radius=0.115)
        b.add_grain(full, theta=30, radius=0.145)
    elif name == "continuous_radius_drift":
        b.add_grain(full, theta=35, radius=lambda x, y: 0.12 * (0.9 + 0.2 * x / max(h - 1, 1)))
    elif name == "dropout":
        b.add_grain(full, theta=40, dropout=0.30)
    elif name == "dense_interior_noisy_edges":
        extent = _ellipse(map_shape, (h / 2, w / 2), (0.42 * h, 0.36 * w), angle=0.35)
        b.add_grain(extent, theta=42, dropout=_edge_dropout_map(extent), theta_noise=2.0)
    elif name == "dense_interior_with_holes":
        extent = _ellipse(map_shape, (h / 2, w / 2), (0.42 * h, 0.36 * w), angle=-0.25)
        dropout = _hole_dropout_map(extent, rng, n_holes=4, radius=max(2, min(h, w) / 14))
        b.add_grain(extent, theta=lambda x, y: 55 + 0.2 * x, dropout=dropout, theta_noise=1.5)
    elif name == "sparse_smooth_percolation":
        # Latent grain is filled, but only a sparse random support is observed. A method
        # should distinguish successful signal association from footprint reconstruction.
        b.add_grain(full, theta=lambda x, y: 25 + 0.1 * x, dropout=0.82, theta_noise=1.0)
    elif name == "radius2_false_bridge":
        left = (xx < h // 2 - 2) & (yy > w // 8) & (yy < 7 * w // 8)
        right = (xx > h // 2 + 2) & (yy > w // 8) & (yy < 7 * w // 8)
        b.add_grain(left, theta=45, dropout=_edge_dropout_map(left, interior=0.05, edge=0.35))
        b.add_grain(right, theta=45, dropout=_edge_dropout_map(right, interior=0.05, edge=0.35))
        # Unlabelled, same-feature detections form a radius-2 chain through the true gap.
        for x in range(h // 2 - 1, h // 2 + 2):
            b.rows.append((x, w // 2, 45.0, 0.13, 0.2, 0, -1))
    elif name == "broad_continuous_gradient":
        b.add_grain(full, theta=lambda x, y: 5 + 100 * x / max(h - 1, 1), theta_noise=1.0)
    elif name == "false_positive_outliers":
        b.add_grain((xx < 3 * h // 4) & (yy < 3 * w // 4), theta=55)
        b.add_outliers(max(10, h * w // 100))
    elif name == "multi_window_mixture":
        b.add_grain(full, theta=20, radius=0.03, window=0)
        b.add_grain(_circle(map_shape, (h // 2, w // 2), min(h, w) // 3), theta=80, radius=0.13, window=1)
        b.add_grain(yy > w // 3, theta=135, radius=0.26, window=2)

    return b.build(name, seed, {"kind": "canonical", "schema_version": SCHEMA_VERSION})


def generate_canonical_suite(
    *, map_shape: tuple[int, int] = (64, 64), seed: int = 42
) -> list[SyntheticGrainSample]:
    return [
        generate_canonical_sample(name, map_shape=map_shape, seed=seed + index)
        for index, name in enumerate(CANONICAL_SCENARIOS)
    ]


def _random_masks(shape, morphology, scale_regime, rng):
    h, w = shape
    if scale_regime == "large":
        n = int(rng.integers(1, 5))
    elif scale_regime == "small":
        n = int(rng.integers(10, 31))
    else:
        n = int(rng.integers(5, 16))
    centers = np.column_stack([rng.uniform(0, h, n), rng.uniform(0, w, n)])

    if morphology in {"voronoi", "junctions"}:
        if morphology == "junctions":
            if len(centers) < 2:
                centers = np.vstack([centers, [rng.uniform(0, h), rng.uniform(0, w)]])
            count = int(rng.integers(2, min(4, len(centers)) + 1))
            centers = centers[:count]
        return _voronoi_masks(shape, centers)
    if morphology == "bands":
        xx, yy = _grid(shape)
        angle = rng.uniform(0, np.pi)
        coordinate = xx * np.cos(angle) + yy * np.sin(angle)
        edges = np.linspace(coordinate.min(), coordinate.max(), n + 1)
        return [(coordinate >= edges[i]) & (coordinate <= edges[i + 1]) for i in range(n)]
    if morphology == "flow":
        return [np.ones(shape, bool)]
    if morphology == "nested":
        masks = []
        occupied = np.zeros(shape, bool)
        for i, center in enumerate(centers):
            radius = max(2, min(h, w) * (0.07 if scale_regime == "small" else 0.16))
            mask = _circle(shape, center, radius * rng.uniform(0.7, 1.3)) & ~occupied
            occupied |= mask
            masks.append(mask)
        masks.insert(0, ~occupied)
        return masks
    if morphology == "islands":
        radius = max(2, min(h, w) * (0.045 if scale_regime == "small" else 0.12))
        return [_circle(shape, center, radius * rng.uniform(0.6, 1.4)) for center in centers]
    if morphology == "blobs":
        scale = max(3, min(h, w) * (0.08 if scale_regime == "small" else 0.22))
        return [_irregular_blob(shape, rng, center, scale * rng.uniform(0.6, 1.4)) for center in centers]
    # Interpenetrating ellipses intentionally overlap before the overlap modifier.
    radii = (max(3, h / 3), max(2, w / 8))
    return [_ellipse(shape, center, radii, rng.uniform(0, np.pi)) for center in centers[: max(2, min(n, 6))]]


def generate_random_sample(
    index: int, *, map_shape: tuple[int, int] = (64, 64), seed: int = 42
) -> SyntheticGrainSample:
    """Generate an order-independent, stratified randomized scenario."""
    morphologies = ("voronoi", "blobs", "nested", "bands", "flow", "junctions", "islands", "interpenetrating")
    scales = ("small", "large", "mixed")
    overlaps = ("none", "partial", "high", "interpenetrating")
    orientation_kinds = ("uniform", "gradient", "curved")
    detection_kinds = ("clean", "dropout", "edge_dropout", "interior_holes", "noisy", "outliers")
    morphology = morphologies[index % len(morphologies)]
    scale_regime = scales[(index // len(morphologies)) % len(scales)]
    overlap_regime = overlaps[(index // (len(morphologies) * len(scales))) % len(overlaps)]
    orientation_kind = orientation_kinds[(index // 3) % len(orientation_kinds)]
    detection_kind = detection_kinds[(index // 5) % len(detection_kinds)]

    child_seed = int(np.random.SeedSequence([seed, int(index)]).generate_state(1)[0])
    rng = np.random.default_rng(child_seed)
    b = _SignalBuilder(map_shape, rng)
    masks = _random_masks(map_shape, morphology, scale_regime, rng)

    if overlap_regime in {"partial", "high"}:
        iterations = 1 if overlap_regime == "partial" else max(2, min(map_shape) // 16)
        masks = [binary_dilation(mask, iterations=iterations) for mask in masks]
    elif overlap_regime == "interpenetrating" and morphology != "interpenetrating":
        h, w = map_shape
        masks.extend(
            [
                _ellipse(map_shape, (h / 2, w / 2), (h / 2.5, max(2, w / 10)), rng.uniform(0, np.pi)),
                _ellipse(map_shape, (h / 2, w / 2), (h / 2.5, max(2, w / 10)), rng.uniform(0, np.pi)),
            ]
        )

    n_windows = int(rng.integers(1, 4))
    radius_centers = (0.03, 0.13, 0.26)
    dropout = rng.uniform(0.08, 0.30) if detection_kind == "dropout" else 0.0
    theta_noise = rng.uniform(1.0, 5.0) if detection_kind == "noisy" else rng.uniform(0, 1.0)
    radius_noise = rng.uniform(0.01, 0.04) if detection_kind == "noisy" else rng.uniform(0, 0.01)
    intensity_noise = rng.uniform(0.2, 0.6)

    for grain_index, mask in enumerate(masks):
        if not np.any(mask):
            continue
        window = int(rng.integers(0, n_windows))
        base_theta = float(rng.uniform(0, 180))
        theta = _orientation_function(orientation_kind, base_theta, map_shape, rng)
        radial_mode = (index + grain_index) % 3
        base_radius = radius_centers[window]
        if radial_mode == 0:
            radius = base_radius
        elif radial_mode == 1:
            slope = rng.uniform(-0.08, 0.08)
            radius = lambda x, y, br=base_radius, sl=slope: br * (1 + sl * (x / max(map_shape[0] - 1, 1) - 0.5))
        else:
            radius = base_radius * rng.choice([0.88, 1.0, 1.12])
        base_intensity = float(np.exp(rng.uniform(np.log(0.08), np.log(1.0))))
        gx, gy = rng.uniform(-0.4, 0.4, size=2)
        intensity = lambda x, y, bi=base_intensity, ax=gx, ay=gy: bi * np.exp(
            ax * (x / max(map_shape[0] - 1, 1) - 0.5)
            + ay * (y / max(map_shape[1] - 1, 1) - 0.5)
        )
        spatial_dropout = dropout
        if detection_kind == "edge_dropout":
            spatial_dropout = _edge_dropout_map(
                mask,
                interior=rng.uniform(0.0, 0.05),
                edge=rng.uniform(0.35, 0.75),
                edge_depth=rng.uniform(2.0, 6.0),
            )
        elif detection_kind == "interior_holes":
            spatial_dropout = _hole_dropout_map(
                mask,
                rng,
                n_holes=int(rng.integers(1, 6)),
                radius=rng.uniform(1.5, max(2.0, min(map_shape) / 10)),
            )
        b.add_grain(
            mask,
            theta=theta,
            radius=radius,
            intensity=intensity,
            window=window,
            dropout=spatial_dropout,
            theta_noise=theta_noise,
            radius_noise_rel=radius_noise,
            intensity_noise=intensity_noise,
        )

    if detection_kind == "outliers":
        b.add_outliers(max(5, int(np.prod(map_shape) * rng.uniform(0.002, 0.02))), windows=n_windows)

    metadata = {
        "kind": "random",
        "schema_version": SCHEMA_VERSION,
        "index": int(index),
        "child_seed": child_seed,
        "morphology": morphology,
        "scale_regime": scale_regime,
        "overlap_regime": overlap_regime,
        "orientation_heterogeneity": orientation_kind,
        "detection_regime": detection_kind,
        "n_windows": n_windows,
    }
    return b.build(f"random_{index:06d}", child_seed, metadata)


def save_sample(sample: SyntheticGrainSample, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pos=sample.signals.pos,
        theta=sample.signals.theta,
        r=sample.signals.r,
        intensity=sample.signals.intensity,
        window=sample.signals.window,
        map_shape=np.asarray(sample.signals.map_shape, dtype=np.int64),
        ground_truth=sample.ground_truth,
        scenario=np.asarray(sample.scenario),
        seed=np.asarray(sample.seed, dtype=np.uint64),
        metadata_json=np.asarray(json.dumps(sample.metadata, default=_as_jsonable)),
        grain_mask_ids=sample.grain_mask_ids,
        grain_masks_truth=sample.grain_masks_truth,
        grain_windows=sample.grain_windows,
    )
    return path


def load_sample(path: str | Path) -> SyntheticGrainSample:
    with np.load(path, allow_pickle=False) as data:
        signals = SignalTable(
            data["pos"], data["theta"], data["r"], data["intensity"], data["window"], tuple(data["map_shape"])
        )
        return SyntheticGrainSample(
            signals,
            data["ground_truth"],
            str(data["scenario"].item()),
            int(data["seed"].item()),
            json.loads(str(data["metadata_json"].item())),
            grain_mask_ids=data["grain_mask_ids"],
            grain_masks_truth=data["grain_masks_truth"],
            grain_windows=data["grain_windows"],
        )


def save_quicklook(sample: SyntheticGrainSample, path: str | Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    primary = sample.primary_label_map()
    membership = sample.membership_count_map()
    observed_membership = sample.observed_membership_count_map()
    orientation = np.zeros((*sample.signals.map_shape, 3), float)
    grain_mean = {row["grain_id"]: row["theta_mean_deg"] for row in sample.grain_table()}
    assigned = primary >= 0
    if np.any(assigned):
        means = np.asarray([grain_mean[int(label)] for label in primary[assigned]])
        orientation[assigned] = orientation_to_rgb(means)
    sizes = {row["grain_id"]: row["n_signals"] for row in sample.grain_table()}
    size_map = np.full(sample.signals.map_shape, np.nan)
    if np.any(assigned):
        size_map[assigned] = [sizes[int(label)] for label in primary[assigned]]
    overlap = membership >= 2

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    panels = [
        (orientation, "primary mean orientation", None, None),
        (membership, "latent memberships/probe", "viridis", None),
        (observed_membership, "observed memberships/probe", "viridis", None),
        (overlap, "latent overlap mask (>=2)", "gray", None),
        (size_map, "primary grain size", "magma", LogNorm(vmin=1, vmax=max(sizes.values(), default=1))),
    ]
    for ax, (image, title, cmap, norm) in zip(axes, panels):
        if cmap is None:
            ax.imshow(image, origin="upper", interpolation="nearest")
        else:
            ax.imshow(image, origin="upper", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(sample.scenario)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _sample_manifest_row(sample, path, split):
    membership = sample.membership_count_map()
    observed_membership = sample.observed_membership_count_map()
    latent_occupied = membership > 0
    observed_occupied = observed_membership > 0
    return {
        "scenario": sample.scenario,
        "split": split,
        "path": str(path),
        "seed": int(sample.seed),
        "map_shape": list(sample.signals.map_shape),
        "n_signals": int(len(sample.signals)),
        "n_grains": int(len(sample.grain_ids)),
        "n_windows": int(len(np.unique(sample.signals.window))),
        "overlap_probes": int(np.count_nonzero(membership >= 2)),
        "latent_occupied_probes": int(np.count_nonzero(latent_occupied)),
        "observed_occupied_probes": int(np.count_nonzero(observed_occupied)),
        "observed_coverage_fraction": float(
            np.count_nonzero(observed_occupied & latent_occupied) / np.count_nonzero(latent_occupied)
        )
        if np.any(latent_occupied)
        else 0.0,
        "noise_signals": int(np.count_nonzero(sample.ground_truth < 0)),
        "grains": sample.grain_table(),
        "tags": sample.metadata,
    }


def generate_dataset(
    output_dir: str | Path,
    *,
    num_random: int = 1000,
    map_shape: tuple[int, int] = (64, 64),
    seed: int = 42,
    quicklooks: str = "canonical",
    resume: bool = False,
) -> dict:
    """Generate or safely resume the canonical suite and 70/15/15 random corpus."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if quicklooks not in {"none", "canonical", "all"}:
        raise ValueError("quicklooks must be one of: none, canonical, all")
    existing_manifest_path = output_dir / "manifest.json"
    if resume and existing_manifest_path.exists():
        existing = json.loads(existing_manifest_path.read_text())
        requested = (int(seed), list(map_shape), int(num_random))
        recorded = (existing.get("seed"), existing.get("map_shape"), existing.get("num_random"))
        if recorded != requested:
            raise ValueError(
                "resume configuration does not match existing manifest: "
                f"requested={requested}, recorded={recorded}"
            )

    def obtain_sample(relative_path, generate, expected_scenario):
        path = output_dir / relative_path
        if resume and path.exists():
            sample = load_sample(path)
            if sample.scenario != expected_scenario or sample.signals.map_shape != tuple(map_shape):
                raise ValueError(f"existing sample does not match requested corpus: {path}")
            resume_stats["reused"] += 1
            return sample
        sample = generate()
        save_sample(sample, path)
        resume_stats["generated"] += 1
        return sample

    rows = []
    resume_stats = {"reused": 0, "generated": 0}
    for canonical_index, scenario in enumerate(CANONICAL_SCENARIOS):
        rel = Path("samples") / "canonical" / f"{scenario}.npz"
        sample = obtain_sample(
            rel,
            lambda scenario=scenario, canonical_index=canonical_index: generate_canonical_sample(
                scenario, map_shape=map_shape, seed=seed + canonical_index
            ),
            scenario,
        )
        if quicklooks in {"canonical", "all"}:
            quicklook_path = output_dir / "quicklooks" / "canonical" / f"{sample.scenario}.png"
            if not (resume and quicklook_path.exists()):
                save_quicklook(sample, quicklook_path)
        rows.append(_sample_manifest_row(sample, rel, "canonical"))

    split_rng = np.random.default_rng(seed)
    order = split_rng.permutation(num_random)
    n_cal = int(round(0.70 * num_random))
    n_val = int(round(0.15 * num_random))
    split_for = {}
    for rank, index in enumerate(order):
        split_for[int(index)] = "calibration" if rank < n_cal else "validation" if rank < n_cal + n_val else "test"
    for index in range(num_random):
        split = split_for[index]
        rel = Path("samples") / split / f"sample_{index:06d}.npz"
        scenario = f"random_{index:06d}"
        sample = obtain_sample(
            rel,
            lambda index=index: generate_random_sample(index, map_shape=map_shape, seed=seed),
            scenario,
        )
        if quicklooks == "all":
            quicklook_path = output_dir / "quicklooks" / split / f"sample_{index:06d}.png"
            if not (resume and quicklook_path.exists()):
                save_quicklook(sample, quicklook_path)
        rows.append(_sample_manifest_row(sample, rel, split))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "quantem.diffraction.grain_clustering_synthetic",
        "seed": int(seed),
        "map_shape": list(map_shape),
        "num_canonical": len(CANONICAL_SCENARIOS),
        "num_random": int(num_random),
        "splits": {
            name: sum(row["split"] == name for row in rows)
            for name in ("canonical", "calibration", "validation", "test")
        },
        "samples": rows,
        "last_run": resume_stats,
    }
    manifest_tmp = output_dir / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, default=_as_jsonable))
    manifest_tmp.replace(output_dir / "manifest.json")
    return manifest
