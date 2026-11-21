import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

PEAK_DTYPE = np.dtype([("x", float), ("y", float), ("intensity", float)])


def _local_maxima(array, min_spacing=0.0, max_num_peaks=None, edge=1, subpixel=True):
    """8-neighbor maxima, brightest first, spacing-pruned, parabola-refined."""
    greater = (
        (array >= np.roll(array, (-1, 0), axis=(0, 1)))
        & (array > np.roll(array, (1, 0), axis=(0, 1)))
        & (array >= np.roll(array, (0, -1), axis=(0, 1)))
        & (array > np.roll(array, (0, 1), axis=(0, 1)))
        & (array >= np.roll(array, (-1, -1), axis=(0, 1)))
        & (array > np.roll(array, (-1, 1), axis=(0, 1)))
        & (array >= np.roll(array, (1, -1), axis=(0, 1)))
        & (array > np.roll(array, (1, 1), axis=(0, 1)))
    )
    greater[:edge, :] = greater[-edge:, :] = False
    greater[:, :edge] = greater[:, -edge:] = False

    rows, cols = np.nonzero(greater)
    order = np.argsort(array[rows, cols])[::-1]
    rows, cols = rows[order], cols[order]

    if min_spacing > 0:
        drop = np.zeros(len(rows), dtype=bool)
        for i in range(len(rows)):
            if not drop[i]:
                close = (rows - rows[i]) ** 2 + (cols - cols[i]) ** 2 < min_spacing**2
                close[: i + 1] = False
                drop[close] = True
        rows, cols = rows[~drop], cols[~drop]
    if max_num_peaks is not None:
        rows, cols = rows[:max_num_peaks], cols[:max_num_peaks]

    peaks = np.zeros(len(rows), dtype=PEAK_DTYPE)
    peaks["x"], peaks["y"] = rows, cols
    peaks["intensity"] = array[rows, cols]
    if not subpixel:
        return peaks

    center = array[rows, cols].astype(float)
    for axis, field in ((0, "x"), (1, "y")):
        step = np.zeros(2, dtype=int)
        step[axis] = 1
        low = array[rows - step[0], cols - step[1]].astype(float)
        high = array[rows + step[0], cols + step[1]].astype(float)
        peaks[field] = peaks[field] + (high - low) / (4 * center - 2 * high - 2 * low)
    peaks["intensity"] = _bilinear(array, peaks["x"], peaks["y"])
    return peaks


def _bilinear(array, x, y):
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    x1, y1 = np.ceil(x).astype(int), np.ceil(y).astype(int)
    return (
        (1 - dx) * (1 - dy) * array[x0, y0]
        + (1 - dx) * dy * array[x0, y1]
        + dx * (1 - dy) * array[x1, y0]
        + dx * dy * array[x1, y1]
    )


def _basis_from_fft(image, num_peaks=20, center_buffer=15, min_spacing=5):
    """Lattice vectors from the two shortest linearly independent Bragg peaks."""
    power = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    peaks = _local_maxima(power, min_spacing=min_spacing, max_num_peaks=num_peaks)
    center = np.array(image.shape) // 2

    offset = np.stack([peaks["x"] - center[0], peaks["y"] - center[1]], axis=1)
    peaks = peaks[np.linalg.norm(offset, axis=1) > center_buffer]
    offset = np.stack([peaks["x"] - center[0], peaks["y"] - center[1]], axis=1)
    order = np.argsort((offset**2).sum(1))
    peaks, offset = peaks[order], offset[order]

    # collinear peaks cross to ~0, so threshold off the smallest cross product
    cross = offset[0, 0] * offset[1:, 1] - offset[0, 1] * offset[1:, 0]
    threshold = 5 * (np.abs(cross).min() + 0.1)
    if not (np.abs(cross) > threshold).any():
        threshold = 2 * (np.abs(cross).min() + 0.1)
    k = np.stack([offset[0], offset[1 + int(np.argmax(np.abs(cross) > threshold))]])
    return np.linalg.inv(k / np.array(image.shape)).T


def _third_vector(u, v):
    """The third nearest-neighbor direction: u + v on an obtuse basis, u - v on acute."""
    cos_uv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    if np.rad2deg(np.arccos(np.clip(cos_uv, -1.0, 1.0))) > 90.0:
        return u + v, 1
    return u - v, -1


def _index_lattice(positions, seed_index, basis, w_sign, tolerance_uvw,
                   recover_duplicates=True):
    """Flood-fill (a, b) indices out from a seed along +-u, +-v, +-w.

    Returns integer indices and a state per candidate: 1 accepted, 2 rejected as
    a duplicate (a, b), 3 recovered by a second pass from the accepted set, 0
    never reached.
    """
    tree = cKDTree(positions)
    radius = float(np.mean(np.linalg.norm(basis, axis=1)) * (tolerance_uvw - 1))
    # (row, col) increments in (a, b) for a step along +u, +v, +w
    steps = [(vector, (int(i == 0) + int(i == 2), int(i == 1) + w_sign * int(i == 2)))
             for i, vector in enumerate(basis)]

    state = np.zeros(len(positions), dtype=int)
    ab = np.zeros((len(positions), 2), dtype=int)
    state[seed_index] = 1
    frontier = np.zeros(len(positions), dtype=bool)
    frontier[seed_index] = True
    settled = frontier.copy()

    def nearest(source, vector, sign):
        """Index of the candidate closest to source + sign*vector, or None."""
        target = positions[source] + sign * np.asarray(vector)
        for distance, index in zip(*tree.query(target, k=2)):
            if index != source:
                return int(index) if distance < radius else None
        return None

    while frontier.any():
        found = np.zeros(len(positions), dtype=bool)
        for source in np.flatnonzero(frontier):
            for vector, (da, db) in steps:
                for sign in (-1, 1):
                    index = nearest(source, vector, sign)
                    if index is not None and state[index] == 0:
                        state[index] = 1
                        ab[index] = ab[source] + sign * np.array([da, db])
                        found[index] = True
        # same (a, b) reached from two directions: keep the settled one
        accepted = np.flatnonzero(state == 1)
        groups = {}
        for index in accepted:
            groups.setdefault(tuple(ab[index]), []).append(index)
        for group in groups.values():
            if len(group) < 2:
                continue
            prior = [i for i in group if settled[i]]
            if len(prior) > 1:
                if seed_index not in group:
                    raise ValueError("two settled candidates share an (a, b) index")
                keep = seed_index
            else:
                keep = prior[0] if prior else [i for i in group if found[i]][0]
            for index in group:
                if index != keep:
                    state[index], ab[index] = 2, 0
                    settled[index] = found[index] = False
        if not found.any():
            break
        settled |= found
        frontier = found

    if recover_duplicates:
        for source in np.flatnonzero(state == 1):
            for vector, (da, db) in steps:
                for sign in (-1, 1):
                    index = nearest(source, vector, sign)
                    if index is not None and state[index] == 2:
                        state[index] = 3
                        ab[index] = ab[source] + sign * np.array([da, db])
    return ab, state


def _disk(image, x, y, radius):
    """Pixels of `image` within `radius` of (x, y): coords, values, patch, mask."""
    height, width = image.shape
    reach = int(np.ceil(radius))
    i0, i1 = max(0, int(x) - reach), min(height - 1, int(x) + reach)
    j0, j1 = max(0, int(y) - reach), min(width - 1, int(y) + reach)
    patch = image[i0:i1 + 1, j0:j1 + 1]
    rows = np.arange(i0, i1 + 1)[:, None] * np.ones(patch.shape[1], dtype=int)
    cols = np.ones((patch.shape[0], 1), dtype=int) * np.arange(j0, j1 + 1)[None, :]
    inside = (rows - x) ** 2 + (cols - y) ** 2 <= radius**2
    return (rows[inside].ravel().astype(float), cols[inside].ravel().astype(float),
            patch[inside].ravel(), patch, inside)


def _solve_gaussian(gx, gy, vals, guess, bounds, max_nfev):
    """Isotropic Gaussian + constant background, soft_l1 / trf, bounded."""
    def residual(theta):
        x_c, y_c, amp, sigma, bg = theta
        rr = (gx - x_c) ** 2 + (gy - y_c) ** 2
        return amp * np.exp(-0.5 * rr / max(sigma, 1e-6) ** 2) + bg - vals
    return least_squares(residual, np.clip(guess, *bounds), bounds=bounds,
                         method="trf", loss="soft_l1", max_nfev=int(max_nfev),
                         xtol=1e-6, ftol=1e-6, gtol=1e-6)


def _fit_one(image, x0, y0, radius, max_move, anchor, max_nfev):
    """One bounded Gaussian on the disk about (x0, y0), bounds anchored on `anchor`."""
    height, width = image.shape
    gx, gy, vals, patch, inside = _disk(image, x0, y0, radius)
    if not vals.size:
        return None
    low, high = vals.min(), vals.max()
    bg = np.median(patch[~inside]) if (~inside).any() else np.median(patch)
    amp = max(image[int(x0), int(y0)] - bg, 1e-6)
    x_a, y_a = anchor
    bounds = (
        np.array([max(x_a - max_move, 0.0), max(y_a - max_move, 0.0),
                  0.0, 0.25, low - (high - low)]),
        np.array([min(x_a + max_move, height - 1.0), min(y_a + max_move, width - 1.0),
                  max(high - low, amp * 4.0), max(2.0 * radius, 1.0), high + (high - low)]),
    )
    return _solve_gaussian(gx, gy, vals, [x0, y0, amp, max(radius * 0.5, 0.5), bg],
                           bounds, max_nfev).x


def _refine_gaussian(image, x, y, fit_radius, max_move, max_nfev=200,
                     window_sigma=None, window_bounds=None, window_iters=5,
                     window_tol=0.01):
    """One isotropic 2-D Gaussian per site.

    With `window_sigma` set, the disk is re-derived from the site's own fitted
    sigma and the fit repeated until the radius moves by less than `window_tol`;
    `window_bounds` clips it, and the railed count is returned.
    """
    out = np.empty((len(x), 5))
    out[:, 0], out[:, 1] = x, y
    out[:, 2:] = np.nan
    iterations = np.zeros(len(x), dtype=int)
    railed = np.zeros(len(x), dtype=bool)

    for site, (x0, y0) in enumerate(zip(x, y)):
        anchor, radius = (x0, y0), float(fit_radius)
        theta = _fit_one(image, x0, y0, radius, max_move, anchor, max_nfev)
        if theta is None:
            continue
        out[site], iterations[site] = theta, 1
        if window_sigma is None:
            continue
        for _ in range(int(window_iters) - 1):
            wanted = window_sigma * theta[3]
            new_radius = wanted if window_bounds is None else float(
                np.clip(wanted, window_bounds[0], window_bounds[1]))
            if abs(new_radius - radius) <= window_tol * radius:
                break
            radius = new_radius
            theta = _fit_one(image, theta[0], theta[1], radius, max_move, anchor, max_nfev)
            if theta is None:
                break
            out[site], iterations[site] = theta, iterations[site] + 1
        railed[site] = window_bounds is not None and (
            window_sigma * out[site, 3] > window_bounds[1] * (1 + 1e-9)
            or window_sigma * out[site, 3] < window_bounds[0] * (1 - 1e-9))
    return out, iterations, railed


def fit_a_sites(image, px_nm, min_spacing_frac=0.70, tolerance_uvw=1.3,
                fit_radius_frac=0.40, max_move_frac=0.15, edge_margin_frac=0.5,
                sigma_min_frac=0.045, min_amplitude_frac=0.35,
                amplitude_radius_frac=3.0, max_nfev=200, max_num_peaks=20000,
                recover_duplicates=True, window_sigma=None, window_bounds_frac=(0.06, 0.50)):
    """Peaks first, indexed, then a 2-D Gaussian each. Returns a dict of arrays."""
    image = np.asarray(image, dtype=float)
    if np.isfinite(image.max()) and image.max() > 0:
        image = image / image.max()
    height, width = image.shape
    center = np.array(image.shape) // 2

    u, v = _basis_from_fft(image)
    w, w_sign = _third_vector(u, v)
    nn_px = float(min(np.linalg.norm(u), np.linalg.norm(v)))

    edge = max(1, int(edge_margin_frac * nn_px / 3))
    peaks = _local_maxima(image, min_spacing=min_spacing_frac * nn_px,
                          max_num_peaks=max_num_peaks, edge=1)
    keep = ((peaks["x"] - edge >= 0) & (peaks["x"] + edge <= height - 1)
            & (peaks["y"] - edge >= 0) & (peaks["y"] + edge <= width - 1))
    if not keep.any():
        raise ValueError("no peak candidates survived the edge margin")
    peaks = peaks[keep]
    positions = np.stack([peaks["x"], peaks["y"]], axis=1)

    half = 60
    smoothed = gaussian_filter(image, 2.0)[center[0] - half:center[0] + half,
                                           center[1] - half:center[1] + half]
    seed = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    seed = np.array([center[0] - half + seed[0], center[1] - half + seed[1]], dtype=float)
    seed_index = int(np.argmin(np.linalg.norm(positions - seed, axis=1)))

    ab, state = _index_lattice(positions, seed_index, np.stack([u, v, w]), w_sign,
                               tolerance_uvw, recover_duplicates=recover_duplicates)
    used = np.isin(state, (1, 3))
    fitted, iterations, railed = _refine_gaussian(
        image, positions[used, 0], positions[used, 1],
        fit_radius=fit_radius_frac * nn_px, max_move=max_move_frac * nn_px,
        max_nfev=max_nfev, window_sigma=window_sigma,
        window_bounds=None if window_sigma is None else
        (window_bounds_frac[0] * nn_px, window_bounds_frac[1] * nn_px))

    out = dict(x=fitted[:, 0], y=fitted[:, 1], amplitude=fitted[:, 2],
               sigma=fitted[:, 3], background=fitted[:, 4],
               a=ab[used, 0], b=ab[used, 1], nn_px=nn_px, px_nm=float(px_nm),
               iterations=iterations, window_railed=railed)

    bad = out["sigma"] <= sigma_min_frac * nn_px
    bad |= ~(np.isfinite(out["x"]) & np.isfinite(out["y"]))
    if min_amplitude_frac is not None and (~bad).any():
        points = np.stack([out["x"], out["y"]], axis=1)
        tree = cKDTree(points)
        reference = np.array([
            np.median(out["amplitude"][
                [j for j in tree.query_ball_point(point, amplitude_radius_frac * nn_px)
                 if not bad[j]] or [i]])
            for i, point in enumerate(points)])
        bad |= (~bad) & (out["amplitude"] < min_amplitude_frac * reference)
    out["good"] = ~bad
    out["third"] = (1, 1) if np.linalg.norm(u + v) < np.linalg.norm(u - v) else (1, -1)
    return out


def refit_adaptive(fit, image, n_sigma=2.0, iterations=3, max_move_frac=0.175,
                   min_radius_px=2.0, max_radius_frac=None, max_nfev=200, verbose=True):
    """Re-fit every accepted site with a disk of `n_sigma` x *its own* sigma."""
    image = np.asarray(image, dtype=float)
    height, width = image.shape
    nn_px = float(fit["nn_px"])
    max_move = max_move_frac * nn_px
    max_radius = np.inf if max_radius_frac is None else max_radius_frac * nn_px

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}
    out["radius_used"] = np.full(len(fit["x"]), np.nan)
    n_failed = 0

    for j in np.flatnonzero(fit["good"]):
        x_a, y_a = float(fit["x"][j]), float(fit["y"][j])
        xc, yc, amp, sigma, bg = x_a, y_a, float(fit["amplitude"][j]), \
            float(fit["sigma"][j]), float(fit["background"][j])
        converged = False
        for _ in range(iterations):
            radius = float(np.clip(n_sigma * sigma, min_radius_px, max_radius))
            gx, gy, vals, _, _ = _disk(image, xc, yc, radius)
            if vals.size < 8:
                break
            span, offset = float(vals.max() - vals.min()), float(vals.min())
            if span <= 0:
                break
            vals = (vals - offset) / span
            amp_n, bg_n = amp / span, (bg - offset) / span
            bounds = (
                np.array([max(x_a - max_move, 0.0), max(y_a - max_move, 0.0), 0.0, 0.25, -1.0]),
                np.array([min(x_a + max_move, height - 1.0), min(y_a + max_move, width - 1.0),
                          max(1.0, 4 * amp_n), max(2.0 * radius, 1.0), 2.0]),
            )
            theta = _solve_gaussian(gx, gy, vals, [xc, yc, amp_n, sigma, bg_n],
                                    bounds, max_nfev).x
            xc, yc, amp_n, sigma, bg_n = theta
            amp, bg = amp_n * span, bg_n * span + offset
            out["radius_used"][j] = radius
            converged = True
        if not converged:
            out["good"][j] = False
            n_failed += 1
            continue
        out["x"][j], out["y"][j], out["sigma"][j] = xc, yc, sigma
        out["amplitude"][j], out["background"][j] = amp, bg

    if verbose:
        print(f"  adaptive {n_sigma:.1f}sigma x{iterations}: median radius "
              f"{np.nanmedian(out['radius_used']):.2f} px, median sigma "
              f"{np.median(out['sigma'][out['good']]):.2f} px, {n_failed} failed")
    return out


def bond_statistics(fit):
    good = fit["good"]
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(fit["a"], fit["b"])) if good[i]}
    scale = fit["px_nm"] * 1000.0
    clouds, lengths = [], []
    third = fit["third"]
    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1), third, (-third[0], -third[1])):
        vectors = [(np.array([fit["x"][j] - fit["x"][i], fit["y"][j] - fit["y"][i]]) * scale)
                   for (a, b), i in index.items()
                   if (j := index.get((a + da, b + db))) is not None]
        if not vectors:
            continue
        vectors = np.asarray(vectors)

        keep = np.ones(len(vectors), dtype=bool)
        for _ in range(3):
            center = vectors[keep].mean(0)
            radius = np.linalg.norm(vectors - center, axis=1)
            rms = np.sqrt(np.mean(radius[keep] ** 2))
            new_keep = radius < 4.0 * rms
            if new_keep.sum() < 16 or (new_keep == keep).all():
                break
            keep = new_keep
        vectors = vectors[keep]
        mean = vectors.mean(0)
        clouds.append(dict(vectors=vectors, mean=mean, deviations=vectors - mean,
                           sigma_2d=float(np.sqrt(((vectors - mean) ** 2).sum(1).mean()))))
        lengths.append(np.linalg.norm(vectors, axis=1))
    return dict(clouds=clouds,
                mean_length=float(np.concatenate(lengths).mean()),
                sigma_mean=float(np.mean([c["sigma_2d"] for c in clouds])),
                deviations=np.concatenate([c["deviations"] for c in clouds]))

def cloud_ellipse(cloud, n_sigma=2.0, ellipse_scale=5.0):
    dev = cloud["deviations"]
    values, vectors = np.linalg.eigh(np.cov(dev[:, 1], dev[:, 0]))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    semi = n_sigma * ellipse_scale * np.sqrt(np.maximum(values, 0))
    return (cloud["mean"][::-1], 2 * semi, np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])),
            semi, vectors)

def cloud_half_extent(stats, **kw):
    out = 0.0
    for cloud in stats["clouds"]:
        _, _, _, semi, vectors = cloud_ellipse(cloud, **kw)
        out = max(out,
                  abs(cloud["mean"][1]) + np.hypot(semi[0] * vectors[0, 0], semi[1] * vectors[0, 1]),
                  abs(cloud["mean"][0]) + np.hypot(semi[0] * vectors[1, 0], semi[1] * vectors[1, 1]))
    return out
