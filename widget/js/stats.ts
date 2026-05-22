/** Find min/max range of a Float32Array, filtering out NaN and Infinity. */
export function findDataRange(data: Float32Array): { min: number; max: number } {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  // If no finite values found, return zeros
  if (min === Infinity) return { min: 0, max: 0 };
  return { min, max };
}

/** Signed log1p. For non-negative inputs identical to log1p(x); for negatives
 *  returns -log1p(|x|) so diff_mode frames don't collapse to zero. */
export function applyLogScale(data: Float32Array): Float32Array {
  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    result[i] = v >= 0 ? Math.log1p(v) : -Math.log1p(-v);
  }
  return result;
}

/** Apply signed log1p scale into a pre-allocated buffer. Avoids per-frame allocation. */
export function applyLogScaleInPlace(data: Float32Array, out: Float32Array): Float32Array {
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    out[i] = v >= 0 ? Math.log1p(v) : -Math.log1p(-v);
  }
  return out;
}

/** Percentile-based clipping using O(n) histogram approach.
 *  Also returns data min/max so callers can skip a redundant findDataRange scan. */
export function percentileClip(
  data: Float32Array, pLow: number, pHigh: number,
): { vmin: number; vmax: number; min: number; max: number } {
  const len = data.length;
  if (len === 0) return { vmin: 0, vmax: 0, min: 0, max: 0 };

  // Pass 1: find min/max
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === max) return { vmin: min, vmax: max, min, max };

  // Pass 2: build histogram
  const NUM_BINS = 1024;
  const bins = new Uint32Array(NUM_BINS);
  const range = max - min;
  const scale = (NUM_BINS - 1) / range;
  for (let i = 0; i < len; i++) {
    bins[Math.floor((data[i] - min) * scale)]++;
  }

  // Walk cumulative histogram to find percentile values
  const lowCount = Math.floor(len * (pLow / 100));
  const highCount = Math.ceil(len * (pHigh / 100));
  let cumSum = 0;
  let vmin = min, vmax = max;
  for (let i = 0; i < NUM_BINS; i++) {
    cumSum += bins[i];
    if (cumSum >= lowCount) { vmin = min + (i / (NUM_BINS - 1)) * range; break; }
  }
  cumSum = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    cumSum += bins[i];
    if (cumSum >= highCount) { vmax = min + (i / (NUM_BINS - 1)) * range; break; }
  }
  return { vmin, vmax, min, max };
}

/** Compute mean, min, max, and standard deviation of a Float32Array. */
export function computeStats(data: Float32Array): { mean: number; min: number; max: number; std: number } {
  if (data.length === 0) return { mean: 0, min: 0, max: 0, std: 0 };
  let sum = 0, min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / data.length;
  let variance = 0;
  for (let i = 0; i < data.length; i++) variance += (data[i] - mean) ** 2;
  const std = Math.sqrt(variance / data.length);
  return { mean, min, max, std };
}

/** Convert histogram slider percentages (0-100) to vmin/vmax in data space. */
export function sliderRange(
  dataMin: number, dataMax: number, vminPct: number, vmaxPct: number,
): { vmin: number; vmax: number } {
  const range = dataMax - dataMin;
  return {
    vmin: dataMin + (vminPct / 100) * range,
    vmax: dataMin + (vmaxPct / 100) * range,
  };
}

/** Compute normalized histogram bins from Float32Array.
 *  fixedMin/fixedMax pin bin edges to a global range (so scrubbing through
 *  a stack doesn't rescale per-frame). Defaults to per-array min/max. */
export function computeHistogramFromBytes(
  data: Float32Array | null,
  numBins = 256,
  fixedMin?: number,
  fixedMax?: number,
): number[] {
  if (!data || data.length === 0) return new Array(numBins).fill(0);
  const bins = new Array(numBins).fill(0);
  let min: number, max: number;
  if (fixedMin !== undefined && fixedMax !== undefined && isFinite(fixedMin) && isFinite(fixedMax) && fixedMin < fixedMax) {
    min = fixedMin;
    max = fixedMax;
  } else {
    min = Infinity; max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
    }
    if (!isFinite(min) || !isFinite(max) || min === max) return bins;
  }
  const range = max - min;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) {
      // Clamp into last bin so max-value pixels aren't silently dropped.
      let idx = Math.floor(((v - min) / range) * numBins);
      if (idx === numBins) idx = numBins - 1;
      if (idx >= 0 && idx < numBins) bins[idx]++;
    }
  }
  const maxCount = Math.max(...bins);
  if (maxCount > 0) for (let i = 0; i < numBins; i++) bins[i] /= maxCount;
  return bins;
}
