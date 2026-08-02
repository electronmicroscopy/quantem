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

/** Apply log1p scale: result[i] = log(1 + max(0, data[i])). Returns a new array. */
export function applyLogScale(data: Float32Array): Float32Array {
  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = Math.log1p(Math.max(0, data[i]));
  }
  return result;
}

/** Apply log1p scale into a pre-allocated buffer. Avoids per-frame allocation. */
export function applyLogScaleInPlace(data: Float32Array, out: Float32Array): Float32Array {
  for (let i = 0; i < data.length; i++) {
    out[i] = Math.log1p(Math.max(0, data[i]));
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

/** Compute normalized histogram bins from Float32Array. Returns array of 0-1 values. */
export function computeHistogramFromBytes(data: Float32Array | null, numBins = 256): number[] {
  if (!data || data.length === 0) return new Array(numBins).fill(0);
  const bins = new Array(numBins).fill(0);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  }
  if (!isFinite(min) || !isFinite(max) || min === max) return bins;
  const range = max - min;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) bins[Math.min(numBins - 1, Math.floor(((v - min) / range) * numBins))]++;
  }
  const maxCount = Math.max(...bins);
  if (maxCount > 0) for (let i = 0; i < numBins; i++) bins[i] /= maxCount;
  return bins;
}
