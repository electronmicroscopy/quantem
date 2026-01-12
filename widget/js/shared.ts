/**
 * Shared utilities for widget components.
 * Contains CPU FFT fallback and band-pass filtering.
 * Re-exports commonly used utilities from core.
 */

// Re-export colormaps from core
export { COLORMAP_NAMES, COLORMAP_POINTS, COLORMAPS, createColormapLUT, applyColormapValue, applyColormapToImage } from "./core/colormaps";

// Re-export fftshift from core (also available here for backward compatibility)
export { fftshift, computeMagnitude, renderFFT, renderHistogram } from "./core/fft-utils";

// Re-export zoom constants from core hooks
export { ZOOM_LIMITS } from "./core/hooks";
export const MIN_ZOOM = 0.5;  // Legacy alias
export const MAX_ZOOM = 10;   // Legacy alias

// ============================================================================
// CPU FFT Implementation (Cooley-Tukey radix-2) - Fallback when WebGPU unavailable
// Supports ANY size via automatic zero-padding to next power of 2
// ============================================================================

/** Get next power of 2 >= n */
function nextPow2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

/** Check if n is a power of 2 */
function isPow2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/** Internal 1D FFT - requires power-of-2 size */
function fft1dPow2(real: Float32Array, imag: Float32Array, inverse: boolean = false) {
  const n = real.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }

  // Cooley-Tukey FFT
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curReal = 1, curImag = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k;
        const oddIdx = i + k + halfLen;

        const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
        const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];

        real[oddIdx] = real[evenIdx] - tReal;
        imag[oddIdx] = imag[evenIdx] - tImag;
        real[evenIdx] += tReal;
        imag[evenIdx] += tImag;

        const newReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = newReal;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

/**
 * 1D FFT - supports ANY size via zero-padding
 * Modifies arrays in-place
 */
export function fft1d(real: Float32Array, imag: Float32Array, inverse: boolean = false) {
  const n = real.length;
  if (isPow2(n)) {
    fft1dPow2(real, imag, inverse);
    return;
  }

  // Pad to next power of 2
  const paddedN = nextPow2(n);
  const paddedReal = new Float32Array(paddedN);
  const paddedImag = new Float32Array(paddedN);
  paddedReal.set(real);
  paddedImag.set(imag);

  fft1dPow2(paddedReal, paddedImag, inverse);

  // Copy back (truncate to original size)
  for (let i = 0; i < n; i++) {
    real[i] = paddedReal[i];
    imag[i] = paddedImag[i];
  }
}

/**
 * 2D FFT - supports ANY size via zero-padding
 * Modifies arrays in-place
 */
export function fft2d(real: Float32Array, imag: Float32Array, width: number, height: number, inverse: boolean = false) {
  const paddedW = nextPow2(width);
  const paddedH = nextPow2(height);
  const needsPadding = paddedW !== width || paddedH !== height;

  // Work arrays (padded if needed)
  let workReal: Float32Array;
  let workImag: Float32Array;

  if (needsPadding) {
    workReal = new Float32Array(paddedW * paddedH);
    workImag = new Float32Array(paddedW * paddedH);
    // Copy original data into top-left corner
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = y * width + x;
        const dstIdx = y * paddedW + x;
        workReal[dstIdx] = real[srcIdx];
        workImag[dstIdx] = imag[srcIdx];
      }
    }
  } else {
    workReal = real;
    workImag = imag;
  }

  // FFT on rows (padded width)
  const rowReal = new Float32Array(paddedW);
  const rowImag = new Float32Array(paddedW);
  for (let y = 0; y < paddedH; y++) {
    const offset = y * paddedW;
    for (let x = 0; x < paddedW; x++) {
      rowReal[x] = workReal[offset + x];
      rowImag[x] = workImag[offset + x];
    }
    fft1dPow2(rowReal, rowImag, inverse);
    for (let x = 0; x < paddedW; x++) {
      workReal[offset + x] = rowReal[x];
      workImag[offset + x] = rowImag[x];
    }
  }

  // FFT on columns (padded height)
  const colReal = new Float32Array(paddedH);
  const colImag = new Float32Array(paddedH);
  for (let x = 0; x < paddedW; x++) {
    for (let y = 0; y < paddedH; y++) {
      colReal[y] = workReal[y * paddedW + x];
      colImag[y] = workImag[y * paddedW + x];
    }
    fft1dPow2(colReal, colImag, inverse);
    for (let y = 0; y < paddedH; y++) {
      workReal[y * paddedW + x] = colReal[y];
      workImag[y * paddedW + x] = colImag[y];
    }
  }

  // Copy back to original arrays if padded
  if (needsPadding) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = y * paddedW + x;
        const dstIdx = y * width + x;
        real[dstIdx] = workReal[srcIdx];
        imag[dstIdx] = workImag[srcIdx];
      }
    }
  }
}

// ============================================================================
// Band-pass Filter
// ============================================================================

/** Apply band-pass filter in frequency domain (keeps frequencies between inner and outer radius) */
export function applyBandPassFilter(
  real: Float32Array,
  imag: Float32Array,
  width: number,
  height: number,
  innerRadius: number,  // High-pass: remove frequencies below this
  outerRadius: number   // Low-pass: remove frequencies above this
) {
  const centerX = width >> 1;
  const centerY = height >> 1;
  const innerSq = innerRadius * innerRadius;
  const outerSq = outerRadius * outerRadius;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      const distSq = dx * dx + dy * dy;
      const idx = y * width + x;

      // Zero out frequencies outside the band
      if (distSq < innerSq || (outerRadius > 0 && distSq > outerSq)) {
        real[idx] = 0;
        imag[idx] = 0;
      }
    }
  }
}

