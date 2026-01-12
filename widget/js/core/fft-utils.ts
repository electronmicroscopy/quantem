/**
 * FFT and histogram rendering utilities.
 * Shared across Show2D and Show3D widgets.
 */

import { COLORMAPS } from "./colormaps";
import { colors } from "./colors";

// ============================================================================
// FFT Rendering
// ============================================================================

/**
 * Render FFT magnitude to canvas with log scale and colormap.
 * @param ctx - Canvas 2D context
 * @param fftMag - FFT magnitude data (Float32Array)
 * @param width - Image width
 * @param height - Image height
 * @param panelSize - Canvas panel size
 * @param zoom - Zoom level (default 3 for center detail)
 * @param panX - Pan X offset
 * @param panY - Pan Y offset
 * @param cmapName - Colormap name (default "inferno")
 */
export function renderFFT(
  ctx: CanvasRenderingContext2D,
  fftMag: Float32Array,
  width: number,
  height: number,
  panelSize: number,
  zoom: number = 3,
  panX: number = 0,
  panY: number = 0,
  cmapName: string = "inferno"
): void {
  // Log scale and normalize
  let min = Infinity;
  let max = -Infinity;
  const logData = new Float32Array(fftMag.length);

  for (let i = 0; i < fftMag.length; i++) {
    logData[i] = Math.log(1 + fftMag[i]);
    if (logData[i] < min) min = logData[i];
    if (logData[i] > max) max = logData[i];
  }

  const lut = COLORMAPS[cmapName] || COLORMAPS.inferno;

  // Create offscreen canvas at native resolution
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const offCtx = offscreen.getContext("2d");
  if (!offCtx) return;

  const imgData = offCtx.createImageData(width, height);
  const range = max - min || 1;

  for (let i = 0; i < logData.length; i++) {
    const v = Math.floor(((logData[i] - min) / range) * 255);
    const j = i * 4;
    imgData.data[j] = lut[v * 3];
    imgData.data[j + 1] = lut[v * 3 + 1];
    imgData.data[j + 2] = lut[v * 3 + 2];
    imgData.data[j + 3] = 255;
  }
  offCtx.putImageData(imgData, 0, 0);

  // Draw with zoom/pan - center the zoomed view
  const scale = panelSize / Math.max(width, height);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, panelSize, panelSize);
  ctx.save();

  const centerOffsetX = (panelSize - width * scale * zoom) / 2 + panX;
  const centerOffsetY = (panelSize - height * scale * zoom) / 2 + panY;

  ctx.translate(centerOffsetX, centerOffsetY);
  ctx.scale(zoom, zoom);
  ctx.drawImage(offscreen, 0, 0, width * scale, height * scale);
  ctx.restore();
}

// ============================================================================
// Histogram Rendering
// ============================================================================

/**
 * Render histogram to canvas.
 * @param ctx - Canvas 2D context
 * @param counts - Histogram bin counts
 * @param panelSize - Canvas panel size
 * @param accentColor - Bar color (default: accent blue)
 * @param bgColor - Background color (default: panel background)
 */
export function renderHistogram(
  ctx: CanvasRenderingContext2D,
  counts: number[],
  panelSize: number,
  accentColor: string = colors.accent,
  bgColor: string = colors.bgPanel
): void {
  const w = panelSize;
  const h = panelSize;

  // Clear and fill background
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, w, h);

  // Only draw bars if we have data
  if (!counts || counts.length === 0) return;

  const maxCount = Math.max(...counts);
  if (maxCount === 0) return;

  // Add padding for centering
  const padding = 8;
  const drawWidth = w - 2 * padding;
  const drawHeight = h - padding - 5; // 5px bottom margin
  const barWidth = drawWidth / counts.length;

  ctx.fillStyle = accentColor;
  for (let i = 0; i < counts.length; i++) {
    const barHeight = (counts[i] / maxCount) * drawHeight;
    ctx.fillRect(padding + i * barWidth, h - padding - barHeight, barWidth - 1, barHeight);
  }
}

// ============================================================================
// FFT Shift (move DC component to center)
// ============================================================================

/**
 * Shift FFT data to center the DC component.
 * Modifies data in place.
 */
export function fftshift(data: Float32Array, width: number, height: number): void {
  const halfW = width >> 1;
  const halfH = height >> 1;
  const temp = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const newY = (y + halfH) % height;
      const newX = (x + halfW) % width;
      temp[newY * width + newX] = data[y * width + x];
    }
  }
  data.set(temp);
}

/**
 * Compute FFT magnitude from real and imaginary parts.
 */
export function computeMagnitude(real: Float32Array, imag: Float32Array): Float32Array {
  const mag = new Float32Array(real.length);
  for (let i = 0; i < real.length; i++) {
    mag[i] = Math.sqrt(real[i] ** 2 + imag[i] ** 2);
  }
  return mag;
}
