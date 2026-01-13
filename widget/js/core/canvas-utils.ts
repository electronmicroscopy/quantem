/**
 * Shared canvas rendering utilities.
 * Used by Show2D, Show3D, Show4DSTEM, and Reconstruct.
 */

import { COLORMAPS } from "./colormaps";
import { colors } from "./colors";

// ============================================================================
// Colormap LUT Application
// ============================================================================

/**
 * Render uint8 data to canvas with colormap LUT.
 */
export function renderWithColormap(
  ctx: CanvasRenderingContext2D,
  data: Uint8Array,
  width: number,
  height: number,
  cmapName: string = "inferno"
): void {
  const lut = COLORMAPS[cmapName] || COLORMAPS.inferno;
  const imgData = ctx.createImageData(width, height);
  const rgba = imgData.data;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    const j = i * 4;
    const lutIdx = v * 3;
    rgba[j] = lut[lutIdx];
    rgba[j + 1] = lut[lutIdx + 1];
    rgba[j + 2] = lut[lutIdx + 2];
    rgba[j + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

/**
 * Render float32 data to canvas with colormap.
 */
export function renderFloat32WithColormap(
  ctx: CanvasRenderingContext2D,
  data: Float32Array,
  width: number,
  height: number,
  cmapName: string = "inferno"
): void {
  const lut = COLORMAPS[cmapName] || COLORMAPS.inferno;

  // Calculate min/max
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const range = max - min || 1;
  const scale = 255 / range;

  const imgData = ctx.createImageData(width, height);
  const rgba = imgData.data;

  for (let i = 0; i < data.length; i++) {
    const v = Math.round((data[i] - min) * scale);
    const lutIdx = Math.max(0, Math.min(255, v)) * 3;
    const j = i * 4;
    rgba[j] = lut[lutIdx];
    rgba[j + 1] = lut[lutIdx + 1];
    rgba[j + 2] = lut[lutIdx + 2];
    rgba[j + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

/**
 * Draw image data to canvas with zoom and pan.
 */
export function drawWithZoomPan(
  ctx: CanvasRenderingContext2D,
  source: HTMLCanvasElement | ImageData,
  canvasWidth: number,
  canvasHeight: number,
  zoom: number,
  panX: number,
  panY: number
): void {
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);
  if (source instanceof ImageData) {
    ctx.putImageData(source, 0, 0);
  } else {
    ctx.drawImage(source, 0, 0);
  }
  ctx.restore();
}

// ============================================================================
// Scale Bar Rendering
// ============================================================================

/** Round to a nice value (1, 2, 5, 10, 20, 50, etc.) */
export function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/** Format scale bar label with appropriate unit */
export function formatScaleLabel(value: number, unit: string): string {
  const nice = roundToNiceValue(value);

  if (unit === "Å") {
    if (nice >= 10) return `${Math.round(nice / 10)} nm`;
    if (nice >= 1) return `${Math.round(nice)} Å`;
    return `${nice.toFixed(2)} Å`;
  } else if (unit === "nm") {
    if (nice >= 1000) return `${Math.round(nice / 1000)} µm`;
    if (nice >= 1) return `${Math.round(nice)} nm`;
    return `${nice.toFixed(2)} nm`;
  } else if (unit === "mrad") {
    if (nice >= 1000) return `${Math.round(nice / 1000)} rad`;
    if (nice >= 1) return `${Math.round(nice)} mrad`;
    return `${nice.toFixed(2)} mrad`;
  } else if (unit === "1/µm") {
    if (nice >= 1000) return `${Math.round(nice / 1000)} 1/nm`;
    if (nice >= 1) return `${Math.round(nice)} 1/µm`;
    return `${nice.toFixed(2)} 1/µm`;
  } else if (unit === "px") {
    return `${Math.round(nice)} px`;
  }
  return `${Math.round(nice)} ${unit}`;
}

/**
 * Draw scale bar on high-DPI canvas.
 */
export function drawScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: string = "nm",
  imageWidth: number,
  imageHeight: number
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const displayScale = Math.min(cssWidth / imageWidth, cssHeight / imageHeight);
  const effectiveZoom = zoom * displayScale;

  // Fixed UI sizes in CSS pixels
  const targetBarPx = 60;
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;

  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  // Draw with shadow for visibility
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  // Draw label
  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  // Draw zoom indicator (bottom left)
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);

  ctx.restore();
}

/**
 * Draw crosshair on high-DPI canvas.
 */
export function drawCrosshairHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  posX: number,
  posY: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  color: string = "rgba(0, 255, 0, 0.9)",
  dragColor: string = "rgba(255, 255, 0, 0.9)"
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const displayScale = Math.min(cssWidth / imageWidth, cssHeight / imageHeight);

  const screenX = posX * zoom * displayScale + panX * displayScale;
  const screenY = posY * zoom * displayScale + panY * displayScale;

  const crosshairSize = 18;
  const lineWidth = 3;
  const dotRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.strokeStyle = isDragging ? dragColor : color;
  ctx.lineWidth = lineWidth;

  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(screenX, screenY, dotRadius, 0, 2 * Math.PI);
  ctx.stroke();

  ctx.restore();
}

// ============================================================================
// Export to Blob/ZIP Helpers
// ============================================================================

/**
 * Convert canvas to PNG blob.
 */
export function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob!), "image/png");
  });
}
