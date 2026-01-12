/**
 * Canvas rendering utilities for image widgets.
 * Scale bar, overlays, ROI drawing, etc.
 */

import { colors } from "./colors";

/** Nice values for scale bar lengths */
const NICE_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];

/**
 * Calculate a "nice" scale bar length.
 * @param imageWidthNm - Total image width in nm
 * @param targetFraction - Target fraction of image width (default 0.2)
 * @returns Scale bar length in nm
 */
export function calculateNiceScaleBar(
  imageWidthNm: number,
  targetFraction: number = 0.2
): number {
  const targetNm = imageWidthNm * targetFraction;
  const magnitude = Math.pow(10, Math.floor(Math.log10(targetNm)));

  let barNm = magnitude;
  for (const v of NICE_VALUES) {
    if (v * magnitude <= targetNm * 1.5) {
      barNm = v * magnitude;
    }
  }
  return barNm;
}

/**
 * Round to a "nice" scale bar value (1, 2, 5, 10, 20, 50, 100, etc.)
 * This ensures scale bars always show clean integer values.
 * @param value - Raw value to round
 * @returns Nice rounded value
 */
function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  // Round to 1, 2, 5, or 10
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/**
 * Format scale bar label with appropriate unit.
 * Always displays integer values (no decimals).
 * @param angstroms - Length in Angstroms
 * @returns Formatted string (e.g., "5 Å", "20 nm", "1 µm")
 */
export function formatScaleBarLabel(angstroms: number): string {
  // Round to nice value first
  const nice = roundToNiceValue(angstroms);
  
  if (nice >= 10000) { // >= 1 µm
    return `${Math.round(nice / 10000)} µm`;
  }
  if (nice >= 100) { // >= 10 nm, show in nm
    return `${Math.round(nice / 10)} nm`;
  }
  return `${Math.round(nice)} Å`;
}

/**
 * Draw scale bar on canvas overlay with nice integer labels.
 * The bar length is dynamically calculated to accurately represent the physical distance.
 * @param ctx - Canvas 2D context
 * @param canvasWidth - Canvas width in pixels
 * @param canvasHeight - Canvas height in pixels
 * @param imageWidth - Image width in data pixels
 * @param pixelSizeAngstrom - Pixel size in Angstroms
 * @param displayScale - Canvas scale factor (includes zoom)
 * @param targetBarLength - Target length of the scale bar in pixels (default 50)
 * @param barThickness - Thickness of the scale bar (default 4)
 * @param fontSize - Font size for the label (default 16)
 */
export function drawScaleBar(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  canvasHeight: number,
  _imageWidth: number,
  pixelSizeAngstrom: number,
  displayScale: number = 1,
  targetBarLength: number = 50,
  barThickness: number = 4,
  fontSize: number = 16
): void {
  // Fallback if pixelSize is missing or invalid: show bar in pixels
  if (pixelSizeAngstrom <= 0) {
    const x = canvasWidth - targetBarLength - 10;
    const y = canvasHeight - 20;

    ctx.fillStyle = colors.textPrimary;
    ctx.fillRect(x, y, targetBarLength, barThickness);

    ctx.font = "11px sans-serif";
    ctx.fillStyle = colors.textPrimary;
    ctx.textAlign = "right";
    ctx.fillText(`${targetBarLength} px`, x + targetBarLength, y - 5);
    return;
  }

  // Calculate what the target bar length represents in Angstroms at current zoom
  const targetAngstroms = targetBarLength * pixelSizeAngstrom / displayScale;
  
  // Round to a nice value
  const niceAngstroms = roundToNiceValue(targetAngstroms);
  
  // Calculate the actual bar length for the nice value
  const barLength = (niceAngstroms / pixelSizeAngstrom) * displayScale;

  const x = canvasWidth - barLength - 10;
  const y = canvasHeight - 20;

  // Draw bar (length matches the nice value)
  ctx.fillStyle = colors.textPrimary;
  ctx.fillRect(x, y, barLength, barThickness);

  // Draw label
  const label = formatScaleBarLabel(niceAngstroms);
  ctx.font = `${fontSize}px sans-serif`;
  ctx.fillStyle = colors.textPrimary;
  ctx.textAlign = "right";
  ctx.fillText(label, x + barLength, y - 5);
}

/**
 * Draw ROI on canvas overlay with different shapes.
 * @param ctx - Canvas 2D context
 * @param x - Center X in canvas pixels
 * @param y - Center Y in canvas pixels
 * @param shape - ROI shape: "circle", "square", or "rectangle"
 * @param radius - Radius for circle, or half-size for square
 * @param width - Width for rectangle
 * @param height - Height for rectangle
 * @param active - Whether ROI is being dragged
 */
export function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: "circle" | "square" | "rectangle",
  radius: number,
  width: number,
  height: number,
  active: boolean = false
): void {
  const strokeColor = active ? colors.accentYellow : colors.accentGreen;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2;

  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape === "square") {
    const size = radius * 2;
    ctx.strokeRect(x - radius, y - radius, size, size);
  } else if (shape === "rectangle") {
    const halfW = width / 2;
    const halfH = height / 2;
    ctx.strokeRect(x - halfW, y - halfH, width, height);
  }

  // Center crosshair - only show while dragging
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
  }
}

/**
 * Draw ROI circle on canvas overlay.
 * @param ctx - Canvas 2D context
 * @param x - Center X in canvas pixels
 * @param y - Center Y in canvas pixels
 * @param radius - Radius in canvas pixels
 * @param active - Whether ROI is being dragged
 */
export function drawROICircle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  active: boolean = false
): void {
  const strokeColor = active ? colors.accentYellow : colors.accentGreen;

  // Circle
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Center crosshair
  ctx.beginPath();
  ctx.moveTo(x - 5, y);
  ctx.lineTo(x + 5, y);
  ctx.moveTo(x, y - 5);
  ctx.lineTo(x, y + 5);
  ctx.stroke();
}

/**
 * Draw crosshair on canvas.
 * @param ctx - Canvas 2D context
 * @param x - Center X
 * @param y - Center Y
 * @param size - Half-length of crosshair arms
 * @param color - Stroke color
 */
export function drawCrosshair(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number = 10,
  color: string = colors.accentGreen
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - size, y);
  ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size);
  ctx.lineTo(x, y + size);
  ctx.stroke();
}

/**
 * Calculate canvas scale factor for display.
 * Aims for approximately targetSize pixels on screen.
 * @param width - Image width
 * @param height - Image height
 * @param targetSize - Target display size in pixels (default 400)
 * @returns Integer scale factor >= 1
 */
export function calculateDisplayScale(
  width: number,
  height: number,
  targetSize: number = 400
): number {
  return Math.max(1, Math.floor(targetSize / Math.max(width, height)));
}

/**
 * Extract bytes from DataView (handles anywidget's byte transfer).
 * @param dataView - DataView from anywidget
 * @returns Uint8Array of bytes
 */
export function extractBytes(dataView: DataView | ArrayBuffer | Uint8Array): Uint8Array {
  if (dataView instanceof Uint8Array) {
    return dataView;
  }
  if (dataView instanceof ArrayBuffer) {
    return new Uint8Array(dataView);
  }
  // DataView from anywidget
  if (dataView && "buffer" in dataView) {
    return new Uint8Array(
      dataView.buffer,
      dataView.byteOffset,
      dataView.byteLength
    );
  }
  return new Uint8Array(0);
}
