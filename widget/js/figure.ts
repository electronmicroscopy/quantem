/**
 * Shared scale bar, colorbar, and overlay utilities for all canvas-based widgets.
 * Provides HiDPI-aware rendering with automatic unit conversion.
 */

import { formatNumber } from "./format";

/** Round a physical value to a "nice" number (1, 2, 5, 10, 20, 50, ...) */
export function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/** Format scale bar label. Unit string is displayed verbatim - no conversion. */
export function formatScaleLabel(value: number, unit: string): string {
  const nice = roundToNiceValue(value);
  return nice >= 1 ? `${Math.round(nice)} ${unit}` : `${nice.toFixed(2)} ${unit}`;
}

const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

/**
 * Draw scale bar and zoom indicator on a high-DPI UI canvas.
 * Renders crisp text/lines independent of the image resolution.
 */
export function drawScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: string,
  imageWidth: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = zoom * scaleX;

  const targetBarPx = 60;
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;

  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px ${FONT}`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);

  ctx.restore();
}

/**
 * Draw reciprocal-space scale bar on an FFT overlay canvas.
 * Only draws when fftPixelSize > 0 (i.e. real-space calibration is available).
 */
export function drawFFTScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  fftZoom: number,
  fftPixelSize: number,
  imageWidth: number,
  unit: string = "1/px",
) {
  const ctx = canvas.getContext("2d");
  if (!ctx || fftPixelSize <= 0) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = fftZoom * scaleX;

  const targetBarPx = 60;
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;

  const targetPhysical = (targetBarPx / effectiveZoom) * fftPixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / fftPixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px ${FONT}`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${fftZoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);

  ctx.restore();
}

/**
 * Draw a vertical colorbar on a canvas context (already DPR-scaled by caller).
 * Gradient strip on right edge with vmin/vmax labels and optional log indicator.
 */
export function drawColorbar(
  ctx: CanvasRenderingContext2D,
  cssW: number,
  cssH: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
) {
  const barW = 12;
  const barH = Math.round(cssH * 0.6);
  const barX = cssW - barW - 12;
  const barY = Math.round((cssH - barH) / 2);

  // Gradient strip (bottom=vmin, top=vmax)
  for (let row = 0; row < barH; row++) {
    const t = 1 - row / (barH - 1);
    const lutIdx = Math.round(t * 255);
    const r = lut[lutIdx * 3];
    const g = lut[lutIdx * 3 + 1];
    const b = lut[lutIdx * 3 + 2];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(barX, barY + row, barW, 1);
  }

  // Border
  ctx.strokeStyle = "rgba(255,255,255,0.5)";
  ctx.lineWidth = 1;
  ctx.strokeRect(barX, barY, barW, barH);

  // Labels with drop shadow
  ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  ctx.font = `11px ${FONT}`;
  ctx.fillStyle = "white";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(formatNumber(vmax), barX - 4, barY + 6);
  ctx.textBaseline = "top";
  ctx.fillText(formatNumber(vmin), barX - 4, barY + barH - 4);
  if (logScale) {
    ctx.textBaseline = "middle";
    ctx.fillText("log", barX - 4, barY + barH / 2);
  }
}

// ============================================================================
// Publication-quality figure export
// ============================================================================

export interface ExportFigureOptions {
  /** Colormapped image canvas at native resolution (no zoom/pan). */
  imageCanvas: HTMLCanvasElement;
  /** Figure title drawn above the image. */
  title?: string;
  /** Colormap LUT (256 × 3 bytes) for the colorbar. */
  lut?: Uint8Array;
  /** Data range for colorbar labels. */
  vmin?: number;
  vmax?: number;
  logScale?: boolean;
  /** Pixel size in user-supplied unit (for scale bar computation). */
  pixelSize?: number;
  /** Unit string for the scale bar label (e.g. "A", "nm", "mrad"). */
  pixelUnit?: string;
  showColorbar?: boolean;
  showScaleBar?: boolean;
  /** Upscale factor for high-resolution output (default 4). Image pixels use nearest-neighbor for sharp edges. */
  scale?: number;
  /** Callback to draw annotations (ROI, profile, markers) on the image. ctx is pre-translated to image origin and scaled. */
  drawAnnotations?: (ctx: CanvasRenderingContext2D) => void;
}

/**
 * Create a publication-quality figure canvas with title, scale bar, colorbar,
 * and baked-in annotations. Returns an HTMLCanvasElement — caller can toBlob() + download.
 */
export function exportFigure(options: ExportFigureOptions): HTMLCanvasElement {
  const {
    imageCanvas,
    title,
    lut,
    vmin = 0,
    vmax = 1,
    logScale = false,
    pixelSize = 0,
    pixelUnit = "pixels",
    showColorbar = true,
    showScaleBar = true,
    scale: s = 4,
    drawAnnotations,
  } = options;

  const imgW = imageCanvas.width;
  const imgH = imageCanvas.height;

  // Layout (in logical coordinates — scaled to canvas pixels by ctx.scale)
  const pad = 20;
  const titleH = title ? 28 : 0;
  const titleGap = title ? 8 : 0;
  const hasCb = showColorbar && lut && vmin !== vmax;
  const cbWidth = hasCb ? 20 : 0;
  const cbGap = hasCb ? 12 : 0;
  const cbLabelW = hasCb ? 60 : 0;

  const totalW = pad + imgW + cbGap + cbWidth + cbLabelW + pad;
  const totalH = pad + titleH + titleGap + imgH + pad;

  const canvas = document.createElement("canvas");
  canvas.width = totalW * s;
  canvas.height = totalH * s;
  const ctx = canvas.getContext("2d")!;

  // Scale all drawing operations
  ctx.scale(s, s);

  // White background
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, totalW, totalH);

  // Title
  if (title) {
    ctx.fillStyle = "black";
    ctx.font = `bold 18px ${FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(title, pad, pad);
  }

  const imgX = pad;
  const imgY = pad + titleH + titleGap;

  // Image (nearest-neighbor for sharp pixels)
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(imageCanvas, imgX, imgY, imgW, imgH);
  ctx.imageSmoothingEnabled = true;

  // Annotations
  if (drawAnnotations) {
    ctx.save();
    ctx.translate(imgX, imgY);
    drawAnnotations(ctx);
    ctx.restore();
  }

  // Scale bar (white with drop shadow, positioned at bottom-right of image)
  if (showScaleBar && pixelSize > 0) {
    const targetBarPx = Math.max(60, imgW * 0.15);
    const barThickness = Math.max(4, Math.round(imgH * 0.012));
    const fontSize = Math.max(14, Math.round(imgH * 0.04));
    const margin = Math.max(12, Math.round(imgW * 0.03));

    const targetPhysical = targetBarPx * pixelSize;
    const nicePhysical = roundToNiceValue(targetPhysical);
    const barPx = nicePhysical / pixelSize;

    const barY = imgY + imgH - margin;
    const barX = imgX + imgW - barPx - margin;

    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
    ctx.shadowBlur = 2;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 1;

    ctx.fillStyle = "white";
    ctx.fillRect(barX, barY, barPx, barThickness);

    const label = formatScaleLabel(nicePhysical, pixelUnit);
    ctx.font = `bold ${fontSize}px ${FONT}`;
    ctx.fillStyle = "white";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(label, barX + barPx / 2, barY - 4);

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
  }

  // Colorbar (vertical gradient strip to the right of image)
  if (hasCb && lut) {
    const cbX = imgX + imgW + cbGap;
    const cbY = imgY;
    const cbH = imgH;

    for (let row = 0; row < cbH; row++) {
      const t = 1 - row / (cbH - 1);
      const lutIdx = Math.round(t * 255);
      const r = lut[lutIdx * 3];
      const g = lut[lutIdx * 3 + 1];
      const b = lut[lutIdx * 3 + 2];
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(cbX, cbY + row, cbWidth, 1);
    }

    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    ctx.strokeRect(cbX, cbY, cbWidth, cbH);

    ctx.fillStyle = "black";
    ctx.font = `12px ${FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(vmax), cbX + cbWidth + 4, cbY);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(vmin), cbX + cbWidth + 4, cbY + cbH);
    if (logScale) {
      ctx.textBaseline = "middle";
      ctx.fillText("log", cbX + cbWidth + 4, cbY + cbH / 2);
    }
  }

  return canvas;
}

/**
 * Convert a canvas to a PDF blob by embedding JPEG data in a minimal PDF.
 * Zero external dependencies — uses the DCTDecode filter (native JPEG in PDF).
 */
export async function canvasToPDF(canvas: HTMLCanvasElement, quality = 0.95): Promise<Blob> {
  const jpegBlob = await new Promise<Blob>((resolve) =>
    canvas.toBlob((b) => resolve(b!), "image/jpeg", quality));
  const jpegBytes = new Uint8Array(await jpegBlob.arrayBuffer());
  const w = canvas.width;
  const h = canvas.height;

  // Build PDF objects
  const contentStream = `q ${w} 0 0 ${h} 0 0 cm /I0 Do Q`;
  const objects: string[] = [];
  const offsets: number[] = [];

  // Helper to track object positions
  let pdf = "%PDF-1.4\n";

  // Object 1: Catalog
  offsets.push(pdf.length);
  objects.push("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
  pdf += objects[0];

  // Object 2: Pages
  offsets.push(pdf.length);
  objects.push("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
  pdf += objects[1];

  // Object 3: Page
  offsets.push(pdf.length);
  objects.push(`3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${w} ${h}] /Contents 4 0 R /Resources << /XObject << /I0 5 0 R >> >> >>\nendobj\n`);
  pdf += objects[2];

  // Object 4: Content stream
  offsets.push(pdf.length);
  objects.push(`4 0 obj\n<< /Length ${contentStream.length} >>\nstream\n${contentStream}\nendstream\nendobj\n`);
  pdf += objects[3];

  // Object 5: Image (JPEG) — build as binary
  const imgHeader = `5 0 obj\n<< /Type /XObject /Subtype /Image /Width ${w} /Height ${h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.length} >>\nstream\n`;
  const imgFooter = "\nendstream\nendobj\n";

  // Convert text part to bytes
  const encoder = new TextEncoder();
  const headerBytes = encoder.encode(pdf + imgHeader);
  const footerBytes = encoder.encode(imgFooter);

  // Build xref
  const imgOffset = pdf.length;
  offsets.push(imgOffset);
  const afterImage = headerBytes.length + jpegBytes.length + footerBytes.length;

  const xrefOffset = afterImage;
  let xref = `xref\n0 6\n0000000000 65535 f \n`;
  for (let i = 0; i < offsets.length; i++) {
    xref += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  }
  xref += `trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  const xrefBytes = encoder.encode(xref);

  // Combine all parts
  const result = new Uint8Array(headerBytes.length + jpegBytes.length + footerBytes.length + xrefBytes.length);
  result.set(headerBytes, 0);
  result.set(jpegBytes, headerBytes.length);
  result.set(footerBytes, headerBytes.length + jpegBytes.length);
  result.set(xrefBytes, headerBytes.length + jpegBytes.length + footerBytes.length);

  return new Blob([result], { type: "application/pdf" });
}
