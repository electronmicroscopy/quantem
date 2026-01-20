import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import JSZip from "jszip";
import { getWebGPUFFT, WebGPUFFT } from "./webgpu-fft";
import { COLORMAPS, fft2d, fftshift, applyBandPassFilter, MIN_ZOOM, MAX_ZOOM } from "./shared";
import { typography, controlPanel, container } from "./CONFIG";
import { upwardMenuProps, switchStyles } from "./components";
import "./show4dstem.css";

// ============================================================================
// Layout Constants - consistent spacing throughout
// ============================================================================
const SPACING = {
  XS: 4,    // Extra small gap
  SM: 8,    // Small gap (default between elements)
  MD: 12,   // Medium gap (between control groups)
  LG: 16,   // Large gap (between major sections)
};

const CANVAS_SIZE = 450;  // Both DP and VI canvases

// Interaction constants
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;  // cos(45°)
const LINE_WIDTH_FRACTION = 0.015;
const LINE_WIDTH_MIN_PX = 1.5;
const LINE_WIDTH_MAX_PX = 3;

// Compact button style for Reset/Export
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
};

/** Round to a nice value (1, 2, 5, 10, 20, 50, etc.) */
function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/** Format scale bar label with appropriate unit */
function formatScaleLabel(value: number, unit: "Å" | "mrad" | "px"): string {
  const nice = roundToNiceValue(value);
  if (unit === "Å") {
    if (nice >= 10) return `${Math.round(nice / 10)} nm`;
    return nice >= 1 ? `${Math.round(nice)} Å` : `${nice.toFixed(2)} Å`;
  }
  if (unit === "px") {
    return nice >= 1 ? `${Math.round(nice)} px` : `${nice.toFixed(1)} px`;
  }
  if (nice >= 1000) return `${Math.round(nice / 1000)} rad`;
  return nice >= 1 ? `${Math.round(nice)} mrad` : `${nice.toFixed(2)} mrad`;
}

/** Format stat value for display (compact scientific notation for small values) */
function formatStat(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) {
    return value.toExponential(2);
  }
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}

/**
 * Draw scale bar and zoom indicator on a high-DPI UI canvas.
 * This renders crisp text/lines independent of the image resolution.
 */
function drawScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: "Å" | "mrad" | "px",
  imageWidth: number,
  imageHeight: number
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Scale context for device pixel ratio
  ctx.save();
  ctx.scale(dpr, dpr);

  // CSS pixel dimensions
  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;

  // Calculate separate X/Y scale factors (canvas stretches to fill, not aspect-preserving)
  const scaleX = cssWidth / imageWidth;
  // Use X scale for horizontal measurements (scale bar is horizontal)
  const effectiveZoom = zoom * scaleX;
  
  // Fixed UI sizes in CSS pixels (always crisp)
  const targetBarPx = 60;  // Target bar length in CSS pixels
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;
  
  // Calculate what physical size the target bar represents
  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  
  // Round to a nice value
  const nicePhysical = roundToNiceValue(targetPhysical);
  
  // Calculate actual bar length for the nice value (in CSS pixels)
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  // Draw bar with shadow for visibility
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  // Draw label (centered above bar)
  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.fillStyle = "white";
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
 * Draw VI crosshair on high-DPI canvas (crisp regardless of image resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawViPositionMarker(
  canvas: HTMLCanvasElement,
  dpr: number,
  posX: number,  // Position in image coordinates
  posY: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  // Convert image coordinates to CSS pixel coordinates
  const screenX = posY * zoom * scaleX + panX * scaleX;
  const screenY = posX * zoom * scaleY + panY * scaleY;

  // Simple crosshair (no circle)
  const crosshairSize = 12;
  const lineWidth = 1.5;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(255, 100, 100, 0.9)";
  ctx.lineWidth = lineWidth;

  // Draw crosshair lines only
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();

  ctx.restore();
}

/**
 * Draw VI ROI overlay on high-DPI canvas for real-space region selection
 * Note: Does NOT clear canvas - should be called after drawViPositionMarker
 */
function drawViRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerX: number,
  centerY: number,
  radius: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isHoveringResize: boolean
) {
  if (roiMode === "off") return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  // Convert image coordinates to screen coordinates (note: Y is row, X is col in image)
  const screenX = centerY * zoom * scaleX + panX * scaleX;
  const screenY = centerX * zoom * scaleY + panY * scaleY;

  const lineWidth = 2.5;
  const crosshairSize = 10;
  const handleRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  // Helper to draw resize handle (purple color for VI ROI to differentiate from DP)
  const drawResizeHandle = (handleX: number, handleY: number) => {
    let handleFill: string;
    let handleStroke: string;

    if (isDraggingResize) {
      handleFill = "rgba(180, 100, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (isHoveringResize) {
      handleFill = "rgba(220, 150, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = "rgba(160, 80, 255, 0.8)";
      handleStroke = "rgba(255, 255, 255, 0.8)";
    }
    ctx.beginPath();
    ctx.arc(handleX, handleY, handleRadius, 0, 2 * Math.PI);
    ctx.fillStyle = handleFill;
    ctx.fill();
    ctx.strokeStyle = handleStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  // Helper to draw center crosshair (purple/magenta for VI ROI)
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSize, screenY);
    ctx.lineTo(screenX + crosshairSize, screenY);
    ctx.moveTo(screenX, screenY - crosshairSize);
    ctx.lineTo(screenX, screenY + crosshairSize);
    ctx.stroke();
  };

  // Purple/magenta color for VI ROI to differentiate from green DP detector
  const strokeColor = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
  const fillColor = isDragging ? "rgba(255, 200, 0, 0.15)" : "rgba(180, 80, 255, 0.15)";

  if (roiMode === "circle" && radius > 0) {
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square uses radius as half-size
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);
  }

  ctx.restore();
}

/**
 * Draw DP crosshair on high-DPI canvas (crisp regardless of detector resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawDpCrosshairHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  kx: number,  // Position in detector coordinates
  ky: number,
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates (no swap - kx is X, ky is Y)
  const screenX = kx * zoom * scaleX + panX * scaleX;
  const screenY = ky * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels (consistent with VI crosshair)
  const crosshairSize = 18;
  const lineWidth = 3;
  const dotRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
  ctx.lineWidth = lineWidth;
  
  // Draw crosshair
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();
  
  // Draw center dot
  ctx.beginPath();
  ctx.arc(screenX, screenY, dotRadius, 0, 2 * Math.PI);
  ctx.stroke();
  
  ctx.restore();
}

/**
 * Draw ROI overlay (circle, square, rect, annular) on high-DPI canvas
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerX: number,
  centerY: number,
  radius: number,
  radiusInner: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isDraggingResizeInner: boolean,
  isHoveringResize: boolean,
  isHoveringResizeInner: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates
  const screenX = centerX * zoom * scaleX + panX * scaleX;
  const screenY = centerY * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels
  const lineWidth = 2.5;
  const crosshairSizeSmall = 10;
  const handleRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  // Helper to draw resize handle
  const drawResizeHandle = (handleX: number, handleY: number, isInner: boolean = false) => {
    let handleFill: string;
    let handleStroke: string;
    const dragging = isInner ? isDraggingResizeInner : isDraggingResize;
    const hovering = isInner ? isHoveringResizeInner : isHoveringResize;
    
    if (dragging) {
      handleFill = "rgba(0, 200, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (hovering) {
      handleFill = "rgba(255, 100, 100, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = isInner ? "rgba(0, 220, 255, 0.8)" : "rgba(0, 255, 0, 0.8)";
      handleStroke = "rgba(255, 255, 255, 0.8)";
    }
    ctx.beginPath();
    ctx.arc(handleX, handleY, handleRadius, 0, 2 * Math.PI);
    ctx.fillStyle = handleFill;
    ctx.fill();
    ctx.strokeStyle = handleStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };
  
  // Helper to draw center crosshair
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSizeSmall, screenY);
    ctx.lineTo(screenX + crosshairSizeSmall, screenY);
    ctx.moveTo(screenX, screenY - crosshairSizeSmall);
    ctx.lineTo(screenX, screenY + crosshairSizeSmall);
    ctx.stroke();
  };
  
  if (roiMode === "circle" && radius > 0) {
    // Use separate X/Y radii for ellipse (handles non-square detectors)
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    // Draw ellipse (becomes circle if scaleX === scaleY)
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Semi-transparent fill
    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square in detector space uses same half-size in both dimensions
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "annular" && radius > 0) {
    // Use separate X/Y radii for ellipses
    const screenRadiusOuterX = radius * zoom * scaleX;
    const screenRadiusOuterY = radius * zoom * scaleY;
    const screenRadiusInnerX = (radiusInner || 0) * zoom * scaleX;
    const screenRadiusInnerY = (radiusInner || 0) * zoom * scaleY;

    // Outer ellipse (green)
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Inner ellipse (cyan)
    ctx.strokeStyle = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(0, 220, 255, 0.9)";
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Fill annular region
    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI, true);
    ctx.fill();

    drawCenterCrosshair();

    // Outer handle at 45° diagonal
    const handleOffsetOuterX = screenRadiusOuterX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetOuterY = screenRadiusOuterY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetOuterX, screenY + handleOffsetOuterY);

    // Inner handle at 45° diagonal
    const handleOffsetInnerX = screenRadiusInnerX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetInnerY = screenRadiusInnerY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetInnerX, screenY + handleOffsetInnerY, true);
  }
  
  ctx.restore();
}

// ============================================================================
// Histogram Component
// ============================================================================

/**
 * Compute histogram from byte data (0-255).
 * Returns 256 bins normalized to 0-1 range.
 */
function computeHistogramFromBytes(data: Uint8Array | Float32Array | null, numBins = 256): number[] {
  if (!data || data.length === 0) {
    return new Array(numBins).fill(0);
  }

  const bins = new Array(numBins).fill(0);

  // For Float32Array, find min/max and bin accordingly
  if (data instanceof Float32Array) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min) || !isFinite(max) || min === max) {
      return bins;
    }
    const range = max - min;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) {
        const binIdx = Math.min(numBins - 1, Math.floor(((v - min) / range) * numBins));
        bins[binIdx]++;
      }
    }
  } else {
    // Uint8Array - values are already 0-255
    for (let i = 0; i < data.length; i++) {
      const binIdx = Math.min(numBins - 1, data[i]);
      bins[binIdx]++;
    }
  }

  // Normalize bins to 0-1
  const maxCount = Math.max(...bins);
  if (maxCount > 0) {
    for (let i = 0; i < numBins; i++) {
      bins[i] /= maxCount;
    }
  }

  return bins;
}

interface HistogramProps {
  data: Uint8Array | Float32Array | null;
  colormap: string;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
}

/**
 * Histogram component with integrated vmin/vmax slider and statistics.
 * Shows data distribution with colormap gradient and adjustable clipping.
 */
function Histogram({
  data,
  colormap,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 120,
  height = 40
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

  // Draw histogram (vertical gray bars)
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear with dark background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, width, height);

    // Reduce to fewer bins for cleaner display
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) {
        sum += bins[i * binRatio + j] || 0;
      }
      reducedBins.push(sum / binRatio);
    }

    // Normalize
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;

    // Calculate which bins are in the clipped range
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);

    // Draw histogram bars (gray)
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;

      // Bars inside range are lighter gray, outside are darker
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? "#888" : "#444";
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }

  }, [bins, colormap, vminPct, vmaxPct, width, height]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, borderRadius: 2, border: "1px solid #333" }}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [newMin, newMax] = v as number[];
          onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1));
        }}
        min={0}
        max={100}
        size="small"
        sx={{
          width,
          py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
        }}
      />
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================
function Show4DSTEM() {
  // ─────────────────────────────────────────────────────────────────────────
  // Model State (synced with Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [shapeX] = useModelState<number>("shape_x");
  const [shapeY] = useModelState<number>("shape_y");
  const [detX] = useModelState<number>("det_x");
  const [detY] = useModelState<number>("det_y");

  const [posX, setPosX] = useModelState<number>("pos_x");
  const [posY, setPosY] = useModelState<number>("pos_y");
  const [roiCenterX, setRoiCenterX] = useModelState<number>("roi_center_x");
  const [roiCenterY, setRoiCenterY] = useModelState<number>("roi_center_y");
  const [, setRoiActive] = useModelState<boolean>("roi_active");

  const [pixelSize] = useModelState<number>("pixel_size");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");

  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");

  // ROI state
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInner, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  // Display options
  const [dpScaleMode, setDpScaleMode] = useModelState<string>("dp_scale_mode");
  const [dpPowerExp, setDpPowerExp] = useModelState<number>("dp_power_exp");

  // Detector calibration (for presets)
  const [bfRadius] = useModelState<number>("bf_radius");
  const [centerX] = useModelState<number>("center_x");
  const [centerY] = useModelState<number>("center_y");

  // Path animation state
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // Auto-detection trigger
  const [, setAutoDetectTrigger] = useModelState<boolean>("auto_detect_trigger");

  // ─────────────────────────────────────────────────────────────────────────
  // Local State (UI-only, not synced to Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [localKx, setLocalKx] = React.useState(roiCenterX);
  const [localKy, setLocalKy] = React.useState(roiCenterY);
  const [localPosX, setLocalPosX] = React.useState(posX);
  const [localPosY, setLocalPosY] = React.useState(posY);
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  const [isDraggingVI, setIsDraggingVI] = React.useState(false);
  const [isDraggingFFT, setIsDraggingFFT] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false); // For annular inner handle
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  // VI ROI drag/resize states (same pattern as DP)
  const [isDraggingViRoi, setIsDraggingViRoi] = React.useState(false);
  const [isDraggingViRoiResize, setIsDraggingViRoiResize] = React.useState(false);
  const [isHoveringViRoiResize, setIsHoveringViRoiResize] = React.useState(false);
  // Independent colormaps for DP and VI panels
  const [dpColormap, setDpColormap] = React.useState("inferno");
  const [viColormap, setViColormap] = React.useState("inferno");
  // vmin/vmax percentile clipping (0-100)
  const [dpVminPct, setDpVminPct] = React.useState(0);
  const [dpVmaxPct, setDpVmaxPct] = React.useState(100);
  const [viVminPct, setViVminPct] = React.useState(0);
  const [viVmaxPct, setViVmaxPct] = React.useState(100);
  // Scale mode: "linear" | "log" | "power"
  const [viScaleMode, setViScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [viPowerExp, setViPowerExp] = React.useState(0.5);

  // VI ROI state (real-space region selection for summed DP) - synced with Python
  const [viRoiMode, setViRoiMode] = useModelState<string>("vi_roi_mode");
  const [viRoiCenterX, setViRoiCenterX] = useModelState<number>("vi_roi_center_x");
  const [viRoiCenterY, setViRoiCenterY] = useModelState<number>("vi_roi_center_y");
  const [viRoiRadius, setViRoiRadius] = useModelState<number>("vi_roi_radius");
  const [viRoiWidth, setViRoiWidth] = useModelState<number>("vi_roi_width");
  const [viRoiHeight, setViRoiHeight] = useModelState<number>("vi_roi_height");
  // Local VI ROI center for smooth dragging
  const [localViRoiCenterX, setLocalViRoiCenterX] = React.useState(viRoiCenterX || 0);
  const [localViRoiCenterY, setLocalViRoiCenterY] = React.useState(viRoiCenterY || 0);
  const [summedDpBytes] = useModelState<DataView>("summed_dp_bytes");
  const [summedDpCount] = useModelState<number>("summed_dp_count");
  const [dpStats] = useModelState<number[]>("dp_stats");  // [mean, min, max, std]
  const [viStats] = useModelState<number[]>("vi_stats");  // [mean, min, max, std]
  const [showFft, setShowFft] = React.useState(false);  // Hidden by default per feedback

  // Histogram data - use state to ensure re-renders
  const [dpHistogramData, setDpHistogramData] = React.useState<Uint8Array | null>(null);
  const [viHistogramData, setViHistogramData] = React.useState<Float32Array | null>(null);

  // Parse DP frame bytes for histogram
  React.useEffect(() => {
    if (!frameBytes) return;
    const bytes = new Uint8Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength);
    // Create a copy to ensure state update triggers re-render
    const copy = new Uint8Array(bytes.length);
    copy.set(bytes);
    setDpHistogramData(copy);
  }, [frameBytes]);

  // Band-pass filter range [innerCutoff, outerCutoff] in pixels - [0, 0] means disabled
  const [bandpass, setBandpass] = React.useState<number[]>([0, 0]);
  const bpInner = bandpass[0];
  const bpOuter = bandpass[1];

  // GPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Path animation timer
  React.useEffect(() => {
    if (!pathPlaying || pathLength === 0) return;

    const timer = setInterval(() => {
      setPathIndex((prev: number) => {
        const next = prev + 1;
        if (next >= pathLength) {
          if (pathLoop) {
            return 0;  // Loop back to start
          } else {
            setPathPlaying(false);  // Stop at end
            return prev;
          }
        }
        return next;
      });
    }, pathIntervalMs);

    return () => clearInterval(timer);
  }, [pathPlaying, pathLength, pathIntervalMs, pathLoop, setPathIndex, setPathPlaying]);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      const step = e.shiftKey ? 10 : 1;

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          setPosX(Math.max(0, posX - step));
          break;
        case 'ArrowDown':
          e.preventDefault();
          setPosX(Math.min(shapeX - 1, posX + step));
          break;
        case 'ArrowLeft':
          e.preventDefault();
          setPosY(Math.max(0, posY - step));
          break;
        case 'ArrowRight':
          e.preventDefault();
          setPosY(Math.min(shapeY - 1, posY + step));
          break;
        case ' ':  // Space bar
          e.preventDefault();
          if (pathLength > 0) {
            setPathPlaying(!pathPlaying);
          }
          break;
        case 'r':  // Reset view
        case 'R':
          setDpZoom(1); setDpPanX(0); setDpPanY(0);
          setViZoom(1); setViPanX(0); setViPanY(0);
          setFftZoom(1); setFftPanX(0); setFftPanY(0);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [posX, posY, shapeX, shapeY, pathPlaying, pathLength, setPosX, setPosY, setPathPlaying]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  // Root element ref (theme-aware styling handled via CSS variables)
  const rootRef = React.useRef<HTMLDivElement>(null);

  // Zoom state
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [viZoom, setViZoom] = React.useState(1);
  const [viPanX, setViPanX] = React.useState(0);
  const [viPanY, setViPanY] = React.useState(0);
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);

  // Sync local state
  React.useEffect(() => {
    if (!isDraggingDP && !isDraggingResize) { setLocalKx(roiCenterX); setLocalKy(roiCenterY); }
  }, [roiCenterX, roiCenterY, isDraggingDP, isDraggingResize]);

  React.useEffect(() => {
    if (!isDraggingVI) { setLocalPosX(posX); setLocalPosY(posY); }
  }, [posX, posY, isDraggingVI]);

  // Sync VI ROI local state
  React.useEffect(() => {
    if (!isDraggingViRoi && !isDraggingViRoiResize) {
      setLocalViRoiCenterX(viRoiCenterX || shapeX / 2);
      setLocalViRoiCenterY(viRoiCenterY || shapeY / 2);
    }
  }, [viRoiCenterX, viRoiCenterY, isDraggingViRoi, isDraggingViRoiResize, shapeX, shapeY]);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const dpImageDataRef = React.useRef<ImageData | null>(null);
  const virtualCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const virtualOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const viUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const viOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const viImageDataRef = React.useRef<ImageData | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftImageDataRef = React.useRef<ImageData | null>(null);

  // Device pixel ratio for high-DPI UI overlays
  const DPR = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // ─────────────────────────────────────────────────────────────────────────
  // Effects: Canvas Rendering & Animation
  // ─────────────────────────────────────────────────────────────────────────

  // Prevent page scroll when scrolling on canvases
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [dpOverlayRef.current, virtualOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, []);

  // Store raw data for histogram visualization
  const dpBytesRef = React.useRef<Uint8Array | null>(null);
  const rawVirtualImageRef = React.useRef<Float32Array | null>(null);
  const viWorkRealRef = React.useRef<Float32Array | null>(null);
  const viWorkImagRef = React.useRef<Float32Array | null>(null);
  const fftWorkRealRef = React.useRef<Float32Array | null>(null);
  const fftWorkImagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeRef = React.useRef<Float32Array | null>(null);

  // Parse virtual image bytes into Float32Array
  React.useEffect(() => {
    if (!virtualImageBytes) return;
    const bytes = new Uint8Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, virtualImageBytes.byteLength);
    let floatData = rawVirtualImageRef.current;
    if (!floatData || floatData.length !== bytes.length) {
      floatData = new Float32Array(bytes.length);
      rawVirtualImageRef.current = floatData;
    }
    for (let i = 0; i < bytes.length; i++) {
      floatData[i] = bytes[i];
    }
    // Update histogram state (triggers re-render)
    setViHistogramData(floatData);
  }, [virtualImageBytes]);

  // Render DP with zoom (use summed DP when VI ROI is active)
  React.useEffect(() => {
    if (!dpCanvasRef.current) return;

    // Determine which bytes to display: summed DP (if VI ROI active) or single frame
    const usesSummedDp = viRoiMode && viRoiMode !== "off" && summedDpBytes && summedDpBytes.byteLength > 0;
    const sourceBytes = usesSummedDp ? summedDpBytes : frameBytes;
    if (!sourceBytes) return;

    const canvas = dpCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bytes = new Uint8Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength);
    dpBytesRef.current = bytes;  // Store for histogram
    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;

    // Apply vmin/vmax percentile clipping
    const vmin = Math.floor(255 * dpVminPct / 100);
    const vmax = Math.ceil(255 * dpVmaxPct / 100);
    const range = vmax > vmin ? vmax - vmin : 1;

    let offscreen = dpOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      dpOffscreenRef.current = offscreen;
    }
    const sizeChanged = offscreen.width !== detY || offscreen.height !== detX;
    if (sizeChanged) {
      offscreen.width = detY;
      offscreen.height = detX;
      dpImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = dpImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(detY, detX);
      dpImageDataRef.current = imgData;
    }
    const rgba = imgData.data;

    for (let i = 0; i < bytes.length; i++) {
      // Apply vmin/vmax clipping and rescaling
      const clamped = Math.max(vmin, Math.min(vmax, bytes[i]));
      const v = Math.round(((clamped - vmin) / range) * 255);
      const j = i * 4;
      const lutIdx = v * 3;
      rgba[j] = lut[lutIdx];
      rgba[j + 1] = lut[lutIdx + 1];
      rgba[j + 2] = lut[lutIdx + 2];
      rgba[j + 3] = 255;
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(dpPanX, dpPanY);
    ctx.scale(dpZoom, dpZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [frameBytes, summedDpBytes, viRoiMode, detX, detY, dpColormap, dpVminPct, dpVmaxPct, dpZoom, dpPanX, dpPanY]);

  // Render DP overlay - just clear (ROI shapes now drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!dpOverlayRef.current) return;
    const canvas = dpOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // All visual overlays (crosshair, ROI shapes, scale bar) are now on dpUiRef for crisp rendering
  }, [localKx, localKy, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, dpZoom, dpPanX, dpPanY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, detX, detY]);

  // Render filtered virtual image
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !virtualCanvasRef.current) return;
    const canvas = virtualCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = shapeY;
    const height = shapeX;

    const renderData = (filtered: Float32Array) => {
      // Normalize and render
      // Apply scale transformation first
      let scaled = filtered;
      if (viScaleMode === "log") {
        scaled = new Float32Array(filtered.length);
        for (let i = 0; i < filtered.length; i++) {
          scaled[i] = Math.log1p(Math.max(0, filtered[i]));
        }
      } else if (viScaleMode === "power") {
        scaled = new Float32Array(filtered.length);
        for (let i = 0; i < filtered.length; i++) {
          scaled[i] = Math.pow(Math.max(0, filtered[i]), viPowerExp);
        }
      }

      // Compute actual min/max of scaled data
      let dataMin = Infinity, dataMax = -Infinity;
      for (let i = 0; i < scaled.length; i++) {
        if (scaled[i] < dataMin) dataMin = scaled[i];
        if (scaled[i] > dataMax) dataMax = scaled[i];
      }

      // Apply vmin/vmax percentile clipping
      const dataRange = dataMax - dataMin;
      const vmin = dataMin + dataRange * viVminPct / 100;
      const vmax = dataMin + dataRange * viVmaxPct / 100;
      const range = vmax > vmin ? vmax - vmin : 1;

      const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;
      let offscreen = viOffscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        viOffscreenRef.current = offscreen;
      }
      const sizeChanged = offscreen.width !== width || offscreen.height !== height;
      if (sizeChanged) {
        offscreen.width = width;
        offscreen.height = height;
        viImageDataRef.current = null;
      }
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      let imageData = viImageDataRef.current;
      if (!imageData) {
        imageData = offCtx.createImageData(width, height);
        viImageDataRef.current = imageData;
      }
      for (let i = 0; i < scaled.length; i++) {
        // Clamp to vmin/vmax and rescale to 0-255
        const clamped = Math.max(vmin, Math.min(vmax, scaled[i]));
        const val = Math.floor(((clamped - vmin) / range) * 255);
        imageData.data[i * 4] = lut[val * 3];
        imageData.data[i * 4 + 1] = lut[val * 3 + 1];
        imageData.data[i * 4 + 2] = lut[val * 3 + 2];
        imageData.data[i * 4 + 3] = 255;
      }
      offCtx.putImageData(imageData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(viPanX, viPanY);
      ctx.scale(viZoom, viZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    };

    if (bpInner > 0 || bpOuter > 0) {
      if (gpuFFTRef.current && gpuReady) {
        // GPU filtering (Async)
        const real = rawVirtualImageRef.current.slice();
        const imag = new Float32Array(real.length);

        // We use a local flag to prevent state updates if the effect has already re-run
        let isCancelled = false;

        const runGpuFilter = async () => {
          // WebGPU version of: Forward -> Filter -> Inverse
          // Note: The provided WebGPUFFT doesn't have shift/unshift built-in yet, 
          // but we can apply the filter in shifted coordinates or modify it.
          // For now, let's keep it simple: Forward -> Filter -> Inverse.
          const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, width, height, false);

          if (isCancelled) return;

          // Shift in CPU for now (future: do this in WGSL)
          fftshift(fReal, width, height);
          fftshift(fImag, width, height);
          applyBandPassFilter(fReal, fImag, width, height, bpInner, bpOuter);
          fftshift(fReal, width, height);
          fftshift(fImag, width, height);

          const { real: invReal } = await gpuFFTRef.current!.fft2D(fReal, fImag, width, height, true);

          if (!isCancelled) renderData(invReal);
        };

        runGpuFilter();
        return () => { isCancelled = true; };
      } else {
        // CPU Fallback (Sync)
        const source = rawVirtualImageRef.current;
        if (!source) return;
        const len = source.length;
        let real = viWorkRealRef.current;
        if (!real || real.length !== len) {
          real = new Float32Array(len);
          viWorkRealRef.current = real;
        }
        real.set(source);
        let imag = viWorkImagRef.current;
        if (!imag || imag.length !== len) {
          imag = new Float32Array(len);
          viWorkImagRef.current = imag;
        } else {
          imag.fill(0);
        }
        fft2d(real, imag, width, height, false);
        fftshift(real, width, height);
        fftshift(imag, width, height);
        applyBandPassFilter(real, imag, width, height, bpInner, bpOuter);
        fftshift(real, width, height);
        fftshift(imag, width, height);
        fft2d(real, imag, width, height, true);
        renderData(real);
      }
    } else {
      if (!rawVirtualImageRef.current) return;
      renderData(rawVirtualImageRef.current);
    }
  }, [virtualImageBytes, shapeX, shapeY, viColormap, viVminPct, viVmaxPct, viScaleMode, viPowerExp, viZoom, viPanX, viPanY, bpInner, bpOuter, gpuReady]);

  // Render virtual image overlay (just clear - crosshair drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!virtualOverlayRef.current) return;
    const canvas = virtualOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Crosshair and scale bar now drawn on high-DPI UI canvas (viUiRef)
  }, [localPosX, localPosY, isDraggingVI, viZoom, viPanX, viPanY, pixelSize, shapeX, shapeY]);

  // Render FFT (WebGPU when available, CPU fallback)
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (!showFft) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const width = shapeY;
    const height = shapeX;
    const sourceData = rawVirtualImageRef.current;
    const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;

    // Helper to render magnitude to canvas
    const renderMagnitude = (real: Float32Array, imag: Float32Array) => {
      // Compute log magnitude
      let magnitude = fftMagnitudeRef.current;
      if (!magnitude || magnitude.length !== real.length) {
        magnitude = new Float32Array(real.length);
        fftMagnitudeRef.current = magnitude;
      }
      for (let i = 0; i < real.length; i++) {
        magnitude[i] = Math.log1p(Math.sqrt(real[i] * real[i] + imag[i] * imag[i]));
      }

      // Normalize
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < magnitude.length; i++) {
        if (magnitude[i] < min) min = magnitude[i];
        if (magnitude[i] > max) max = magnitude[i];
      }

      let offscreen = fftOffscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        fftOffscreenRef.current = offscreen;
      }
      const sizeChanged = offscreen.width !== width || offscreen.height !== height;
      if (sizeChanged) {
        offscreen.width = width;
        offscreen.height = height;
        fftImageDataRef.current = null;
      }
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      let imgData = fftImageDataRef.current;
      if (!imgData) {
        imgData = offCtx.createImageData(width, height);
        fftImageDataRef.current = imgData;
      }
      const rgba = imgData.data;
      const range = max > min ? max - min : 1;

      for (let i = 0; i < magnitude.length; i++) {
        const v = Math.round(((magnitude[i] - min) / range) * 255);
        const j = i * 4;
        const lutIdx = Math.max(0, Math.min(255, v)) * 3;
        rgba[j] = lut[lutIdx];
        rgba[j + 1] = lut[lutIdx + 1];
        rgba[j + 2] = lut[lutIdx + 2];
        rgba[j + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(fftPanX, fftPanY);
      ctx.scale(fftZoom, fftZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    };

    // Try WebGPU first, fall back to CPU
    if (gpuFFTRef.current && gpuReady) {
      // WebGPU path (async)
      let isCancelled = false;
      const runGpuFFT = async () => {
        const real = sourceData.slice();
        const imag = new Float32Array(real.length);
        
        const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, width, height, false);
        if (isCancelled) return;
        
        // Shift in CPU (TODO: move to GPU shader)
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);
        
        renderMagnitude(fReal, fImag);
      };
      runGpuFFT();
      return () => { isCancelled = true; };
    } else {
      // CPU fallback (sync)
      const len = sourceData.length;
      let real = fftWorkRealRef.current;
      if (!real || real.length !== len) {
        real = new Float32Array(len);
        fftWorkRealRef.current = real;
      }
      real.set(sourceData);
      let imag = fftWorkImagRef.current;
      if (!imag || imag.length !== len) {
        imag = new Float32Array(len);
        fftWorkImagRef.current = imag;
      } else {
        imag.fill(0);
      }
      fft2d(real, imag, width, height, false);
      fftshift(real, width, height);
      fftshift(imag, width, height);
      renderMagnitude(real, imag);
    }
  }, [virtualImageBytes, shapeX, shapeY, viColormap, fftZoom, fftPanX, fftPanY, gpuReady, showFft]);

  // Render FFT overlay with high-pass filter circle
  React.useEffect(() => {
    if (!fftOverlayRef.current) return;
    const canvas = fftOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!showFft) return;

    // Draw band-pass filter circles (inner = HP, outer = LP)
    const centerX = (shapeY / 2) * fftZoom + fftPanX;
    const centerY = (shapeX / 2) * fftZoom + fftPanY;
    const minScanSize = Math.min(shapeX, shapeY);
    const fftLineWidth = Math.max(LINE_WIDTH_MIN_PX, Math.min(LINE_WIDTH_MAX_PX, minScanSize * LINE_WIDTH_FRACTION));

    if (bpInner > 0) {
      ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
      ctx.lineWidth = fftLineWidth;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, bpInner * fftZoom, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (bpOuter > 0) {
      ctx.strokeStyle = "rgba(0, 150, 255, 0.8)";
      ctx.lineWidth = fftLineWidth;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, bpOuter * fftZoom, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [fftZoom, fftPanX, fftPanY, pixelSize, shapeX, shapeY, bpInner, bpOuter, showFft]);

  // ─────────────────────────────────────────────────────────────────────────
  // High-DPI Scale Bar UI Overlays
  // ─────────────────────────────────────────────────────────────────────────
  
  // DP scale bar + crosshair + ROI overlay (high-DPI)
  React.useEffect(() => {
    if (!dpUiRef.current) return;
    // Draw scale bar first (clears canvas)
    const kUnit = kCalibrated ? "mrad" : "px";
    drawScaleBarHiDPI(dpUiRef.current, DPR, dpZoom, kPixelSize || 1, kUnit, detY, detX);
    // Draw ROI overlay (circle, square, rect, annular) or point crosshair
    if (roiMode === "point") {
      drawDpCrosshairHiDPI(dpUiRef.current, DPR, localKx, localKy, dpZoom, dpPanX, dpPanY, detY, detX, isDraggingDP);
    } else {
      drawRoiOverlayHiDPI(
        dpUiRef.current, DPR, roiMode,
        localKx, localKy, roiRadius, roiRadiusInner, roiWidth, roiHeight,
        dpZoom, dpPanX, dpPanY, detY, detX,
        isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner
      );
    }
  }, [dpZoom, dpPanX, dpPanY, kPixelSize, kCalibrated, detX, detY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, localKx, localKy, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner]);
  
  // VI scale bar + crosshair + ROI (high-DPI)
  React.useEffect(() => {
    if (!viUiRef.current) return;
    // Draw scale bar first (clears canvas)
    drawScaleBarHiDPI(viUiRef.current, DPR, viZoom, pixelSize || 1, "Å", shapeY, shapeX);
    // Draw crosshair only when ROI is off (ROI replaces the crosshair)
    if (!viRoiMode || viRoiMode === "off") {
      drawViPositionMarker(viUiRef.current, DPR, localPosX, localPosY, viZoom, viPanX, viPanY, shapeY, shapeX, isDraggingVI);
    } else {
      // Draw VI ROI instead of crosshair
      drawViRoiOverlayHiDPI(
        viUiRef.current, DPR, viRoiMode,
        localViRoiCenterX, localViRoiCenterY, viRoiRadius || 5, viRoiWidth || 10, viRoiHeight || 10,
        viZoom, viPanX, viPanY, shapeY, shapeX,
        isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize
      );
    }
  }, [viZoom, viPanX, viPanY, pixelSize, shapeX, shapeY, localPosX, localPosY, isDraggingVI,
      viRoiMode, localViRoiCenterX, localViRoiCenterY, viRoiRadius, viRoiWidth, viRoiHeight,
      isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize]);
  
  // Generic zoom handler
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanX: React.Dispatch<React.SetStateAction<number>>,
    setPanY: React.Dispatch<React.SetStateAction<number>>,
    zoom: number, panX: number, panY: number,
    canvasRef: React.RefObject<HTMLCanvasElement | null>
  ) => (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanX(mouseX - (mouseX - panX) * zoomRatio);
    setPanY(mouseY - (mouseY - panY) * zoomRatio);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Mouse Handlers
  // ─────────────────────────────────────────────────────────────────────────

  // Helper: check if point is near the outer resize handle
  const isNearResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "rect") {
      // For rectangle, check near bottom-right corner
      const handleX = roiCenterX + roiWidth / 2;
      const handleY = roiCenterY + roiHeight / 2;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      return dist < RESIZE_HIT_AREA_PX / dpZoom;
    }
    if ((roiMode !== "circle" && roiMode !== "square" && roiMode !== "annular") || !roiRadius) return false;
    const offset = roiMode === "square" ? roiRadius : roiRadius * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterX + offset;
    const handleY = roiCenterY + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (imgX: number, imgY: number): boolean => {
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterX + offset;
    const handleY = roiCenterY + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Helper: check if point is near VI ROI resize handle (same logic as DP)
  // Hit area is capped to avoid overlap with center for small ROIs
  const isNearViRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    if (viRoiMode === "rect") {
      const halfH = (viRoiHeight || 10) / 2;
      const halfW = (viRoiWidth || 10) / 2;
      const handleX = localViRoiCenterX + halfH;
      const handleY = localViRoiCenterY + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / viZoom, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (viRoiMode === "circle" || viRoiMode === "square") {
      const radius = viRoiRadius || 5;
      const offset = viRoiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localViRoiCenterX + offset;
      const handleY = localViRoiCenterY + offset;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      // Cap hit area to 50% of radius so center remains draggable
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / viZoom, radius * 0.5);
      return dist < hitArea;
    }
    return false;
  };

  // Mouse handlers
  const handleDpMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Check if clicking on resize handle (inner first, then outer)
    if (isNearResizeHandleInner(imgX, imgY)) {
      setIsDraggingResizeInner(true);
      return;
    }
    if (isNearResizeHandle(imgX, imgY)) {
      setIsDraggingResize(true);
      return;
    }

    setIsDraggingDP(true);
    setLocalKx(imgX); setLocalKy(imgY);
    setRoiActive(true);
    setRoiCenterX(Math.round(Math.max(0, Math.min(detY - 1, imgX))));
    setRoiCenterY(Math.round(Math.max(0, Math.min(detX - 1, imgY))));
  };

  const handleDpMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Handle inner resize dragging (annular mode)
    if (isDraggingResizeInner) {
      const dx = Math.abs(imgX - roiCenterX);
      const dy = Math.abs(imgY - roiCenterY);
      const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
      // Inner radius must be less than outer radius
      setRoiRadiusInner(Math.max(1, Math.min(roiRadius - 1, Math.round(newRadius))));
      return;
    }

    // Handle outer resize dragging - use model state center, not local values
    if (isDraggingResize) {
      const dx = Math.abs(imgX - roiCenterX);
      const dy = Math.abs(imgY - roiCenterY);
      if (roiMode === "rect") {
        // For rectangle, update width and height independently
        setRoiWidth(Math.max(2, Math.round(dx * 2)));
        setRoiHeight(Math.max(2, Math.round(dy * 2)));
      } else {
        const newRadius = roiMode === "square" ? Math.max(dx, dy) : Math.sqrt(dx ** 2 + dy ** 2);
        // For annular mode, outer radius must be greater than inner radius
        const minRadius = roiMode === "annular" ? (roiRadiusInner || 0) + 1 : 1;
        setRoiRadius(Math.max(minRadius, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles
    if (!isDraggingDP) {
      setIsHoveringResizeInner(isNearResizeHandleInner(imgX, imgY));
      setIsHoveringResize(isNearResizeHandle(imgX, imgY));
      return;
    }

    setLocalKx(imgX); setLocalKy(imgY);
    setRoiCenterX(Math.round(Math.max(0, Math.min(detY - 1, imgX))));
    setRoiCenterY(Math.round(Math.max(0, Math.min(detX - 1, imgY))));
  };

  const handleDpMouseUp = () => { setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false); };
  const handleDpMouseLeave = () => { setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false); setIsHoveringResize(false); setIsHoveringResizeInner(false); };
  const handleDpDoubleClick = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  const handleViMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // Check if VI ROI mode is active - same logic as DP
    if (viRoiMode && viRoiMode !== "off") {
      // Check if clicking on resize handle
      if (isNearViRoiResizeHandle(imgX, imgY)) {
        setIsDraggingViRoiResize(true);
        return;
      }

      // Otherwise, move ROI center to click position (same as DP)
      setIsDraggingViRoi(true);
      setLocalViRoiCenterX(imgX);
      setLocalViRoiCenterY(imgY);
      setViRoiCenterX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
      setViRoiCenterY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
      return;
    }

    // Regular position selection (when ROI is off)
    setIsDraggingVI(true);
    setLocalPosX(imgX); setLocalPosY(imgY);
    setPosX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
    setPosY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
  };

  const handleViMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // Handle VI ROI resize dragging (same pattern as DP)
    if (isDraggingViRoiResize) {
      const dx = Math.abs(imgX - localViRoiCenterX);
      const dy = Math.abs(imgY - localViRoiCenterY);
      if (viRoiMode === "rect") {
        setViRoiWidth(Math.max(2, Math.round(dy * 2)));
        setViRoiHeight(Math.max(2, Math.round(dx * 2)));
      } else if (viRoiMode === "square") {
        const newHalfSize = Math.max(dx, dy);
        setViRoiRadius(Math.max(1, Math.round(newHalfSize)));
      } else {
        // circle
        const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
        setViRoiRadius(Math.max(1, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles (same as DP)
    if (!isDraggingViRoi) {
      setIsHoveringViRoiResize(isNearViRoiResizeHandle(imgX, imgY));
      if (viRoiMode && viRoiMode !== "off") return;  // Don't update position when ROI active
    }

    // Handle VI ROI center dragging (same as DP)
    if (isDraggingViRoi) {
      setLocalViRoiCenterX(imgX);
      setLocalViRoiCenterY(imgY);
      setViRoiCenterX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
      setViRoiCenterY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
      return;
    }

    // Handle regular position dragging (when ROI is off)
    if (!isDraggingVI) return;
    setLocalPosX(imgX); setLocalPosY(imgY);
    setPosX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
    setPosY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
  };

  const handleViMouseUp = () => {
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
  };
  const handleViMouseLeave = () => {
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
    setIsHoveringViRoiResize(false);
  };
  const handleViDoubleClick = () => { setViZoom(1); setViPanX(0); setViPanY(0); };
  const handleFftDoubleClick = () => { setFftZoom(1); setFftPanX(0); setFftPanY(0); };

  // FFT drag-to-pan handlers
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDraggingFFT(true);
    setFftDragStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingFFT || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - fftDragStart.x) * scaleX;
    const dy = (e.clientY - fftDragStart.y) * scaleY;
    setFftPanX(fftDragStart.panX + dx);
    setFftPanY(fftDragStart.panY + dy);
  };

  const handleFftMouseUp = () => { setIsDraggingFFT(false); setFftDragStart(null); };
  const handleFftMouseLeave = () => { setIsDraggingFFT(false); setFftDragStart(null); };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  // Export DP handler
  const handleExportDP = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      exported_at: new Date().toISOString(),
      type: "diffraction_pattern",
      scan_position: { x: posX, y: posY },
      scan_shape: { x: shapeX, y: shapeY },
      detector_shape: { x: detX, y: detY },
      roi: { mode: roiMode, center_x: roiCenterX, center_y: roiCenterY, radius_outer: roiRadius, radius_inner: roiRadiusInner },
      display: { colormap: dpColormap, vmin_pct: dpVminPct, vmax_pct: dpVmaxPct, scale_mode: dpScaleMode },
      calibration: { bf_radius: bfRadius, center_x: centerX, center_y: centerY, k_pixel_size: kPixelSize, k_calibrated: kCalibrated },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const link = document.createElement('a');
    link.download = `dp_export_${timestamp}.zip`;
    link.href = URL.createObjectURL(zipBlob);
    link.click();
    URL.revokeObjectURL(link.href);
  };

  // Export VI handler
  const handleExportVI = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      exported_at: new Date().toISOString(),
      scan_position: { x: posX, y: posY },
      scan_shape: { x: shapeX, y: shapeY },
      detector_shape: { x: detX, y: detY },
      roi: { mode: roiMode, center_x: roiCenterX, center_y: roiCenterY, radius_outer: roiRadius, radius_inner: roiRadiusInner },
      display: { dp_colormap: dpColormap, vi_colormap: viColormap, dp_scale_mode: dpScaleMode, vi_scale_mode: viScaleMode },
      calibration: { bf_radius: bfRadius, center_x: centerX, center_y: centerY, pixel_size: pixelSize, k_pixel_size: kPixelSize },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (virtualCanvasRef.current) zip.file("virtual_image.png", await canvasToBlob(virtualCanvasRef.current));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    if (fftCanvasRef.current) zip.file("fft.png", await canvasToBlob(fftCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const link = document.createElement('a');
    link.download = `4dstem_export_${timestamp}.zip`;
    link.href = URL.createObjectURL(zipBlob);
    link.click();
    URL.revokeObjectURL(link.href);
  };

  // Common styles for panel control groups (fills parent width = canvas width)
  const panelControlStyle = {
    display: "flex",
    alignItems: "center",
    gap: `${SPACING.SM}px`,
    width: "100%",
    boxSizing: "border-box",
  };

  return (
    <Box ref={rootRef} className="show4dstem-root" sx={{ p: `${SPACING.LG}px`, bgcolor: "#1e1e1e", color: "#e0e0e0" }}>
      {/* HEADER */}
      <Typography variant="h6" sx={{ ...typography.title, mb: `${SPACING.SM}px` }}>
        4D-STEM Explorer
      </Typography>

      {/* MAIN CONTENT: Two columns */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* LEFT COLUMN: DP Panel */}
        <Box sx={{ width: CANVAS_SIZE }}>
          {/* DP Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typography.label }}>
              DP at ({Math.round(localPosX)}, {Math.round(localPosY)})
              <span style={{ color: "#0f0", marginLeft: SPACING.SM }}>k: ({Math.round(localKx)}, {Math.round(localKy)})</span>
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`}>
              <Button size="small" sx={compactButton} onClick={() => { setDpZoom(1); setDpPanX(0); setDpPanY(0); setRoiCenterX(centerX); setRoiCenterY(centerY); }}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleExportDP}>Export</Button>
            </Stack>
          </Stack>

          {/* DP Canvas */}
          <Box sx={{ ...container.imageBox, width: CANVAS_SIZE, height: CANVAS_SIZE }}>
            <canvas ref={dpCanvasRef} width={detY} height={detX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={dpOverlayRef} width={detY} height={detX}
              onMouseDown={handleDpMouseDown} onMouseMove={handleDpMouseMove}
              onMouseUp={handleDpMouseUp} onMouseLeave={handleDpMouseLeave}
              onWheel={createZoomHandler(setDpZoom, setDpPanX, setDpPanY, dpZoom, dpPanX, dpPanY, dpOverlayRef)}
              onDoubleClick={handleDpDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: isHoveringResize || isDraggingResize ? "nwse-resize" : "crosshair" }}
            />
            <canvas ref={dpUiRef} width={CANVAS_SIZE * DPR} height={CANVAS_SIZE * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
          </Box>

          {/* DP Stats Bar */}
          {dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: "#1a1a1a", borderRadius: "2px", display: "flex", gap: 2 }}>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Mean <Box component="span" sx={{ color: "#5af" }}>{formatStat(dpStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Min <Box component="span" sx={{ color: "#5af" }}>{formatStat(dpStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Max <Box component="span" sx={{ color: "#5af" }}>{formatStat(dpStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Std <Box component="span" sx={{ color: "#5af" }}>{formatStat(dpStats[3])}</Box></Typography>
            </Box>
          )}

          {/* DP Controls - two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, border: "1px solid #3a3a3a", borderRadius: "2px", px: 1, py: 0.5, bgcolor: "#252525", width: "100%", boxSizing: "border-box" }}>
            {/* Left: two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1 }}>
              {/* Row 1: Detector + slider */}
              <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Detector:</Typography>
                <Select value={roiMode || "point"} onChange={(e) => setRoiMode(e.target.value)} size="small" sx={{ ...controlPanel.select, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="point">Point</MenuItem>
                  <MenuItem value="circle">Circle</MenuItem>
                  <MenuItem value="square">Square</MenuItem>
                  <MenuItem value="rect">Rect</MenuItem>
                  <MenuItem value="annular">Annular</MenuItem>
                </Select>
                {(roiMode === "circle" || roiMode === "square" || roiMode === "annular") && (
                  <>
                    <Slider
                      value={roiMode === "annular" ? [roiRadiusInner, roiRadius] : [roiRadius]}
                      onChange={(_, v) => {
                        if (roiMode === "annular") {
                          const [inner, outer] = v as number[];
                          setRoiRadiusInner(Math.min(inner, outer - 1));
                          setRoiRadius(Math.max(outer, inner + 1));
                        } else {
                          setRoiRadius(v as number);
                        }
                      }}
                      min={1}
                      max={Math.min(detX, detY) / 2}
                      size="small"
                      sx={{
                        width: roiMode === "annular" ? 100 : 70,
                        mx: 1,
                        "& .MuiSlider-thumb": { width: 14, height: 14 }
                      }}
                    />
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      {roiMode === "annular" ? `${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : `${Math.round(roiRadius)}px`}
                    </Typography>
                  </>
                )}
              </Box>
              {/* Row 2: Presets + Color + Scale */}
              <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                <Typography component="span" onClick={() => { setRoiMode("circle"); setRoiRadius(bfRadius || 10); setRoiCenterX(centerX); setRoiCenterY(centerY); }} sx={{ color: "#4f4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>BF</Typography>
                <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner((bfRadius || 10) * 0.5); setRoiRadius(bfRadius || 10); setRoiCenterX(centerX); setRoiCenterY(centerY); }} sx={{ color: "#4af", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ABF</Typography>
                <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner(bfRadius || 10); setRoiRadius(Math.min((bfRadius || 10) * 3, Math.min(detX, detY) / 2 - 2)); setRoiCenterX(centerX); setRoiCenterY(centerY); }} sx={{ color: "#fa4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ADF</Typography>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={dpColormap} onChange={(e) => setDpColormap(String(e.target.value))} size="small" sx={{ ...controlPanel.select, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...controlPanel.select, minWidth: 50, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={dpHistogramData} colormap={dpColormap} vminPct={dpVminPct} vmaxPct={dpVmaxPct} onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }} width={100} height={50} />
            </Box>
          </Box>
        </Box>

        {/* RIGHT COLUMN: VI Panel + FFT (when shown) */}
        <Box sx={{ width: CANVAS_SIZE }}>
          {/* VI Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typography.label }}>Virtual Image</Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typography.label, color: "#666", fontSize: 10 }}>
                {shapeX}×{shapeY} | {detX}×{detY}
              </Typography>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} onClick={() => { setViZoom(1); setViPanX(0); setViPanY(0); }}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleExportVI}>Export</Button>
            </Stack>
          </Stack>

          {/* VI Canvas */}
          <Box sx={{ ...container.imageBox, width: CANVAS_SIZE, height: CANVAS_SIZE }}>
            <canvas ref={virtualCanvasRef} width={shapeY} height={shapeX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={virtualOverlayRef} width={shapeY} height={shapeX}
              onMouseDown={handleViMouseDown} onMouseMove={handleViMouseMove}
              onMouseUp={handleViMouseUp} onMouseLeave={handleViMouseLeave}
              onWheel={createZoomHandler(setViZoom, setViPanX, setViPanY, viZoom, viPanX, viPanY, virtualOverlayRef)}
              onDoubleClick={handleViDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: "crosshair" }}
            />
            <canvas ref={viUiRef} width={CANVAS_SIZE * DPR} height={CANVAS_SIZE * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
          </Box>

          {/* VI Stats Bar */}
          {viStats && viStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: "#1a1a1a", borderRadius: "2px", display: "flex", gap: 2 }}>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Mean <Box component="span" sx={{ color: "#5af" }}>{formatStat(viStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Min <Box component="span" sx={{ color: "#5af" }}>{formatStat(viStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Max <Box component="span" sx={{ color: "#5af" }}>{formatStat(viStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: "#888" }}>Std <Box component="span" sx={{ color: "#5af" }}>{formatStat(viStats[3])}</Box></Typography>
            </Box>
          )}

          {/* VI Controls - Two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.MD}px`, border: "1px solid #3a3a3a", borderRadius: "2px", px: 1, py: 0.5, bgcolor: "#252525", width: "100%", boxSizing: "border-box" }}>
            {/* Left: Two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1 }}>
              {/* Row 1: ROI selector */}
              <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
                <Select value={viRoiMode || "off"} onChange={(e) => setViRoiMode(e.target.value)} size="small" sx={{ ...controlPanel.select, minWidth: 60, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="off">Off</MenuItem>
                  <MenuItem value="circle">Circle</MenuItem>
                  <MenuItem value="square">Square</MenuItem>
                  <MenuItem value="rect">Rect</MenuItem>
                </Select>
                {viRoiMode && viRoiMode !== "off" && (
                  <>
                    {(viRoiMode === "circle" || viRoiMode === "square") && (
                      <>
                        <Slider
                          value={viRoiRadius || 5}
                          onChange={(_, v) => setViRoiRadius(v as number)}
                          min={1}
                          max={Math.min(shapeX, shapeY) / 2}
                          size="small"
                          sx={{ width: 80, mx: 1 }}
                        />
                        <Typography sx={{ ...typography.value, fontSize: 10, minWidth: 30 }}>
                          {Math.round(viRoiRadius || 5)}px
                        </Typography>
                      </>
                    )}
                    {summedDpCount > 0 && (
                      <Typography sx={{ ...typography.label, fontSize: 9, color: "#a6f" }}>
                        {summedDpCount} pos
                      </Typography>
                    )}
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale */}
              <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={viColormap} onChange={(e) => setViColormap(String(e.target.value))} size="small" sx={{ ...controlPanel.select, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={viScaleMode} onChange={(e) => setViScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...controlPanel.select, minWidth: 50, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={viHistogramData} colormap={viColormap} vminPct={viVminPct} vmaxPct={viVmaxPct} onRangeChange={(min, max) => { setViVminPct(min); setViVmaxPct(max); }} width={100} height={50} />
            </Box>
          </Box>

          {/* FFT Panel (conditionally shown) */}
          {showFft && (
            <Box sx={{ mt: `${SPACING.LG}px` }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
                <Typography variant="caption" sx={{ ...typography.label }}>FFT</Typography>
                <Button size="small" sx={compactButton} onClick={() => { setFftZoom(1); setFftPanX(0); setFftPanY(0); }}>Reset</Button>
              </Stack>
              <Box sx={{ ...container.imageBox, width: CANVAS_SIZE, height: CANVAS_SIZE }}>
                <canvas ref={fftCanvasRef} width={shapeY} height={shapeX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
                <canvas
                  ref={fftOverlayRef} width={shapeY} height={shapeX}
                  onMouseDown={handleFftMouseDown} onMouseMove={handleFftMouseMove}
                  onMouseUp={handleFftMouseUp} onMouseLeave={handleFftMouseLeave}
                  onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftZoom, fftPanX, fftPanY, fftOverlayRef)}
                  onDoubleClick={handleFftDoubleClick}
                  style={{ position: "absolute", width: "100%", height: "100%", cursor: isDraggingFFT ? "grabbing" : "grab" }}
                />
              </Box>
            </Box>
          )}
        </Box>
      </Stack>

      {/* BOTTOM CONTROLS - Path only (FFT toggle moved to VI panel) */}
      {pathLength > 0 && (
        <Stack direction="row" spacing={`${SPACING.MD}px`} sx={{ mt: `${SPACING.LG}px` }}>
          <Box className="show4dstem-control-group" sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
            <Typography sx={{ ...typography.label }}>Path:</Typography>
            <Typography component="span" onClick={() => { setPathPlaying(false); setPathIndex(0); }} sx={{ color: "#888", fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }} title="Stop">⏹</Typography>
            <Typography component="span" onClick={() => setPathPlaying(!pathPlaying)} sx={{ color: pathPlaying ? "#0f0" : "#888", fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }} title={pathPlaying ? "Pause" : "Play"}>{pathPlaying ? "⏸" : "▶"}</Typography>
            <Typography sx={{ ...typography.value, minWidth: 60 }}>{pathIndex + 1}/{pathLength}</Typography>
            <Slider value={pathIndex} onChange={(_, v) => { setPathPlaying(false); setPathIndex(v as number); }} min={0} max={Math.max(0, pathLength - 1)} size="small" sx={{ width: 100 }} />
          </Box>
        </Stack>
      )}
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
