/// <reference types="@webgpu/types" />
import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import StopIcon from "@mui/icons-material/Stop";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import JSZip from "jszip";
import { useTheme } from "../theme";
import { COLORMAPS, applyColormap, renderToOffscreen } from "../colormaps";
import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, autoEnhanceFFT, nextPow2, applyHannWindow2D } from "../fft";
import { drawScaleBarHiDPI, drawColorbar, roundToNiceValue, exportFigure, canvasToPDF } from "../figure";
import { findDataRange, sliderRange, computeStats, applyLogScale, computeHistogramFromBytes, percentileClip } from "../stats";
import { downloadBlob, formatNumber, downloadDataView } from "../format";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// UI Styles - component styling helpers
// ============================================================================
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
  medium: { '& .MuiSwitch-thumb': { width: 14, height: 14 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

const sliderStyles = {
  small: {
    "& .MuiSlider-thumb": { width: 12, height: 12 },
    "& .MuiSlider-rail": { height: 3 },
    "& .MuiSlider-track": { height: 3 },
  },
};

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

// Theme-aware ROI colors for DP detector overlay
interface RoiColors {
  stroke: string;
  strokeDragging: string;
  fill: string;
  fillDragging: string;
  handleFill: string;
  innerStroke: string;
  innerStrokeDragging: string;
  innerHandleFill: string;
  textColor: string;
}
const DARK_ROI_COLORS: RoiColors = {
  stroke: "rgba(0, 255, 0, 0.9)",
  strokeDragging: "rgba(255, 255, 0, 0.9)",
  fill: "rgba(0, 255, 0, 0.12)",
  fillDragging: "rgba(255, 255, 0, 0.12)",
  handleFill: "rgba(0, 255, 0, 0.8)",
  innerStroke: "rgba(0, 220, 255, 0.9)",
  innerStrokeDragging: "rgba(255, 200, 0, 0.9)",
  innerHandleFill: "rgba(0, 220, 255, 0.8)",
  textColor: "#0f0",
};
const LIGHT_ROI_COLORS: RoiColors = {
  stroke: "rgba(0, 140, 0, 0.9)",
  strokeDragging: "rgba(200, 160, 0, 0.9)",
  fill: "rgba(0, 140, 0, 0.15)",
  fillDragging: "rgba(200, 160, 0, 0.15)",
  handleFill: "rgba(0, 140, 0, 0.85)",
  innerStroke: "rgba(0, 160, 200, 0.9)",
  innerStrokeDragging: "rgba(200, 160, 0, 0.9)",
  innerHandleFill: "rgba(0, 160, 200, 0.85)",
  textColor: "#0a0",
};

// Interaction constants
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;  // cos(45°)
// Compact button style for Reset/Export
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

// Control row style — bordered container per row.
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};

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


// ============================================================================
// FFT peak finder (snap to Bragg spot with sub-pixel centroid refinement)
// ============================================================================
function findFFTPeak(mag: Float32Array, width: number, height: number, col: number, row: number, radius: number): { row: number; col: number } {
  const c0 = Math.max(0, Math.floor(col) - radius);
  const r0 = Math.max(0, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  const wc0 = Math.max(0, bestCol - 1), wc1 = Math.min(width - 1, bestCol + 1);
  const wr0 = Math.max(0, bestRow - 1), wr1 = Math.min(height - 1, bestRow + 1);
  let sumW = 0, sumWC = 0, sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w; sumWC += w * ic; sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}
const FFT_SNAP_RADIUS = 5;

/**
 * Draw VI crosshair on high-DPI canvas (crisp regardless of image resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawViPositionMarker(
  canvas: HTMLCanvasElement,
  dpr: number,
  posRow: number,  // Position in image coordinates
  posCol: number,
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
  const screenX = posCol * zoom * scaleX + panX * scaleX;
  const screenY = posRow * zoom * scaleY + panY * scaleY;

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
  centerRow: number,
  centerCol: number,
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

  // Convert image coordinates to screen coordinates (row→screenY, col→screenX)
  const screenX = centerCol * zoom * scaleX + panX * scaleX;
  const screenY = centerRow * zoom * scaleY + panY * scaleY;

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
  kCol: number,  // Column position in detector coordinates
  kRow: number,  // Row position in detector coordinates
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean,
  roiColors: RoiColors = DARK_ROI_COLORS
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
  const screenX = kCol * zoom * scaleX + panX * scaleX;
  const screenY = kRow * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels (consistent with VI crosshair)
  const crosshairSize = 18;
  const lineWidth = 3;
  const dotRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
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
  centerCol: number,
  centerRow: number,
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
  isHoveringResizeInner: boolean,
  roiColors: RoiColors = DARK_ROI_COLORS
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
  const screenX = centerCol * zoom * scaleX + panX * scaleX;
  const screenY = centerRow * zoom * scaleY + panY * scaleY;
  
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
      handleFill = isInner ? roiColors.innerHandleFill : roiColors.handleFill;
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
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
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
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Semi-transparent fill
    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
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

    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "annular" && radius > 0) {
    // Use separate X/Y radii for ellipses
    const screenRadiusOuterX = radius * zoom * scaleX;
    const screenRadiusOuterY = radius * zoom * scaleY;
    const screenRadiusInnerX = (radiusInner || 0) * zoom * scaleX;
    const screenRadiusInnerY = (radiusInner || 0) * zoom * scaleY;

    // Outer ellipse
    ctx.strokeStyle = isDragging ? roiColors.strokeDragging : roiColors.stroke;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Inner ellipse
    ctx.strokeStyle = isDragging ? roiColors.innerStrokeDragging : roiColors.innerStroke;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Fill annular region
    ctx.fillStyle = isDragging ? roiColors.fillDragging : roiColors.fill;
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

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

/**
 * Info tooltip component - small ⓘ icon with hover tooltip
 */
function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content}
      arrow
      placement="bottom"
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: isDark ? "#333" : "#fff",
            color: isDark ? "#ddd" : "#333",
            border: `1px solid ${isDark ? "#555" : "#ccc"}`,
            maxWidth: 280,
            p: 1,
          },
        },
        arrow: {
          sx: {
            color: isDark ? "#333" : "#fff",
            "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` },
          },
        },
      }}
    >
      <Typography
        component="span"
        sx={{
          fontSize: 12,
          color: isDark ? "#888" : "#666",
          cursor: "help",
          ml: 0.5,
          "&:hover": { color: isDark ? "#aaa" : "#444" },
        }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box component="table" sx={{ borderCollapse: "collapse", "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" }, "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" } }}>
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

/**
 * Histogram component with integrated vmin/vmax slider and statistics.
 * Shows data distribution with adjustable clipping.
 */
function Histogram({
  data,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 120,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

  // Theme-aware colors
  const colors = theme === "dark" ? {
    bg: "#1a1a1a",
    barActive: "#888",
    barInactive: "#444",
    border: "#333",
  } : {
    bg: "#f0f0f0",
    barActive: "#666",
    barInactive: "#bbb",
    border: "#ccc",
  };

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

    // Clear with theme background
    ctx.fillStyle = colors.bg;
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

    // Draw histogram bars
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;

      // Bars inside range are highlighted, outside are dimmed
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }

  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
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
        valueLabelDisplay="auto"
        valueLabelFormat={(pct) => {
          const val = dataMin + (pct / 100) * (dataMax - dataMin);
          return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
        }}
        sx={{
          width,
          py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vminPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography></Box>
    </Box>
  );
}

// ============================================================================
// Line Profile Sampling
// ============================================================================

function sampleSingleLine(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(len));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = col0 + t * dc;
    const r = row0 + t * dr;
    const ci = Math.floor(c), ri = Math.floor(r);
    const cf = c - ci, rf = r - ri;
    const c0c = Math.max(0, Math.min(w - 1, ci));
    const c1c = Math.max(0, Math.min(w - 1, ci + 1));
    const r0c = Math.max(0, Math.min(h - 1, ri));
    const r1c = Math.max(0, Math.min(h - 1, ri + 1));
    out[i] = data[r0c * w + c0c] * (1 - cf) * (1 - rf) +
             data[r0c * w + c1c] * cf * (1 - rf) +
             data[r1c * w + c0c] * (1 - cf) * rf +
             data[r1c * w + c1c] * cf * rf;
  }
  return out;
}

function sampleLineProfile(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number, profileWidth: number = 1): Float32Array {
  if (profileWidth <= 1) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const dc = col1 - col0;
  const dr = row1 - row0;
  const len = Math.sqrt(dc * dc + dr * dr);
  if (len < 1e-8) return sampleSingleLine(data, w, h, row0, col0, row1, col1);
  const perpR = -dc / len;
  const perpC = dr / len;
  const half = (profileWidth - 1) / 2;
  let accumulated: Float32Array | null = null;
  for (let k = 0; k < profileWidth; k++) {
    const off = -half + k;
    const vals = sampleSingleLine(data, w, h, row0 + off * perpR, col0 + off * perpC, row1 + off * perpR, col1 + off * perpC);
    if (!accumulated) {
      accumulated = vals;
    } else {
      for (let i = 0; i < vals.length; i++) accumulated[i] += vals[i];
    }
  }
  if (accumulated) for (let i = 0; i < accumulated.length; i++) accumulated[i] /= profileWidth;
  return accumulated || new Float32Array(0);
}

function pointToSegmentDistance(col: number, row: number, col0: number, row0: number, col1: number, row1: number): number {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const lenSq = dc * dc + dr * dr;
  if (lenSq <= 1e-12) return Math.sqrt((col - col0) ** 2 + (row - row0) ** 2);
  const tRaw = ((col - col0) * dc + (row - row0) * dr) / lenSq;
  const t = Math.max(0, Math.min(1, tRaw));
  const projCol = col0 + t * dc;
  const projRow = row0 + t * dr;
  return Math.sqrt((col - projCol) ** 2 + (row - projRow) ** 2);
}

// ============================================================================
// Crop single-mode ROI region from raw float32 data for ROI-scoped FFT
// ============================================================================
function cropSingleROI(
  data: Float32Array, imgW: number, imgH: number,
  mode: string, centerRow: number, centerCol: number,
  radius: number, roiW: number, roiH: number,
): { cropped: Float32Array; cropW: number; cropH: number } | null {
  if (mode === "off") return null;
  let x0: number, y0: number, x1: number, y1: number;

  if (mode === "rect") {
    const hw = roiW / 2, hh = roiH / 2;
    x0 = Math.max(0, Math.floor(centerCol - hw));
    y0 = Math.max(0, Math.floor(centerRow - hh));
    x1 = Math.min(imgW, Math.ceil(centerCol + hw));
    y1 = Math.min(imgH, Math.ceil(centerRow + hh));
  } else {
    x0 = Math.max(0, Math.floor(centerCol - radius));
    y0 = Math.max(0, Math.floor(centerRow - radius));
    x1 = Math.min(imgW, Math.ceil(centerCol + radius));
    y1 = Math.min(imgH, Math.ceil(centerRow + radius));
  }

  const cropW = x1 - x0, cropH = y1 - y0;
  if (cropW < 2 || cropH < 2) return null;

  const cropped = new Float32Array(cropW * cropH);
  if (mode === "circle") {
    const rSq = radius * radius;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const ix = x0 + dx, iy = y0 + dy;
        const distSq = (ix - centerCol) * (ix - centerCol) + (iy - centerRow) * (iy - centerRow);
        cropped[dy * cropW + dx] = distSq <= rSq ? data[iy * imgW + ix] : 0;
      }
    }
  } else {
    for (let dy = 0; dy < cropH; dy++) {
      const srcOff = (y0 + dy) * imgW + x0;
      cropped.set(data.subarray(srcOff, srcOff + cropW), dy * cropW);
    }
  }
  return { cropped, cropW, cropH };
}

// ============================================================================
// Main Component
// ============================================================================
function Show4DSTEM() {
  // Direct model access for batched updates
  const model = useModel();

  // ─────────────────────────────────────────────────────────────────────────
  // Model State (synced with Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [shapeRows] = useModelState<number>("shape_rows");
  const [shapeCols] = useModelState<number>("shape_cols");
  const [detRows] = useModelState<number>("det_rows");
  const [detCols] = useModelState<number>("det_cols");

  const [posRow, setPosRow] = useModelState<number>("pos_row");
  const [posCol, setPosCol] = useModelState<number>("pos_col");
  const [roiCenterCol, setRoiCenterCol] = useModelState<number>("roi_center_col");
  const [roiCenterRow, setRoiCenterRow] = useModelState<number>("roi_center_row");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kPixelUnit] = useModelState<string>("k_pixel_unit");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");
  const [widgetVersion] = useModelState<string>("widget_version");
  const [title] = useModelState<string>("title");

  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");

  // ROI state
  const [roiRadiusModel, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInnerModel, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  // Global min/max for DP normalization (from Python)
  const [dpGlobalMin] = useModelState<number>("dp_global_min");
  const [dpGlobalMax] = useModelState<number>("dp_global_max");

  // VI min/max for normalization (from Python)
  // viDataMin/viDataMax are derived JS-side from virtual_image_bytes (computed below).
  // Keeping them out of Python traits avoids a comm-message ordering race where
  // bytes from click N arrive with min/max from click N-1.

  // Detector calibration (for presets)
  const [bfRadius] = useModelState<number>("bf_radius");
  const [centerCol] = useModelState<number>("center_col");
  const [centerRow] = useModelState<number>("center_row");

  // Path animation state
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // Frame animation state (5D time/tilt series)
  const [frameIdx, setFrameIdx] = useModelState<number>("frame_idx");
  const [nFrames] = useModelState<number>("n_frames");
  const [frameDimLabel] = useModelState<string>("frame_dim_label");
  const [frameLabels] = useModelState<string[]>("frame_labels");
  const [framePlaying, setFramePlaying] = useModelState<boolean>("frame_playing");
  const [frameLoop, setFrameLoop] = useModelState<boolean>("frame_loop");
  const [frameFps, setFrameFps] = useModelState<number>("frame_fps");
  const [frameReverse, setFrameReverse] = useModelState<boolean>("frame_reverse");
  const [frameBoomerang, setFrameBoomerang] = useModelState<boolean>("frame_boomerang");

  // Profile line state (synced with Python)
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth] = useModelState<number>("profile_width");

  // Auto-detection trigger
  // ─────────────────────────────────────────────────────────────────────────
  // Local State (UI-only, not synced to Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [localKCol, setLocalKCol] = React.useState(roiCenterCol);
  const [localKRow, setLocalKRow] = React.useState(roiCenterRow);
  const [localPosRow, setLocalPosRow] = React.useState(posRow);
  const [localPosCol, setLocalPosCol] = React.useState(posCol);
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  // rAF coalescing for ROI drag: collapse rapid mousemove events into ≤1
  // Python comm message per animation frame. Without this, drag fires 60+
  // events/sec at >100ms Python compute each → queue piles up → laggy UX.
  const roiCenterPendingRef = React.useRef<[number, number] | null>(null);
  const roiCenterRafRef = React.useRef<number | null>(null);
  const flushRoiCenter = React.useCallback(() => {
    if (roiCenterPendingRef.current) {
      const [r, c] = roiCenterPendingRef.current;
      model.set("roi_center", [r, c]);
      model.save_changes();
      roiCenterPendingRef.current = null;
    }
    roiCenterRafRef.current = null;
  }, [model]);
  const queueRoiCenter = React.useCallback((row: number, col: number) => {
    roiCenterPendingRef.current = [row, col];
    if (roiCenterRafRef.current === null) {
      roiCenterRafRef.current = requestAnimationFrame(flushRoiCenter);
    }
  }, [flushRoiCenter]);
  // rAF coalescing for ROI RADIUS drag — same reason as center: a no-bin BF/DF
  // recompute is ~100ms in Python, and a resize-drag fires 60+ mousemoves/sec.
  // Without coalescing every move becomes a queued comm message + recompute, so
  // the image lags ~1s behind the cursor. Collapse to <=1 radius per frame.
  // Local radius drives the ring/handle render INSTANTLY during a resize drag, so
  // the ring tracks the cursor with no snap-back, while the model trait (which
  // triggers the ~100ms Python recompute) is sent at most once per recompute.
  const [localRoiRadius, setLocalRoiRadius] = React.useState<number | null>(null);
  // Effective radius used by ALL render/hit-test code below: the live local value
  // while dragging, else the model value. Keeps the ring glued to the cursor.
  const roiRadius = localRoiRadius != null ? localRoiRadius : roiRadiusModel;
  const roiRadiusPendingRef = React.useRef<number | null>(null);
  const roiRadiusRafRef = React.useRef<number | null>(null);
  const flushRoiRadius = React.useCallback(() => {
    if (roiRadiusPendingRef.current !== null) {
      const r = roiRadiusPendingRef.current;
      roiRadiusPendingRef.current = null;
      model.set("roi_radius", r);
      model.save_changes();
    }
    roiRadiusRafRef.current = null;
  }, [model]);
  const sendRoiRadius = React.useCallback((radius: number) => {
    roiRadiusPendingRef.current = radius;
    if (roiRadiusRafRef.current === null) {
      roiRadiusRafRef.current = requestAnimationFrame(flushRoiRadius);
    }
  }, [flushRoiRadius]);
  React.useEffect(() => {
    if (localRoiRadius != null && Math.abs(localRoiRadius - roiRadiusModel) < 0.5) {
      setLocalRoiRadius(null);
    }
  }, [localRoiRadius, roiRadiusModel]);
  const [localRoiRadiusInner, setLocalRoiRadiusInner] = React.useState<number | null>(null);
  const roiRadiusInner = localRoiRadiusInner != null ? localRoiRadiusInner : roiRadiusInnerModel;
  const roiRadiusInnerPendingRef = React.useRef<number | null>(null);
  const roiRadiusInnerRafRef = React.useRef<number | null>(null);
  const flushRoiRadiusInner = React.useCallback(() => {
    if (roiRadiusInnerPendingRef.current !== null) {
      const r = roiRadiusInnerPendingRef.current;
      roiRadiusInnerPendingRef.current = null;
      model.set("roi_radius_inner", r);
      model.save_changes();
    }
    roiRadiusInnerRafRef.current = null;
  }, [model]);
  const sendRoiRadiusInner = React.useCallback((radius: number) => {
    roiRadiusInnerPendingRef.current = radius;
    if (roiRadiusInnerRafRef.current === null) {
      roiRadiusInnerRafRef.current = requestAnimationFrame(flushRoiRadiusInner);
    }
  }, [flushRoiRadiusInner]);
  React.useEffect(() => {
    if (localRoiRadiusInner != null && Math.abs(localRoiRadiusInner - roiRadiusInnerModel) < 0.5) {
      setLocalRoiRadiusInner(null);
    }
  }, [localRoiRadiusInner, roiRadiusInnerModel]);
  React.useEffect(() => () => {
    if (roiRadiusRafRef.current !== null) cancelAnimationFrame(roiRadiusRafRef.current);
    if (roiRadiusInnerRafRef.current !== null) cancelAnimationFrame(roiRadiusInnerRafRef.current);
  }, []);
  const [isDraggingVI, setIsDraggingVI] = React.useState(false);
  const [isDraggingFFT, setIsDraggingFFT] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false); // For annular inner handle
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  // VI ROI drag/resize states (same pattern as DP)
  const [isDraggingViRoi, setIsDraggingViRoi] = React.useState(false);
  const [isDraggingViRoiResize, setIsDraggingViRoiResize] = React.useState(false);
  const [isHoveringViRoiResize, setIsHoveringViRoiResize] = React.useState(false);
  // Independent colormaps for DP and VI panels
  const [showDpColorbar, setShowDpColorbar] = useModelState<boolean>("dp_show_colorbar");
  const [dpColormap, setDpColormap] = useModelState<string>("dp_colormap");
  const [viColormap, setViColormap] = useModelState<string>("vi_colormap");
  // vmin/vmax percentile clipping (0-100)
  const [dpVminPct, setDpVminPct] = useModelState<number>("dp_vmin_pct");
  const [dpVmaxPct, setDpVmaxPct] = useModelState<number>("dp_vmax_pct");
  const [viVminPct, setViVminPct] = useModelState<number>("vi_vmin_pct");
  const [viVmaxPct, setViVmaxPct] = useModelState<number>("vi_vmax_pct");
  // Absolute intensity bounds (override percentile sliders when both set)
  const [traitDpVmin] = useModelState<number | null>("dp_vmin");
  const [traitDpVmax] = useModelState<number | null>("dp_vmax");
  const [traitViVmin] = useModelState<number | null>("vi_vmin");
  const [traitViVmax] = useModelState<number | null>("vi_vmax");
  // Scale mode: "linear" | "log"
  const [dpScaleMode, setDpScaleMode] = useModelState<"linear" | "log">("dp_scale_mode");
  const [viScaleMode, setViScaleMode] = useModelState<"linear" | "log">("vi_scale_mode");
  // VI auto-contrast (1st/99th percentile clip) + Smooth (CSS bilinear blit).
  // DP doesn't need them — Bragg spots read best with the slider's percentile
  // range and nearest-neighbor blit.
  const [viAutoContrast, setViAutoContrast] = useModelState<boolean>("vi_auto_contrast");
  const [viSmooth, setViSmooth] = useModelState<boolean>("vi_smooth");

  // VI ROI state (real-space region selection for summed DP) - synced with Python
  const [viRoiMode, setViRoiMode] = useModelState<string>("vi_roi_mode");
  const [viRoiCenterRow, setViRoiCenterRow] = useModelState<number>("vi_roi_center_row");
  const [viRoiCenterCol, setViRoiCenterCol] = useModelState<number>("vi_roi_center_col");
  const [viRoiRadius, setViRoiRadius] = useModelState<number>("vi_roi_radius");
  const [viRoiWidth, setViRoiWidth] = useModelState<number>("vi_roi_width");
  const [viRoiHeight, setViRoiHeight] = useModelState<number>("vi_roi_height");
  // Local VI ROI center for smooth dragging
  const [localViRoiCenterRow, setLocalViRoiCenterRow] = React.useState(viRoiCenterRow || 0);
  const [localViRoiCenterCol, setLocalViRoiCenterCol] = React.useState(viRoiCenterCol || 0);
  const [viRoiDpBytes] = useModelState<DataView>("vi_roi_dp_bytes");
  const [viRoiReduce, setViRoiReduce] = useModelState<string>("vi_roi_reduce");
  // dp_stats are computed in JS from frameBytes (Python side no longer
  // syncs a dp_stats trait — saves 4 trait sync round-trips per click).
  const [viStats, setViStats] = React.useState<number[]>([0, 0, 0, 0]);
  const [viDataMin, setViDataMin] = React.useState<number>(0);
  const [viDataMax, setViDataMax] = React.useState<number>(1);
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [showControls] = useModelState<boolean>("show_controls");

  const effectiveShowFft = showFft;

  // ROI FFT state (VI ROI crops virtual image for FFT)
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);
  const roiFftActive = effectiveShowFft && viRoiMode !== "off";

  // Canvas resize state
  const [canvasSize, setCanvasSize] = React.useState(CANVAS_SIZE);
  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeCanvasStart, setResizeCanvasStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  // Export
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [gifMetadataJson] = useModelState<string>("_gif_metadata_json");
  const [exporting, setExporting] = React.useState(false);
  const [dpExportAnchor, setDpExportAnchor] = React.useState<HTMLElement | null>(null);
  const [viExportAnchor, setViExportAnchor] = React.useState<HTMLElement | null>(null);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; panel: string } | null>(null);

  // DP Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const profileResizeStart = React.useRef<{ startY: number; startHeight: number } | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);
  const profilePoints = profileLine || [];
  const rawDpDataRef = React.useRef<Float32Array | null>(null);
  const dpClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingDpProfileEndpoint, setDraggingDpProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingDpProfileLine, setIsDraggingDpProfileLine] = React.useState(false);
  const [hoveredDpProfileEndpoint, setHoveredDpProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringDpProfileLine, setIsHoveringDpProfileLine] = React.useState(false);
  const dpProfileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);
  const dpDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });

  // VI Line profile state
  const [viProfileActive, setViProfileActive] = React.useState(false);
  const [viProfileData, setViProfileData] = React.useState<Float32Array | null>(null);
  const [viProfilePoints, setViProfilePoints] = React.useState<Array<{ row: number; col: number }>>([]);
  const [viProfileHeight, setViProfileHeight] = React.useState(76);
  const [isResizingViProfile, setIsResizingViProfile] = React.useState(false);
  const viProfileResizeStart = React.useRef<{ startY: number; startHeight: number } | null>(null);
  const viProfileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const viProfileBaseImageRef = React.useRef<ImageData | null>(null);
  const viProfileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);
  const rawViDataRef = React.useRef<Float32Array | null>(null);
  const viClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingViProfileEndpoint, setDraggingViProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingViProfileLine, setIsDraggingViProfileLine] = React.useState(false);
  const [hoveredViProfileEndpoint, setHoveredViProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringViProfileLine, setIsHoveringViProfileLine] = React.useState(false);
  const viProfileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);
  const viRoiDragOffsetRef = React.useRef<{ dRow: number; dCol: number }>({ dRow: 0, dCol: 0 });

  // Theme detection
  const { themeInfo, colors: themeColors } = useTheme();
  const roiColors = themeInfo.theme === "dark" ? DARK_ROI_COLORS : LIGHT_ROI_COLORS;
  const accentGreen = themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a";

  // Themed typography — applies theme colors to module-level font sizes
  const typo = React.useMemo(() => ({
    label: { ...typography.label, color: themeColors.textMuted },
    labelSmall: { ...typography.labelSmall, color: themeColors.textMuted },
    value: { ...typography.value, color: themeColors.textMuted },
    title: { ...typography.title, color: themeColors.accent },
  }), [themeColors]);

  // Compute VI canvas dimensions to respect aspect ratio of rectangular scans
  const viCanvasWidth = shapeRows > shapeCols ? Math.round(canvasSize * (shapeCols / shapeRows)) : canvasSize;
  const viCanvasHeight = shapeCols > shapeRows ? Math.round(canvasSize * (shapeRows / shapeCols)) : canvasSize;

  // Histogram data - use state to ensure re-renders (both are Float32Array now)
  const [dpHistogramData, setDpHistogramData] = React.useState<Float32Array | null>(null);
  const [viHistogramData, setViHistogramData] = React.useState<Float32Array | null>(null);

  // DP stats computed JS-side from frame_bytes (was Python trait pre-refactor;
  // moving to JS skips 4 sync trait round-trips per scan-position click).
  const [dpStats, setDpStats] = React.useState<number[]>([0, 0, 0, 0]);

  // Parse DP frame bytes for histogram (float32 now)
  React.useEffect(() => {
    if (!frameBytes) return;
    // Parse as Float32Array since Python now sends raw float32
    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    // Store raw data for profile sampling
    if (!rawDpDataRef.current || rawDpDataRef.current.length !== rawData.length) {
      rawDpDataRef.current = new Float32Array(rawData.length);
    }
    rawDpDataRef.current.set(rawData);
    // Compute stats JS-side (replaces removed Python dp_stats trait)
    const s = computeStats(rawData);
    setDpStats([s.mean, s.min, s.max, s.std]);
    // Apply scale transformation for histogram display
    const scaledData = new Float32Array(rawData.length);
    if (dpScaleMode === "log") {
      for (let i = 0; i < rawData.length; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else {
      scaledData.set(rawData);
    }
    setDpHistogramData(scaledData);
  }, [frameBytes, dpScaleMode]);

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

  // Frame animation timer (5D time/tilt series)
  const frameBounceDir = React.useRef(1);
  React.useEffect(() => {
    frameBounceDir.current = frameReverse ? -1 : 1;
  }, [frameReverse]);

  React.useEffect(() => {
    if (!framePlaying || nFrames <= 1) return;

    const intervalMs = 1000 / Math.max(0.1, frameFps);
    const timer = setInterval(() => {
      setFrameIdx((prev: number) => {
        let next: number;
        if (frameBoomerang) {
          next = prev + frameBounceDir.current;
          if (next >= nFrames) { frameBounceDir.current = -1; next = nFrames - 2; }
          if (next < 0) { frameBounceDir.current = 1; next = 1; }
          next = Math.max(0, Math.min(nFrames - 1, next));
        } else {
          next = prev + (frameReverse ? -1 : 1);
          if (next >= nFrames) {
            if (frameLoop) return 0;
            setFramePlaying(false);
            return prev;
          }
          if (next < 0) {
            if (frameLoop) return nFrames - 1;
            setFramePlaying(false);
            return prev;
          }
        }
        return next;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [framePlaying, nFrames, frameFps, frameLoop, frameReverse, frameBoomerang, setFrameIdx, setFramePlaying]);

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
  const [fftScaleMode, setFftScaleMode] = useModelState<"linear" | "log">("fft_scale_mode");
  const [fftColormap, setFftColormap] = useModelState<string>("fft_colormap");
  const [fftAuto, setFftAuto] = useModelState<boolean>("fft_auto");
  const [fftVminPct, setFftVminPct] = useModelState<number>("fft_vmin_pct");
  const [fftVmaxPct, setFftVmaxPct] = useModelState<number>("fft_vmax_pct");
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);  // [mean, min, max, std]
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataMin, setFftDataMin] = React.useState(0);
  const [fftDataMax, setFftDataMax] = React.useState(1);
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);

  const isTypingTarget = React.useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    return target.closest("input, textarea, select, [role='textbox'], [contenteditable='true']") !== null;
  }, []);

  const handleRootMouseDownCapture = React.useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus();
  }, []);

  const handleKeyDown = React.useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (isTypingTarget(e.target)) return;

    const step = e.shiftKey ? 10 : 1;
    let handled = false;

    switch (e.key) {
        case "ArrowUp":
          setPosRow(Math.max(0, posRow - step));
          handled = true;
          break;
        case "ArrowDown":
          setPosRow(Math.min(shapeRows - 1, posRow + step));
          handled = true;
          break;
        case "ArrowLeft":
          setPosCol(Math.max(0, posCol - step));
          handled = true;
          break;
        case "ArrowRight":
          setPosCol(Math.min(shapeCols - 1, posCol + step));
          handled = true;
          break;
        case " ": // Space bar
          if (pathLength > 0) {
            setPathPlaying(!pathPlaying);
            handled = true;
          }
          break;
        case "r":
        case "R":
          setDpZoom(1); setDpPanX(0); setDpPanY(0);
          setViZoom(1); setViPanX(0); setViPanY(0);
          setFftZoom(1); setFftPanX(0); setFftPanY(0);
          handled = true;
          break;
        case "[":
          if (nFrames > 1) {
            setFrameIdx(Math.max(0, frameIdx - 1));
            handled = true;
          }
          break;
        case "]":
          if (nFrames > 1) {
            setFrameIdx(Math.min(nFrames - 1, frameIdx + 1));
            handled = true;
          }
          break;
        case "Escape":
          rootRef.current?.blur();
          handled = true;
          break;
    }

    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, [
    frameIdx, isTypingTarget, nFrames, pathLength,
    pathPlaying, posCol, posRow, setFrameIdx, setPathPlaying, setPosCol, setPosRow, shapeCols, shapeRows,
  ]);

  // Sync local state
  React.useEffect(() => {
    if (!isDraggingDP && !isDraggingResize) { setLocalKCol(roiCenterCol); setLocalKRow(roiCenterRow); }
  }, [roiCenterCol, roiCenterRow, isDraggingDP, isDraggingResize]);

  React.useEffect(() => {
    if (!isDraggingVI) { setLocalPosRow(posRow); setLocalPosCol(posCol); }
  }, [posRow, posCol, isDraggingVI]);

  // Sync VI ROI local state
  React.useEffect(() => {
    if (!isDraggingViRoi && !isDraggingViRoiResize) {
      setLocalViRoiCenterRow(viRoiCenterRow || shapeRows / 2);
      setLocalViRoiCenterCol(viRoiCenterCol || shapeCols / 2);
    }
  }, [viRoiCenterRow, viRoiCenterCol, isDraggingViRoi, isDraggingViRoiResize, shapeRows, shapeCols]);

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

  // Offscreen version counters — bump when colormap/data changes, cheap draw effects depend on these
  const [dpOffscreenVersion, setDpOffscreenVersion] = React.useState(0);
  const [viOffscreenVersion, setViOffscreenVersion] = React.useState(0);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);

  // Cached colorbar vmin/vmax — computed in expensive DP effect, reused in UI overlay without recomputing
  const dpColorbarVminRef = React.useRef(0);
  const dpColorbarVmaxRef = React.useRef(1);

  // Device pixel ratio for high-DPI UI overlays
  const DPR = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // ─────────────────────────────────────────────────────────────────────────
  // Effects: Canvas Rendering & Animation
  // ─────────────────────────────────────────────────────────────────────────

  // Prevent page scroll when scrolling on canvases
  // Re-run when showFft changes since FFT canvas is conditionally rendered
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [dpOverlayRef.current, virtualOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [effectiveShowFft]);

  // Store raw data for filtering/FFT
  const rawVirtualImageRef = React.useRef<Float32Array | null>(null);
  const fftWorkRealRef = React.useRef<Float32Array | null>(null);
  const fftWorkImagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeRef = React.useRef<Float32Array | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // Parse virtual image bytes into Float32Array and apply scale for histogram
  React.useEffect(() => {
    if (!virtualImageBytes) return;
    // Parse as Float32Array
    const numFloats = virtualImageBytes.byteLength / 4;
    const rawData = new Float32Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, numFloats);

    // Store a copy for filtering/FFT (rawData is a view, we need a copy)
    let storedData = rawVirtualImageRef.current;
    if (!storedData || storedData.length !== numFloats) {
      storedData = new Float32Array(numFloats);
      rawVirtualImageRef.current = storedData;
    }
    storedData.set(rawData);

    // Also store for VI profile sampling
    if (!rawViDataRef.current || rawViDataRef.current.length !== numFloats) {
      rawViDataRef.current = new Float32Array(numFloats);
    }
    rawViDataRef.current.set(rawData);

    // Compute stats + min/max JS-side (replaces removed Python vi_stats / vi_data_min / vi_data_max traits).
    // Python sending bytes + 4 separate stat traits caused a comm-message ordering race on rapid
    // preset clicks: bytes from click N could arrive with min/max from click N-1, normalizing
    // the colormap to the wrong range and producing a uniform-color VI flash.
    const s = computeStats(rawData);
    setViStats([s.mean, s.min, s.max, s.std]);
    setViDataMin(s.min);
    setViDataMax(s.max);

    // Apply scale transformation for histogram display
    const scaledData = new Float32Array(numFloats);
    if (viScaleMode === "log") {
      for (let i = 0; i < numFloats; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else {
      scaledData.set(rawData);
    }
    setViHistogramData(scaledData);
  }, [virtualImageBytes, viScaleMode]);

  // Render DP with zoom (use summed DP when VI ROI is active)
  // Expensive: colormap + data processing → cached offscreen canvas
  React.useEffect(() => {
    // Determine which bytes to display: summed DP (if VI ROI active) or single frame
    const usesViRoiDp = viRoiMode && viRoiMode !== "off" && viRoiDpBytes && viRoiDpBytes.byteLength > 0;
    const sourceBytes = usesViRoiDp ? viRoiDpBytes : frameBytes;
    if (!sourceBytes) return;

    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;

    // Parse raw float32 data and apply scale transformation
    const rawData = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
    let scaled: Float32Array;
    if (dpScaleMode === "log") {
      scaled = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) {
        scaled[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else {
      scaled = rawData;
    }

    const { min: dataMin, max: dataMax } = findDataRange(scaled);

    let vmin: number, vmax: number;
    if (traitDpVmin != null && traitDpVmax != null) {
      if (dpScaleMode === "log") {
        vmin = Math.log1p(Math.max(traitDpVmin, 0));
        vmax = Math.log1p(Math.max(traitDpVmax, 0));
      } else {
        vmin = traitDpVmin;
        vmax = traitDpVmax;
      }
    } else {
      ({ vmin, vmax } = sliderRange(dataMin, dataMax, dpVminPct, dpVmaxPct));
    }

    let offscreen = dpOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      dpOffscreenRef.current = offscreen;
    }
    const sizeChanged = offscreen.width !== detCols || offscreen.height !== detRows;
    if (sizeChanged) {
      offscreen.width = detCols;
      offscreen.height = detRows;
      dpImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = dpImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(detCols, detRows);
      dpImageDataRef.current = imgData;
    }
    applyColormap(scaled, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);
    // Cache colorbar range for the UI overlay (avoids recomputing findDataRange on every zoom/pan)
    dpColorbarVminRef.current = vmin;
    dpColorbarVmaxRef.current = vmax;
    setDpOffscreenVersion(v => v + 1);
  }, [frameBytes, viRoiDpBytes, viRoiMode, detRows, detCols, dpColormap, dpVminPct, dpVmaxPct, dpScaleMode, traitDpVmin, traitDpVmax]);

  // Cheap: zoom/pan redraw — just drawImage from cached offscreen
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  React.useLayoutEffect(() => {
    const offscreen = dpOffscreenRef.current;
    if (!offscreen || !dpCanvasRef.current) return;
    const canvas = dpCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(dpPanX, dpPanY);
    ctx.scale(dpZoom, dpZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [dpOffscreenVersion, dpZoom, dpPanX, dpPanY]);

  // Render DP overlay - just clear (ROI shapes now drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!dpOverlayRef.current) return;
    const canvas = dpOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // All visual overlays (crosshair, ROI shapes, scale bar) are now on dpUiRef for crisp rendering
  }, [localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, dpZoom, dpPanX, dpPanY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, detRows, detCols]);

  // Expensive: VI colormap + data processing → cached offscreen canvas
  React.useEffect(() => {
    if (!rawVirtualImageRef.current) return;

    const width = shapeCols;
    const height = shapeRows;
    const filtered = rawVirtualImageRef.current;

    let scaled = filtered;
    if (viScaleMode === "log") {
      scaled = new Float32Array(filtered.length);
      for (let i = 0; i < filtered.length; i++) {
        scaled[i] = Math.log1p(Math.max(0, filtered[i]));
      }
    }

    // Compute min/max from the data we just received. Do NOT use Python's
    // viDataMin/viDataMax traits here: they arrive as separate comm messages
    // and can be stale on rapid preset clicks (BF↔ABF), causing the render
    // to apply the WRONG normalization range and produce a uniform white/black
    // VI panel until comm catches up. findDataRange on a scan-shape buffer
    // (~64K-256K floats) is sub-millisecond.
    const r = findDataRange(scaled);
    const dataMin = r.min;
    const dataMax = r.max;

    // Apply absolute bounds or percentile clipping
    let vmin: number, vmax: number;
    if (traitViVmin != null && traitViVmax != null) {
      if (viScaleMode === "log") {
        vmin = Math.log1p(Math.max(traitViVmin, 0));
        vmax = Math.log1p(Math.max(traitViVmax, 0));
      } else {
        vmin = traitViVmin;
        vmax = traitViVmax;
      }
    } else if (viAutoContrast) {
      ({ vmin, vmax } = percentileClip(scaled, 1, 99));
      // Reflect the auto-chosen clip on the histogram slider so the user sees
      // where Auto landed (and can grab it from there). Map the absolute
      // vmin/vmax back to slider percent; guard against a redundant write that
      // would loop this effect.
      const span = dataMax - dataMin;
      if (span > 0) {
        const lo = Math.max(0, Math.min(100, ((vmin - dataMin) / span) * 100));
        const hi = Math.max(0, Math.min(100, ((vmax - dataMin) / span) * 100));
        if (Math.abs(lo - viVminPct) > 0.5) setViVminPct(lo);
        if (Math.abs(hi - viVmaxPct) > 0.5) setViVmaxPct(hi);
      }
    } else {
      ({ vmin, vmax } = sliderRange(dataMin, dataMax, viVminPct, viVmaxPct));
    }

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
    applyColormap(scaled, imageData.data, lut, vmin, vmax);
    offCtx.putImageData(imageData, 0, 0);
    setViOffscreenVersion(v => v + 1);
  }, [virtualImageBytes, shapeRows, shapeCols, viColormap, viVminPct, viVmaxPct, viScaleMode, traitViVmin, traitViVmax, viAutoContrast]);

  // Cheap: VI zoom/pan redraw — just drawImage from cached offscreen
  React.useLayoutEffect(() => {
    const offscreen = viOffscreenRef.current;
    if (!offscreen || !virtualCanvasRef.current) return;
    const canvas = virtualCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = viSmooth;
    if (viSmooth) ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(viPanX, viPanY);
    ctx.scale(viZoom, viZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [viOffscreenVersion, viZoom, viPanX, viPanY, viSmooth]);

  // Render virtual image overlay (just clear - crosshair drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!virtualOverlayRef.current) return;
    const canvas = virtualOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Crosshair and scale bar now drawn on high-DPI UI canvas (viUiRef)
  }, [localPosRow, localPosCol, isDraggingVI, viZoom, viPanX, viPanY, pixelSize, shapeRows, shapeCols]);

  // Compute FFT (expensive, async — only re-run on data/GPU changes)
  const fftRealRef = React.useRef<Float32Array | null>(null);
  const fftImagRef = React.useRef<Float32Array | null>(null);
  const [fftVersion, setFftVersion] = React.useState(0);

  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !effectiveShowFft) { setFftCropDims(null); return; }
    let cancelled = false;
    let width = shapeCols;
    let height = shapeRows;
    let sourceData = rawVirtualImageRef.current;
    let origCropW = 0, origCropH = 0;

    // ROI FFT: crop virtual image to VI ROI region and pre-pad to power-of-2.
    // Use localViRoiCenter* (updated immediately on drag) instead of the synced
    // model traits, which lag by one comm roundtrip after a compound trait write.
    // Without this, FFT visibly stalls during rapid VI ROI drag.
    if (roiFftActive) {
      const cRow = localViRoiCenterRow ?? viRoiCenterRow;
      const cCol = localViRoiCenterCol ?? viRoiCenterCol;
      const crop = cropSingleROI(sourceData, shapeCols, shapeRows, viRoiMode, cRow, cCol, viRoiRadius, viRoiWidth, viRoiHeight);
      if (crop) {
        origCropW = crop.cropW;
        origCropH = crop.cropH;
        // Apply Hann window to crop at native dimensions BEFORE zero-padding
        if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
        const padW = nextPow2(crop.cropW);
        const padH = nextPow2(crop.cropH);
        const padded = new Float32Array(padW * padH);
        for (let y = 0; y < crop.cropH; y++) {
          for (let x = 0; x < crop.cropW; x++) {
            padded[y * padW + x] = crop.cropped[y * crop.cropW + x];
          }
        }
        sourceData = padded;
        width = padW;
        height = padH;
      }
    }

    // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
    if (!roiFftActive) {
      const padW = nextPow2(width);
      const padH = nextPow2(height);
      if (padW !== width || padH !== height) {
        const padded = new Float32Array(padW * padH);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            padded[y * padW + x] = sourceData[y * width + x];
          }
        }
        sourceData = padded;
        width = padW;
        height = padH;
      }
    }

    const fftW = width, fftH = height;
    if (gpuFFTRef.current && gpuReady) {
      const runGpuFFT = async () => {
        const real = sourceData.slice();
        const imag = new Float32Array(real.length);
        const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, fftW, fftH, false);
        if (cancelled) return;
        fftshift(fReal, fftW, fftH);
        fftshift(fImag, fftW, fftH);
        fftRealRef.current = fReal;
        fftImagRef.current = fImag;
        if (origCropW > 0) {
          setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
        } else if (fftW !== shapeCols || fftH !== shapeRows) {
          setFftCropDims({ cropWidth: shapeCols, cropHeight: shapeRows, fftWidth: fftW, fftHeight: fftH });
        } else {
          setFftCropDims(null);
        }
        setFftVersion(v => v + 1);
      };
      runGpuFFT();
      return () => { cancelled = true; };
    } else {
      const len = sourceData.length;
      let real = fftWorkRealRef.current;
      if (!real || real.length !== len) { real = new Float32Array(len); fftWorkRealRef.current = real; }
      real.set(sourceData);
      let imag = fftWorkImagRef.current;
      if (!imag || imag.length !== len) { imag = new Float32Array(len); fftWorkImagRef.current = imag; } else { imag.fill(0); }
      fft2d(real, imag, fftW, fftH, false);
      fftshift(real, fftW, fftH);
      fftshift(imag, fftW, fftH);
      fftRealRef.current = real;
      fftImagRef.current = imag;
      if (origCropW > 0) {
        setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
      } else if (fftW !== shapeCols || fftH !== shapeRows) {
        setFftCropDims({ cropWidth: shapeCols, cropHeight: shapeRows, fftWidth: fftW, fftHeight: fftH });
      } else {
        setFftCropDims(null);
      }
      setFftVersion(v => v + 1);
    }
  }, [virtualImageBytes, shapeRows, shapeCols, gpuReady, effectiveShowFft, roiFftActive, viRoiMode, viRoiCenterRow, viRoiCenterCol, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight, fftWindow]);

  // Expensive: FFT magnitude + histogram + colormap → cached offscreen canvas
  React.useEffect(() => {
    if (!fftRealRef.current || !fftImagRef.current) return;
    if (!effectiveShowFft) return;

    const width = fftCropDims?.fftWidth ?? shapeCols;
    const height = fftCropDims?.fftHeight ?? shapeRows;
    const real = fftRealRef.current;
    const imag = fftImagRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Compute magnitude with scale mode
    let magnitude = fftMagnitudeRef.current;
    if (!magnitude || magnitude.length !== real.length) {
      magnitude = new Float32Array(real.length);
      fftMagnitudeRef.current = magnitude;
    }
    // Cache raw magnitude for peak-snap before applying scale transform
    let rawMag = fftMagCacheRef.current;
    if (!rawMag || rawMag.length !== real.length) {
      rawMag = new Float32Array(real.length);
      fftMagCacheRef.current = rawMag;
    }
    for (let i = 0; i < real.length; i++) {
      const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
      rawMag[i] = mag;
      if (fftScaleMode === "log") { magnitude[i] = Math.log1p(mag); }
      else { magnitude[i] = mag; }
    }

    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, width, height));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
    }
    setFftDataMin(displayMin);
    setFftDataMax(displayMax);
    const magStats = computeStats(magnitude);
    setFftStats([magStats.mean, displayMin, displayMax, magStats.std]);
    setFftHistogramData(magnitude.slice());

    // Render to offscreen canvas
    let offscreen = fftOffscreenRef.current;
    if (!offscreen) { offscreen = document.createElement("canvas"); fftOffscreenRef.current = offscreen; }
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width; offscreen.height = height; fftImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;
    let imgData = fftImageDataRef.current;
    if (!imgData) { imgData = offCtx.createImageData(width, height); fftImageDataRef.current = imgData; }

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    applyColormap(magnitude, imgData.data, lut, vmin, vmax);
    offCtx.putImageData(imgData, 0, 0);
    setFftOffscreenVersion(v => v + 1);
  }, [effectiveShowFft, fftVersion, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct, fftColormap, shapeRows, shapeCols, fftCropDims]);

  // Cheap: FFT zoom/pan redraw — just drawImage from cached offscreen
  React.useLayoutEffect(() => {
    if (!fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const offscreen = fftOffscreenRef.current;
    if (!offscreen || !effectiveShowFft) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    const fftW = offscreen.width;
    const fftH = offscreen.height;
    const canvasW = canvas.width;
    const canvasH = canvas.height;
    // Use bilinear smoothing when FFT dims differ from canvas (non-pow2 padding or ROI crop).
    // Stretch offscreen to fill canvas via the 9-arg drawImage form: ROI FFT crops produce a
    // small offscreen (e.g. 64×64) that would otherwise blit at native size in the corner.
    ctx.imageSmoothingEnabled = fftW !== canvasW || fftH !== canvasH;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [fftOffscreenVersion, fftZoom, fftPanX, fftPanY, effectiveShowFft]);

  // Render FFT overlay with d-spacing crosshair marker
  React.useEffect(() => {
    if (!fftOverlayRef.current) return;
    const canvas = fftOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // D-spacing crosshair marker
    if (fftClickInfo && effectiveShowFft) {
      const fftW = fftCropDims?.fftWidth ?? shapeCols;
      const fftH = fftCropDims?.fftHeight ?? shapeRows;
      ctx.save();
      // Forward mapping: image col/row → canvas x/y (matches stretched drawImage).
      const screenX = fftPanX + fftZoom * (fftClickInfo.col * canvas.width / fftW);
      const screenY = fftPanY + fftZoom * (fftClickInfo.row * canvas.height / fftH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
      ctx.shadowBlur = 2;
      ctx.lineWidth = 1.5;
      // Scale crosshair size relative to canvas (not zoom-dependent)
      const r = 8 * Math.max(fftW, fftH) / 450;
      const gap = 3 * Math.max(fftW, fftH) / 450;
      const dotR = 4 * Math.max(fftW, fftH) / 450;
      ctx.beginPath();
      ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - gap, screenY);
      ctx.moveTo(screenX + gap, screenY); ctx.lineTo(screenX + r, screenY);
      ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - gap);
      ctx.moveTo(screenX, screenY + gap); ctx.lineTo(screenX, screenY + r);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, dotR, 0, Math.PI * 2);
      ctx.stroke();
      if (fftClickInfo.dSpacing != null) {
        const d = fftClickInfo.dSpacing;
        const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} \u00C5`;
        const fontSize = Math.max(10, Math.round(11 * Math.max(fftW, fftH) / 450));
        ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
        ctx.fillStyle = "white";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, screenX + r + 4, screenY - gap);
      }
      ctx.restore();
    }
  }, [fftZoom, fftPanX, fftPanY, effectiveShowFft, fftClickInfo, shapeCols, shapeRows, fftCropDims]);

  // Clear FFT click info when virtual image changes (scan position, VI ROI, etc.)
  React.useEffect(() => {
    setFftClickInfo(null);
  }, [virtualImageBytes]);

  // ─────────────────────────────────────────────────────────────────────────
  // High-DPI Scale Bar UI Overlays
  // ─────────────────────────────────────────────────────────────────────────
  
  // DP scale bar + crosshair + ROI overlay + profile line (high-DPI)
  React.useEffect(() => {
    if (!dpUiRef.current) return;
    // Draw scale bar first (clears canvas)
    const kUnit = kCalibrated ? kPixelUnit : "px";
    drawScaleBarHiDPI(dpUiRef.current, DPR, dpZoom, kPixelSize || 1, kUnit, detCols);
    // Draw ROI overlay (circle, square, rect, annular) or point crosshair
    if (roiMode === "point") {
      drawDpCrosshairHiDPI(dpUiRef.current, DPR, localKCol, localKRow, dpZoom, dpPanX, dpPanY, detCols, detRows, isDraggingDP, roiColors);
    } else {
      drawRoiOverlayHiDPI(
        dpUiRef.current, DPR, roiMode,
        localKCol, localKRow, roiRadius, roiRadiusInner, roiWidth, roiHeight,
        dpZoom, dpPanX, dpPanY, detCols, detRows,
        isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner,
        roiColors
      );
    }

    // Profile line overlay
    if (profileActive && profilePoints.length > 0) {
      const canvas = dpUiRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(DPR, DPR);
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const scaleX = cssW / detCols;
        const scaleY = cssH / detRows;
        const toScreenX = (col: number) => col * dpZoom * scaleX + dpPanX * scaleX;
        const toScreenY = (row: number) => row * dpZoom * scaleY + dpPanY * scaleY;

        // Draw point A
        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);

          // Draw band when profile width > 1
          if (profileWidth > 1) {
            const dc = profilePoints[1].col - profilePoints[0].col;
            const dr = profilePoints[1].row - profilePoints[0].row;
            const lineLen = Math.sqrt(dc * dc + dr * dr);
            if (lineLen > 0) {
              const halfW = (profileWidth - 1) / 2;
              const perpR = -dc / lineLen * halfW;
              const perpC = dr / lineLen * halfW;
              ctx.fillStyle = themeColors.accent + "20";
              ctx.strokeStyle = themeColors.accent;
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(toScreenX(profilePoints[0].col + perpC), toScreenY(profilePoints[0].row + perpR));
              ctx.lineTo(toScreenX(profilePoints[1].col + perpC), toScreenY(profilePoints[1].row + perpR));
              ctx.lineTo(toScreenX(profilePoints[1].col - perpC), toScreenY(profilePoints[1].row - perpR));
              ctx.lineTo(toScreenX(profilePoints[0].col - perpC), toScreenY(profilePoints[0].row - perpR));
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }

          // Draw line A->B
          ctx.strokeStyle = themeColors.accent;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          // Draw point B
          ctx.fillStyle = themeColors.accent;
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
    }

    // Colorbar overlay — uses cached vmin/vmax from the expensive DP offscreen effect
    if (showDpColorbar) {
      const canvas = dpUiRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(DPR, DPR);
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;
        drawColorbar(ctx, cssW, cssH, lut, dpColorbarVminRef.current, dpColorbarVmaxRef.current, dpScaleMode === "log");
        ctx.restore();
      }
    }
  }, [dpZoom, dpPanX, dpPanY, kPixelSize, kCalibrated, detRows, detCols, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, localKCol, localKRow, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner,
      profileActive, profilePoints, profileWidth, themeColors, showDpColorbar, dpColormap, dpScaleMode, dpVminPct, dpVmaxPct, canvasSize, roiColors]);
  
  // VI scale bar + crosshair + ROI + profile lines (high-DPI)
  React.useEffect(() => {
    if (!viUiRef.current) return;
    // Draw scale bar first (clears canvas)
    drawScaleBarHiDPI(viUiRef.current, DPR, viZoom, pixelSize || 1, pixelUnit || "px", shapeCols);
    // Draw crosshair only when ROI is off (ROI replaces the crosshair)
    if (!viRoiMode || viRoiMode === "off") {
      drawViPositionMarker(viUiRef.current, DPR, localPosRow, localPosCol, viZoom, viPanX, viPanY, shapeCols, shapeRows, isDraggingVI);
    } else {
      // Draw VI ROI instead of crosshair
      drawViRoiOverlayHiDPI(
        viUiRef.current, DPR, viRoiMode,
        localViRoiCenterRow, localViRoiCenterCol, viRoiRadius || 5, viRoiWidth || 10, viRoiHeight || 10,
        viZoom, viPanX, viPanY, shapeCols, shapeRows,
        isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize
      );
    }
    // Draw VI profile lines
    if (viProfileActive && viProfilePoints.length > 0) {
      const canvas = viUiRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const cssW = canvas.width / DPR;
        const cssH = canvas.height / DPR;
        const scaleX = cssW / shapeCols;
        const scaleY = cssH / shapeRows;
        ctx.save();
        ctx.scale(DPR, DPR);
        ctx.strokeStyle = "#a0f";
        ctx.lineWidth = 2;
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 2;
        if (viProfilePoints.length >= 1) {
          const p0 = viProfilePoints[0];
          const x0 = p0.col * viZoom * scaleX + viPanX * scaleX;
          const y0 = p0.row * viZoom * scaleY + viPanY * scaleY;
          ctx.beginPath();
          ctx.arc(x0, y0, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText("1", x0 + 6, y0 - 6);
        }
        if (viProfilePoints.length === 2) {
          const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
          const x0 = p0.col * viZoom * scaleX + viPanX * scaleX;
          const y0 = p0.row * viZoom * scaleY + viPanY * scaleY;
          const x1 = p1.col * viZoom * scaleX + viPanX * scaleX;
          const y1 = p1.row * viZoom * scaleY + viPanY * scaleY;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x1, y1, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.fillText("2", x1 + 6, y1 - 6);
        }
        ctx.restore();
      }
    }
  }, [viZoom, viPanX, viPanY, pixelSize, shapeRows, shapeCols, localPosRow, localPosCol, isDraggingVI,
      viRoiMode, localViRoiCenterRow, localViRoiCenterCol, viRoiRadius, viRoiWidth, viRoiHeight,
      isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize, canvasSize, viProfileActive, viProfilePoints]);

  // ── DP Profile computation ──
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawDpDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, p1.row, p1.col, profileWidth));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // ── VI Profile computation ──
  React.useEffect(() => {
    if (viProfilePoints.length === 2 && rawViDataRef.current && shapeCols > 0 && shapeRows > 0) {
      const p0 = viProfilePoints[0], p1 = viProfilePoints[1];
      setViProfileData(sampleLineProfile(rawViDataRef.current, shapeCols, shapeRows, p0.row, p0.col, p1.row, p1.col, 1));
    } else {
      setViProfileData(null);
    }
  }, [viProfilePoints, virtualImageBytes, shapeCols, shapeRows]);

  // ── Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasSize;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!profileData || profileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click two points on the DP to draw a profile", cssW / 2, cssH / 2);
      profileBaseImageRef.current = null;
      profileLayoutRef.current = null;
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < profileData.length; i++) {
      if (profileData[i] < gMin) gMin = profileData[i];
      if (profileData[i] > gMax) gMax = profileData[i];
    }
    const range = gMax - gMin || 1;

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (kCalibrated && kPixelSize > 0) {
        totalDist = distPx * kPixelSize;
        xUnit = kPixelUnit;
      } else {
        totalDist = distPx;
      }
    }

    // Draw axes
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Draw profile curve
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < profileData.length; i++) {
      const x = padLeft + (i / (profileData.length - 1)) * plotW;
      const y = padTop + plotH - ((profileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw x-axis ticks
    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(plotW / 70));
    const tickStep = roundToNiceValue(totalDist / idealTicks);
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textBaseline = "top";
    const ticks: number[] = [];
    for (let v = 0; v <= totalDist + tickStep * 0.01; v += tickStep) {
      if (v > totalDist * 1.001) break;
      ticks.push(v);
    }
    for (let i = 0; i < ticks.length; i++) {
      const v = ticks[i];
      const frac = totalDist > 0 ? v / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const label = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${label} ${xUnit}` : label, x, tickY + 4);
    }

    // Y-axis min/max labels (left margin)
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH);

    // Save base image and layout for hover
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileData, profilePoints, kPixelSize, kCalibrated, themeInfo.theme, themeColors.accent, canvasSize, profileHeight]);

  // DP Profile hover handlers
  const handleProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout || !profileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    // Restore base image
    ctx.putImageData(base, 0, 0);

    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    const isDark = themeInfo.theme === "dark";
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on curve + value
    const dataIdx = Math.min(profileData.length - 1, Math.max(0, Math.round(frac * (profileData.length - 1))));
    const val = profileData[dataIdx];
    const y = padTop + plotH - ((val - gMin) / range) * plotH;
    ctx.fillStyle = themeColors.accent;
    ctx.beginPath();
    ctx.arc(cssX, y, 3, 0, Math.PI * 2);
    ctx.fill();

    // Value readout label
    const dist = frac * totalDist;
    const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
    ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    const textW = ctx.measureText(label).width;
    const labelX = Math.min(cssX + 6, padLeft + plotW - textW - 2);
    const labelY = padTop + 2;
    ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
    ctx.fillRect(labelX - 2, labelY - 1, textW + 4, 11);
    ctx.fillStyle = isDark ? "#fff" : "#000";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(label, labelX, labelY);

    ctx.restore();
  }, [profileData, themeInfo.theme, themeColors.accent]);

  const handleProfileMouseLeave = React.useCallback(() => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // DP Profile resize handlers
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart.current) return;
      const deltaY = e.clientY - profileResizeStart.current.startY;
      const newHeight = Math.max(40, Math.min(300, profileResizeStart.current.startHeight + deltaY));
      setProfileHeight(newHeight);
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      profileResizeStart.current = null;
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile]);

  // ── VI Profile sparkline rendering ──
  React.useEffect(() => {
    const canvas = viProfileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = viCanvasWidth;
    const cssH = viProfileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    if (!viProfileData || viProfileData.length < 2) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = isDark ? "#555" : "#999";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Click two points on the VI to draw a profile", cssW / 2, cssH / 2);
      viProfileBaseImageRef.current = null;
      viProfileLayoutRef.current = null;
      return;
    }

    const padLeft = 40;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < viProfileData.length; i++) {
      if (viProfileData[i] < gMin) gMin = viProfileData[i];
      if (viProfileData[i] > gMax) gMax = viProfileData[i];
    }
    const range = gMax - gMin || 1;

    // X-axis: calibrated distance
    let totalDist = viProfileData.length - 1;
    let xUnit = "px";
    if (viProfilePoints.length === 2 && pixelSize > 0) {
      const dx = viProfilePoints[1].col - viProfilePoints[0].col;
      const dy = viProfilePoints[1].row - viProfilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      totalDist = distPx * pixelSize;
      xUnit = pixelUnit;
    }

    // Draw axes
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Draw profile curve
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < viProfileData.length; i++) {
      const x = padLeft + (i / (viProfileData.length - 1)) * plotW;
      const y = padTop + plotH - ((viProfileData[i] - gMin) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw x-axis ticks
    const tickY = padTop + plotH;
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    const idealTicks = Math.max(2, Math.floor(plotW / 70));
    const tickStep = roundToNiceValue(totalDist / idealTicks);
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textBaseline = "top";
    const ticks: number[] = [];
    for (let v = 0; v <= totalDist + tickStep * 0.01; v += tickStep) {
      if (v > totalDist * 1.001) break;
      ticks.push(v);
    }
    for (let i = 0; i < ticks.length; i++) {
      const v = ticks[i];
      const frac = totalDist > 0 ? v / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath(); ctx.moveTo(x, tickY); ctx.lineTo(x, tickY + 3); ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const label = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${label} ${xUnit}` : label, x, tickY + 4);
    }

    // Y-axis min/max labels (left margin)
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), 2, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), 2, padTop + plotH);

    // Save base image and layout for hover
    viProfileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    viProfileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [viProfileData, viProfilePoints, pixelSize, themeInfo.theme, themeColors.accent, viCanvasWidth, viProfileHeight]);

  // VI Profile hover handlers
  const handleViProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = viProfileCanvasRef.current;
    const base = viProfileBaseImageRef.current;
    const layout = viProfileLayoutRef.current;
    if (!canvas || !base || !layout || !viProfileData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    // Restore base image
    ctx.putImageData(base, 0, 0);

    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    const isDark = themeInfo.theme === "dark";
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on curve + value
    const dataIdx = Math.min(viProfileData.length - 1, Math.max(0, Math.round(frac * (viProfileData.length - 1))));
    const val = viProfileData[dataIdx];
    const y = padTop + plotH - ((val - gMin) / range) * plotH;
    ctx.fillStyle = themeColors.accent;
    ctx.beginPath();
    ctx.arc(cssX, y, 3, 0, Math.PI * 2);
    ctx.fill();

    // Value readout label
    const dist = frac * totalDist;
    const label = `${formatNumber(val)}  @  ${dist.toFixed(1)} ${xUnit}`;
    ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    const textW = ctx.measureText(label).width;
    const labelX = Math.min(cssX + 6, padLeft + plotW - textW - 2);
    const labelY = padTop + 2;
    ctx.fillStyle = isDark ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.8)";
    ctx.fillRect(labelX - 2, labelY - 1, textW + 4, 11);
    ctx.fillStyle = isDark ? "#fff" : "#000";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(label, labelX, labelY);

    ctx.restore();
  }, [viProfileData, themeInfo.theme, themeColors.accent]);

  const handleViProfileMouseLeave = React.useCallback(() => {
    const canvas = viProfileCanvasRef.current;
    const base = viProfileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // VI Profile resize handlers
  React.useEffect(() => {
    if (!isResizingViProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!viProfileResizeStart.current) return;
      const deltaY = e.clientY - viProfileResizeStart.current.startY;
      const newHeight = Math.max(40, Math.min(300, viProfileResizeStart.current.startHeight + deltaY));
      setViProfileHeight(newHeight);
    };
    const handleMouseUp = () => {
      setIsResizingViProfile(false);
      viProfileResizeStart.current = null;
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingViProfile]);

  // Generic zoom handler
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanX: React.Dispatch<React.SetStateAction<number>>,
    setPanY: React.Dispatch<React.SetStateAction<number>>,
    zoom: number, panX: number, panY: number,
    canvasRef: React.RefObject<HTMLCanvasElement | null>,
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

  // Helper: convert screen-pixel hit radius to image-pixel radius
  // handleRadius=6 CSS px drawn, hit area ~10 CSS px → convert to image coords
  const dpHitRadius = RESIZE_HIT_AREA_PX * Math.max(detCols, detRows) / canvasSize / dpZoom;

  // Helper: check if point is near the outer resize handle
  const isNearResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "rect") {
      // For rectangle, check near bottom-right corner
      const handleX = roiCenterCol + roiWidth / 2;
      const handleY = roiCenterRow + roiHeight / 2;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      return dist < dpHitRadius;
    }
    if ((roiMode !== "circle" && roiMode !== "square" && roiMode !== "annular") || !roiRadius) return false;
    const offset = roiMode === "square" ? roiRadius : roiRadius * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterCol + offset;
    const handleY = roiCenterRow + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < dpHitRadius;
  };

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (imgX: number, imgY: number): boolean => {
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterCol + offset;
    const handleY = roiCenterRow + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < dpHitRadius;
  };

  // Helper: check if point is near VI ROI resize handle (same logic as DP)
  // Hit area is capped to avoid overlap with center for small ROIs
  const viHitRadius = RESIZE_HIT_AREA_PX * Math.max(shapeRows, shapeCols) / canvasSize / viZoom;
  const isNearViRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    if (viRoiMode === "rect") {
      const halfH = (viRoiHeight || 10) / 2;
      const halfW = (viRoiWidth || 10) / 2;
      const handleX = localViRoiCenterRow + halfH;
      const handleY = localViRoiCenterCol + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(viHitRadius, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (viRoiMode === "circle" || viRoiMode === "square") {
      const radius = viRoiRadius || 5;
      const offset = viRoiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localViRoiCenterRow + offset;
      const handleY = localViRoiCenterCol + offset;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      // Cap hit area to 50% of radius so center remains draggable
      const hitArea = Math.min(viHitRadius, radius * 0.5);
      return dist < hitArea;
    }
    return false;
  };

  // Helper: check if point is inside the DP ROI area
  const isInsideDpRoi = (imgX: number, imgY: number): boolean => {
    if (roiMode === "point") return false;
    const dx = imgX - roiCenterCol;
    const dy = imgY - roiCenterRow;
    if (roiMode === "circle") return Math.sqrt(dx * dx + dy * dy) <= (roiRadius || 5);
    if (roiMode === "square") return Math.abs(dx) <= (roiRadius || 5) && Math.abs(dy) <= (roiRadius || 5);
    if (roiMode === "annular") { const d = Math.sqrt(dx * dx + dy * dy); return d <= (roiRadius || 20) && d >= (roiRadiusInner || 5); }
    if (roiMode === "rect") return Math.abs(dx) <= (roiWidth || 10) / 2 && Math.abs(dy) <= (roiHeight || 10) / 2;
    return false;
  };

  // Helper: check if point is inside the VI ROI area
  const isInsideViRoi = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    const dx = imgY - localViRoiCenterCol;
    const dy = imgX - localViRoiCenterRow;
    if (viRoiMode === "circle") return Math.sqrt(dx * dx + dy * dy) <= (viRoiRadius || 5);
    if (viRoiMode === "square") return Math.abs(dx) <= (viRoiRadius || 5) && Math.abs(dy) <= (viRoiRadius || 5);
    if (viRoiMode === "rect") return Math.abs(dx) <= (viRoiWidth || 10) / 2 && Math.abs(dy) <= (viRoiHeight || 10) / 2;
    return false;
  };

  // Pointer handlers
  const handleDpMouseDown = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    if ("pointerId" in e) {
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
    }
    dpClickStartRef.current = { x: e.clientX, y: e.clientY };
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // When profile mode is active, use profile interactions only
    if (profileActive) {
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / dpZoom;
        const d0 = Math.sqrt((imgX - p0.col) ** 2 + (imgY - p0.row) ** 2);
        const d1 = Math.sqrt((imgX - p1.col) ** 2 + (imgY - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingDpProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingDP(false);
          return;
        }
        if (pointToSegmentDistance(imgX, imgY, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingDpProfileLine(true);
          dpProfileDragStartRef.current = {
            row: imgY,
            col: imgX,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingDP(false);
          return;
        }
      }
      setIsDraggingDP(false);
      return;
    }

    // Check if clicking on resize handle (inner first, then outer)
    if (isNearResizeHandleInner(imgX, imgY)) {
      setIsDraggingResizeInner(true);
      return;
    }
    if (isNearResizeHandle(imgX, imgY)) {
      e.preventDefault();
      resizeAspectRef.current = roiMode === "rect" && roiWidth > 0 && roiHeight > 0 ? roiWidth / roiHeight : null;
      setIsDraggingResize(true);
      return;
    }

    setIsDraggingDP(true);
    // If clicking inside the ROI, drag with offset (grab-and-drag)
    if (roiMode !== "off" && roiMode !== "point" && isInsideDpRoi(imgX, imgY)) {
      dpDragOffsetRef.current = { dRow: imgY - roiCenterRow, dCol: imgX - roiCenterCol };
      return;
    }
    // Clicking outside ROI — teleport center to click position
    dpDragOffsetRef.current = { dRow: 0, dCol: 0 };
    setLocalKCol(imgX); setLocalKRow(imgY);
    // Use compound roi_center trait [row, col] - single observer fires in Python
    const newCol = Math.round(Math.max(0, Math.min(detCols - 1, imgX)));
    const newRow = Math.round(Math.max(0, Math.min(detRows - 1, imgY)));
    model.set("roi_active", true);
    model.set("roi_center", [newRow, newCol]);
    model.save_changes();
  };

  const handleDpMouseMove = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Fast path: skip cursor readout during any active drag — avoids setCursorInfo re-renders
    const anyDrag = isDraggingDP || isDraggingResize || isDraggingResizeInner
      || draggingDpProfileEndpoint !== null || isDraggingDpProfileLine;

    // Cursor readout: look up raw DP value at pixel position
    if (!anyDrag) {
      const pxCol = Math.floor(imgX);
      const pxRow = Math.floor(imgY);
      if (pxCol >= 0 && pxCol < detCols && pxRow >= 0 && pxRow < detRows && frameBytes) {
        const usesViRoiDp = viRoiMode && viRoiMode !== "off" && viRoiDpBytes && viRoiDpBytes.byteLength > 0;
        const sourceBytes = usesViRoiDp ? viRoiDpBytes : frameBytes;
        const raw = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
        setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * detCols + pxCol], panel: "DP" });
      } else {
        setCursorInfo(null);
      }
    }

    if (profileActive && profilePoints.length === 2) {
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = 10 / dpZoom;
      const d0 = Math.sqrt((imgX - p0.col) ** 2 + (imgY - p0.row) ** 2);
      const d1 = Math.sqrt((imgX - p1.col) ** 2 + (imgY - p1.row) ** 2);
      if (draggingDpProfileEndpoint !== null) {
        if (!rawDpDataRef.current) return;
        const clampedRow = Math.max(0, Math.min(detRows - 1, imgY));
        const clampedCol = Math.max(0, Math.min(detCols - 1, imgX));
        const next = [
          draggingDpProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingDpProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
        return;
      }
      if (isDraggingDpProfileLine && dpProfileDragStartRef.current) {
        if (!rawDpDataRef.current) return;
        const drag = dpProfileDragStartRef.current;
        let deltaRow = imgY - drag.row;
        let deltaCol = imgX - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (detRows - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (detCols - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgX, imgY, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredDpProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringDpProfileLine(nextHoverLine);
      return;
    } else {
      if (hoveredDpProfileEndpoint !== null) setHoveredDpProfileEndpoint(null);
      if (isHoveringDpProfileLine) setIsHoveringDpProfileLine(false);
    }

    // Handle inner resize dragging (annular mode)
    if (isDraggingResizeInner) {
      const dx = Math.abs(imgX - roiCenterCol);
      const dy = Math.abs(imgY - roiCenterRow);
      const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
      // Inner radius must be less than outer radius
      const rad = Math.max(1, Math.min(roiRadius - 1, Math.round(newRadius)));
      setLocalRoiRadiusInner(rad);
      sendRoiRadiusInner(rad);
      return;
    }

    // Handle outer resize dragging - use model state center, not local values
    if (isDraggingResize) {
      const dx = Math.abs(imgX - roiCenterCol);
      const dy = Math.abs(imgY - roiCenterRow);
      if (roiMode === "rect") {
        let newW = Math.max(2, Math.round(dx * 2));
        let newH = Math.max(2, Math.round(dy * 2));
        if (e.shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
          else newW = Math.max(2, Math.round(newH * aspect));
        }
        setRoiWidth(newW);
        setRoiHeight(newH);
      } else {
        const newRadius = roiMode === "square" ? Math.max(dx, dy) : Math.sqrt(dx ** 2 + dy ** 2);
        // For annular mode, outer radius must be greater than inner radius
        const minRadius = roiMode === "annular" ? (roiRadiusInner || 0) + 1 : 1;
        const rad = Math.max(minRadius, Math.round(newRadius));
        setLocalRoiRadius(rad);   // ring follows the cursor instantly
        sendRoiRadius(rad);       // coalesced Python recompute (latest-only)
      }
      return;
    }

    // Check hover state for resize handles
    if (!isDraggingDP) {
      setIsHoveringResizeInner(isNearResizeHandleInner(imgX, imgY));
      setIsHoveringResize(isNearResizeHandle(imgX, imgY));
      return;
    }

    const centerCol = imgX - dpDragOffsetRef.current.dCol;
    const centerRow = imgY - dpDragOffsetRef.current.dRow;
    setLocalKCol(centerCol); setLocalKRow(centerRow);
    // rAF-coalesced — sends only the latest roi_center per frame.
    const newCol = Math.round(Math.max(0, Math.min(detCols - 1, centerCol)));
    const newRow = Math.round(Math.max(0, Math.min(detRows - 1, centerRow)));
    queueRoiCenter(newRow, newCol);
  };

  const handleDpMouseUp = (e: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>) => {
    if ("pointerId" in e) {
      try { e.currentTarget.releasePointerCapture(e.pointerId); } catch {}
    }
    if (draggingDpProfileEndpoint !== null || isDraggingDpProfileLine) {
      setDraggingDpProfileEndpoint(null);
      setIsDraggingDpProfileLine(false);
      dpProfileDragStartRef.current = null;
      dpClickStartRef.current = null;
      setIsDraggingDP(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setHoveredDpProfileEndpoint(null);
      setIsHoveringDpProfileLine(false);
      return;
    }

    // Profile click capture
    if (profileActive && dpClickStartRef.current) {
      const dx = e.clientX - dpClickStartRef.current.x;
      const dy = e.clientY - dpClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = dpOverlayRef.current;
        if (canvas && rawDpDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const imgCol = (screenX - dpPanX) / dpZoom;
          const imgRow = (screenY - dpPanY) / dpZoom;
          if (imgCol >= 0 && imgCol < detCols && imgRow >= 0 && imgRow < detRows) {
            const pt = { row: imgRow, col: imgCol };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleLineProfile(rawDpDataRef.current, detCols, detRows, p0.row, p0.col, pt.row, pt.col, profileWidth));
            }
          }
        }
      }
    }
    dpClickStartRef.current = null;
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
    setDraggingDpProfileEndpoint(null);
    setIsDraggingDpProfileLine(false);
    setHoveredDpProfileEndpoint(null);
    setIsHoveringDpProfileLine(false);
    dpProfileDragStartRef.current = null;
  };
  const handleDpMouseLeave = () => {
    if (isDraggingDP || isDraggingResize || isDraggingResizeInner || draggingDpProfileEndpoint !== null || isDraggingDpProfileLine) {
      setCursorInfo(prev => prev?.panel === "DP" ? null : prev);
      return;
    }
    dpClickStartRef.current = null;
    setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false);
    setDraggingDpProfileEndpoint(null);
    setIsDraggingDpProfileLine(false);
    setHoveredDpProfileEndpoint(null);
    setIsHoveringDpProfileLine(false);
    dpProfileDragStartRef.current = null;
    setIsHoveringResize(false); setIsHoveringResizeInner(false);
    setCursorInfo(prev => prev?.panel === "DP" ? null : prev);
  };
  const handleDpDoubleClick = () => {
    setDpZoom(1);
    setDpPanX(0);
    setDpPanY(0);
  };

  const handleViMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // VI Profile mode - click to set points
    if (viProfileActive) {
      viClickStartRef.current = { x: screenX, y: screenY };
      if (viProfilePoints.length === 2) {
        const p0 = viProfilePoints[0];
        const p1 = viProfilePoints[1];
        const hitRadius = 10 / viZoom;
        const d0 = Math.sqrt((imgY - p0.col) ** 2 + (imgX - p0.row) ** 2);
        const d1 = Math.sqrt((imgY - p1.col) ** 2 + (imgX - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingViProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingVI(false);
          return;
        }
        if (pointToSegmentDistance(imgY, imgX, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingViProfileLine(true);
          viProfileDragStartRef.current = {
            row: imgX,
            col: imgY,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingVI(false);
          return;
        }
      }
      return;
    }

    // Check if VI ROI mode is active - same logic as DP
    if (viRoiMode && viRoiMode !== "off") {
      // Check if clicking on resize handle
      if (isNearViRoiResizeHandle(imgX, imgY)) {
        setIsDraggingViRoiResize(true);
        return;
      }

      // Grab-and-drag if clicking inside VI ROI, otherwise teleport
      setIsDraggingViRoi(true);
      if (isInsideViRoi(imgX, imgY)) {
        viRoiDragOffsetRef.current = { dRow: imgX - localViRoiCenterRow, dCol: imgY - localViRoiCenterCol };
      } else {
        viRoiDragOffsetRef.current = { dRow: 0, dCol: 0 };
        setLocalViRoiCenterRow(imgX);
        setLocalViRoiCenterCol(imgY);
        setViRoiCenterRow(Math.round(Math.max(0, Math.min(shapeRows - 1, imgX))));
        setViRoiCenterCol(Math.round(Math.max(0, Math.min(shapeCols - 1, imgY))));
      }
      return;
    }

    // Regular position selection (when ROI is off)
    setIsDraggingVI(true);
    setLocalPosRow(imgX); setLocalPosCol(imgY);
    // Batch X and Y updates into a single sync
    const newX = Math.round(Math.max(0, Math.min(shapeRows - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(shapeCols - 1, imgY)));
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleViMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // Fast path: skip cursor readout during any active drag — avoids setCursorInfo re-renders
    const anyViDrag = isDraggingVI || isDraggingViRoi || isDraggingViRoiResize
      || draggingViProfileEndpoint !== null || isDraggingViProfileLine;

    // Cursor readout: look up raw VI value at pixel position
    // imgX = row, imgY = col (swapped coordinate convention)
    if (!anyViDrag) {
      const pxRow = Math.floor(imgX);
      const pxCol = Math.floor(imgY);
      if (pxRow >= 0 && pxRow < shapeRows && pxCol >= 0 && pxCol < shapeCols && rawVirtualImageRef.current) {
        const raw = rawVirtualImageRef.current;
        setCursorInfo({ row: pxRow, col: pxCol, value: raw[pxRow * shapeCols + pxCol], panel: "VI" });
      } else {
        setCursorInfo(prev => prev?.panel === "VI" ? null : prev);
      }
    }

    if (viProfileActive && viProfilePoints.length === 2) {
      const p0 = viProfilePoints[0];
      const p1 = viProfilePoints[1];
      const hitRadius = 10 / viZoom;
      const d0 = Math.sqrt((imgY - p0.col) ** 2 + (imgX - p0.row) ** 2);
      const d1 = Math.sqrt((imgY - p1.col) ** 2 + (imgX - p1.row) ** 2);
      if (draggingViProfileEndpoint !== null) {
        const clampedRow = Math.max(0, Math.min(shapeRows - 1, imgX));
        const clampedCol = Math.max(0, Math.min(shapeCols - 1, imgY));
        const next = [
          draggingViProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : viProfilePoints[0],
          draggingViProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : viProfilePoints[1],
        ];
        setViProfilePoints(next);
        return;
      }
      if (isDraggingViProfileLine && viProfileDragStartRef.current) {
        const drag = viProfileDragStartRef.current;
        let deltaRow = imgX - drag.row;
        let deltaCol = imgY - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (shapeRows - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (shapeCols - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setViProfilePoints(next);
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgY, imgX, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredViProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringViProfileLine(nextHoverLine);
      return;
    } else {
      if (hoveredViProfileEndpoint !== null) setHoveredViProfileEndpoint(null);
      if (isHoveringViProfileLine) setIsHoveringViProfileLine(false);
    }

    // Handle VI ROI resize dragging (same pattern as DP)
    if (isDraggingViRoiResize) {
      const dx = Math.abs(imgX - localViRoiCenterRow);
      const dy = Math.abs(imgY - localViRoiCenterCol);
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

    // Handle VI ROI center dragging (same as DP — with offset)
    if (isDraggingViRoi) {
      const centerRow = imgX - viRoiDragOffsetRef.current.dRow;
      const centerCol = imgY - viRoiDragOffsetRef.current.dCol;
      setLocalViRoiCenterRow(centerRow);
      setLocalViRoiCenterCol(centerCol);
      // Compound trait update — single observer fires Python-side; reduced DP is
      // never computed against split-trait state (old col + new row, or vice versa).
      const newViX = Math.round(Math.max(0, Math.min(shapeRows - 1, centerRow)));
      const newViY = Math.round(Math.max(0, Math.min(shapeCols - 1, centerCol)));
      model.set("vi_roi_center", [newViX, newViY]);
      model.save_changes();
      return;
    }

    // Handle regular position dragging (when ROI is off)
    if (!isDraggingVI) return;
    setLocalPosRow(imgX); setLocalPosCol(imgY);
    // Batch position updates into a single sync
    const newX = Math.round(Math.max(0, Math.min(shapeRows - 1, imgX)));
    const newY = Math.round(Math.max(0, Math.min(shapeCols - 1, imgY)));
    model.set("pos_row", newX);
    model.set("pos_col", newY);
    model.save_changes();
  };

  const handleViMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingViProfileEndpoint !== null || isDraggingViProfileLine) {
      setDraggingViProfileEndpoint(null);
      setIsDraggingViProfileLine(false);
      viProfileDragStartRef.current = null;
      viClickStartRef.current = null;
      setIsDraggingVI(false);
      setIsDraggingViRoi(false);
      setIsDraggingViRoiResize(false);
      setHoveredViProfileEndpoint(null);
      setIsHoveringViProfileLine(false);
      return;
    }

    // VI Profile mode - complete point selection
    if (viProfileActive && viClickStartRef.current) {
      const canvas = virtualOverlayRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const endX = (e.clientX - rect.left) * (canvas.width / rect.width);
        const endY = (e.clientY - rect.top) * (canvas.height / rect.height);
        const dx = endX - viClickStartRef.current.x;
        const dy = endY - viClickStartRef.current.y;
        const wasDrag = Math.sqrt(dx * dx + dy * dy) > 3;

        if (!wasDrag) {
          // Click to add point
          const imgX = (endY - viPanY) / viZoom;
          const imgY = (endX - viPanX) / viZoom;
          const pt = { row: Math.round(Math.max(0, Math.min(shapeRows - 1, imgX))), col: Math.round(Math.max(0, Math.min(shapeCols - 1, imgY))) };
          if (viProfilePoints.length < 2) {
            setViProfilePoints([...viProfilePoints, pt]);
          } else {
            setViProfilePoints([pt]);
          }
        }
      }
      viClickStartRef.current = null;
    }

    setDraggingViProfileEndpoint(null);
    setIsDraggingViProfileLine(false);
    setHoveredViProfileEndpoint(null);
    setIsHoveringViProfileLine(false);
    viProfileDragStartRef.current = null;
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
  };
  const handleViMouseLeave = () => {
    viClickStartRef.current = null;
    setDraggingViProfileEndpoint(null);
    setIsDraggingViProfileLine(false);
    setHoveredViProfileEndpoint(null);
    setIsHoveringViProfileLine(false);
    viProfileDragStartRef.current = null;
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
    setIsHoveringViRoiResize(false);
    setCursorInfo(prev => prev?.panel === "VI" ? null : prev);
  };
  const handleViDoubleClick = () => {
    setViZoom(1);
    setViPanX(0);
    setViPanY(0);
  };
  const handleFftDoubleClick = () => {
    setFftZoom(1);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  // FFT drag-to-pan handlers
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
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

  const handleFftMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        // Convert screen coords to FFT image coords
        const canvas = fftOverlayRef.current;
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvas.width / rect.width;
          const scaleY = canvas.height / rect.height;
          const canvasX = (e.clientX - rect.left) * scaleX;
          const canvasY = (e.clientY - rect.top) * scaleY;
          const fftW = fftCropDims?.fftWidth ?? shapeCols;
          const fftH = fftCropDims?.fftHeight ?? shapeRows;
          // Reverse the render transform: canvas coords -> image coords.
          // Render: translate(panX, panY); scale(zoom); drawImage(offscreen, 0,0,fftW,fftH, 0,0,canvasW,canvasH)
          // So: canvasX = panX + zoom * (imgCol * canvasW / fftW)  →  imgCol = (canvasX - panX) / zoom * fftW / canvasW
          let imgCol = ((canvasX - fftPanX) / fftZoom) * (fftW / canvas.width);
          let imgRow = ((canvasY - fftPanY) / fftZoom) * (fftH / canvas.height);
          // Bounds check
          if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            // Snap to nearest peak in FFT magnitude
            if (fftMagCacheRef.current) {
              const snapped = findFFTPeak(fftMagCacheRef.current, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
              imgCol = snapped.col;
              imgRow = snapped.row;
            }
            const halfW = Math.floor(fftW / 2);
            const halfH = Math.floor(fftH / 2);
            const dcol = imgCol - halfW;
            const drow = imgRow - halfH;
            const distPx = Math.sqrt(dcol * dcol + drow * drow);
            if (distPx < 1) {
              setFftClickInfo(null); // Clicked on DC center
            } else {
              let spatialFreq: number | null = null;
              let dSpacing: number | null = null;
              if (pixelSize > 0) {
                const paddedW = nextPow2(fftW);
                const paddedH = nextPow2(fftH);
                const binC = ((Math.round(imgCol) - halfW) % fftW + fftW) % fftW;
                const binR = ((Math.round(imgRow) - halfH) % fftH + fftH) % fftH;
                const freqC = binC <= paddedW / 2 ? binC / (paddedW * pixelSize) : (binC - paddedW) / (paddedW * pixelSize);
                const freqR = binR <= paddedH / 2 ? binR / (paddedH * pixelSize) : (binR - paddedH) / (paddedH * pixelSize);
                spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
                dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
              }
              setFftClickInfo({ row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
            }
          }
        }
      }
      fftClickStartRef.current = null;
    }
    setIsDraggingFFT(false);
    setFftDragStart(null);
  };
  const handleFftMouseLeave = () => { fftClickStartRef.current = null; setIsDraggingFFT(false); setFftDragStart(null); };

  // ── Canvas resize handlers ──
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeCanvasStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeCanvasStart ? resizeCanvasStart.size : canvasSize;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeCanvasStart) return;
      const delta = Math.max(e.clientX - resizeCanvasStart.x, e.clientY - resizeCanvasStart.y);
      latestSize = Math.max(CANVAS_SIZE, resizeCanvasStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setCanvasSize(latestSize);
      setIsResizingCanvas(false);
      setResizeCanvasStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeCanvasStart]);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  // Export DP handler
  const handleExportDP = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      metadata_version: "1.0",
      widget_name: "Show4DSTEM",
      widget_version: widgetVersion || "unknown",
      exported_at: new Date().toISOString(),
      view: "diffraction",
      format: "zip",
      export_kind: "single_view_png_zip",
      position: { row: posRow, col: posCol },
      frame_idx: frameIdx,
      n_frames: nFrames,
      scan_shape: { rows: shapeRows, cols: shapeCols },
      detector_shape: { rows: detRows, cols: detCols },
      roi: {
        active: roiMode !== "off",
        mode: roiMode,
        center_row: roiCenterRow,
        center_col: roiCenterCol,
        radius: roiRadius,
        radius_inner: roiRadiusInner,
        width: roiWidth,
        height: roiHeight,
      },
      vi_roi: {
        mode: viRoiMode,
        center_row: viRoiCenterRow,
        center_col: viRoiCenterCol,
        radius: viRoiRadius,
        width: viRoiWidth,
        height: viRoiHeight,
      },
      calibration: {
        pixel_size_angstrom: pixelSize,
        pixel_size_unit: "Å/px",
        k_pixel_size: kPixelSize,
        k_pixel_size_unit: kCalibrated ? "mrad/px" : "px/px",
        k_calibrated: kCalibrated,
        center_row: centerRow,
        center_col: centerCol,
        bf_radius: bfRadius,
      },
      display: {
        diffraction: {
          colormap: dpColormap,
          scale_mode: dpScaleMode,
          vmin_pct: dpVminPct,
          vmax_pct: dpVmaxPct,
        },
      },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    downloadBlob(zipBlob, `dp_export_${timestamp}.zip`);
  };

  // Export VI handler
  const handleExportVI = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      metadata_version: "1.0",
      widget_name: "Show4DSTEM",
      widget_version: widgetVersion || "unknown",
      exported_at: new Date().toISOString(),
      view: "all",
      format: "zip",
      export_kind: "multi_panel_png_zip",
      position: { row: posRow, col: posCol },
      frame_idx: frameIdx,
      n_frames: nFrames,
      scan_shape: { rows: shapeRows, cols: shapeCols },
      detector_shape: { rows: detRows, cols: detCols },
      roi: {
        active: roiMode !== "off",
        mode: roiMode,
        center_row: roiCenterRow,
        center_col: roiCenterCol,
        radius: roiRadius,
        radius_inner: roiRadiusInner,
        width: roiWidth,
        height: roiHeight,
      },
      vi_roi: {
        mode: viRoiMode,
        center_row: viRoiCenterRow,
        center_col: viRoiCenterCol,
        radius: viRoiRadius,
        width: viRoiWidth,
        height: viRoiHeight,
      },
      calibration: {
        pixel_size_angstrom: pixelSize,
        pixel_size_unit: "Å/px",
        k_pixel_size: kPixelSize,
        k_pixel_size_unit: kCalibrated ? "mrad/px" : "px/px",
        k_calibrated: kCalibrated,
        center_row: centerRow,
        center_col: centerCol,
        bf_radius: bfRadius,
      },
      display: {
        diffraction: {
          colormap: dpColormap,
          scale_mode: dpScaleMode,
          vmin_pct: dpVminPct,
          vmax_pct: dpVmaxPct,
        },
        virtual: {
          colormap: viColormap,
          scale_mode: viScaleMode,
          vmin_pct: viVminPct,
          vmax_pct: viVmaxPct,
        },
        fft: {
          colormap: fftColormap,
          scale_mode: fftScaleMode,
          auto: fftAuto,
          vmin_pct: fftVminPct,
          vmax_pct: fftVmaxPct,
        },
      },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (virtualCanvasRef.current) zip.file("virtual_image.png", await canvasToBlob(virtualCanvasRef.current));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    if (fftCanvasRef.current) zip.file("fft.png", await canvasToBlob(fftCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    downloadBlob(zipBlob, `4dstem_export_${timestamp}.zip`);
  };

  // ── DP Figure Export ──
  const handleDpExportFigure = (withColorbar: boolean) => {
    setDpExportAnchor(null);
    const frameData = rawDpDataRef.current;
    if (!frameData) return;
    const processed = dpScaleMode === "log" ? applyLogScale(frameData) : frameData;
    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;
    const { min: dMin, max: dMax } = findDataRange(processed);
    let vmin: number, vmax: number;
    if (traitDpVmin != null && traitDpVmax != null) {
      if (dpScaleMode === "log") {
        vmin = Math.log1p(Math.max(traitDpVmin, 0));
        vmax = Math.log1p(Math.max(traitDpVmax, 0));
      } else {
        vmin = traitDpVmin;
        vmax = traitDpVmax;
      }
    } else {
      ({ vmin, vmax } = sliderRange(dMin, dMax, dpVminPct, dpVmaxPct));
    }
    const offscreen = renderToOffscreen(processed, detCols, detRows, lut, vmin, vmax);
    if (!offscreen) return;
    const kPxAngstrom = kPixelSize > 0 && kCalibrated ? kPixelSize : 0;
    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: `DP at (${posRow}, ${posCol})`,
      lut,
      vmin,
      vmax,
      logScale: dpScaleMode === "log",
      pixelSize: kPxAngstrom > 0 ? kPxAngstrom : undefined,
      showColorbar: withColorbar,
      showScaleBar: kPxAngstrom > 0,
    });
    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, "show4dstem_dp_figure.pdf")).catch(console.error);
  };

  const handleDpExportPng = () => {
    setDpExportAnchor(null);
    if (!dpCanvasRef.current) return;
    dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_dp.png"); }, "image/png");
  };

  const handleDpExportGif = () => {
    setDpExportAnchor(null);
    setExporting(true);
    setGifExportRequested(true);
  };

  // ── VI Figure Export ──
  const handleViExportFigure = (withColorbar: boolean) => {
    setViExportAnchor(null);
    if (!virtualCanvasRef.current) return;
    const viCanvas = virtualCanvasRef.current;
    const pixelSizeAngstrom = pixelSize > 0 ? pixelSize : 0;
    const figCanvas = exportFigure({
      imageCanvas: viCanvas,
      title: "Virtual Image",
      showColorbar: withColorbar,
      showScaleBar: pixelSizeAngstrom > 0,
      pixelSize: pixelSizeAngstrom > 0 ? pixelSizeAngstrom : undefined,
    });
    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, "show4dstem_vi_figure.pdf")).catch(console.error);
  };

  const handleViExportPng = () => {
    setViExportAnchor(null);
    if (!virtualCanvasRef.current) return;
    virtualCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_vi.png"); }, "image/png");
  };

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show4dstem_dp_animation.gif", "image/gif");
    const metaText = (gifMetadataJson || "").trim();
    if (metaText) {
      downloadBlob(new Blob([metaText], { type: "application/json" }), "show4dstem_dp_animation.json");
    }
    setExporting(false);
  }, [gifData, gifMetadataJson]);


  // Theme-aware select style
  const themedSelect = {
    ...controlPanel.select,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  const keyboardShortcutItems: [string, string][] = [
    ["↑ / ↓", "Move scan row"],
    ["← / →", "Move scan col"],
    ["Shift+Arrows", "Move ×10"],
    ...(nFrames > 1 ? [["[ / ]", `Prev / next ${frameDimLabel.toLowerCase()}`] as [string, string]] : []),
    ["Space", "Play / pause"],
    ["R", "Reset all zoom/pan"],
    ["Esc", "Release keyboard focus"],
    ["Scroll", "Zoom"],
    ["Dbl-click", "Reset view"],
  ];

  return (
    <Box
      ref={rootRef}
      className="show4dstem-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, outline: "none", borderRadius: "2px" }}
    >
      {/* HEADER */}
      <Typography variant="h6" sx={{ ...typo.title, mb: `${SPACING.SM}px` }}>
        {title || "4D-STEM Explorer"}
        {nFrames > 1 && <span style={{ fontWeight: "normal", fontSize: 13, marginLeft: 8, opacity: 0.7 }}>({frameLabels && frameLabels.length > frameIdx ? frameLabels[frameIdx] : `${frameDimLabel} ${frameIdx + 1}/${nFrames}`})</span>}
        <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>DP: Diffraction pattern I(kx,ky) at scan position. Drag to move ROI center.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Detector: ROI mask shape — defines which DP pixels are integrated for the virtual image.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>BF/ABF/ADF: Preset detector configurations (bright-field, annular bright-field, annular dark-field).</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Image: Virtual image — integrated intensity within detector ROI at each scan position.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Spatial frequency content of the virtual image. Auto masks DC + clips to 99.9th percentile.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Smooth: CSS bilinear blit on the VI canvas. No data change — browser smooths the upscale visually. Off = nearest-neighbor (sharp pixel boundaries).</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile contrast (1st–99th). Clips outliers automatically.</Typography>
          <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on DP to draw a line intensity profile.</Typography>
          {nFrames > 1 && <>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Frame Playback ({frameDimLabel})</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Bounce: Ping-pong — alternates forward and reverse.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FPS: Adjust playback speed (1–30 frames per second).</Typography>
          </>}
          <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
          <KeyboardShortcuts items={keyboardShortcutItems} />
        </Box>} theme={themeInfo.theme} />
      </Typography>

      {/* MAIN CONTENT: DP | VI | FFT (three columns when FFT shown) */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* LEFT COLUMN: DP Panel */}
        <Box sx={{ width: canvasSize }}>
          {/* DP Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typo.label }}>
              DP at ({Math.round(localPosRow)}, {Math.round(localPosCol)})
              <span style={{ color: roiColors.textColor, marginLeft: SPACING.SM }}>k: ({Math.round(localKRow)}, {Math.round(localKCol)})</span>
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile:</Typography>
              <Switch checked={profileActive} onChange={(e) => {
                const on = e.target.checked;
                setProfileActive(on);
                if (!on) {
                  setProfileLine([]);
                  setProfileData(null);
                  setHoveredDpProfileEndpoint(null);
                  setIsHoveringDpProfileLine(false);
                }
              }} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} disabled={dpZoom === 1 && dpPanX === 0 && dpPanY === 0 && roiCenterCol === centerCol && roiCenterRow === centerRow} onClick={() => { setDpZoom(1); setDpPanX(0); setDpPanY(0); setRoiCenterCol(centerCol); setRoiCenterRow(centerRow); }}>Reset</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                if (!dpCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => dpCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  dpCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_dp.png"); }, "image/png");
                }
              }}>COPY</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setDpExportAnchor(e.currentTarget)} disabled={exporting}>{exporting ? "..." : "Export"}</Button>
              <Menu anchorEl={dpExportAnchor} open={Boolean(dpExportAnchor)} onClose={() => setDpExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleDpExportFigure(true)} sx={{ fontSize: 12 }}>PDF + colorbar</MenuItem>
                <MenuItem onClick={() => handleDpExportFigure(false)} sx={{ fontSize: 12 }}>PDF</MenuItem>
                <MenuItem onClick={handleDpExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
                <MenuItem onClick={() => { setDpExportAnchor(null); handleExportDP(); }} sx={{ fontSize: 12 }}>ZIP (PNG + metadata)</MenuItem>
                {pathLength > 0 && <MenuItem onClick={handleDpExportGif} sx={{ fontSize: 12 }}>GIF (path animation)</MenuItem>}
              </Menu>
            </Stack>
          </Stack>

          {/* DP Canvas */}
          <Box sx={{ ...container.imageBox, width: canvasSize, height: canvasSize }}>
            <canvas ref={dpCanvasRef} width={detCols} height={detRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={dpOverlayRef} width={detCols} height={detRows}
              onPointerDown={handleDpMouseDown} onPointerMove={handleDpMouseMove}
              onPointerUp={handleDpMouseUp} onPointerCancel={handleDpMouseUp}
              onMouseLeave={handleDpMouseLeave}
              onWheel={createZoomHandler(setDpZoom, setDpPanX, setDpPanY, dpZoom, dpPanX, dpPanY, dpOverlayRef)}
              onDoubleClick={handleDpDoubleClick}
              style={{
                position: "absolute",
                width: "100%",
                height: "100%",
                cursor: (draggingDpProfileEndpoint !== null || isDraggingDpProfileLine)
                  ? "grabbing"
                  : (profileActive && (hoveredDpProfileEndpoint !== null || isHoveringDpProfileLine))
                    ? "grab"
                    : isHoveringResize || isDraggingResize
                      ? "nwse-resize"
                      : "crosshair",
              }}
            />
            <canvas ref={dpUiRef} width={canvasSize * DPR} height={canvasSize * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "DP" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* DP Stats Bar */}
          {dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[3])}</Box></Typography>
              <Box sx={{ flex: 1 }} />
              <Typography component="span" onClick={() => { model.set("_preset_request", "bf"); model.save_changes(); }} sx={{ color: roiColors.textColor, fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>BF</Typography>
              <Typography component="span" onClick={() => { model.set("_preset_request", "abf"); model.save_changes(); }} sx={{ color: "#4af", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ABF</Typography>
              <Typography component="span" onClick={() => { model.set("_preset_request", "adf"); model.save_changes(); }} sx={{ color: "#fa4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ADF</Typography>
            </Box>
          )}

          {/* Profile sparkline */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: canvasSize, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: canvasSize, height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <Box
                onMouseDown={(e) => {
                  setIsResizingProfile(true);
                  profileResizeStart.current = { startY: e.clientY, startHeight: profileHeight };
                }}
                sx={{ width: canvasSize, height: 4, cursor: "ns-resize", borderTop: `1px solid ${themeColors.border}`, borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, "&:hover": { bgcolor: themeColors.accent } }}
              />
            </Box>
          )}

          {/* DP Controls - two rows with histogram on right */}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Detector + slider */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Detector:</Typography>
                  <Select value={roiMode || "point"} onChange={(e) => setRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
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
                            const next = Array.isArray(v) ? v[0] : v;
                            setRoiRadius(next);
                          }
                        }}
                        min={1}
                        max={Math.min(detRows, detCols) / 2}
                        size="small"
                        sx={{ ...sliderStyles.small, width: roiMode === "annular" ? 67 : 47, mx: 1 }}
                      />
                      <Typography sx={{ ...typo.label, fontSize: 10 }}>
                        {roiMode === "annular" ? `${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : `${Math.round(roiRadius)}px`}
                      </Typography>
                    </>
                  )}
                </Box>
                {/* Row 2: Color + Scale + Colorbar */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={dpColormap} onChange={(e) => setDpColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="inferno">Inferno</MenuItem>
                    <MenuItem value="viridis">Viridis</MenuItem>
                    <MenuItem value="plasma">Plasma</MenuItem>
                    <MenuItem value="magma">Magma</MenuItem>
                    <MenuItem value="hot">Hot</MenuItem>
                    <MenuItem value="gray">Gray</MenuItem>
                  </Select>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>

                  </Select>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Colorbar:</Typography>
                  <Switch checked={showDpColorbar} onChange={(e) => setShowDpColorbar(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram data={dpHistogramData} vminPct={dpVminPct} vmaxPct={dpVmaxPct} onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={dpGlobalMin} dataMax={dpGlobalMax} />
              </Box>
            </Box>
          )}
        </Box>

        {/* SECOND COLUMN: VI Panel */}
        <Box sx={{ width: viCanvasWidth }}>
          {/* VI Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography sx={{ ...typo.label, color: themeColors.textMuted }}>
              {shapeRows}×{shapeCols} | {detRows}×{detCols}
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typo.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={effectiveShowFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              <Typography sx={{ ...typo.label, fontSize: 10 }}>Profile:</Typography>
              <Switch checked={viProfileActive} onChange={(e) => {
                const on = e.target.checked;
                setViProfileActive(on);
                if (!on) {
                  setViProfilePoints([]);
                  setHoveredViProfileEndpoint(null);
                  setIsHoveringViProfileLine(false);
                }
              }} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} disabled={viZoom === 1 && viPanX === 0 && viPanY === 0} onClick={() => { setViZoom(1); setViPanX(0); setViPanY(0); }}>Reset</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={async () => {
                if (!virtualCanvasRef.current) return;
                try {
                  const blob = await new Promise<Blob | null>(resolve => virtualCanvasRef.current!.toBlob(resolve, "image/png"));
                  if (!blob) return;
                  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                } catch {
                  virtualCanvasRef.current.toBlob((b) => { if (b) downloadBlob(b, "show4dstem_vi.png"); }, "image/png");
                }
              }}>COPY</Button>
              <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => setViExportAnchor(e.currentTarget)}>Export</Button>
              <Menu anchorEl={viExportAnchor} open={Boolean(viExportAnchor)} onClose={() => setViExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                <MenuItem onClick={() => handleViExportFigure(true)} sx={{ fontSize: 12 }}>PDF + colorbar</MenuItem>
                <MenuItem onClick={() => handleViExportFigure(false)} sx={{ fontSize: 12 }}>PDF</MenuItem>
                <MenuItem onClick={handleViExportPng} sx={{ fontSize: 12 }}>PNG</MenuItem>
                <MenuItem onClick={() => { setViExportAnchor(null); handleExportVI(); }} sx={{ fontSize: 12 }}>ZIP (all panels + metadata)</MenuItem>
              </Menu>
            </Stack>
          </Stack>

          {/* VI Canvas */}
          <Box sx={{ ...container.imageBox, width: viCanvasWidth, height: viCanvasHeight }}>
            <canvas ref={virtualCanvasRef} width={shapeCols} height={shapeRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={virtualOverlayRef} width={shapeCols} height={shapeRows}
              onMouseDown={handleViMouseDown} onMouseMove={handleViMouseMove}
              onMouseUp={handleViMouseUp} onMouseLeave={handleViMouseLeave}
              onWheel={createZoomHandler(setViZoom, setViPanX, setViPanY, viZoom, viPanX, viPanY, virtualOverlayRef)}
              onDoubleClick={handleViDoubleClick}
              style={{
                position: "absolute",
                width: "100%",
                height: "100%",
                cursor: (draggingViProfileEndpoint !== null || isDraggingViProfileLine)
                  ? "grabbing"
                  : (viProfileActive && (hoveredViProfileEndpoint !== null || isHoveringViProfileLine))
                    ? "grab"
                    : "crosshair",
              }}
            />
            <canvas ref={viUiRef} width={viCanvasWidth * DPR} height={viCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
            {cursorInfo && cursorInfo.panel === "VI" && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
          </Box>

          {/* VI Stats Bar — stats on left, Auto/Smooth toggles on right edge */}
          {viStats && viStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[3])}</Box></Typography>
              <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: "2px" }}>
                <Typography sx={{ ...typo.label, fontSize: 10 }}>Auto:</Typography>
                <Switch checked={viAutoContrast} onChange={(e) => setViAutoContrast(e.target.checked)} size="small" sx={switchStyles.small} />
                <Typography sx={{ ...typo.label, fontSize: 10 }} title="CSS bilinear interpolation. Same data, browser smooths visually.">Smooth:</Typography>
                <Switch checked={viSmooth} onChange={(e) => setViSmooth(e.target.checked)} size="small" sx={switchStyles.small} />
              </Box>
            </Box>
          )}

          {/* VI Profile sparkline */}
          {viProfileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: viCanvasWidth, boxSizing: "border-box" }}>
              <canvas
                ref={viProfileCanvasRef}
                onMouseMove={handleViProfileMouseMove}
                onMouseLeave={handleViProfileMouseLeave}
                style={{ width: viCanvasWidth, height: viProfileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <Box
                onMouseDown={(e) => {
                  setIsResizingViProfile(true);
                  viProfileResizeStart.current = { startY: e.clientY, startHeight: viProfileHeight };
                }}
                sx={{ width: viCanvasWidth, height: 4, cursor: "ns-resize", borderTop: `1px solid ${themeColors.border}`, borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, "&:hover": { bgcolor: themeColors.accent } }}
              />
            </Box>
          )}

          {/* VI Controls - Two rows with histogram on right */}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              {/* Left: Two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: ROI selector */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>ROI:</Typography>
                  <Select value={viRoiMode || "off"} onChange={(e) => setViRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
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
                            max={Math.min(shapeRows, shapeCols) / 2}
                            size="small"
                            sx={{ ...sliderStyles.small, width: 53, mx: 1 }}
                          />
                          <Typography sx={{ ...typo.value, fontSize: 10, minWidth: 30 }}>
                            {Math.round(viRoiRadius || 5)}px
                          </Typography>
                        </>
                      )}
                      <Select value={viRoiReduce || "mean"} onChange={(e) => setViRoiReduce(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps}>
                        <MenuItem value="mean">Mean</MenuItem>
                        <MenuItem value="sum">Sum</MenuItem>
                        <MenuItem value="max">Max</MenuItem>
                      </Select>
                    </>
                  )}
                </Box>
                {/* Row 2: Color + Scale */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={viColormap} onChange={(e) => setViColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="inferno">Inferno</MenuItem>
                    <MenuItem value="viridis">Viridis</MenuItem>
                    <MenuItem value="plasma">Plasma</MenuItem>
                    <MenuItem value="magma">Magma</MenuItem>
                    <MenuItem value="hot">Hot</MenuItem>
                    <MenuItem value="gray">Gray</MenuItem>
                  </Select>
                  <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={viScaleMode} onChange={(e) => setViScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                <Histogram data={viHistogramData} vminPct={viVminPct} vmaxPct={viVmaxPct} onRangeChange={(min, max) => { if (viAutoContrast) setViAutoContrast(false); setViVminPct(min); setViVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={viDataMin} dataMax={viDataMax} />
              </Box>
            </Box>
          )}
        </Box>

        {/* THIRD COLUMN: FFT Panel (conditionally shown) */}
        {effectiveShowFft && (
          <Box sx={{ width: viCanvasWidth }}>
            {/* FFT Header */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography variant="caption" sx={{ ...typo.label, color: roiFftActive && fftCropDims ? accentGreen : themeColors.textMuted }}>{roiFftActive && fftCropDims ? `ROI FFT (${fftCropDims.cropWidth}\u00D7${fftCropDims.cropHeight})` : "FFT"}</Typography>
              <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
                <Button size="small" sx={compactButton} disabled={fftZoom === 1 && fftPanX === 0 && fftPanY === 0} onClick={() => { setFftZoom(1); setFftPanX(0); setFftPanY(0); }}>Reset</Button>
              </Stack>
            </Stack>

            {/* FFT Canvas */}
            <Box sx={{ ...container.imageBox, width: viCanvasWidth, height: viCanvasHeight }}>
              <canvas ref={fftCanvasRef} width={shapeCols} height={shapeRows} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
              <canvas
                ref={fftOverlayRef} width={shapeCols} height={shapeRows}
                onMouseDown={handleFftMouseDown} onMouseMove={handleFftMouseMove}
                onMouseUp={handleFftMouseUp} onMouseLeave={handleFftMouseLeave}
                onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftZoom, fftPanX, fftPanY, fftOverlayRef)}
                onDoubleClick={handleFftDoubleClick}
                style={{ position: "absolute", width: "100%", height: "100%", cursor: isDraggingFFT ? "grabbing" : "grab" }}
              />
              <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, "&:hover": { opacity: 1 } }} />
            </Box>

            {/* FFT Stats Bar */}
            {fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[3])}</Box></Typography>
              </Box>
            )}

            {/* FFT D-spacing readout */}
            {fftClickInfo && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center" }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  Spot <Box component="span" sx={{ color: themeColors.accent }}>({fftClickInfo.row.toFixed(1)}, {fftClickInfo.col.toFixed(1)})</Box>
                </Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                  dist <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box>
                </Typography>
                {fftClickInfo.dSpacing != null && (
                  <Typography sx={{ fontSize: 11, fontWeight: "bold", color: themeColors.accent }}>
                    d = {fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}
                  </Typography>
                )}
                {fftClickInfo.spatialFreq != null && (
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                    q = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq.toFixed(4)} {"\u00C5\u207B\u00B9"}</Box>
                  </Typography>
                )}
              </Box>
            )}

            {/* FFT Controls - Two rows with histogram on right */}
            {showControls && (
              <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
                {/* Left: Two rows of controls */}
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                  {/* Row 1: Scale + Clip */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Scale:</Typography>
                    <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>

                    </Select>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Auto:</Typography>
                    <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                    {fftCropDims && (
                      <>
                        <Typography sx={{ ...typo.label, fontSize: 10 }}>Win:</Typography>
                        <Switch checked={fftWindow} onChange={(e) => setFftWindow(e.target.checked)} size="small" sx={switchStyles.small} />
                      </>
                    )}
                  </Box>
                  {/* Row 2: Color */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                    <Typography sx={{ ...typo.label, fontSize: 10 }}>Color:</Typography>
                    <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="inferno">Inferno</MenuItem>
                      <MenuItem value="viridis">Viridis</MenuItem>
                      <MenuItem value="plasma">Plasma</MenuItem>
                      <MenuItem value="magma">Magma</MenuItem>
                      <MenuItem value="hot">Hot</MenuItem>
                      <MenuItem value="gray">Gray</MenuItem>
                    </Select>
                  </Box>
                </Box>
                {/* Right: Histogram spanning both rows */}
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                  {fftHistogramData && (
                    <Histogram data={fftHistogramData} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={fftDataMin} dataMax={fftDataMax} />
                  )}
                </Box>
              </Box>
            )}
          </Box>
        )}
      </Stack>

      {/* BOTTOM CONTROLS */}

      {/* Frame controls (5D time/tilt series) — matches Show3D playback */}
      {showControls && nFrames > 1 && (<>
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Typography sx={{ ...typo.label, fontSize: 10, flexShrink: 0 }}>{frameDimLabel}:</Typography>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
            <IconButton size="small" onClick={() => { setFrameReverse(true); setFramePlaying(true); }} sx={{ color: frameReverse && framePlaying ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
              <FastRewindIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" onClick={() => setFramePlaying(!framePlaying)} sx={{ color: themeColors.accent, p: 0.25 }}>
              {framePlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" onClick={() => { setFrameReverse(false); setFramePlaying(true); }} sx={{ color: !frameReverse && framePlaying ? themeColors.accent : themeColors.textMuted, p: 0.25 }}>
              <FastForwardIcon sx={{ fontSize: 18 }} />
            </IconButton>
            <IconButton size="small" onClick={() => { setFramePlaying(false); setFrameIdx(0); }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider value={frameIdx} onChange={(_, v) => { setFramePlaying(false); setFrameIdx(v as number); }} min={0} max={Math.max(0, nFrames - 1)} size="small" sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
          <Typography sx={{ ...typo.value, minWidth: 50, textAlign: "right", flexShrink: 0 }}>{frameLabels && frameLabels.length > frameIdx ? frameLabels[frameIdx] : `${frameIdx + 1}/${nFrames}`}</Typography>
        </Box>
        <Box sx={{ ...controlRow, mt: `${SPACING.XS}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
          <Slider value={frameFps} min={1} max={30} step={1} onChange={(_, v) => setFrameFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} />
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(frameFps)}</Typography>
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>Loop</Typography>
          <Switch size="small" checked={frameLoop} onChange={() => setFrameLoop(!frameLoop)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
          <Typography sx={{ ...typo.label, fontSize: 10, color: themeColors.textMuted, flexShrink: 0 }}>Bounce</Typography>
          <Switch size="small" checked={frameBoomerang} onChange={() => setFrameBoomerang(!frameBoomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} />
        </Box>
      </>)}

      {/* Path animation slider */}
      {showControls && pathLength > 0 && (
        <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
          <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
            <IconButton size="small" onClick={() => setPathPlaying(!pathPlaying)} sx={{ color: themeColors.accent, p: 0.25 }}>
              {pathPlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
            </IconButton>
            <IconButton size="small" onClick={() => { setPathPlaying(false); setPathIndex(0); }} sx={{ color: themeColors.textMuted, p: 0.25 }}>
              <StopIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Stack>
          <Slider value={pathIndex} onChange={(_, v) => { setPathPlaying(false); setPathIndex(v as number); }} min={0} max={Math.max(0, pathLength - 1)} size="small" sx={{ flex: 1, minWidth: 60, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
          <Typography sx={{ ...typo.value, minWidth: 50, textAlign: "right", flexShrink: 0 }}>{pathIndex + 1}/{pathLength}</Typography>
          <Typography sx={{ ...typo.label, fontSize: 10 }}>Loop:</Typography>
          <Switch checked={pathLoop} onChange={(_, v) => { model.set("path_loop", v); model.save_changes(); }} size="small" sx={switchStyles.small} />
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
