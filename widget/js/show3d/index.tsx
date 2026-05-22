/// <reference types="@webgpu/types" />
/**
 * Show3D - Interactive 3D stack viewer with playback controls.
 *
 * Features:
 * - Scroll to zoom, double-click to reset
 * - Adjustable ROI size via slider
 * - FPS slider control
 * - WebGPU-accelerated FFT
 * - Equal-sized FFT and histogram panels
 * - Automatic theme detection (light/dark mode)
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import IconButton from "@mui/material/IconButton";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import StopIcon from "@mui/icons-material/Stop";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar, roundToNiceValue, exportFigure, canvasToPDF } from "../figure";
import { extractFloat32, formatNumber, downloadBlob, downloadDataView } from "../format";
import { findDataRange, applyLogScale, applyLogScaleInPlace, percentileClip, sliderRange, computeStats, computeHistogramFromBytes } from "../stats";
// ============================================================================
// Style tokens (inlined - matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};
const compactButton = {
  fontSize: 10,
  textTransform: "none" as const,
  letterSpacing: 0,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};
const sliderStyles = {
  small: {
    py: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  },
};
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

// ============================================================================
// Inlined utilities (matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const signedLog1p = (x: number): number => x >= 0 ? Math.log1p(x) : -Math.log1p(-x);

function shouldIgnoreWidgetShortcut(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return target.closest([
    "input", "textarea", "button", "select",
    "[contenteditable='true']", "[role='button']", "[role='slider']",
    "[role='switch']", "[role='textbox']", "[role='combobox']", "[role='menuitem']",
    ".MuiSlider-root", ".MuiSelect-select",
  ].join(",")) !== null;
}

function findFFTPeak(
  mag: Float32Array, width: number, height: number,
  col: number, row: number, radius: number,
): { row: number; col: number } {
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

function resolveDisplayRange(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean, vminPct: number, vmaxPct: number,
): { vmin: number; vmax: number } {
  const baseMin = logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin);
  const baseMax = logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax);
  return sliderRange(baseMin, baseMax, vminPct, vmaxPct);
}

function resolveDisplayBounds(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean,
): { min: number; max: number } {
  return {
    min: logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin),
    max: logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax),
  };
}

// ============================================================================
// Inlined components (matches Show2D single-file convention)
// ============================================================================
function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content} arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 280, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography component="span" sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}>
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box
      component="table"
      sx={{
        borderCollapse: "collapse",
        "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" },
        "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" },
      }}
    >
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

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
  pinBinsToRange?: boolean;
  ariaHidden?: boolean;
}

function Histogram({
  data, vminPct, vmaxPct, onRangeChange,
  width = 110, height = 40, theme = "dark",
  dataMin = 0, dataMax = 1, pinBinsToRange = true, ariaHidden = false,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(
    () => pinBinsToRange
      ? computeHistogramFromBytes(data, 256, dataMin, dataMax)
      : computeHistogramFromBytes(data),
    [data, dataMin, dataMax, pinBinsToRange],
  );
  const colors = React.useMemo(
    () => theme === "dark"
      ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
      : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" },
    [theme],
  );
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    // setTransform (not scale) so React 19 StrictMode double-invoke doesn't stack.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.max(1, Math.floor(bins.length / displayBins));
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);
  const formatValue = (pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
  };
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
        role={ariaHidden ? undefined : "img"}
        aria-hidden={ariaHidden ? "true" : undefined}
        aria-label={ariaHidden ? undefined : "Histogram of intensity values with min and max clip handles"}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [newMin, newMax] = v as number[];
          onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1));
        }}
        min={0} max={100} size="small"
        valueLabelDisplay="auto" valueLabelFormat={formatValue}
        aria-label="Histogram intensity clip range"
        sx={{
          width, py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(vminPct)}</Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(vmaxPct)}</Typography>
      </Box>
    </Box>
  );
}

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  // Match Show2D root font stack so Reset/Copy/Export buttons render in the
  // same system sans-serif. Items that genuinely want monospace (stats values,
  // numerical readouts, scale-bar labels) declare it explicitly.
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", overflow: "visible" },
  imageBox: { bgcolor: "transparent", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse, getGPUColormapEngine, GPUColormapEngine } from "../colormaps";

const DPR = window.devicePixelRatio || 1;
const RESIZE_HIT_AREA_PX = 10;
// ROI drawing
function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: "circle" | "square" | "rectangle" | "annular",
  radius: number,
  width: number,
  height: number,
  activeColor: string,
  inactiveColor: string,
  active: boolean = false,
  innerRadius: number = 0
): void {
  const strokeColor = active ? activeColor : inactiveColor;
  ctx.strokeStyle = strokeColor;
  // Caller sets ctx.lineWidth from roi.line_width; don't clobber.
  if (shape === "circle") {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - width / 2, y - height / 2, width, height);
  } else if (shape === "annular") {
    // Outer circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    // Inner circle (cyan)
    ctx.strokeStyle = active ? "#0ff" : inactiveColor;
    ctx.beginPath();
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2);
    ctx.stroke();
    // Annular fill
    ctx.fillStyle = (active ? activeColor : inactiveColor) + "15";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.arc(x, y, innerRadius, 0, Math.PI * 2, true);
    ctx.fill();
    ctx.strokeStyle = strokeColor;
  }
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
  }
}

import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D } from "../fft";

const FFT_SNAP_RADIUS = 5;

/** Sample intensity values along a line using bilinear interpolation. */
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

/** Sample intensity along a line, averaging over profileWidth perpendicular pixels. */
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
// Constants
// ============================================================================
const CANVAS_TARGET_SIZE = 550;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 30;

type ROIItem = {
  row: number;
  col: number;
  shape: string;
  radius: number;
  radius_inner: number;
  width: number;
  height: number;
  color: string;
  line_width: number;
  highlight: boolean;
};
const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];

function createROI(row: number, col: number, shape: string, index: number, imgW: number = 0, imgH: number = 0): ROIItem {
  const defR = imgW > 0 && imgH > 0 ? Math.max(10, Math.round(Math.min(imgW, imgH) * 0.05)) : 10;
  return {
    row,
    col,
    shape,
    radius: defR,
    radius_inner: Math.max(5, Math.round(defR * 0.5)),
    width: defR * 2,
    height: defR * 2,
    color: ROI_COLORS[index % ROI_COLORS.length],
    line_width: 2,
    highlight: false,
  };
}

function normalizeROI(roi: ROIItem, index: number): ROIItem {
  return {
    ...roi,
    color: roi.color || ROI_COLORS[index % ROI_COLORS.length],
    shape: roi.shape || "circle",
    radius: roi.radius ?? 10,
    radius_inner: roi.radius_inner ?? 5,
    width: roi.width ?? 20,
    height: roi.height ?? 20,
    line_width: roi.line_width ?? 2,
    highlight: !!roi.highlight,
  };
}

/** Extract a single frame from the playback buffer (zero-copy subarray). */
function getFrameFromBuffer(
  buffer: Float32Array | null,
  bufStart: number,
  bufCount: number,
  nSlices: number,
  frameIdx: number,
  frameSize: number,
): Float32Array | null {
  if (!buffer || bufCount === 0) return null;
  let offset = frameIdx - bufStart;
  if (offset < 0) offset += nSlices;
  if (offset < 0 || offset >= bufCount) return null;
  const start = offset * frameSize;
  const end = start + frameSize;
  if (end > buffer.length) return null;
  return buffer.subarray(start, end);
}

/** Fused single-pass render: optional log scale + normalize + colormap → RGBA.
 *  Eliminates multiple data passes during playback for maximum frame rate. */
function renderFramePlayback(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): void {
  const range = vmax - vmin;
  const invRange = range > 0 ? 255 / range : 0;
  if (logScale) {
    for (let i = 0; i < data.length; i++) {
      const d = data[i];
      const v = d >= 0 ? Math.log1p(d) : -Math.log1p(-d);
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  } else {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const idx = v <= vmin ? 0 : v >= vmax ? 255 : ((v - vmin) * invRange) | 0;
      const j = i << 2;
      const k = idx * 3;
      rgba[j] = lut[k];
      rgba[j + 1] = lut[k + 1];
      rgba[j + 2] = lut[k + 2];
      rgba[j + 3] = 255;
    }
  }
}

// ============================================================================
// Crop ROI region from raw float32 data for ROI-scoped FFT
// ============================================================================
function cropROIRegion(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROIItem,
): { cropped: Float32Array; cropW: number; cropH: number } | null {
  const shape = roi.shape || "circle";
  let x0: number, y0: number, x1: number, y1: number;

  if (shape === "rectangle") {
    const hw = roi.width / 2;
    const hh = roi.height / 2;
    x0 = Math.max(0, Math.floor(roi.col - hw));
    y0 = Math.max(0, Math.floor(roi.row - hh));
    x1 = Math.min(imgW, Math.ceil(roi.col + hw));
    y1 = Math.min(imgH, Math.ceil(roi.row + hh));
  } else {
    const r = roi.radius;
    x0 = Math.max(0, Math.floor(roi.col - r));
    y0 = Math.max(0, Math.floor(roi.row - r));
    x1 = Math.min(imgW, Math.ceil(roi.col + r));
    y1 = Math.min(imgH, Math.ceil(roi.row + r));
  }

  const cropW = x1 - x0;
  const cropH = y1 - y0;
  if (cropW < 2 || cropH < 2) return null;

  const cropped = new Float32Array(cropW * cropH);

  if (shape === "circle" || shape === "annular") {
    const r = roi.radius;
    const rSq = r * r;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgX = x0 + dx;
        const imgY = y0 + dy;
        const distSq = (imgX - roi.col) * (imgX - roi.col) + (imgY - roi.row) * (imgY - roi.row);
        cropped[dy * cropW + dx] = distSq <= rSq ? data[imgY * imgW + imgX] : 0;
      }
    }
  } else {
    for (let dy = 0; dy < cropH; dy++) {
      const srcOffset = (y0 + dy) * imgW + x0;
      cropped.set(data.subarray(srcOffset, srcOffset + cropW), dy * cropW);
    }
  }

  return { cropped, cropW, cropH };
}

// ============================================================================
// Compute stats for pixels inside a single ROI (mean/min/max/std)
// ============================================================================
function computeROIPixelStats(
  data: Float32Array, imgW: number, imgH: number,
  roi: ROIItem,
): { mean: number; min: number; max: number; std: number } | null {
  const shape = roi.shape || "circle";
  let x0: number, y0: number, x1: number, y1: number;

  if (shape === "rectangle") {
    const hw = roi.width / 2;
    const hh = roi.height / 2;
    x0 = Math.max(0, Math.floor(roi.col - hw));
    y0 = Math.max(0, Math.floor(roi.row - hh));
    x1 = Math.min(imgW, Math.ceil(roi.col + hw));
    y1 = Math.min(imgH, Math.ceil(roi.row + hh));
  } else {
    const r = roi.radius;
    x0 = Math.max(0, Math.floor(roi.col - r));
    y0 = Math.max(0, Math.floor(roi.row - r));
    x1 = Math.min(imgW, Math.ceil(roi.col + r));
    y1 = Math.min(imgH, Math.ceil(roi.row + r));
  }

  const cropW = x1 - x0;
  const cropH = y1 - y0;
  if (cropW < 1 || cropH < 1) return null;

  let sum = 0, sumSq = 0, mn = Infinity, mx = -Infinity, n = 0;

  if (shape === "circle") {
    const rSq = roi.radius * roi.radius;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgX = x0 + dx, imgY = y0 + dy;
        const distSq = (imgX - roi.col) ** 2 + (imgY - roi.row) ** 2;
        if (distSq > rSq) continue;
        const v = data[imgY * imgW + imgX];
        sum += v; sumSq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        n++;
      }
    }
  } else if (shape === "annular") {
    const rSq = roi.radius * roi.radius;
    const riSq = (roi.radius_inner || 0) ** 2;
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const imgX = x0 + dx, imgY = y0 + dy;
        const distSq = (imgX - roi.col) ** 2 + (imgY - roi.row) ** 2;
        if (distSq > rSq || distSq < riSq) continue;
        const v = data[imgY * imgW + imgX];
        sum += v; sumSq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        n++;
      }
    }
  } else {
    // square or rectangle - all pixels in bounding box
    for (let dy = 0; dy < cropH; dy++) {
      for (let dx = 0; dx < cropW; dx++) {
        const v = data[(y0 + dy) * imgW + (x0 + dx)];
        sum += v; sumSq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        n++;
      }
    }
  }

  if (n === 0) return null;
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  return { mean, min: mn, max: mx, std };
}

// ============================================================================
// Main Component
// ============================================================================
function Show3D() {
  // Theme detection
  const { themeInfo, colors: baseColors } = useTheme();
  const themeColors = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };

  // Theme-aware select style (matching Show4DSTEM)
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

  // Model state (synced with Python)
  const [sliceIdx, setSliceIdx] = useModelState<number>("slice_idx");
  const [nSlices] = useModelState<number>("n_slices");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  // Defensive: traitlets.Bytes can identity-suppress trait events when content
  // and length are similar. frame_seq is incremented Python-side on every write
  // so JS effects always see a change. Use it in dep arrays alongside frameBytes.
  const [frameSeq] = useModelState<number>("frame_seq");

  // Truthful first-render signal: flipped ONCE after the first frame_bytes
  // arrives and the browser has had time to composite two frames.  Python side
  // observes `_js_rendered` and prints the real end-to-end wall clock, not the
  // misleading Python-only __init__ number.
  const [, setJsRendered] = useModelState<boolean>("_js_rendered");
  const firstRenderFiredRef = React.useRef(false);
  React.useEffect(() => {
    if (firstRenderFiredRef.current) return;
    if (!frameBytes || frameBytes.byteLength === 0) return;
    firstRenderFiredRef.current = true;
    requestAnimationFrame(() => requestAnimationFrame(() => setJsRendered(true)));
  }, [frameBytes, setJsRendered]);

  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [dimLabel] = useModelState<string>("dim_label");
  const [nPanels] = useModelState<number>("n_panels");
  const [panelTitles] = useModelState<string[]>("panel_titles");
  const [panelRealFrames] = useModelState<number[]>("panel_real_frames");
  const [hiddenIndices] = useModelState<number[]>("hidden_indices");
  const hiddenSet = React.useMemo(() => new Set(hiddenIndices || []), [hiddenIndices]);
  const nextVisible = (from: number, dir: 1 | -1, allowWrap = true): number => {
    if (!hiddenSet.size) return from + dir;
    let n = from + dir;
    while (n >= 0 && n < nSlices) {
      if (!hiddenSet.has(n)) return n;
      n += dir;
    }
    if (!allowWrap) return from;
    n = dir > 0 ? 0 : nSlices - 1;
    while (n !== from) {
      if (!hiddenSet.has(n)) return n;
      n += dir;
      if (n < 0 || n >= nSlices) return from;
    }
    return from;
  };
  const visibleCount = nSlices - hiddenSet.size;
  // If the user hides the currently-displayed slice, snap to next visible.
  React.useEffect(() => {
    if (!hiddenSet.has(sliceIdx)) return;
    const next = nextVisible(sliceIdx, 1, true);
    if (next !== sliceIdx) setSliceIdx(next);
  }, [hiddenIndices]);
  const [maxCols] = useModelState<number>("max_cols");
  const [linkPanels, setLinkPanels] = useModelState<boolean>("link_panels");
  const [showResizeHandles] = useModelState<boolean>("show_resize_handles");
  const [showZoomIndicator] = useModelState<boolean>("show_zoom_indicator");
  const [panelTitleFontSize] = useModelState<number>("panel_title_font_size");
  const [panelGapTrait] = useModelState<number>("panel_gap");
  const [linkContrast, setLinkContrast] = useModelState<boolean>("link_contrast");
  // Back-compat: both old axis flags follow the single Link toggle.
  const linkZoom = linkPanels;
  const linkPan = linkPanels;
  const [cmap, setCmap] = useModelState<string>("cmap");

  // Playback
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const [fps, setFps] = useModelState<number>("fps");
  const [loop, setLoop] = useModelState<boolean>("loop");
  const [loopStart, setLoopStart] = useModelState<number>("loop_start");
  const [loopEnd, setLoopEnd] = useModelState<number>("loop_end");
  const [bookmarkedFrames] = useModelState<number[]>("bookmarked_frames");
  const [playbackPath] = useModelState<number[]>("playback_path");

  // Boomerang direction ref (avoids stale closure in setInterval)
  const bounceDirRef = React.useRef<1 | -1>(1);

  // Stats
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [percentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh] = useModelState<number>("percentile_high");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [dataMin] = useModelState<number>("data_min");
  const [dataMax] = useModelState<number>("data_max");
  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [imageRotation] = useModelState<number>("image_rotation");

  // Customization
  const [canvasSizeTrait] = useModelState<number>("size");

  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<ROIItem[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [roiPlotData] = useModelState<DataView>("roi_plot_data");
  const [newRoiShape, setNewRoiShape] = React.useState<"circle" | "square" | "rectangle" | "annular">("square");

  // Diff mode
  const [diffMode, setDiffMode] = useModelState<string>("diff_mode");

  // FFT
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const hideDisplay = false;
  const hideHistogram = false;
  const hideStats = false;
  const hidePlayback = false;
  const hideView = false;
  const hideExport = false;
  const hideRoi = false;
  const hideProfile = false;

  const lockDisplay = false;
  const lockHistogram = false;
  const lockStats = false;
  const lockPlayback = false;
  const lockView = false;
  const lockExport = false;
  const lockRoi = false;
  const lockProfile = false;
  const effectiveShowFft = showFft;

  // Export
  const [, setGifExportRequested] = useModelState<boolean>("_gif_export_requested");
  const [gifData] = useModelState<DataView>("_gif_data");
  const [gifMetadataJson] = useModelState<string>("_gif_metadata_json");
  const [, setZipExportRequested] = useModelState<boolean>("_zip_export_requested");
  const [zipData] = useModelState<DataView>("_zip_data");
  const [, setBundleExportRequested] = useModelState<boolean>("_bundle_export_requested");
  const [bundleData] = useModelState<DataView>("_bundle_data");
  const [exporting, setExporting] = React.useState(false);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);

  // Playback buffer (sliding prefetch)
  const [bufferBytes] = useModelState<DataView>("_buffer_bytes");
  const [bufferStart] = useModelState<number>("_buffer_start");
  const [bufferCount] = useModelState<number>("_buffer_count");
  const [, setPrefetchRequest] = useModelState<number>("_prefetch_request");

  // Canvas refs
  const rootRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const uiRef = React.useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = React.useRef<HTMLDivElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);

  // Local state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const roiItems = React.useMemo(() => (roiList || []).map((roi, i) => normalizeROI(roi, i)), [roiList]);
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < roiItems.length ? roiItems[roiSelectedIdx] : null;
  const [showRoiResizeHint, setShowRoiResizeHint] = React.useState(true);
  const pendingRoiAddRef = React.useRef<{ row: number; col: number } | null>(null);

  // Preview panel state (JS-only, shows ROI crop at full resolution - auto-shows when ROI selected)
  const [previewZoom, setPreviewZoom] = React.useState({ zoom: 1, panX: 0, panY: 0 });
  const previewCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const previewOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const previewContainerRef = React.useRef<HTMLDivElement>(null);
  const [isDraggingPreviewPan, setIsDraggingPreviewPan] = React.useState(false);
  const [previewPanStart, setPreviewPanStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  const [previewCropDims, setPreviewCropDims] = React.useState<{ w: number; h: number } | null>(null);
  const previewOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const [previewVersion, setPreviewVersion] = React.useState(0);

  const updateSelectedRoi = React.useCallback((updates: Partial<ROIItem>) => {
    if (roiSelectedIdx < 0 || !roiList) return;
    const newList = [...roiList];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList);
  }, [roiList, roiSelectedIdx, setRoiList]);
  // Per-panel zoom/pan: index 0 is also used as the SHARED state for
  // single-panel widgets, and as the linked state when link_zoom or
  // link_pan are on. Each panel keeps its own state when unlinked.
  type ZoomState = { zoom: number; panX: number; panY: number };
  const initialState: ZoomState = { zoom: 1, panX: 0, panY: 0 };
  const [linkedState, setLinkedState] = React.useState<ZoomState>(initialState);
  const [panelStates, setPanelStates] = React.useState<ZoomState[]>([initialState]);
  React.useEffect(() => {
    const n = Math.max(1, nPanels || 1);
    setPanelStates(prev => {
      if (prev.length === n) return prev;
      const next = Array.from({ length: n }, (_, i) => prev[i] || { ...initialState });
      return next;
    });
  }, [nPanels]);
  // Seamless toggle: on link→unlink, copy linkedState into every panel; on
  // unlink→link, copy panel 0 into linkedState. Single effect so both axes
  // sync atomically.
  const prevLinkRef = React.useRef(linkPanels);
  React.useEffect(() => {
    if (prevLinkRef.current && !linkPanels) {
      // Linked → unlinked: distribute linkedState to all panels
      const s = linkedState;
      setPanelStates(arr => arr.map(() => ({ zoom: s.zoom, panX: s.panX, panY: s.panY })));
    } else if (!prevLinkRef.current && linkPanels) {
      // Unlinked → linked: adopt panel 0's state as the shared linked state
      const s0 = panelStates[0] || initialState;
      setLinkedState({ zoom: s0.zoom, panX: s0.panX, panY: s0.panY });
    }
    prevLinkRef.current = linkPanels;
  }, [linkPanels]);
  const getState = (panelIdx: number, axis: "zoom" | "pan"): ZoomState => {
    const linked = axis === "zoom" ? linkZoom : linkPan;
    return linked ? linkedState : (panelStates[panelIdx] || initialState);
  };
  const stateFor = (panelIdx: number): ZoomState => ({
    zoom: getState(panelIdx, "zoom").zoom,
    panX: getState(panelIdx, "pan").panX,
    panY: getState(panelIdx, "pan").panY,
  });
  // Back-compat aliases for the single-panel code paths (ROI, profile, etc.)
  // which still expect plain zoom/panX/panY. Use panel 0's state.
  const zoom = stateFor(0).zoom;
  const panX = stateFor(0).panX;
  const panY = stateFor(0).panY;
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);
  const panStartPanelRef = React.useRef<number>(0);
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const [isResizingMain, setIsResizingMain] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number, y: number, size: number } | null>(null);
  const rawFrameDataRef = React.useRef<Float32Array | null>(null);
  const initialCanvasSizeRef = React.useRef<number>(canvasSizeTrait > 0 ? canvasSizeTrait : CANVAS_TARGET_SIZE);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);
  const [showRoiPlot, setShowRoiPlot] = React.useState(true);
  const roiPlotCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Lens (magnifier inset)
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128);
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null);
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const lensCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const lensDragStartRef = React.useRef<{ mx: number; my: number; ax: number; ay: number } | null>(null);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensResizeStartRef = React.useRef<{ my: number; startSize: number } | null>(null);

  // Reusable rendering buffers (avoid per-frame allocation)
  const mainOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const mainImgDataRef = React.useRef<ImageData | null>(null);
  const logBufferRef = React.useRef<Float32Array | null>(null);

  // Playback buffer refs (double-buffer: current + next to avoid overwrite stalls)
  const bufferRef = React.useRef<Float32Array | null>(null);
  const bufferStartRef = React.useRef(0);
  const bufferCountRef = React.useRef(0);
  const nextBufferRef = React.useRef<Float32Array | null>(null);
  const nextBufferStartRef = React.useRef(0);
  const nextBufferCountRef = React.useRef(0);
  const prefetchPendingRef = React.useRef(false);
  const playbackIdxRef = React.useRef(0);
  const [displaySliceIdx, setDisplaySliceIdx] = React.useState(sliceIdx);
  const [localStats, setLocalStats] = React.useState<{ mean: number; min: number; max: number; std: number } | null>(null);
  const [localPanelStats, setLocalPanelStats] = React.useState<{ mean: number; min: number; max: number; std: number }[] | null>(null);

  // WebGPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  // WebGPU colormap engine (GPU-accelerated colormap for 4K frames)
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);

  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
    getGPUColormapEngine().then(engine => {
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
      }
    });
  }, []);

  // Parse incoming playback buffer (double-buffer to avoid overwrite stalls)
  React.useEffect(() => {
    if (!bufferBytes || bufferBytes.byteLength === 0) return;
    const parsed = extractFloat32(bufferBytes);
    if (!parsed) return;
    if (!bufferRef.current || bufferCountRef.current === 0) {
      // No active buffer - use as current (initial load)
      bufferRef.current = parsed;
      bufferStartRef.current = bufferStart;
      bufferCountRef.current = bufferCount;
    } else {
      // Active buffer exists - store as next (prefetch)
      nextBufferRef.current = parsed;
      nextBufferStartRef.current = bufferStart;
      nextBufferCountRef.current = bufferCount;
    }
    prefetchPendingRef.current = false;
  }, [bufferBytes, bufferStart, bufferCount]);

  // Sync displaySliceIdx with model when not playing
  React.useEffect(() => {
    if (!playing) setDisplaySliceIdx(sliceIdx);
  }, [sliceIdx, playing]);

  // Histogram state for main image
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // Histogram state for FFT
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftStats, setFftStats] = React.useState<{ mean: number; min: number; max: number; std: number }>({ mean: 0, min: 0, max: 0, std: 0 });
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);  // Auto: mask DC + 99.9% clipping
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);
  const [showColorbar, setShowColorbar] = React.useState(false);

  const handleRootMouseDownCapture = React.useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("canvas")) rootRef.current?.focus();
  }, []);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);

  // ROI FFT state: when ROI + FFT are both active, compute FFT of cropped ROI region
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);

  // FFT zoom/pan state
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const fftContainerRef = React.useRef<HTMLDivElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{row: number; col: number}[]>("profile_line");
  const [profileWidth, setProfileWidth] = useModelState<number>("profile_width");
  const [profileData, setProfileData] = React.useState<Float32Array | null>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profilePoints = profileLine || [];
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);

  React.useEffect(() => {
    if (hideRoi && roiActive) {
      setRoiActive(false);
      setRoiSelectedIdx(-1);
    }
  }, [hideRoi, roiActive, setRoiActive, setRoiSelectedIdx]);

  React.useEffect(() => {
    if (hideProfile && profileActive) {
      setProfileActive(false);
      setProfileLine([]);
      setProfileData(null);
    }
  }, [hideProfile, profileActive, setProfileLine]);

  React.useEffect(() => {
    if (hideDisplay && showLens) {
      setShowLens(false);
      setLensPos(null);
    }
  }, [hideDisplay, showLens]);

  // Sync sizes from Python and set initial minimum. In multi-panel mode the user
  // is comparing N images side-by-side; default ~300 px per panel so each is
  // readable instead of crushed when the widget concatenates them into one wide
  // canvas (e.g. 4 panels at 500 px total → 125 px per panel = too small).
  React.useEffect(() => {
    // size is PER PANEL. For multi-panel, total canvas width = size * cols.
    // NEVER BIN rule: data is never averaged. CSS canvas scales the painted
    // image for display, source pixels stay intact. 400 px/panel default
    // gives 4 cols → 1600 px wide which fits a typical monitor; operator
    // drags the resize handle larger when they want pixel-1:1.
    const n = nPanels || 1;
    const cols = (maxCols && maxCols > 0) ? Math.min(maxCols, n) : n;
    const perPanel = canvasSizeTrait > 0 ? canvasSizeTrait : (n > 1 ? 450 : CANVAS_TARGET_SIZE);
    const target = perPanel * cols;
    setMainCanvasSize(target);
    if (initialCanvasSizeRef.current === CANVAS_TARGET_SIZE) {
      initialCanvasSizeRef.current = target;
    }
  }, [canvasSizeTrait, nPanels, maxCols]);

  // Calculate display scale
  const displayScale = mainCanvasSize / Math.max(width, height);
  // For 90°/270° rotations, swap canvas dims so non-square images fit without clipping.
  const rotSwap = (imageRotation % 2) !== 0;
  const canvasW = Math.round((rotSwap ? height : width) * displayScale);
  // Grid layout: when max_cols wraps panels into multiple rows, canvasH grows to fit `rows` rows.
  const _nPanelsLocal = Math.max(1, nPanels || 1);
  const _colsLocal = (maxCols && maxCols > 0) ? Math.min(maxCols, _nPanelsLocal) : _nPanelsLocal;
  const _rowsLocal = Math.ceil(_nPanelsLocal / _colsLocal);
  const _canvasHSingleRow = Math.round((rotSwap ? width : height) * displayScale);
  // In single-row mode the canvas spans the full concatenated width.
  // In multi-row mode, derive canvas height by scaling the single-row height proportionally:
  // each row gets canvasW/cols of width; for square per-panel, height per row = canvasW/cols.
  const canvasH = _rowsLocal > 1 ? Math.round((canvasW / _colsLocal) * _rowsLocal * (height / (width / _nPanelsLocal))) : _canvasHSingleRow;
  const effectiveLoopEnd = loopEnd < 0 ? nSlices - 1 : loopEnd;

  // ROI FFT active: both ROI and FFT on, with a selected ROI
  const roiFftActive = effectiveShowFft && roiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);

  // Preview panel visible: auto-shows when ROI active with a selected ROI
  const previewVisible = roiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);
  const selectedRoiKey = React.useMemo(() => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return "";
    const r = roiList[roiSelectedIdx];
    return `${r.row},${r.col},${r.radius},${r.radius_inner},${r.width},${r.height},${r.shape}`;
  }, [roiList, roiSelectedIdx]);

  // Compute stats for ALL ROIs (memoized, recomputes on frame/ROI geometry change)
  const allRoiStats = React.useMemo(() => {
    const raw = rawFrameDataRef.current;
    if (!roiActive || !roiItems.length || !raw || !width || !height) return [];
    return roiItems.map(roi => computeROIPixelStats(raw, width, height, roi));
    // frameBytes triggers recompute on frame change; displaySliceIdx triggers recompute during playback
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roiActive, roiItems, width, height, frameBytes, displaySliceIdx]);

  // Initialize reusable offscreen canvas + ImageData (resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    mainOffscreenRef.current = canvas;
    mainImgDataRef.current = canvas.getContext("2d")!.createImageData(width, height);
    logBufferRef.current = new Float32Array(width * height);
  }, [width, height]);

  // Prevent page scroll on canvas containers (but don't stop propagation so React handlers work)
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const el1 = canvasContainerRef.current;
    const el2 = fftContainerRef.current;
    const el3 = previewContainerRef.current;
    el1?.addEventListener("wheel", preventDefault, { passive: false });
    el2?.addEventListener("wheel", preventDefault, { passive: false });
    el3?.addEventListener("wheel", preventDefault, { passive: false });
    return () => {
      el1?.removeEventListener("wheel", preventDefault);
      el2?.removeEventListener("wheel", preventDefault);
      el3?.removeEventListener("wheel", preventDefault);
    };
  }, [effectiveShowFft, previewVisible]);


  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // All playback params as a single ref (avoids stale closures in rAF loop)
  const pathIdxRef = React.useRef(0);
  const playRef = React.useRef({
    fps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth,
    traitVmin, traitVmax, smooth, imageRotation,
  });
  React.useEffect(() => {
    playRef.current = {
      fps, reverse, boomerang, loop, loopStart, loopEnd: effectiveLoopEnd,
      nSlices, width, height, displayScale, canvasW, canvasH,
      logScale, autoContrast, percentileLow, percentileHigh,
      dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
      zoom, panX, panY, playbackPath,
      profileActive, profilePoints, profileWidth,
      traitVmin, traitVmax, smooth, imageRotation,
    };
  }, [fps, reverse, boomerang, loop, loopStart, effectiveLoopEnd,
    nSlices, width, height, displayScale, canvasW, canvasH,
    logScale, autoContrast, percentileLow, percentileHigh,
    dataMin, dataMax, cmap, imageVminPct, imageVmaxPct,
    zoom, panX, panY, playbackPath,
    profileActive, profilePoints, profileWidth,
    traitVmin, traitVmax, smooth, imageRotation]);

  // Playback logic - rAF-driven, zero React re-renders in hot path
  React.useEffect(() => {
    if (!playing) {
      // Playback stopped - sync final position to Python
      if (playbackIdxRef.current !== sliceIdx && bufferRef.current) {
        setSliceIdx(playbackIdxRef.current);
      }
      setLocalStats(null);
      bufferRef.current = null;
      bufferCountRef.current = 0;
      nextBufferRef.current = null;
      nextBufferCountRef.current = 0;
      prefetchPendingRef.current = false;
      return;
    }

    // === PLAYBACK START ===
    // Snap slice_idx into [loop_start, loop_end] before first tick, otherwise
    // playback walked outside the loop range on the first frame.
    {
      const c0 = playRef.current;
      const rs0 = c0.loop ? Math.max(0, Math.min(c0.loopStart, c0.nSlices - 1)) : 0;
      const re0 = c0.loop ? Math.max(rs0, Math.min(c0.loopEnd, c0.nSlices - 1)) : c0.nSlices - 1;
      playbackIdxRef.current = Math.max(rs0, Math.min(re0, sliceIdx));
    }
    const pathLen = playRef.current.playbackPath?.length ?? 0;
    pathIdxRef.current = pathLen > 0 ? (playRef.current.reverse ? pathLen : -1) : 0;
    bounceDirRef.current = playRef.current.reverse ? -1 : 1;
    let lastFrameTime = 0;
    let lastUIUpdate = 0;
    let animId: number;

    const tick = (now: number) => {
      const c = playRef.current;
      const intervalMs = 1000 / c.fps;

      // First tick - just record time
      if (lastFrameTime === 0) {
        lastFrameTime = now;
        lastUIUpdate = now;
        animId = requestAnimationFrame(tick);
        return;
      }

      const elapsed = now - lastFrameTime;
      if (elapsed < intervalMs) {
        animId = requestAnimationFrame(tick);
        return;
      }
      lastFrameTime = now - (elapsed % intervalMs);

      // Advance frame
      let next: number;
      if (c.playbackPath && c.playbackPath.length > 0) {
        // Custom playback path
        const pp = c.playbackPath;
        let pi = pathIdxRef.current;
        if (c.boomerang) {
          // Visit endpoints once (matches grid-mode boomerang). Earlier code
          // jumped to pp.length-2 / 1 on overshoot, skipping endpoints.
          pi += bounceDirRef.current;
          if (pi >= pp.length) { bounceDirRef.current = -1; pi = pp.length - 1; }
          else if (pi < 0) { bounceDirRef.current = 1; pi = 0; }
        } else {
          pi += (c.reverse ? -1 : 1);
          if (pi >= pp.length) { if (!c.loop) { setPlaying(false); return; } pi = 0; }
          if (pi < 0) { if (!c.loop) { setPlaying(false); return; } pi = pp.length - 1; }
        }
        pi = Math.max(0, Math.min(pp.length - 1, pi));
        pathIdxRef.current = pi;
        next = pp[pi];
      } else {
        const rangeStart = c.loop ? Math.max(0, Math.min(c.loopStart, c.nSlices - 1)) : 0;
        const rangeEnd = c.loop ? Math.max(rangeStart, Math.min(c.loopEnd, c.nSlices - 1)) : c.nSlices - 1;
        const prev = playbackIdxRef.current;

        if (c.boomerang) {
          next = prev + bounceDirRef.current;
          if (next > rangeEnd) { bounceDirRef.current = -1; next = prev - 1 >= rangeStart ? prev - 1 : prev; }
          else if (next < rangeStart) { bounceDirRef.current = 1; next = prev + 1 <= rangeEnd ? prev + 1 : prev; }
        } else {
          next = prev + (c.reverse ? -1 : 1);
          if (c.reverse) {
            if (next < rangeStart) { if (!c.loop) { setPlaying(false); return; } next = rangeEnd; }
          } else {
            if (next > rangeEnd) { if (!c.loop) { setPlaying(false); return; } next = rangeStart; }
          }
        }
      }

      // Try buffer path (zero round-trip) with double-buffer swap
      const frameSize = c.width * c.height;
      let frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
      if (!frame && nextBufferRef.current) {
        // Current buffer doesn't have this frame - swap to next buffer
        bufferRef.current = nextBufferRef.current;
        bufferStartRef.current = nextBufferStartRef.current;
        bufferCountRef.current = nextBufferCountRef.current;
        nextBufferRef.current = null;
        nextBufferCountRef.current = 0;
        frame = getFrameFromBuffer(bufferRef.current, bufferStartRef.current, bufferCountRef.current, c.nSlices, next, frameSize);
      }
      if (!frame) {
        // Buffer not ready yet - keep requesting frames
        animId = requestAnimationFrame(tick);
        return;
      }

      playbackIdxRef.current = next;
      rawFrameDataRef.current = frame;

      // Render frame - fused single-pass when possible
      const lut = COLORMAPS[c.cmap] || COLORMAPS.inferno;
      if (mainOffscreenRef.current && mainImgDataRef.current) {
        let vmin: number, vmax: number;
        if (c.autoContrast) {
          // Auto-contrast needs per-frame percentile (2 passes), but no stats
          if (c.logScale && logBufferRef.current) {
            applyLogScaleInPlace(frame, logBufferRef.current);
            ({ vmin, vmax } = percentileClip(logBufferRef.current, c.percentileLow, c.percentileHigh));
            renderToOffscreenReuse(logBufferRef.current, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
          } else {
            ({ vmin, vmax } = percentileClip(frame, c.percentileLow, c.percentileHigh));
            renderToOffscreenReuse(frame, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
          }
        } else {
          // Global range + slider - fused single-pass render (fastest path)
          ({ vmin, vmax } = resolveDisplayRange(
            c.dataMin,
            c.dataMax,
            c.traitVmin,
            c.traitVmax,
            c.logScale,
            c.imageVminPct,
            c.imageVmaxPct,
          ));
          renderFramePlayback(frame, mainImgDataRef.current.data, lut, vmin, vmax, c.logScale);
          mainOffscreenRef.current.getContext("2d")!.putImageData(mainImgDataRef.current, 0, 0);
        }

        // Draw to display canvas. Apply image_rotation so playback matches the
        // static render path (lines 1444-1453); otherwise rotated stacks
        // silently lose their rotation when the user hits Play.
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.imageSmoothingEnabled = c.smooth;
            ctx.clearRect(0, 0, c.canvasW, c.canvasH);
            ctx.save();
            ctx.translate(c.panX, c.panY);
            ctx.scale(c.zoom, c.zoom);
            const dw = c.width * c.displayScale, dh = c.height * c.displayScale;
            if (c.imageRotation % 4 !== 0) {
              const cx = c.canvasW / 2 / c.zoom, cy = c.canvasH / 2 / c.zoom;
              ctx.translate(cx, cy);
              ctx.rotate((c.imageRotation * Math.PI) / 2);
              ctx.translate(-dw / 2, -dh / 2);
              ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
            } else {
              ctx.drawImage(mainOffscreenRef.current, 0, 0, dw, dh);
            }
            ctx.restore();
          }
        }
      }

      // Throttled UI updates - 10 FPS for slider/stats/profile (avoids costly MUI re-renders)
      if (now - lastUIUpdate > 100) {
        lastUIUpdate = now;
        setDisplaySliceIdx(next);
        setLocalStats(computeStats(frame));
        if (c.profileActive && c.profilePoints.length === 2) {
          const p0 = c.profilePoints[0], p1 = c.profilePoints[1];
          setProfileData(sampleLineProfile(frame, c.width, c.height, p0.row, p0.col, p1.row, p1.col, c.profileWidth));
        }
        // Histogram refresh during playback. The non-playback effect path is keyed on
        // frameBytes/frameSeq which DON'T change during rAF playback (frames come from
        // the prefetch buffer, not via Comm), so we drive histogram updates directly
        // here at the same 10 Hz cadence. Skip every 2nd tick → ~5 Hz refresh.
        playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
        if (playbackHistogramCounterRef.current === 0) {
          const histInput = c.logScale ? applyLogScale(frame) : frame;
          setImageHistogramData(histInput);
        }
      }

      // Prefetch at 25% buffer consumed - only if no next buffer is already queued.
      // Respect loop range so we don't fetch frames outside [loop_start, loop_end].
      if (!prefetchPendingRef.current && !nextBufferRef.current && bufferCountRef.current > 0) {
        let idxInBuffer = next - bufferStartRef.current;
        if (idxInBuffer < 0) idxInBuffer += c.nSlices;
        if (idxInBuffer >= Math.floor(bufferCountRef.current / 4)) {
          let prefetchStart = (bufferStartRef.current + bufferCountRef.current) % c.nSlices;
          // If loop range is constrained, snap prefetch start into it so we
          // don't waste buffer on frames the loop will never display.
          if (c.loop && (c.loopStart > 0 || c.loopEnd >= 0)) {
            const rs = Math.max(0, Math.min(c.loopStart, c.nSlices - 1));
            const re = c.loopEnd < 0 ? c.nSlices - 1 : Math.max(rs, Math.min(c.loopEnd, c.nSlices - 1));
            if (prefetchStart < rs || prefetchStart > re) prefetchStart = rs;
          }
          prefetchPendingRef.current = true;
          setPrefetchRequest(prefetchStart);
        }
      }

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // Update frame ref when frame changes
  React.useEffect(() => {
    const parsed = extractFloat32(frameBytes);
    if (!parsed || parsed.length === 0) return;
    rawFrameDataRef.current = parsed;
    // Recompute stats JS-side (mean/min/max/std + per-panel slices). Cheap
    // (single pass over the frame), and avoids the Python round-trip.
    const n = Math.max(1, nPanels || 1);
    const total = computeStats(parsed);
    setLocalStats(total);
    if (n > 1 && height > 0 && width > 0 && width % n === 0) {
      const pw = width / n;
      const panels: { mean: number; min: number; max: number; std: number }[] = [];
      for (let p = 0; p < n; p++) {
        // Slice columns [p*pw, (p+1)*pw) for all rows.
        const slab = new Float32Array(height * pw);
        for (let r = 0; r < height; r++) {
          const srcOff = r * width + p * pw;
          slab.set(parsed.subarray(srcOff, srcOff + pw), r * pw);
        }
        panels.push(computeStats(slab));
      }
      setLocalPanelStats(panels);
    } else {
      setLocalPanelStats(null);
    }
  }, [frameBytes, frameSeq, nPanels, width, height]);

  // Update histogram data (reflects log scale state). During playback we throttle so
  // a full O(N) bin pass on multi-MB frames doesn't blow the frame budget. We rely on
  // `displaySliceIdx` ticking at ~10 Hz (the playback rAF's React-state throttle) and
  // only update histogram every 2nd tick → ~5 Hz refresh, ~10 % CPU budget.
  const playbackHistogramCounterRef = React.useRef(0);
  React.useEffect(() => {
    const raw = rawFrameDataRef.current;
    if (!raw || raw.length === 0) return;
    if (playing) {
      playbackHistogramCounterRef.current = (playbackHistogramCounterRef.current + 1) % 2;
      if (playbackHistogramCounterRef.current !== 0) return;
    } else {
      playbackHistogramCounterRef.current = 0;
    }
    const data = logScale ? applyLogScale(raw) : raw;
    setImageHistogramData(data);
    setImageDataRange(findDataRange(data));
  }, [frameBytes, frameSeq, playing, logScale, displaySliceIdx]);

  // Auto-snap thumbs to percentile-clip values while Auto is on. Fires once at mount
  // (so the slider visually reflects the percentile-clipped contrast that Python applies
  // when auto_contrast=True), and re-fires when logScale flips (linear vs log percentile
  // give different clip values, so the thumbs must follow). The lastLogScaleRef tracks
  // the previous logScale value so we only re-snap on transitions, not on every render.
  const initialAutoSnappedRef = React.useRef(false);
  const lastLogScaleRef = React.useRef(logScale);
  React.useEffect(() => {
    const logScaleChanged = lastLogScaleRef.current !== logScale;
    lastLogScaleRef.current = logScale;
    if (!autoContrast || !imageHistogramData || imageHistogramData.length === 0) return;
    // Skip initial snap if user already moved thumbs (e.g. loaded from saved state).
    if (!initialAutoSnappedRef.current && (imageVminPct !== 0 || imageVmaxPct !== 100)) {
      initialAutoSnappedRef.current = true;
      return;
    }
    // After first snap, only re-snap on logScale transitions.
    if (initialAutoSnappedRef.current && !logScaleChanged) return;
    const span = dataMax - dataMin;
    if (span <= 0) return;
    const { vmin: pmin, vmax: pmax } = percentileClip(imageHistogramData, percentileLow, percentileHigh);
    setImageVminPct(Math.max(0, Math.min(100, ((pmin - dataMin) / span) * 100)));
    setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - dataMin) / span) * 100)));
    initialAutoSnappedRef.current = true;
  }, [autoContrast, imageHistogramData, dataMin, dataMax, percentileLow, percentileHigh, logScale, imageVminPct, imageVmaxPct]);

  React.useEffect(() => {
    if (!roiActive || roiItems.length === 0 || !showRoiResizeHint) return;
    const timer = window.setTimeout(() => setShowRoiResizeHint(false), 6000);
    return () => window.clearTimeout(timer);
  }, [roiActive, roiItems.length, showRoiResizeHint]);

  // Data effect: normalize + colormap → reusable offscreen canvas, then draw
  React.useEffect(() => {
    const frameData = rawFrameDataRef.current;
    if (!frameData || frameData.length === 0) return;
    if (!mainOffscreenRef.current || !mainImgDataRef.current) return;

    // Apply log scale using reusable buffer
    const processed = logScale && logBufferRef.current
      ? applyLogScaleInPlace(frameData, logBufferRef.current)
      : frameData;

    const nP = Math.max(1, nPanels || 1);
    const perPanelContrast = nP > 1 && !linkContrast && width % nP === 0 && height > 0;

    // Compute vmin/vmax (per-panel branch uses GPU multi-slot below)
    let vmin: number, vmax: number;
    const hasTraitRange = traitVmin != null || traitVmax != null;
    if (hasTraitRange) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else {
      const { min: pMin, max: pMax } = findDataRange(processed);
      ({ vmin, vmax } = sliderRange(pMin, pMax, imageVminPct, imageVmaxPct));
    }

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    // GPU colormap path (single frame) - zero-copy via OffscreenCanvas→ImageBitmap
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      engine.uploadLUT(cmap, lut);
      // Per-panel contrast: upload the FULL frame ONCE as slot 0, then run a
      // fused GPU pipeline that, per panel: reduces a sub-region → vmin/vmax,
      // colormaps the panel sub-image using those values + slider pcts, and
      // blits to a panel-sized OffscreenCanvas. No JS slab extraction, no
      // findDataRange loop, no CPU readback between range and colormap.
      const dataForGpu = logScale ? processed : frameData;
      if (perPanelContrast) {
        const pw = width / nP;
        engine.uploadData(0, dataForGpu, width, height);
        const regions = Array.from({ length: nP }, (_, p) => ({
          x: p * pw, y: 0, width: pw, height,
        }));
        const lowPct = Array(nP).fill(imageVminPct);
        const highPct = Array(nP).fill(imageVmaxPct);
        requestAnimationFrame(() => {
          if (!mainOffscreenRef.current) return;
          const bitmaps = engine.renderPerPanelGpu(0, regions, lowPct, highPct, false);
          if (bitmaps) {
            const ctx = mainOffscreenRef.current.getContext("2d");
            if (ctx) {
              for (let p = 0; p < nP; p++) {
                if (bitmaps[p]) {
                  ctx.drawImage(bitmaps[p], p * pw, 0);
                  bitmaps[p].close();
                }
              }
            }
          }
          const canvas = canvasRef.current;
          if (!canvas) return;
          const ctx2 = canvas.getContext("2d");
          if (ctx2 && mainOffscreenRef.current) drawMain(ctx2, mainOffscreenRef.current);
        });
        return;
      }
      engine.uploadData(0, dataForGpu, width, height);
      const capturedVmin = vmin, capturedVmax = vmax;
      requestAnimationFrame(async () => {
        if (!mainOffscreenRef.current) return;
        // Zero-copy: GPU → OffscreenCanvas → ImageBitmap → drawImage
        const bitmaps = engine.renderSlotsToImageBitmap([0], [{ vmin: capturedVmin, vmax: capturedVmax }], false);
        if (bitmaps && bitmaps[0]) {
          const ctx = mainOffscreenRef.current.getContext("2d");
          if (ctx) ctx.drawImage(bitmaps[0], 0, 0);
          // ImageBitmap holds external GPU/CPU memory not reclaimed by GC. Must close()
          // explicitly or repeated render calls (cmap/contrast/scrub) leak ~MB per call.
          bitmaps[0].close();
        } else {
          // Fallback: mapAsync path
          if (mainImgDataRef.current) {
            const rendered = await engine.renderSlots(
              [0], [{ vmin: capturedVmin, vmax: capturedVmax }],
              [mainOffscreenRef.current], [mainImgDataRef.current], false,
            );
            if (rendered === 0) {
              renderToOffscreenReuse(processed, lut, capturedVmin, capturedVmax, mainOffscreenRef.current!, mainImgDataRef.current!);
            }
          }
        }
        // Redraw main canvas (per-panel)
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (ctx && mainOffscreenRef.current) drawMain(ctx, mainOffscreenRef.current);
      });
    } else {
      // CPU fallback
      renderToOffscreenReuse(processed, lut, vmin, vmax, mainOffscreenRef.current, mainImgDataRef.current);
    }

    // Draw to main canvas (CPU path only - GPU path draws in its own rAF above)
    if (!engine || !gpuCmapReadyRef.current) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (ctx && mainOffscreenRef.current) drawMain(ctx, mainOffscreenRef.current);
    }
  }, [frameBytes, frameSeq, width, height, cmap, displayScale, canvasW, canvasH, imageVminPct, imageVmaxPct, logScale, autoContrast, percentileLow, percentileHigh, traitVmin, traitVmax, dataMin, dataMax, smooth, imageRotation, nPanels, linkContrast]);

  // Per-panel render: each slot gets its own zoom/pan transform. 2px gap
  // between slots painted as the canvas bg (transparent through clearRect).
  const drawMain = React.useCallback((ctx: CanvasRenderingContext2D, offscreen: HTMLCanvasElement | OffscreenCanvas) => {
    ctx.imageSmoothingEnabled = smooth;
    // Clear entire canvas. Slot-level bg fill happens inside the per-panel
    // loop so empty grid cells (partial last row) stay transparent - the
    // page bg shows through instead of a dead white block.
    ctx.clearRect(0, 0, canvasW, canvasH);
    const n = Math.max(1, nPanels || 1);
    const cols = (maxCols && maxCols > 0) ? Math.min(maxCols, n) : n;
    const rows = Math.ceil(n / cols);
    const srcPanelW = offscreen.width / n;
    const srcH = offscreen.height;
    const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
    const outPanelW = (canvasW - gap * (cols - 1)) / cols;
    const outPanelH = (canvasH - gap * (rows - 1)) / rows;
    for (let i = 0; i < n; i++) {
      const s = stateFor(i);
      const col = i % cols;
      const row = Math.floor(i / cols);
      const slotX = col * (outPanelW + gap);
      const slotY = row * (outPanelH + gap);
      // Per-slot bg fill - only real panels get the theme bg; empty grid
      // cells in a partial last row stay transparent.
      ctx.fillStyle = themeColors.bg;
      ctx.fillRect(slotX, slotY, outPanelW, outPanelH);
      // End-of-stack: when current frame exceeds this panel's real frame
      // count, blur the (repeated last) frame + draw "end ({real}/{real})"
      // badge so operator sees they're scrubbing past real data.
      const realN = panelRealFrames && panelRealFrames[i];
      const pastEnd = !!(realN && (playing ? displaySliceIdx : sliceIdx) >= realN);
      ctx.save();
      ctx.beginPath();
      ctx.rect(slotX, slotY, outPanelW, outPanelH);
      ctx.clip();
      ctx.translate(slotX + s.panX, slotY + s.panY);
      ctx.scale(s.zoom, s.zoom);
      const w = outPanelW, h = outPanelH;
      if (imageRotation % 4 !== 0) {
        const cx = w / 2 / s.zoom, cy = h / 2 / s.zoom;
        ctx.translate(cx, cy);
        ctx.rotate((imageRotation * Math.PI) / 2);
        ctx.translate(-w / 2, -h / 2);
      }
      if (pastEnd) ctx.filter = "blur(4px)";
      ctx.drawImage(offscreen as CanvasImageSource, i * srcPanelW, 0, srcPanelW, srcH, 0, 0, w, h);
      ctx.restore();
      // Per-panel title - drawn on canvas at top-center of each panel slot.
      // Lives on the canvas (not below it) so it follows grid layout when
      // panels wrap into multiple rows. Clipped to slot so long titles
      // don't bleed into the next column.
      if ((nPanels || 1) > 1 && panelTitles && panelTitles[i]) {
        const realN2 = panelRealFrames && panelRealFrames[i];
        const cur = (playing ? displaySliceIdx : sliceIdx) + 1;
        const total = realN2 || nSlices;
        const shown = realN2 ? Math.min(cur, realN2) : cur;
        const label = `${panelTitles[i]}  ${shown}/${total}`;
        ctx.save();
        ctx.beginPath();
        ctx.rect(slotX, slotY, outPanelW, outPanelH);
        ctx.clip();
        ctx.font = `bold ${Math.max(8, panelTitleFontSize || 11)}px ui-monospace, monospace`;
        const tw = ctx.measureText(label).width;
        const lx = slotX + (outPanelW - tw) / 2;
        const ly = slotY + 14;
        // Offset shadow via two paints (cheaper than canvas shadowBlur):
        ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
        ctx.fillText(label, lx + 1, ly + 1);
        ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
        ctx.fillText(label, lx, ly);
        ctx.restore();
      }
      // No end badge - blur alone signals past-real-frame.
    }
  }, [smooth, canvasW, canvasH, nPanels, maxCols, imageRotation, panelStates, linkedState, linkZoom, linkPan, themeColors.bg, panelRealFrames, panelTitles, panelGapTrait, panelTitleFontSize, sliceIdx, displaySliceIdx, playing, nSlices]);

  React.useLayoutEffect(() => {
    if (!mainOffscreenRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (ctx) drawMain(ctx, mainOffscreenRef.current);
  }, [drawMain]);

  // Render overlay (ROI only) - HiDPI aware
  React.useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, canvasW, canvasH);
    // Match the main image's rotation so ROIs / profile sit on the right pixels.
    // Image draw applies `translate(panX,panY) → scale(zoom) → rotate(around cx)`,
    // so the rotation pivot in screen pixels is (canvasW/2+panX, canvasH/2+panY).
    // Overlay must use the SAME screen-space pivot - earlier bug used (canvasW/2,
    // canvasH/2) without pan offset, drifting ROIs when user panned + rotated.
    if (imageRotation % 4 !== 0) {
      const cx = canvasW / 2 + panX;
      const cy = canvasH / 2 + panY;
      ctx.translate(cx, cy);
      ctx.rotate((imageRotation * Math.PI) / 2);
      ctx.translate(-cx, -cy);
    }
    if (!hideRoi && roiActive && roiItems.length > 0) {
      const highlightedRois = roiItems.filter(r => r.highlight);
      if (highlightedRois.length > 0) {
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(0, 0, canvasW, canvasH);
        ctx.globalCompositeOperation = "destination-out";
        for (const roi of highlightedRois) {
          const sx = roi.col * displayScale * zoom + panX;
          const sy = roi.row * displayScale * zoom + panY;
          const sr = roi.radius * displayScale * zoom;
          const shape = roi.shape || "circle";
          ctx.fillStyle = "rgba(0,0,0,1)";
          if (shape === "circle") {
            ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
          } else if (shape === "square") {
            ctx.fillRect(sx - sr, sy - sr, sr * 2, sr * 2);
          } else if (shape === "rectangle") {
            const sw = roi.width * displayScale * zoom;
            const sh = roi.height * displayScale * zoom;
            ctx.fillRect(sx - sw / 2, sy - sh / 2, sw, sh);
          } else if (shape === "annular") {
            ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "source-over";
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            const sir = roi.radius_inner * displayScale * zoom;
            ctx.beginPath(); ctx.arc(sx, sy, sir, 0, Math.PI * 2); ctx.fill();
            ctx.globalCompositeOperation = "destination-out";
          }
        }
        ctx.restore();
      }

      for (let ri = 0; ri < roiItems.length; ri++) {
        const roi = roiItems[ri];
        const isSelected = ri === roiSelectedIdx;
        const screenX = roi.col * displayScale * zoom + panX;
        const screenY = roi.row * displayScale * zoom + panY;
        const screenRadius = roi.radius * displayScale * zoom;
        const screenWidth = roi.width * displayScale * zoom;
        const screenHeight = roi.height * displayScale * zoom;
        const screenRadiusInner = roi.radius_inner * displayScale * zoom;
        const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
        ctx.lineWidth = roi.line_width || 2;
        const color = roi.color || ROI_COLORS[ri % ROI_COLORS.length];
        drawROI(ctx, screenX, screenY, shape, screenRadius, screenWidth, screenHeight, color, color, isSelected && isDraggingROI, screenRadiusInner);
        if (isSelected) {
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1;
          if (shape === "circle" || shape === "annular") {
            ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius + 3, 0, Math.PI * 2); ctx.stroke();
          } else if (shape === "square") {
            ctx.strokeRect(screenX - screenRadius - 3, screenY - screenRadius - 3, (screenRadius + 3) * 2, (screenRadius + 3) * 2);
          } else if (shape === "rectangle") {
            ctx.strokeRect(screenX - screenWidth / 2 - 3, screenY - screenHeight / 2 - 3, screenWidth + 6, screenHeight + 6);
          }
          ctx.setLineDash([]);
        }
      }
    }

    // Line profile overlay
    if (!hideProfile && profileActive && profilePoints.length > 0) {
      const toScreenX = (col: number) => col * displayScale * zoom + panX;
      const toScreenY = (row: number) => row * displayScale * zoom + panY;

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

        ctx.strokeStyle = themeColors.accent;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(bx, by, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [roiActive, roiItems, roiSelectedIdx, isDraggingROI, canvasW, canvasH, displayScale, zoom, panX, panY, themeColors, profileActive, profilePoints, profileWidth, hideRoi, hideProfile, nPanels, panelTitles, imageRotation, width, height]);

  // Lens inset rendering
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lctx = lensCanvas.getContext("2d");
      if (lctx) lctx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || hideDisplay || !lensPos || !rawFrameDataRef.current) return;
    if ((nPanels || 1) > 1) return;  // Lens disabled in multi-panel mode
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = rawFrameDataRef.current;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (traitVmin != null || traitVmax != null) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else if (imageDataRange.min !== imageDataRange.max) {
      ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
    } else {
      const r = findDataRange(processed);
      vmin = r.min; vmax = r.max;
    }

    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const margin = 12;
    const half = Math.floor(regionSize / 2);
    const r0 = lensPos.row - half;
    const c0 = lensPos.col - half;

    const regionCanvas = document.createElement("canvas");
    regionCanvas.width = regionSize;
    regionCanvas.height = regionSize;
    const rctx = regionCanvas.getContext("2d");
    if (!rctx) return;
    const imgData = rctx.createImageData(regionSize, regionSize);
    const range = vmax - vmin || 1;
    for (let dr = 0; dr < regionSize; dr++) {
      for (let dc = 0; dc < regionSize; dc++) {
        const sr = r0 + dr;
        const sc = c0 + dc;
        const idx = (dr * regionSize + dc) * 4;
        if (sr < 0 || sr >= height || sc < 0 || sc >= width) {
          imgData.data[idx] = 0; imgData.data[idx + 1] = 0; imgData.data[idx + 2] = 0; imgData.data[idx + 3] = 255;
        } else {
          const val = processed[sr * width + sc];
          const t = Math.max(0, Math.min(1, (val - vmin) / range));
          const li = Math.round(t * 255);
          imgData.data[idx] = lut[li * 3]; imgData.data[idx + 1] = lut[li * 3 + 1]; imgData.data[idx + 2] = lut[li * 3 + 2]; imgData.data[idx + 3] = 255;
        }
      }
    }
    rctx.putImageData(imgData, 0, 0);

    ctx.save();
    ctx.scale(DPR, DPR);
    // Clamp anchor + default position to canvas bounds. Without clamp a small canvas
    // (e.g. multi-panel 100 px tall) puts the inset off-screen (-60 px) because
    // default ly = canvasH - lensSize - margin - 20 goes negative.
    const cssH = canvasH;
    const cssW = canvasW;
    const rawLx = lensAnchor ? lensAnchor.x : margin;
    const rawLy = lensAnchor ? lensAnchor.y : cssH - lensSize - margin - 20;
    const lx = Math.max(0, Math.min(cssW - lensSize, rawLx));
    const ly = Math.max(0, Math.min(cssH - lensSize, rawLy));
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(regionCanvas, lx, ly, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, lensSize, lensSize);
    const cx = lx + lensSize / 2;
    const cy = ly + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lx + 4, ly + lensSize - 4);
    ctx.restore();
  }, [showLens, hideDisplay, lensPos, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, dataMin, dataMax, traitVmin, traitVmax, width, height, canvasW, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor, percentileLow, percentileHigh, frameBytes, sliceIdx, displaySliceIdx, nPanels]);

  // ROI sparkline plot
  React.useEffect(() => {
    const canvas = roiPlotCanvasRef.current;
    if (!canvas || !showRoiPlot || !roiActive || hideRoi) return;
    const plotW = canvasW;
    const plotH = 76;
    canvas.width = Math.round(plotW * DPR);
    canvas.height = Math.round(plotH * DPR);
    canvas.style.width = `${plotW}px`;
    canvas.style.height = `${plotH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, plotW, plotH);

    if (!roiPlotData || roiPlotData.byteLength < 4) return;
    const values = extractFloat32(roiPlotData);
    if (!values || values.length === 0) return;
    let min = values[0], max = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    const range = max - min || 1;
    const padY = 14;
    const drawH = plotH - padY * 2;

    // Draw plot line
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const denom = Math.max(1, values.length - 1);
    for (let i = 0; i < values.length; i++) {
      const x = (i / denom) * plotW;
      const y = padY + drawH - ((values[i] - min) / range) * drawH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw current frame marker
    const activeIdx = playing ? displaySliceIdx : sliceIdx;
    const markerIdx = Math.max(0, Math.min(values.length - 1, activeIdx));
    const mx = (markerIdx / denom) * plotW;
    ctx.strokeStyle = themeColors.textMuted;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(mx, padY);
    ctx.lineTo(mx, padY + drawH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Current value dot
    if (values.length > 0) {
      const cy = padY + drawH - ((values[markerIdx] - min) / range) * drawH;
      ctx.fillStyle = themeColors.accent;
      ctx.beginPath();
      ctx.arc(mx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Y-axis labels
    ctx.fillStyle = themeColors.textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(formatNumber(max), 2, padY - 2);
    ctx.fillText(formatNumber(min), 2, padY + drawH + 10);
  }, [roiPlotData, roiActive, showRoiPlot, canvasW, themeColors, sliceIdx, displaySliceIdx, playing, hideRoi]);

  // Compute profile data when points/width/frame change
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawFrameDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      const data = rawFrameDataRef.current;
      setProfileData(sampleLineProfile(data, width, height, p0.row, p0.col, p1.row, p1.col, profileWidth));
      if (!profileActive) setProfileActive(true);
    } else {
      setProfileData(null);
    }
  }, [profilePoints, profileWidth, frameBytes]);

  // Render profile sparkline
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvasW;
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
      ctx.fillText("Click two points on the image to draw a profile", cssW / 2, cssH / 2);
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

    // Draw profile line
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

    // X-axis: calibrated distance
    let totalDist = profileData.length - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (pixelSize > 0) {
        const distA = distPx * pixelSize;
        if (distA >= 10) { totalDist = distA / 10; xUnit = "nm"; }
        else { totalDist = distA; xUnit = "Å"; }
      } else {
        totalDist = distPx;
      }
    }

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
      const valStr = v % 1 === 0 ? v.toFixed(0) : v.toFixed(1);
      ctx.fillText(i === ticks.length - 1 ? `${valStr} ${xUnit}` : valStr, x, tickY + 4);
    }

    // Y-axis min/max labels
    ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = isDark ? "#888" : "#666";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatNumber(gMax), padLeft - 3, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatNumber(gMin), padLeft - 3, padTop + plotH);

    // Draw axis lines
    ctx.strokeStyle = isDark ? "#555" : "#bbb";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileData, profilePoints, pixelSize, canvasW, themeInfo.theme, themeColors.accent, profileHeight]);

  // Profile hover handler - draws crosshair + value readout
  const handleProfileMouseMove = React.useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    const layout = profileLayoutRef.current;
    if (!canvas || !base || !layout) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit } = layout;
    const range = gMax - gMin || 1;

    ctx.putImageData(base, 0, 0);
    if (cssX < padLeft || cssX > padLeft + plotW) return;
    const frac = (cssX - padLeft) / plotW;

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Vertical crosshair
    ctx.strokeStyle = themeInfo.theme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(cssX, padTop);
    ctx.lineTo(cssX, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dot on profile line + value
    if (profileData && profileData.length >= 2) {
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
      const isDark = themeInfo.theme === "dark";
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
    }

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

  // Profile height resize
  React.useEffect(() => {
    if (!isResizingProfile) return;
    const handleMouseMove = (e: MouseEvent) => {
      if (!profileResizeStart) return;
      const delta = e.clientY - profileResizeStart.y;
      setProfileHeight(Math.max(40, Math.min(300, profileResizeStart.height + delta)));
    };
    const handleMouseUp = () => {
      setIsResizingProfile(false);
      setProfileResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingProfile, profileResizeStart]);

  // Render HiDPI scale bar + zoom indicator + colorbar
  React.useEffect(() => {
    if (!uiRef.current) return;
    const ctx = uiRef.current.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, uiRef.current.width, uiRef.current.height);
    if (scaleBarVisible) {
      const unit = pixelSize > 0 ? pixelUnit : "px";
      const pxSize = pixelSize > 0 ? pixelSize : 1;
      // Per-panel scale bar + zoom indicator. Each panel slot uses its
      // own panelStates[i].zoom so panels at different zoom levels show
      // their own length bar.
      const n = Math.max(1, nPanels || 1);
      const cols = (maxCols && maxCols > 0) ? Math.min(maxCols, n) : n;
      const rows = Math.ceil(n / cols);
      const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      const slotW = (cssW - gap * (cols - 1)) / cols;
      const slotH = (cssH - gap * (rows - 1)) / rows;
      ctx.save();
      ctx.scale(DPR, DPR);
      // Exact Show2D drawScaleBarHiDPI style: 60 px target, 5 px thickness,
      // 16 px font, 12 px margin. Per-panel: each slot acts as its own
      // canvas region with width=slotW, image source width=`width`.
      const targetBarPxSpec = 60;
      const barThickness = 5;
      const fontSize = 16;
      const margin = 12;
      ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      for (let i = 0; i < n; i++) {
        const s = stateFor(i);
        const col = i % cols;
        const row = Math.floor(i / cols);
        const slotX = col * (slotW + gap);
        const slotY = row * (slotH + gap);
        // Cap bar at 25% of slot width so it never overflows a small slot.
        const targetBarPx = Math.min(targetBarPxSpec, slotW * 0.25);
        const slotScale = slotW / width;
        const effectiveZoom = s.zoom * slotScale;
        const targetPhysical = (targetBarPx / effectiveZoom) * pxSize;
        const nicePhysical = (function (v: number) {
          if (v <= 0) return 1;
          const mag = Math.pow(10, Math.floor(Math.log10(v)));
          const norm = v / mag;
          if (norm < 1.5) return mag;
          if (norm < 3.5) return 2 * mag;
          if (norm < 7.5) return 5 * mag;
          return 10 * mag;
        })(targetPhysical);
        const barPx = (nicePhysical / pxSize) * effectiveZoom;
        const barY = slotY + slotH - margin;
        const barX = slotX + slotW - barPx - margin;
        ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
        ctx.shadowBlur = 2;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        ctx.fillStyle = "white";
        ctx.fillRect(barX, barY, barPx, barThickness);
        const label = nicePhysical >= 1 ? `${Math.round(nicePhysical)} ${unit}` : `${nicePhysical.toFixed(2)} ${unit}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, barX + barPx / 2, barY - 4);
        if (showZoomIndicator !== false) {
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(`${s.zoom.toFixed(1)}×`, slotX + margin, slotY + slotH - margin + barThickness);
        }
      }
      ctx.restore();
    }
    if (!hideDisplay && showColorbar) {
      const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
      // Colorbar must match what's painted on the image, not the raw data range.
      // When autoContrast is on, the image uses percentileClip(low, high) of the
      // current frame - show that range. Otherwise use slider range over data.
      let vmin: number, vmax: number;
      if (traitVmin != null || traitVmax != null) {
        ({ vmin, vmax } = resolveDisplayRange(
          dataMin,
          dataMax,
          traitVmin,
          traitVmax,
          logScale,
          imageVminPct,
          imageVmaxPct,
        ));
      } else if (autoContrast && imageHistogramData && imageHistogramData.length > 0) {
        ({ vmin, vmax } = percentileClip(imageHistogramData, percentileLow, percentileHigh));
      } else {
        ({ vmin, vmax } = sliderRange(imageDataRange.min, imageDataRange.max, imageVminPct, imageVmaxPct));
      }
      const cssW = uiRef.current.width / DPR;
      const cssH = uiRef.current.height / DPR;
      ctx.save();
      ctx.scale(DPR, DPR);
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, logScale);
      ctx.restore();
    }
  }, [pixelSize, pixelUnit, scaleBarVisible, width, canvasW, canvasH, displayScale, zoom, nPanels, maxCols, panelStates, linkedState, linkZoom, panelGapTrait, showZoomIndicator, showColorbar, hideDisplay, cmap, imageDataRange, imageVminPct, imageVmaxPct, logScale, autoContrast, imageHistogramData, percentileLow, percentileHigh, dataMin, dataMax, traitVmin, traitVmax]);

  // Compute FFT magnitude (expensive, async - only re-run on data/GPU changes)
  // Supports ROI-scoped FFT: when ROI is active with a selected ROI, compute
  // FFT of the cropped region instead of the full frame.
  const fftMagRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);

  React.useEffect(() => {
    if (!effectiveShowFft || !rawFrameDataRef.current) return;
    let cancelled = false;

    const doCompute = async () => {
      const data = rawFrameDataRef.current!;
      let fftW = width;
      let fftH = height;
      let inputData = data;

      // ROI crop: extract bounding box and optionally zero-mask outside radius
      let origCropW = 0, origCropH = 0;
      if (roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length) {
        const roi = roiList[roiSelectedIdx];
        const crop = cropROIRegion(data, width, height, roi);
        if (crop) {
          origCropW = crop.cropW;
          origCropH = crop.cropH;
          // Apply Hann window to crop at native dimensions BEFORE zero-padding
          if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
          // Pad to next power-of-2 so fft2d doesn't truncate frequency data
          const padW = nextPow2(crop.cropW);
          const padH = nextPow2(crop.cropH);
          const padded = new Float32Array(padW * padH);
          for (let y = 0; y < crop.cropH; y++) {
            for (let x = 0; x < crop.cropW; x++) {
              padded[y * padW + x] = crop.cropped[y * crop.cropW + x];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      // Pre-pad non-power-of-2 full images so fft2d doesn't truncate frequency data
      if (origCropW === 0) {
        const padW = nextPow2(fftW);
        const padH = nextPow2(fftH);
        if (padW !== fftW || padH !== fftH) {
          const padded = new Float32Array(padW * padH);
          for (let y = 0; y < fftH; y++) {
            for (let x = 0; x < fftW; x++) {
              padded[y * padW + x] = inputData[y * fftW + x];
            }
          }
          inputData = padded;
          fftW = padW;
          fftH = padH;
        }
      }

      let real: Float32Array, imag: Float32Array;

      if (gpuReady && gpuFFTRef.current) {
        const gpuReal = inputData.slice();
        const gpuImag = new Float32Array(inputData.length);
        const result = await gpuFFTRef.current.fft2D(gpuReal, gpuImag, fftW, fftH, false);
        real = result.real;
        imag = result.imag;
      } else {
        real = inputData.slice();
        imag = new Float32Array(inputData.length);
        fft2d(real, imag, fftW, fftH, false);
      }

      if (cancelled) return;
      fftshift(real, fftW, fftH);
      fftshift(imag, fftW, fftH);

      fftMagRef.current = computeMagnitude(real, imag);
      fftMagCacheRef.current = fftMagRef.current;
      // Track FFT dimensions when they differ from image dimensions (ROI crop or non-pow2 padding)
      if (origCropW > 0) {
        setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
      } else if (fftW !== width || fftH !== height) {
        setFftCropDims({ cropWidth: width, cropHeight: height, fftWidth: fftW, fftHeight: fftH });
      } else {
        setFftCropDims(null);
      }
      setFftMagVersion(v => v + 1);
    };

    doCompute();

    return () => { cancelled = true; };
  }, [effectiveShowFft, frameBytes, displaySliceIdx, width, height, gpuReady, roiFftActive, roiList, roiSelectedIdx, fftWindow]);

  // Clear FFT measurement when ROI FFT state changes
  React.useEffect(() => { setFftClickInfo(null); }, [roiFftActive, roiSelectedIdx]);

  // Process FFT magnitude → histogram + colormap rendering (cheap, sync)
  React.useEffect(() => {
    const mag = fftMagRef.current;
    if (!effectiveShowFft || !mag) return;

    // Use crop dimensions when ROI FFT is active
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;

    let displayMin: number, displayMax: number;
    if (fftAuto) {
      ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, fftW, fftH));
    } else {
      ({ min: displayMin, max: displayMax } = findDataRange(mag));
    }

    const displayData = fftLogScale ? applyLogScale(mag) : mag;
    if (fftLogScale) {
      displayMin = Math.log1p(displayMin);
      displayMax = Math.log1p(displayMax);
    }

    setFftHistogramData(displayData);
    setFftDataRange({ min: displayMin, max: displayMax });
    setFftStats(computeStats(displayData));

    const { vmin, vmax } = sliderRange(displayMin, displayMax, fftVminPct, fftVmaxPct);
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const offscreen = renderToOffscreen(displayData, fftW, fftH, lut, vmin, vmax);
    if (!offscreen) return;

    fftOffscreenRef.current = offscreen;

    if (fftCanvasRef.current) {
      const ctx = fftCanvasRef.current.getContext("2d");
      if (ctx) {
        // Use bilinear smoothing when FFT is smaller than canvas (avoids blocky upscaling)
        ctx.imageSmoothingEnabled = fftW < canvasW || fftH < canvasH;
        ctx.clearRect(0, 0, canvasW, canvasH);
        ctx.save();
        ctx.translate(fftPanX, fftPanY);
        ctx.scale(fftZoom, fftZoom);
        // Stretch cropped FFT to fill the full canvas
        ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
        ctx.restore();
      }
    }
  }, [effectiveShowFft, fftMagVersion, fftLogScale, fftAuto, fftVminPct, fftVmaxPct, fftColormap, width, height, canvasW, canvasH, fftCropDims]);

  // Redraw cached FFT with zoom/pan (cheap - no recomputation)
  React.useEffect(() => {
    if (!effectiveShowFft || !fftCanvasRef.current || !fftOffscreenRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const offW = fftOffscreenRef.current.width;
    const offH = fftOffscreenRef.current.height;
    // Use bilinear smoothing when FFT is smaller than canvas (avoids blocky upscaling)
    ctx.imageSmoothingEnabled = offW < canvasW || offH < canvasH;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(fftOffscreenRef.current, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [effectiveShowFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH]);

  // Render FFT overlay (reciprocal-space scale bar + colorbar)
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    overlay.width = Math.round(canvasW * DPR);
    overlay.height = Math.round(canvasH * DPR);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Use crop dimensions for reciprocal-space calculations
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;

    // Reciprocal-space scale bar (pixelSize is in Å)
    if (pixelSize > 0) {
      const fftPixelSize = 1 / (fftW * pixelSize);
      drawFFTScaleBarHiDPI(overlay, DPR, fftZoom, fftPixelSize, fftW, `${pixelUnit || "px"}⁻¹`);
    }

    // FFT colorbar
    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftLogScale);
      ctx.restore();
    }

    // D-spacing crosshair marker - use crop dims for coordinate mapping
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const screenX = fftPanX + fftZoom * (fftClickInfo.col / fftW * canvasW);
      const screenY = fftPanY + fftZoom * (fftClickInfo.row / fftH * canvasH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
      ctx.shadowBlur = 2;
      ctx.lineWidth = 1.5;
      const r = 8;
      ctx.beginPath();
      ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - 3, screenY);
      ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + r, screenY);
      ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - 3);
      ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + r);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
      ctx.stroke();
      if (fftClickInfo.dSpacing != null) {
        const d = fftClickInfo.dSpacing;
        const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} Å`;
        ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        ctx.fillStyle = "white";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(label, screenX + 10, screenY - 4);
      }
      ctx.restore();
    }
  }, [effectiveShowFft, fftZoom, fftPanX, fftPanY, canvasW, canvasH, pixelSize, width, height, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftLogScale, fftShowColorbar, fftClickInfo, fftCropDims]);

  // -------------------------------------------------------------------------
  // Preview panel - cache colormapped offscreen (only recomputes when ROI
  // geometry, data, or display settings change - NOT on zoom/pan)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!previewVisible || !rawFrameDataRef.current) {
      previewOffscreenRef.current = null;
      return;
    }

    const raw = rawFrameDataRef.current;
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;

    const roi = roiList[roiSelectedIdx];
    const crop = cropROIRegion(raw, width, height, roi);
    if (!crop) {
      previewOffscreenRef.current = null;
      setPreviewCropDims(null);
      setPreviewVersion(v => v + 1);
      return;
    }

    setPreviewCropDims({ w: crop.cropW, h: crop.cropH });

    const processed = logScale ? applyLogScale(crop.cropped) : crop.cropped;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    if (traitVmin != null || traitVmax != null) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (imageDataRange.min !== imageDataRange.max && (imageVminPct > 0 || imageVmaxPct < 100)) {
      const mainProcessed = logScale ? applyLogScale(raw) : raw;
      const mainRange = findDataRange(mainProcessed);
      ({ vmin, vmax } = sliderRange(mainRange.min, mainRange.max, imageVminPct, imageVmaxPct));
    } else if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else {
      const r = findDataRange(processed);
      vmin = r.min;
      vmax = r.max;
    }

    const offscreen = renderToOffscreen(processed, crop.cropW, crop.cropH, lut, vmin, vmax);
    previewOffscreenRef.current = offscreen;
    setPreviewVersion(v => v + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewVisible, selectedRoiKey, cmap, logScale, autoContrast, imageVminPct, imageVmaxPct, imageDataRange, dataMin, dataMax, traitVmin, traitVmax, percentileLow, percentileHigh, width, height, frameBytes, displaySliceIdx]);

  // -------------------------------------------------------------------------
  // Preview panel - compute aspect-ratio-aware canvas dimensions
  // -------------------------------------------------------------------------
  const previewCanvasDims = React.useMemo(() => {
    if (!previewCropDims) return { w: canvasW, h: canvasH };
    const { w: cropW, h: cropH } = previewCropDims;
    const aspect = cropW / cropH;
    if (aspect >= 1) {
      return { w: canvasW, h: Math.max(20, Math.round(canvasW / aspect)) };
    } else {
      return { w: Math.max(20, Math.round(canvasH * aspect)), h: canvasH };
    }
  }, [previewCropDims, canvasW, canvasH]);

  // -------------------------------------------------------------------------
  // Preview panel - draw cached offscreen with zoom/pan (fast, no recompute)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !previewVisible) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pw = previewCanvasDims.w;
    const ph = previewCanvasDims.h;
    const offscreen = previewOffscreenRef.current;
    if (!offscreen || !previewCropDims) {
      ctx.clearRect(0, 0, pw, ph);
      return;
    }

    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, pw, ph);

    const { zoom: pz, panX: ppX, panY: ppY } = previewZoom;
    if (pz !== 1 || ppX !== 0 || ppY !== 0) {
      ctx.save();
      const cx = pw / 2;
      const cy = ph / 2;
      ctx.translate(cx + ppX, cy + ppY);
      ctx.scale(pz, pz);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, pw, ph);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, previewCropDims.w, previewCropDims.h, 0, 0, pw, ph);
    }
  }, [previewVisible, previewVersion, previewZoom, previewCanvasDims, previewCropDims]);

  // Preview overlay - scale bar + zoom indicator
  React.useEffect(() => {
    const overlay = previewOverlayRef.current;
    if (!overlay || !previewVisible) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (previewCropDims && pixelSize > 0) {
      const unit = "Å" as const;
      drawScaleBarHiDPI(overlay, DPR, previewZoom.zoom, pixelSize, unit, previewCropDims.w);
    }
  }, [previewVisible, previewZoom, previewCropDims, previewCanvasDims, pixelSize]);

  // Mouse handlers
  const panelIdxFromXY = (cssX: number, cssY: number): number => {
    const n = Math.max(1, nPanels || 1);
    if (n === 1) return 0;
    const cols = (maxCols && maxCols > 0) ? Math.min(maxCols, n) : n;
    const rows = Math.ceil(n / cols);
    const slotW = canvasW / cols;
    const slotH = canvasH / rows;
    const col = Math.floor(cssX / Math.max(1, slotW));
    const row = Math.floor(cssY / Math.max(1, slotH));
    if (col < 0 || col >= cols || row < 0 || row >= rows) return -1;
    const idx = row * cols + col;
    // Empty grid cells past N panels (partial last row) are not panels.
    return idx >= n ? -1 : idx;
  };
  const panelIdxFromEvent = (e: React.MouseEvent): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const cssX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cssY = (e.clientY - rect.top) * (canvas.height / rect.height);
    return panelIdxFromXY(cssX, cssY);
  };
  const beginPan = (e: React.MouseEvent) => {
    const idx = panelIdxFromEvent(e);
    if (idx < 0) return;
    panStartPanelRef.current = idx;
    const s = stateFor(idx);
    setIsDraggingPan(true);
    setPanStart({ x: e.clientX, y: e.clientY, pX: s.panX, pY: s.panY });
  };
  const handleWheel = (e: React.WheelEvent) => {
    if (lockView) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const panelIdx = panelIdxFromXY(mouseX, mouseY);
    if (panelIdx < 0) return;
    const cur = stateFor(panelIdx);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, cur.zoom * zoomFactor));
    const zoomRatio = newZoom / cur.zoom;
    // Mouse position relative to this panel's slot (so zoom anchors to cursor within slot)
    const n = Math.max(1, nPanels || 1);
    const cols = (maxCols && maxCols > 0) ? Math.min(maxCols, n) : n;
    const rows = Math.ceil(n / cols);
    const slotW = canvasW / cols;
    const slotH = canvasH / rows;
    const col = panelIdx % cols;
    const row = Math.floor(panelIdx / cols);
    const localX = mouseX - col * slotW;
    const localY = mouseY - row * slotH;
    const newPanX = localX - (localX - cur.panX) * zoomRatio;
    const newPanY = localY - (localY - cur.panY) * zoomRatio;
    // Combine into one update per state slot to avoid stale linkedState when
    // only one of (zoom, pan) is linked.
    if (linkZoom || linkPan) {
      setLinkedState(s => ({
        zoom: linkZoom ? newZoom : s.zoom,
        panX: linkPan ? newPanX : s.panX,
        panY: linkPan ? newPanY : s.panY,
      }));
    }
    if (!linkZoom || !linkPan) {
      setPanelStates(arr => arr.map((s, i) => i === panelIdx ? {
        zoom: linkZoom ? s.zoom : newZoom,
        panX: linkPan ? s.panX : newPanX,
        panY: linkPan ? s.panY : newPanY,
      } : s));
    }
  };

  const handleDoubleClick = () => {
    if (lockView) return;
    setLinkedState({ zoom: 1, panX: 0, panY: 0 });
    setPanelStates(arr => arr.map(() => ({ zoom: 1, panX: 0, panY: 0 })));
  };

  const addROIAt = React.useCallback((row: number, col: number, shape: "circle" | "square" | "rectangle" | "annular" = newRoiShape) => {
    if (lockRoi) return;
    const clampedRow = Math.max(0, Math.min(height - 1, Math.round(row)));
    const clampedCol = Math.max(0, Math.min(width - 1, Math.round(col)));
    const next = [...roiItems, createROI(clampedRow, clampedCol, shape, roiItems.length, width, height)];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
    setShowRoiResizeHint(true);
  }, [height, width, newRoiShape, roiItems, setRoiList, setRoiSelectedIdx, lockRoi]);

  const deleteSelectedROI = React.useCallback(() => {
    if (lockRoi) return;
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return;
    const next = roiList.filter((_, i) => i !== roiSelectedIdx);
    setRoiList(next);
    setRoiSelectedIdx(next.length > 0 ? Math.min(roiSelectedIdx, next.length - 1) : -1);
  }, [roiList, roiSelectedIdx, setRoiList, setRoiSelectedIdx, lockRoi]);

  const duplicateSelectedROI = React.useCallback(() => {
    if (lockRoi) return;
    if (!selectedRoi) return;
    const duplicated: ROIItem = {
      ...selectedRoi,
      row: Math.max(0, Math.min(height - 1, Math.round(selectedRoi.row + 3))),
      col: Math.max(0, Math.min(width - 1, Math.round(selectedRoi.col + 3))),
      shape: selectedRoi.shape,
      radius: selectedRoi.radius,
      radius_inner: selectedRoi.radius_inner,
      width: selectedRoi.width,
      height: selectedRoi.height,
      color: ROI_COLORS[roiItems.length % ROI_COLORS.length],
      line_width: selectedRoi.line_width,
      highlight: false,
    };
    const next = [...roiItems, duplicated];
    setRoiList(next);
    setRoiSelectedIdx(next.length - 1);
  }, [selectedRoi, height, width, roiItems, setRoiList, setRoiSelectedIdx, lockRoi]);


  const handleExportPng = async () => {
    if (lockExport) return;
    setExportAnchor(null);
    if (!canvasRef.current) return;
    const blob = await new Promise<Blob>((resolve) =>
      canvasRef.current!.toBlob((b) => resolve(b!), "image/png"));
    const label = labels?.[sliceIdx] || String(sliceIdx);
    downloadBlob(blob, "show3d_frame_" + label + ".png");
  };

  const handleExportPngAll = () => {
    if (lockExport) return;
    setExportAnchor(null);
    setExporting(true);
    setZipExportRequested(true);
  };

  const handleExportGif = () => {
    if (lockExport) return;
    setExportAnchor(null);
    setExporting(true);
    setGifExportRequested(true);
  };

  const handleExportBundle = () => {
    if (lockExport) return;
    setExportAnchor(null);
    setExporting(true);
    setBundleExportRequested(true);
  };

  const handleCopy = async () => {
    if (lockExport) return;
    if (!canvasRef.current) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvasRef.current!.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      // Fallback: download if clipboard API unavailable
      canvasRef.current.toBlob((b) => {
        if (b) {
          const label = labels?.[sliceIdx] || String(sliceIdx);
          downloadBlob(b, "show3d_frame_" + label + ".png");
        }
      }, "image/png");
    }
  };

  // Export publication-quality figure
  const handleExportFigure = (withColorbar: boolean) => {
    if (lockExport) return;
    setExportAnchor(null);
    const frameData = rawFrameDataRef.current;
    if (!frameData) return;

    const processed = logScale ? applyLogScale(frameData) : frameData;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    if (traitVmin != null || traitVmax != null) {
      ({ vmin, vmax } = resolveDisplayRange(
        dataMin,
        dataMax,
        traitVmin,
        traitVmax,
        logScale,
        imageVminPct,
        imageVmaxPct,
      ));
    } else if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, percentileLow, percentileHigh));
    } else {
      const { min: pMin, max: pMax } = findDataRange(processed);
      ({ vmin, vmax } = sliderRange(pMin, pMax, imageVminPct, imageVmaxPct));
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    // pixelSize is in Å

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      pixelUnit: pixelSize > 0 ? (pixelUnit || "Å") : "px",
      showColorbar: withColorbar,
      showScaleBar: pixelSize > 0,
      drawAnnotations: (ctx) => {
        if (!hideRoi && roiActive && roiItems.length > 0) {
          for (let i = 0; i < roiItems.length; i++) {
            const roi = roiItems[i];
            const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
            const color = roi.color || ROI_COLORS[i % ROI_COLORS.length];
            ctx.lineWidth = roi.line_width || 2;
            drawROI(ctx, roi.col, roi.row, shape, roi.radius, roi.width, roi.height, color, color, false, roi.radius_inner);
          }
        }
      },
    });

    const label = labels?.[sliceIdx] || String(sliceIdx);
    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, `show3d_figure_${label}.pdf`));
  };

  // Download GIF when data arrives from Python
  React.useEffect(() => {
    if (!gifData || gifData.byteLength === 0) return;
    downloadDataView(gifData, "show3d_animation.gif", "image/gif");
    const metaText = (gifMetadataJson || "").trim();
    if (metaText) {
      downloadBlob(new Blob([metaText], { type: "application/json" }), "show3d_animation.json");
    }
    setExporting(false);
  }, [gifData, gifMetadataJson]);

  // Download ZIP when data arrives from Python
  React.useEffect(() => {
    if (!zipData || zipData.byteLength === 0) return;
    downloadDataView(zipData, "show3d_frames.zip", "application/zip");
    setExporting(false);
  }, [zipData]);

  // Download export bundle when data arrives from Python
  React.useEffect(() => {
    if (!bundleData || bundleData.byteLength === 0) return;
    downloadDataView(bundleData, "show3d_bundle.zip", "application/zip");
    setExporting(false);
  }, [bundleData]);

  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  const screenToImg = (e: React.MouseEvent): { imgCol: number; imgRow: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { imgCol: 0, imgRow: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const screenX = (e.clientX - rect.left) * scaleX;
    const screenY = (e.clientY - rect.top) * scaleY;
    // Undo pan + zoom first.
    let col = (screenX - panX) / (displayScale * zoom);
    let row = (screenY - panY) / (displayScale * zoom);
    // Undo image_rotation. Image rotates around its own center (w/2, h/2)
    // in image-space. To invert, rotate the (col, row) by -rotation around
    // the center of the visible rotated bbox: which is (rotW/2, rotH/2)
    // where rotW = (rotSwap ? height : width), rotH swapped accordingly.
    const r = (((imageRotation % 4) + 4) % 4) | 0;
    if (r !== 0) {
      const rotSwap = (r % 2) !== 0;
      const visW = rotSwap ? height : width;
      const visH = rotSwap ? width : height;
      // Translate to bbox center
      const cx = col - visW / 2;
      const cy = row - visH / 2;
      // Inverse rotation: r quarter-turns CW means undo with -r CW = r CCW
      let ux: number, uy: number;
      if (r === 1) { ux = cy; uy = -cx; }              // CCW 90°
      else if (r === 2) { ux = -cx; uy = -cy; }        // 180°
      else { ux = -cy; uy = cx; }                       // CW 90° (undo 270°)
      // Translate to source image center
      col = ux + width / 2;
      row = uy + height / 2;
    }
    return { imgCol: col, imgRow: row };
  };

  const hitTestROI = React.useCallback((imgCol: number, imgRow: number): number => {
    if (!roiActive || roiItems.length === 0) return -1;
    for (let ri = roiItems.length - 1; ri >= 0; ri--) {
      const roi = roiItems[ri];
      const shape = roi.shape || "circle";
      if (shape === "circle" || shape === "annular") {
        if (Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2) <= roi.radius) return ri;
      } else if (shape === "square") {
        if (Math.abs(imgCol - roi.col) <= roi.radius && Math.abs(imgRow - roi.row) <= roi.radius) return ri;
      } else if (shape === "rectangle") {
        if (Math.abs(imgCol - roi.col) <= roi.width / 2 && Math.abs(imgRow - roi.row) <= roi.height / 2) return ri;
      }
    }
    return -1;
  }, [roiActive, roiItems]);

  const getHitArea = React.useCallback(() => RESIZE_HIT_AREA_PX / (displayScale * zoom), [displayScale, zoom]);

  const isNearEdge = React.useCallback((imgCol: number, imgRow: number, roi: ROIItem): boolean => {
    const hitArea = getHitArea();
    const shape = roi.shape || "circle";
    if (shape === "circle" || shape === "annular") {
      const dist = Math.sqrt((imgCol - roi.col) ** 2 + (imgRow - roi.row) ** 2);
      return Math.abs(dist - roi.radius) < hitArea;
    }
    if (shape === "square") {
      const dx = Math.abs(imgCol - roi.col);
      const dy = Math.abs(imgRow - roi.row);
      const r = roi.radius;
      return (dx <= r + hitArea && dy <= r + hitArea) && (Math.abs(dx - r) < hitArea || Math.abs(dy - r) < hitArea);
    }
    if (shape === "rectangle") {
      const dx = Math.abs(imgCol - roi.col);
      const dy = Math.abs(imgRow - roi.row);
      const hw = roi.width / 2;
      const hh = roi.height / 2;
      return (dx <= hw + hitArea && dy <= hh + hitArea) && (Math.abs(dx - hw) < hitArea || Math.abs(dy - hh) < hitArea);
    }
    return false;
  }, [getHitArea]);

  const isNearResizeHandle = React.useCallback((imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !selectedRoi) return false;
    return isNearEdge(imgCol, imgRow, selectedRoi);
  }, [roiActive, selectedRoi, isNearEdge]);

  const isNearAnyEdge = React.useCallback((imgCol: number, imgRow: number): boolean => {
    if (!roiActive || roiItems.length === 0) return false;
    return roiItems.some(roi => isNearEdge(imgCol, imgRow, roi));
  }, [roiActive, roiItems, isNearEdge]);

  const isNearResizeHandleInner = React.useCallback((imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !selectedRoi || selectedRoi.shape !== "annular") return false;
    const hitArea = getHitArea();
    const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
    return Math.abs(dist - selectedRoi.radius_inner) < hitArea;
  }, [roiActive, selectedRoi, getHitArea]);

  const updateROI = (e: React.MouseEvent) => {
    if (!selectedRoi) return;
    const { imgCol, imgRow } = screenToImg(e);
    updateSelectedRoi({
      col: Math.max(0, Math.min(width - 1, Math.floor(imgCol))),
      row: Math.max(0, Math.min(height - 1, Math.floor(imgRow))),
    });
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    // Ignore clicks in empty grid cells (partial last row when N isn't a
    // multiple of max_cols). Otherwise the click attributes to the last
    // real panel and zoom/pan jumps unexpectedly.
    if (panelIdxFromEvent(e) < 0) return;
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    pendingRoiAddRef.current = null;
    // Check if clicking on lens inset for drag or resize
    if (showLens && !lockDisplay) {
      const rect = canvasContainerRef.current?.getBoundingClientRect();
      if (rect) {
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        if (cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize) {
          const edgeHit = 8;
          const nearEdge = cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit ||
                           cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit;
          if (nearEdge) {
            setIsResizingLens(true);
            lensResizeStartRef.current = { my: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mx: e.clientX, my: e.clientY, ax: lx, ay: ly };
          }
          return;
        }
      }
    }
    if (profileActive) {
      if (lockProfile) {
        if (!lockView) {
          beginPan(e);
        }
        return;
      }
      const { imgCol, imgRow } = screenToImg(e);
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / (displayScale * zoom);
        const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingPan(false);
          setPanStart(null);
          return;
        }
        if (pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius) {
          setIsDraggingProfileLine(true);
          profileDragStartRef.current = {
            row: imgRow,
            col: imgCol,
            p0: { row: p0.row, col: p0.col },
            p1: { row: p1.row, col: p1.col },
          };
          setIsDraggingPan(false);
          setPanStart(null);
          return;
        }
      }
      if (!lockView) {
        beginPan(e);
      }
      return;
    }
    if (roiActive) {
      if (lockRoi) {
        if (!lockView) {
          beginPan(e);
        }
        return;
      }
      const { imgCol, imgRow } = screenToImg(e);
      if (isNearResizeHandleInner(imgCol, imgRow)) {
        setIsDraggingResizeInner(true);
        return;
      }
      if (isNearResizeHandle(imgCol, imgRow)) {
        e.preventDefault();
        resizeAspectRef.current = selectedRoi && (selectedRoi.shape === "rectangle") && selectedRoi.width > 0 && selectedRoi.height > 0 ? selectedRoi.width / selectedRoi.height : null;
        setIsDraggingResize(true);
        return;
      }
      if (roiItems.length > 0) {
        for (let ri = roiItems.length - 1; ri >= 0; ri--) {
          const roi = roiItems[ri];
          if (isNearEdge(imgCol, imgRow, roi)) {
            e.preventDefault();
            resizeAspectRef.current = roi && (roi.shape === "rectangle") && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(ri);
            setIsDraggingResize(true);
            return;
          }
        }
      }
      const hitIdx = hitTestROI(imgCol, imgRow);
      if (hitIdx >= 0) {
        setRoiSelectedIdx(hitIdx);
        setIsDraggingROI(true);
        return;
      }
      setRoiSelectedIdx(-1);
      pendingRoiAddRef.current = {
        row: Math.max(0, Math.min(height - 1, Math.round(imgRow))),
        col: Math.max(0, Math.min(width - 1, Math.round(imgCol))),
      };
      return;
    }
    if (!lockView) {
      beginPan(e);
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    // Fast path: during pan drag, skip all cursor/hover/lens work - just update pan
    if (isDraggingPan && panStart && !lockView) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      const newPanX = panStart.pX + dx;
      const newPanY = panStart.pY + dy;
      if (linkPan) {
        setLinkedState(s => ({ ...s, panX: newPanX, panY: newPanY }));
      } else {
        const idx = panStartPanelRef.current;
        setPanelStates(arr => arr.map((s, i) => i === idx ? { ...s, panX: newPanX, panY: newPanY } : s));
      }
      return;
    }

    // Cursor readout: convert screen position to image pixel coordinates.
    // Skip when hovering an empty grid cell (partial last row when nPanels
    // isn't a multiple of max_cols) so dead space doesn't flash row/col
    // numbers from a phantom panel.
    const canvas = canvasRef.current;
    const hoverPanelIdx = panelIdxFromEvent(e);
    if (hoverPanelIdx < 0) {
      setCursorInfo(null);
      if (showLens && !lockDisplay) setLensPos(null);
    } else if (canvas && rawFrameDataRef.current) {
      const { imgRow, imgCol } = screenToImg(e);
      const imgX = Math.floor(imgCol);
      const imgY = Math.floor(imgRow);
      if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
        const rawData = rawFrameDataRef.current;
        setCursorInfo({ row: imgY, col: imgX, value: rawData[imgY * width + imgX] });
        if (showLens && !lockDisplay) setLensPos({ row: imgY, col: imgX });
      } else {
        setCursorInfo(null);
        if (showLens && !lockDisplay) setLensPos(null);
      }
    }

    // Lens edge hover detection
    if (showLens && !lockDisplay) {
      const rect2 = canvasContainerRef.current?.getBoundingClientRect();
      if (rect2) {
        const cssX2 = e.clientX - rect2.left;
        const cssY2 = e.clientY - rect2.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        const inside = cssX2 >= lx && cssX2 <= lx + lensDisplaySize && cssY2 >= ly && cssY2 <= ly + lensDisplaySize;
        const edgeHit = 8;
        const nearEdge = inside && (cssX2 - lx < edgeHit || lx + lensDisplaySize - cssX2 < edgeHit ||
                                     cssY2 - ly < edgeHit || ly + lensDisplaySize - cssY2 < edgeHit);
        setIsHoveringLensEdge(nearEdge);
      }
    } else {
      setIsHoveringLensEdge(false);
    }

    // Lens drag
    if (!lockDisplay && isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mx;
      const dy = e.clientY - lensDragStartRef.current.my;
      setLensAnchor({ x: lensDragStartRef.current.ax + dx, y: lensDragStartRef.current.ay + dy });
      return;
    }

    // Lens resize drag
    if (!lockDisplay && isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.my;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    if (profileActive && !lockProfile && profilePoints.length === 2) {
      const { imgCol, imgRow } = screenToImg(e);
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const hitRadius = 10 / (displayScale * zoom);
      const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
      const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
      if (draggingProfileEndpoint !== null) {
        if (!rawFrameDataRef.current) return;
        const clampedRow = Math.max(0, Math.min(height - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(width - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawFrameDataRef.current, width, height, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
        return;
      }
      if (isDraggingProfileLine && profileDragStartRef.current) {
        if (!rawFrameDataRef.current) return;
        const drag = profileDragStartRef.current;
        let deltaRow = imgRow - drag.row;
        let deltaCol = imgCol - drag.col;
        const minRow = Math.min(drag.p0.row, drag.p1.row);
        const maxRow = Math.max(drag.p0.row, drag.p1.row);
        const minCol = Math.min(drag.p0.col, drag.p1.col);
        const maxCol = Math.max(drag.p0.col, drag.p1.col);
        deltaRow = Math.max(deltaRow, -minRow);
        deltaRow = Math.min(deltaRow, (height - 1) - maxRow);
        deltaCol = Math.max(deltaCol, -minCol);
        deltaCol = Math.min(deltaCol, (width - 1) - maxCol);
        const next = [
          { row: drag.p0.row + deltaRow, col: drag.p0.col + deltaCol },
          { row: drag.p1.row + deltaRow, col: drag.p1.col + deltaCol },
        ];
        setProfileLine(next);
        setProfileData(sampleLineProfile(rawFrameDataRef.current, width, height, next[0].row, next[0].col, next[1].row, next[1].col, profileWidth));
        return;
      }
      const nextHoveredEndpoint: 0 | 1 | null = d0 <= hitRadius ? 0 : d1 <= hitRadius ? 1 : null;
      const nextHoverLine = nextHoveredEndpoint === null && pointToSegmentDistance(imgCol, imgRow, p0.col, p0.row, p1.col, p1.row) <= hitRadius;
      setHoveredProfileEndpoint(nextHoveredEndpoint);
      setIsHoveringProfileLine(nextHoverLine);
    } else {
      if (hoveredProfileEndpoint !== null) setHoveredProfileEndpoint(null);
      if (isHoveringProfileLine) setIsHoveringProfileLine(false);
    }

    // Resize handle dragging
    if (isDraggingResizeInner && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const newR = Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
      updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newR))) });
      setShowRoiResizeHint(false);
      return;
    }
    if (isDraggingResize && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const shape = selectedRoi.shape || "circle";
      if (shape === "rectangle") {
        let newW = Math.max(2, Math.round(Math.abs(ic - selectedRoi.col) * 2));
        let newH = Math.max(2, Math.round(Math.abs(ir - selectedRoi.row) * 2));
        if (e.shiftKey && resizeAspectRef.current != null) {
          const aspect = resizeAspectRef.current;
          if (newW / newH > aspect) newH = Math.max(2, Math.round(newW / aspect));
          else newW = Math.max(2, Math.round(newH * aspect));
        }
        updateSelectedRoi({ width: newW, height: newH });
      } else {
        const newR = shape === "square"
          ? Math.max(Math.abs(ic - selectedRoi.col), Math.abs(ir - selectedRoi.row))
          : Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
        const minR = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
        updateSelectedRoi({ radius: Math.max(minR, Math.round(newR)) });
      }
      setShowRoiResizeHint(false);
      return;
    }

    // Hover state for resize handles
    if (roiActive && !lockRoi && !isDraggingROI && !isDraggingPan) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e);
      const hoveringInner = isNearResizeHandleInner(ic, ir);
      const hoveringOuter = isNearAnyEdge(ic, ir);
      setIsHoveringResizeInner(hoveringInner);
      setIsHoveringResize(hoveringOuter);
      if (hoveringInner || hoveringOuter) setShowRoiResizeHint(false);
    }

    if (!lockRoi && isDraggingROI) {
      updateROI(e);
    }
  };

  const handleCanvasMouseUp = (e: React.MouseEvent) => {
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      clickStartRef.current = null;
      pendingRoiAddRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsDraggingLens(false);
      lensDragStartRef.current = null;
      setIsResizingLens(false);
      lensResizeStartRef.current = null;
      setIsDraggingPan(false);
      setPanStart(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }

    // Profile click capture
    if (profileActive && !lockProfile && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = canvasRef.current;
        if (canvas && rawFrameDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const imgX = (mouseCanvasX - panX) / zoom / displayScale;
          const imgY = (mouseCanvasY - panY) / zoom / displayScale;
          if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
            const pt = { row: imgY, col: imgX };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              setProfileLine([pt]);
              setProfileData(null);
            } else {
              const p0 = profilePoints[0];
              setProfileLine([p0, pt]);
              setProfileData(sampleLineProfile(rawFrameDataRef.current, width, height, p0.row, p0.col, pt.row, pt.col, profileWidth));
            }
          }
        }
      }
    }

    // ROI click-to-add (empty-area click)
    if (roiActive && !lockRoi && pendingRoiAddRef.current && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        addROIAt(pendingRoiAddRef.current.row, pendingRoiAddRef.current.col);
      }
    }
    clickStartRef.current = null;
    pendingRoiAddRef.current = null;
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsDraggingPan(false);
    setPanStart(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  const handleCanvasMouseLeave = () => {
    setCursorInfo(null);
    // Lens persists at last position when cursor exits main canvas. Wiping on every
    // leave kills the inset whenever the user touches a slider, FFT panel, or any
    // sibling control - surprising "lens vanished" footgun. User explicitly turns
    // lens off via the Lens switch.
    pendingRoiAddRef.current = null;
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingLens(false);
    lensDragStartRef.current = null;
    setIsResizingLens(false);
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    setIsDraggingPan(false);
    setPanStart(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
  };

  // FFT mouse handlers
  const [isFftDragging, setIsFftDragging] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  const handleFftWheel = (e: React.WheelEvent) => {
    if (lockView) return;
    const canvas = fftCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor));
    const zoomRatio = newZoom / fftZoom;
    setFftZoom(newZoom);
    setFftPanX(mouseX - (mouseX - fftPanX) * zoomRatio);
    setFftPanY(mouseY - (mouseY - fftPanY) * zoomRatio);
  };

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const imgCol = ((mouseX - fftPanX) / fftZoom) / canvasW * fftW;
    const imgRow = ((mouseY - fftPanY) / fftZoom) / canvasH * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    if (lockView) return;
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsFftDragging(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (lockView) return;
    if (isFftDragging && fftPanStart) {
      const canvas = fftCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - fftPanStart.x) * scaleX;
      const dy = (e.clientY - fftPanStart.y) * scaleY;
      setFftPanX(fftPanStart.pX + dx);
      setFftPanY(fftPanStart.pY + dy);
    }
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
    if (lockView) {
      fftClickStartRef.current = null;
      setIsFftDragging(false);
      setFftPanStart(null);
      return;
    }
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const pos = fftScreenToImg(e);
        if (pos) {
          // Use crop dimensions when ROI FFT is active
          const fftW = fftCropDims?.fftWidth ?? width;
          const fftH = fftCropDims?.fftHeight ?? height;
          let imgCol = pos.col;
          let imgRow = pos.row;
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
            setFftClickInfo(null);
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
      fftClickStartRef.current = null;
    }
    setIsFftDragging(false);
    setFftPanStart(null);
  };

  const handleFftReset = () => {
    if (lockView) return;
    setFftZoom(1);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  const fftNeedsReset = fftZoom !== 1 || fftPanX !== 0 || fftPanY !== 0;

  // Preview panel zoom/pan handlers
  const handlePreviewWheel = (e: React.WheelEvent) => {
    if (lockView) return;
    e.preventDefault();
    const canvas = previewCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const pw = previewCanvasDims.w;
    const ph = previewCanvasDims.h;
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = pw / 2;
    const cy = ph / 2;
    const mouseImageX = (mouseCanvasX - cx - previewZoom.panX) / previewZoom.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - previewZoom.panY) / previewZoom.zoom + cy;
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, previewZoom.zoom * zoomFactor));
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;
    setPreviewZoom({ zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handlePreviewMouseDown = (e: React.MouseEvent) => {
    if (lockView) return;
    setIsDraggingPreviewPan(true);
    setPreviewPanStart({ x: e.clientX, y: e.clientY, pX: previewZoom.panX, pY: previewZoom.panY });
  };

  const handlePreviewMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingPreviewPan || !previewPanStart) return;
    const dx = e.clientX - previewPanStart.x;
    const dy = e.clientY - previewPanStart.y;
    setPreviewZoom(prev => ({ ...prev, panX: previewPanStart.pX + dx, panY: previewPanStart.pY + dy }));
  };

  const handlePreviewMouseUp = () => {
    setIsDraggingPreviewPan(false);
    setPreviewPanStart(null);
  };

  const handlePreviewDoubleClick = () => {
    if (lockView) return;
    setPreviewZoom({ zoom: 1, panX: 0, panY: 0 });
  };

  // Resize handlers
  const handleMainResizeStart = (e: React.MouseEvent) => {
    if (lockView) return;
    e.stopPropagation();
    e.preventDefault();
    setIsResizingMain(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: mainCanvasSize });
  };

  React.useEffect(() => {
    if (!isResizingMain) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : mainCanvasSize;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(initialCanvasSizeRef.current, resizeStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          setMainCanvasSize(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      cancelAnimationFrame(rafId);
      setMainCanvasSize(latestSize);
      setIsResizingMain(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingMain, resizeStart]);

  // Keyboard
  const handleKeyDown = React.useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    if (shouldIgnoreWidgetShortcut(e.target)) return;

    let handled = false;

    switch (e.key) {
        case " ":
          if (!lockPlayback) {
            setPlaying(!playing);
            handled = true;
          }
          break;
        case "ArrowLeft":
          if (!lockPlayback) {
            const lo = loop ? Math.max(0, loopStart) : 0;
            const candidate = hiddenSet.size ? nextVisible(sliceIdx, -1, false) : sliceIdx - 1;
            setSliceIdx(Math.max(lo, candidate));
            handled = true;
          }
          break;
        case "ArrowRight":
          if (!lockPlayback) {
            const hi = loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1;
            const candidate = hiddenSet.size ? nextVisible(sliceIdx, 1, false) : sliceIdx + 1;
            setSliceIdx(Math.min(hi, candidate));
            handled = true;
          }
          break;
        case "Home":
          if (!lockPlayback) {
            setSliceIdx(loop ? Math.max(0, loopStart) : 0);
            handled = true;
          }
          break;
        case "End":
          if (!lockPlayback) {
            setSliceIdx(loop ? Math.min(effectiveLoopEnd, nSlices - 1) : nSlices - 1);
            handled = true;
          }
          break;
        case "r":
        case "R":
          if (!lockView) {
            handleDoubleClick();
            handled = true;
          }
          break;
        case "c":
        case "C":
          if (!lockExport && cursorInfo) {
            navigator.clipboard.writeText(`(${cursorInfo.row}, ${cursorInfo.col}, ${cursorInfo.value})`);
            handled = true;
          }
          break;
        case "Delete":
        case "Backspace":
          if (!lockRoi && roiActive && roiSelectedIdx >= 0) {
            deleteSelectedROI();
            handled = true;
          }
          break;
        case "d":
        case "D":
          if (!lockRoi && roiActive && roiSelectedIdx >= 0 && (e.metaKey || e.ctrlKey || e.shiftKey)) {
            duplicateSelectedROI();
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
    cursorInfo,
    deleteSelectedROI,
    duplicateSelectedROI,
    effectiveLoopEnd,
    effectiveShowFft,
    handleDoubleClick,
    lockExport,
    lockPlayback,
    lockRoi,
    lockView,
    loop,
    loopStart,
    nSlices,
    playing,
    roiActive,
    roiSelectedIdx,
    setPlaying,
    setSliceIdx,
    sliceIdx,
  ]);

  // Check if view needs reset
  const needsReset = zoom !== 1 || panX !== 0 || panY !== 0;

  return (
    <Box
      ref={rootRef}
      className="show3d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseDownCapture={handleRootMouseDownCapture}
      sx={{ ...container.root, bgcolor: themeColors.bg, color: themeColors.text, outline: "none", "&:focus": { outline: "2px solid #0af", outlineOffset: 2 }, "& canvas": { display: "block" } }}
    >
      <Stack direction="row" spacing={`${SPACING.SM}px`}>
        <Box>
          {/* Title row */}
          <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
            {title || "Image"}
            {diffMode !== "off" && (
              <Typography component="span" sx={{ fontSize: 9, fontWeight: "bold", color: "#fff", bgcolor: "#e65100", px: 0.5, py: 0.125, ml: 0.5, verticalAlign: "middle" }}>
                {diffMode === "previous" ? "\u0394-PREV" : "\u0394-FIRST"}
              </Typography>
            )}
            <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Scale: Linear or logarithmic intensity mapping.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (clips outliers). FFT Auto masks DC + clips to 99.9th.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Click empty image to add at cursor, click ROI to select, drag to move, hover edge to resize. Del removes selected; Ctrl/⌘+D duplicates.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop: Loop playback. Drag end markers on slider for loop range.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce: Ping-pong playback - alternates forward and reverse.</Typography>
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={[["Space", "Play / Pause"], ["← / →", `Prev / Next ${dimLabel.toLowerCase()}`], ["Home / End", `First / Last ${dimLabel.toLowerCase()}`], ["R", "Reset zoom"], ["C", "Copy cursor coords"], ["Del", "Delete selected ROI"], ["Ctrl/⌘+D", "Duplicate selected ROI"], ["Esc", "Release keyboard focus"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
            </Box>} theme={themeInfo.theme} />
          </Typography>
          {/* Controls row */}
          {(!hideDisplay || !hideProfile || !hideRoi || !hideExport || !hideView) && (
            <Box sx={{ display: "flex", alignItems: "center", gap: "4px", mb: `${SPACING.XS}px`, height: 28 }}>
              {!hideDisplay && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
                  <Switch checked={showFft} onChange={(e) => { if (!lockDisplay) setShowFft(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle FFT power spectrum panel" }} />
                </>
              )}
              {!hideProfile && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Profile:</Typography>
                  <Switch checked={profileActive} onChange={(e) => {
                    if (lockProfile) return;
                    const on = e.target.checked;
                    setProfileActive(on);
                    if (on) {
                      if (!lockRoi) { setRoiActive(false); setRoiSelectedIdx(-1); }
                    } else {
                      // Toggle OFF hides overlay (gated by `profileActive` in render) but keeps
                      // the line + sampled data so re-enable restores instantly. Use the Clear
                      // button below to actively wipe.
                      setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false);
                    }
                  }} disabled={lockProfile} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle line intensity profile tool" }} />
                  {profileActive && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10, ml: "4px" }}>W:</Typography>
                      <Slider value={profileWidth} min={1} max={15} step={1} onChange={(_, v) => { if (!lockProfile) setProfileWidth(v as number); }} disabled={lockProfile} size="small" valueLabelDisplay="auto" sx={{ width: 60, ml: "2px" }} aria-label={`Profile width ${profileWidth} px`} />
                    </>
                  )}
                </>
              )}
              {!hideDisplay && (nPanels || 1) === 1 && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Lens:</Typography>
                  <Switch
                    checked={showLens}
                    onChange={() => {
                      if (lockDisplay) return;
                      if (!showLens) { setShowLens(true); setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) }); }
                      else { setShowLens(false); setLensPos(null); }
                    }}
                    disabled={lockDisplay}
                    size="small"
                    sx={switchStyles.small}
                    inputProps={{ "aria-label": "Toggle magnifier lens" }}
                  />
                </>
              )}
              {!hideRoi && (nPanels || 1) === 1 && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>ROI:</Typography>
                  <Switch checked={roiActive} onChange={(e) => {
                    if (lockRoi) return;
                    const on = e.target.checked;
                    if (on) {
                      setRoiActive(true); setShowRoiResizeHint(true);
                      if (!lockProfile) { setProfileActive(false); setProfileLine([]); setProfileData(null); setHoveredProfileEndpoint(null); setIsHoveringProfileLine(false); }
                    } else {
                      setRoiActive(false); setRoiSelectedIdx(-1); pendingRoiAddRef.current = null;
                    }
                  }} disabled={lockRoi} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle ROI selection tool" }} />
                </>
              )}
              {(nPanels || 1) > 1 && (
                <>
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Link:</Typography>
                  <Switch checked={linkPanels} onChange={(e) => setLinkPanels(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Link zoom and pan across panels" }} />
                  <Typography sx={{ ...typography.label, fontSize: 10, ml: "2px" }}>Link contrast:</Typography>
                  <Switch checked={linkContrast} onChange={(e) => setLinkContrast(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Link contrast across panels" }} />
                </>
              )}
              <Box sx={{ flex: 1 }} />
              <Box sx={{ display: "flex", alignItems: "center", gap: "6px" }}>
                {!hideExport && (
                  <>
                    <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => { if (!lockExport) setExportAnchor(e.currentTarget); }} disabled={lockExport || exporting} aria-label="Open export menu (PDF, PNG, GIF, bundle)">{exporting ? "..." : "Export"}</Button>
                    <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                      <MenuItem disabled={lockExport} onClick={() => handleExportFigure(true)} sx={{ fontSize: 12 }}>PDF + colorbar</MenuItem>
                      <MenuItem disabled={lockExport} onClick={() => handleExportFigure(false)} sx={{ fontSize: 12 }}>PDF</MenuItem>
                      <MenuItem disabled={lockExport} onClick={handleExportBundle} sx={{ fontSize: 12 }}>Bundle (PNG + ROI CSV + state)</MenuItem>
                      <MenuItem disabled={lockExport} onClick={handleExportPng} sx={{ fontSize: 12 }}>PNG (current frame)</MenuItem>
                      <MenuItem disabled={lockExport} onClick={handleExportPngAll} sx={{ fontSize: 12 }}>PNG (all frames .zip)</MenuItem>
                      <MenuItem disabled={lockExport} onClick={handleExportGif} sx={{ fontSize: 12 }}>GIF (fps: {fps})</MenuItem>
                    </Menu>
                    <Button size="small" sx={compactButton} disabled={lockExport} onClick={handleCopy} aria-label="Copy current frame to clipboard as PNG">Copy</Button>
                  </>
                )}
                {!hideView && (
                  <Button size="small" sx={compactButton} disabled={lockView || !needsReset} onClick={handleDoubleClick} aria-label="Reset zoom and pan">Reset</Button>
                )}
              </Box>
            </Box>
          )}
          <Box
            ref={canvasContainerRef}
            sx={{
              ...container.imageBox,
              width: canvasW,
              height: canvasH,
              cursor: (!lockDisplay && isHoveringLensEdge)
                ? "nwse-resize"
                : (!lockRoi && (isHoveringResize || isDraggingResize || isHoveringResizeInner || isDraggingResizeInner))
                  ? "nwse-resize"
                  : (!lockProfile && (draggingProfileEndpoint !== null || isDraggingProfileLine))
                    ? "grabbing"
                    : (!lockProfile && profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine))
                      ? "grab"
                      : ((!hideRoi && roiActive && !lockRoi) || (!hideProfile && profileActive && !lockProfile))
                        ? "crosshair"
                        : (lockView ? "default" : "grab"),
            }}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseLeave}
            onWheel={handleWheel}
            onDoubleClick={handleDoubleClick}
          >
            <canvas ref={canvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: smooth ? "auto" : "pixelated" }} role="img" aria-label={`Slice image ${(playing ? displaySliceIdx : sliceIdx) + 1} of ${nSlices}${title ? `: ${title}` : ""} (${width} by ${height} pixels). Use arrow keys to scrub frames.`} />
            <canvas ref={overlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} aria-hidden="true" />
            <canvas ref={uiRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} aria-hidden="true" />
            <canvas ref={lensCanvasRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} aria-hidden="true" />
            {/* Zoom indicator now drawn on the ui canvas in the scale-bar
                pass (Show2D-matching style: white, sans, Unicode ×). */}
            {/* Cursor readout overlay */}
            {cursorInfo && (
              <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                  ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                </Typography>
              </Box>
            )}
            {!hideRoi && !lockRoi && roiActive && roiItems.length > 0 && showRoiResizeHint && (
              <Box sx={{ position: "absolute", left: 6, top: 6, px: 0.6, py: 0.25, bgcolor: "rgba(0,0,0,0.45)", pointerEvents: "none" }}>
                <Typography sx={{ fontSize: 9, color: "rgba(255,255,255,0.8)", lineHeight: 1.1 }}>
                  Hover ROI edge to resize
                </Typography>
              </Box>
            )}
            {/* Per-panel resize corner. Empty cells (partial last row) get
                no handle. Each handle scales the whole multi-panel canvas
                (linked behavior). Match Show2D gallery: 16x16, grey, 0.6/1.0.
                User trait `show_resize_handles` toggles visibility. */}
            {!hideView && showResizeHandles !== false && (() => {
              const n = Math.max(1, nPanels || 1);
              const cols = (maxCols && maxCols > 0) ? Math.min(maxCols, n) : n;
              const rows = Math.ceil(n / cols);
              const gap = n > 1 ? (panelGapTrait ?? 10) : 0;
              const outPanelW = (canvasW - gap * (cols - 1)) / cols;
              const outPanelH = (canvasH - gap * (rows - 1)) / rows;
              return Array.from({ length: n }).map((_, i) => {
                const col = i % cols;
                const row = Math.floor(i / cols);
                const slotX = col * (outPanelW + gap);
                const slotY = row * (outPanelH + gap);
                return (
                  <Box
                    key={`resize-${i}`}
                    onMouseDown={handleMainResizeStart}
                    sx={{
                      position: "absolute",
                      left: slotX + outPanelW - 16,
                      top: slotY + outPanelH - 16,
                      width: 16,
                      height: 16,
                      cursor: lockView ? "default" : "nwse-resize",
                      opacity: lockView ? 0.3 : 0.6,
                      pointerEvents: lockView ? "none" : "auto",
                      background: `linear-gradient(135deg, transparent 50%, ${themeColors.border} 50%)`,
                      borderRadius: "0 0 4px 0",
                      "&:hover": { opacity: lockView ? 0.3 : 1 },
                    }}
                  />
                );
              });
            })()}
          </Box>
          {/* Panel titles render ON canvas inside drawMain - follows grid layout. */}
          {/* Statistics bar - right below the image. Multi-panel = one row per panel. */}
          {showStats && !hideStats && (
            (localPanelStats && (nPanels || 1) > 1) ? (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", flexDirection: "column", gap: 0.25, width: "fit-content", boxSizing: "border-box", opacity: lockStats ? 0.7 : 1, fontFamily: "ui-monospace, monospace" }}>
                {localPanelStats.map((st, i) => (
                  <Box key={i} sx={{ display: "flex", gap: 2, alignItems: "center" }}>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted, minWidth: 80, fontFamily: "ui-monospace, monospace" }}>{(panelTitles && panelTitles[i]) || `Panel ${i + 1}`}</Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.mean)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.min)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.max)}</Box></Typography>
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(st.std)}</Box></Typography>
                  </Box>
                ))}
              </Box>
            ) : (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", width: "fit-content", boxSizing: "border-box", opacity: lockStats ? 0.7 : 1 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.mean : statsMean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.min : statsMin)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.max : statsMax)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(localStats ? localStats.std : statsStd)}</Box></Typography>
              </Box>
            )
          )}
          {/* Line profile sparkline */}
          {!hideProfile && profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: canvasW, height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
                role="img"
                aria-label="Line intensity profile along the drawn line"
              />
              <div
                onMouseDown={(e) => { if (lockProfile) return; e.preventDefault(); setIsResizingProfile(true); setProfileResizeStart({ y: e.clientY, height: profileHeight }); }}
                style={{ width: canvasW, height: 4, cursor: lockProfile ? "default" : "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)`, opacity: lockProfile ? 0.5 : 1, pointerEvents: lockProfile ? "none" : "auto" }}
              />
            </Box>
          )}
          {/* ROI sparkline plot */}
          {!hideRoi && roiActive && showRoiPlot && roiPlotData && roiPlotData.byteLength >= 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              <canvas
                ref={roiPlotCanvasRef}
                style={{ width: canvasW, height: 76, display: "block", border: `1px solid ${themeColors.border}` }}
                role="img"
                aria-label="ROI mean intensity over frames"
              />
            </Box>
          )}
          {/* Image Controls - Display / Histogram / Playback in one row, each
              spanning two control-row heights so the three blocks line up. */}
          {showControls && (!hideDisplay || !hideHistogram || !hidePlayback) && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, alignItems: "stretch", width: canvasW, boxSizing: "border-box", flexWrap: "wrap" }}>
              {!hideDisplay && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: "0 0 auto", justifyContent: "center", opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                  {/* Row 1: Scale + Auto + Color */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale:</Typography>
                    <Select disabled={lockDisplay} value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Intensity scale (linear or logarithmic)" }}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:</Typography>
                    <Switch checked={autoContrast} onChange={(e) => {
                      if (lockDisplay) return;
                      const on = e.target.checked;
                      setAutoContrast(on);
                      if (on && imageHistogramData) {
                        // ON → snap slider thumbs to actual percentile clip so slider shows what's rendered.
                        const { vmin: pmin, vmax: pmax } = percentileClip(imageHistogramData, percentileLow, percentileHigh);
                        const span = dataMax - dataMin;
                        if (span > 0) {
                          setImageVminPct(Math.max(0, Math.min(100, ((pmin - dataMin) / span) * 100)));
                          setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - dataMin) / span) * 100)));
                        }
                      } else {
                        // OFF → reset slider to full range so user sees raw data.
                        setImageVminPct(0);
                        setImageVmaxPct(100);
                      }
                    }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle automatic percentile-based contrast" }} />
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar:</Typography>
                    <Switch checked={showColorbar} onChange={(e) => { if (!lockDisplay) setShowColorbar(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle colorbar overlay" }} />
                  </Box>
                  {/* Row 2: Color + Smooth + Diff + zoom indicator */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                    <Select disabled={lockDisplay} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} inputProps={{ "aria-label": "Image colormap" }}>
                      {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Smooth:</Typography>
                    <Switch checked={smooth} onChange={(e) => { if (!lockDisplay) setSmooth(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle bilinear smoothing" }} />
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Diff:</Typography>
                    <Select disabled={lockDisplay} value={diffMode} onChange={(e) => setDiffMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "Difference mode (off, previous frame, first frame)" }}>
                      <MenuItem value="off">Off</MenuItem>
                      <MenuItem value="previous">Prev</MenuItem>
                      <MenuItem value="first">First</MenuItem>
                    </Select>
                    {/* zoom indicator moved onto the canvas overlay */}
                  </Box>
                </Box>
              )}
              {/* Playback: 2 rows side-by-side with Display + Histogram. */}
              {!hidePlayback && (() => { const activeIdx = playing ? displaySliceIdx : sliceIdx; return (
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, minWidth: 320, justifyContent: "center", opacity: lockPlayback ? 0.5 : 1, pointerEvents: lockPlayback ? "none" : "auto" }}>
                  <Box sx={{ ...controlRow, width: 360, maxWidth: 360, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                    <Stack direction="row" spacing={0} sx={{ flexShrink: 0, mr: 0.5 }}>
                      <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setReverse(true); setPlaying(true); } }} sx={{ color: reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play in reverse" title="Play reverse">
                        <FastRewindIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) setPlaying(!playing); }} sx={{ color: themeColors.accent, p: 0.25 }} aria-label={playing ? "Pause playback" : "Play"} title={playing ? "Pause (Space)" : "Play (Space)"}>
                        {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                      <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setReverse(false); setPlaying(true); } }} sx={{ color: !reverse && playing ? themeColors.accent : themeColors.textMuted, p: 0.25 }} aria-label="Play forward" title="Play forward">
                        <FastForwardIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                      <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setPlaying(false); setSliceIdx(loop ? Math.max(0, loopStart) : 0); } }} sx={{ color: themeColors.textMuted, p: 0.25 }} aria-label="Stop and rewind to start" title="Stop">
                        <StopIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Stack>
                    {loop ? (
                      <Slider value={[loopStart, activeIdx, effectiveLoopEnd]} onChange={(_, v) => { if (lockPlayback) return; const vals = v as number[]; setLoopStart(vals[0]); if (playing) setPlaying(false); setSliceIdx(vals[1]); setLoopEnd(vals[2]); }} disabled={lockPlayback} disableSwap min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => `${v + 1}`} marks={bookmarkedFrames.map(f => ({ value: f }))} aria-label={`Loop range and current ${dimLabel.toLowerCase()} (frame ${activeIdx + 1} of ${nSlices}, loop ${loopStart + 1} to ${effectiveLoopEnd + 1})`} sx={{ ...sliderStyles.small, flex: 1, minWidth: 40, "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 }, "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: themeColors.textMuted }, "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }} />
                    ) : (
                      <Slider value={activeIdx} onChange={(_, v) => { if (lockPlayback) return; if (playing) setPlaying(false); setSliceIdx(v as number); }} disabled={lockPlayback} min={0} max={nSlices - 1} size="small" valueLabelDisplay="auto" valueLabelFormat={(v) => `${v + 1}`} marks={bookmarkedFrames.map(f => ({ value: f }))} aria-label={`Current ${dimLabel.toLowerCase()} (${activeIdx + 1} of ${nSlices})`} sx={{ ...sliderStyles.small, flex: 1, minWidth: 40, "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" } }} />
                    )}
                    <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: `${String(nSlices).length * 2 + 2}ch`, textAlign: "right", flexShrink: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{hiddenSet.size ? `${activeIdx + 1}/${visibleCount} (${nSlices})` : `${activeIdx + 1}/${nSlices}`}</Typography>
                  </Box>
                  <Box sx={{ ...controlRow, width: 360, maxWidth: 360, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
                    <Slider disabled={lockPlayback} value={fps} min={1} max={60} step={1} onChange={(_, v) => { if (!lockPlayback) setFps(v as number); }} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} aria-label="Playback frames per second" valueLabelDisplay="auto" />
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted, minWidth: 14, flexShrink: 0 }}>{Math.round(fps)}</Typography>
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>Loop</Typography>
                    <Switch size="small" checked={loop} onChange={() => { if (!lockPlayback) setLoop(!loop); }} disabled={lockPlayback} sx={{ ...switchStyles.small, flexShrink: 0 }} inputProps={{ "aria-label": "Toggle loop playback" }} />
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>Bounce</Typography>
                    <Switch size="small" checked={boomerang} onChange={() => { if (!lockPlayback) setBoomerang(!boomerang); }} disabled={lockPlayback} sx={{ ...switchStyles.small, flexShrink: 0 }} inputProps={{ "aria-label": "Toggle bounce playback" }} />
                    <Box sx={{ flex: 1 }} />
                  </Box>
                </Box>
              ); })()}
              {!hideHistogram && (() => {
                // Global stack range from Python (data_min/data_max trait), not per-frame.
                // Log mode: log1p the range so bins line up with the log-scaled frame data.
                const { min: histMin, max: histMax } = resolveDisplayBounds(dataMin, dataMax, traitVmin, traitVmax, logScale);
                return (
                <Box sx={{
                  // Match Show2D histogram shell exactly so visual stays consistent
                  // across widgets. alignItems: flex-end (not stretch) prevents the
                  // inner Slider thumbs from overflowing onto the canvas, which was
                  // the source of the "2.8 tooltip overlaps bars" overlap bug.
                  display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5,
                  opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto",
                }}>
                  <Histogram
                    data={imageHistogramData}
                    vminPct={imageVminPct}
                    vmaxPct={imageVmaxPct}
                    onRangeChange={(min, max) => {
                      if (lockHistogram) return;
                      setImageVminPct(min);
                      setImageVmaxPct(max);
                      if (autoContrast) setAutoContrast(false);
                    }}
                    width={110}
                    height={58}
                    theme={themeInfo.theme === "dark" ? "dark" : "light"}
                    dataMin={histMin}
                    dataMax={histMax}
                  />
                </Box>
                );
              })()}
            </Box>
          )}
          {/* Lens settings row (when Lens is active) */}
          {!hideDisplay && showLens && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Lens {lensMag}×</Typography>
                <Slider disabled={lockDisplay} value={lensMag} min={2} max={8} step={1} onChange={(_, v) => setLensMag(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens magnification" valueLabelDisplay="auto" />
                <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>{lensDisplaySize}px</Typography>
                <Slider disabled={lockDisplay} value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, v) => setLensDisplaySize(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} aria-label="Lens display size in pixels" valueLabelDisplay="auto" />
              </Box>
            </Box>
          )}
          {/* ROI settings row (when ROI is active) */}
          {!hideRoi && roiActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: "fit-content" }}>
              <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, px: 1, py: 0.5, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, opacity: lockRoi ? 0.5 : 1, pointerEvents: lockRoi ? "none" : "auto" }}>
                {/* ROI: shape + add/duplicate + plot + dim */}
                <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>ROI:</Typography>
                  <Select
                    size="small"
                    value={newRoiShape}
                    onChange={(e) => setNewRoiShape(e.target.value as "circle" | "square" | "rectangle" | "annular")}
                    MenuProps={themedMenuProps}
                    sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                    inputProps={{ "aria-label": "New ROI shape" }}
                  >
                    {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                  </Select>
                  <Button size="small" sx={compactButton} onClick={() => addROIAt(height / 2, width / 2)} aria-label="Add ROI at image center">ADD</Button>
                  <Button size="small" sx={compactButton} disabled={!selectedRoi} onClick={duplicateSelectedROI} aria-label="Duplicate selected ROI">DUP</Button>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Plot:</Typography>
                  <Switch checked={showRoiPlot} onChange={(e) => setShowRoiPlot(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle ROI intensity plot" }} />
                  <Box sx={{ flex: 1 }} />
                  <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 24, color: "#ef5350" }} disabled={!roiItems.length} onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }} aria-label="Clear all ROIs">CLEAR</Button>
                </Box>

                {/* Selected ROI details */}
                {selectedRoi && (
                  <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: selectedRoi.color }}>#{roiSelectedIdx + 1}/{roiItems.length}</Typography>
                    <Select
                      size="small"
                      value={selectedRoi.shape || "circle"}
                      onChange={(e) => updateSelectedRoi({ shape: String(e.target.value) })}
                      MenuProps={themedMenuProps}
                      sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                      inputProps={{ "aria-label": "Selected ROI shape" }}
                    >
                      {(["square", "rectangle", "circle", "annular"] as const).map((shape) => (<MenuItem key={shape} value={shape}>{shape.charAt(0).toUpperCase() + shape.slice(1)}</MenuItem>))}
                    </Select>
                    {selectedRoi.shape === "rectangle" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>W</Typography>
                        <Slider value={selectedRoi.width} min={5} max={width} onChange={(_, v) => updateSelectedRoi({ width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI width" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>H</Typography>
                        <Slider value={selectedRoi.height} min={5} max={height} onChange={(_, v) => updateSelectedRoi({ height: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="ROI height" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape === "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Inner</Typography>
                        <Slider value={selectedRoi.radius_inner} min={1} max={Math.max(2, selectedRoi.radius - 1)} onChange={(_, v) => updateSelectedRoi({ radius_inner: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI inner radius" valueLabelDisplay="auto" />
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Outer</Typography>
                        <Slider value={selectedRoi.radius} min={selectedRoi.radius_inner + 1} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} aria-label="Annular ROI outer radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    {selectedRoi.shape !== "rectangle" && selectedRoi.shape !== "annular" && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Size</Typography>
                        <Slider value={selectedRoi.radius} min={5} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 50 }} aria-label="ROI radius" valueLabelDisplay="auto" />
                      </>
                    )}
                    <Box sx={{ display: "flex", gap: "2px" }}>
                      {ROI_COLORS.map(c => (
                        <Box key={c} onClick={() => updateSelectedRoi({ color: c })} sx={{ width: 12, height: 12, bgcolor: c, cursor: "pointer", border: c === selectedRoi.color ? `2px solid ${themeColors.text}` : "1px solid transparent", "&:hover": { opacity: 0.8 } }} />
                      ))}
                    </Box>
                    <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Border</Typography>
                    <Slider value={selectedRoi.line_width} min={1} max={6} step={1} onChange={(_, v) => updateSelectedRoi({ line_width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 30 }} aria-label="ROI border line width" valueLabelDisplay="auto" />
                    <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 20, color: "#ef5350" }} onClick={deleteSelectedROI} aria-label="Delete selected ROI">&times;</Button>
                  </Box>
                )}

                {/* ROI list */}
                {roiItems.length > 0 && (
                  <Box sx={{ display: "flex", flexDirection: "column", borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                    {roiItems.map((roi, i) => {
                      const c = roi.color || ROI_COLORS[i % ROI_COLORS.length];
                      const isSelected = i === roiSelectedIdx;
                      const shapeLabel = roi.shape === "rectangle" ? `${roi.width}×${roi.height}` : roi.shape === "annular" ? `r${roi.radius_inner}-${roi.radius}` : `r${roi.radius}`;
                      return (
                        <Box key={i} onClick={() => setRoiSelectedIdx(i)} sx={{ display: "flex", alignItems: "center", gap: "3px", lineHeight: 1.6, cursor: "pointer", "&:hover .roi-delete": { opacity: 1 } }}>
                          <Box sx={{ width: 8, height: 8, borderRadius: roi.shape === "square" || roi.shape === "rectangle" ? 0 : "50%", bgcolor: c, border: isSelected ? "2px solid #fff" : "1px solid transparent", flexShrink: 0 }} />
                          <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: isSelected ? themeColors.text : themeColors.textMuted, fontWeight: isSelected ? "bold" : "normal" }}>
                            <Box component="span" sx={{ color: c }}>{i + 1}</Box>{" "}
                            {roi.shape} ({Math.round(roi.row)}, {Math.round(roi.col)}) {shapeLabel}
                          </Typography>
                          <Box
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.map((r, j) => ({ ...r, highlight: j === i ? !r.highlight : false })); setRoiList(newList); }}
                            sx={{ cursor: "pointer", fontSize: 10, color: roi.highlight ? themeColors.accentGreen : themeColors.textMuted, lineHeight: 1, opacity: roi.highlight ? 1 : 0.5, "&:hover": { opacity: 1 } }}
                            title="Focus (dim outside)"
                          >{roi.highlight ? "\u25C9" : "\u25CB"}</Box>
                          <Box
                            className="roi-delete"
                            onClick={(e) => { e.stopPropagation(); const newList = roiItems.filter((_, j) => j !== i); setRoiList(newList); setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1); }}
                            sx={{ opacity: 0, cursor: "pointer", fontSize: 10, color: themeColors.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: "#f44336" } }}
                          >&times;</Box>
                        </Box>
                      );
                    })}
                  </Box>
                )}
              </Box>
            </Box>
          )}
        </Box>

        {/* Preview Panel - ROI crop at full resolution with aspect ratio */}
        {previewVisible && (
          <Box sx={{ width: canvasW }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Header row - matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                Preview{previewCropDims ? ` (${previewCropDims.w}\u00d7${previewCropDims.h})` : ""}
              </Typography>
              {!hideView && (
                <Button size="small" sx={compactButton} disabled={lockView || (previewZoom.zoom === 1 && previewZoom.panX === 0 && previewZoom.panY === 0)} onClick={handlePreviewDoubleClick} aria-label="Reset preview zoom and pan">Reset</Button>
              )}
            </Stack>
            <Box
              ref={previewContainerRef}
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, cursor: lockView ? "default" : "grab", width: previewCanvasDims.w, height: previewCanvasDims.h }}
              onWheel={lockView ? undefined : handlePreviewWheel}
              onDoubleClick={lockView ? undefined : handlePreviewDoubleClick}
              onMouseDown={lockView ? undefined : handlePreviewMouseDown}
              onMouseMove={lockView ? undefined : handlePreviewMouseMove}
              onMouseUp={handlePreviewMouseUp}
              onMouseLeave={handlePreviewMouseUp}
            >
              <canvas ref={previewCanvasRef} width={previewCanvasDims.w} height={previewCanvasDims.h} style={{ width: previewCanvasDims.w, height: previewCanvasDims.h, imageRendering: "pixelated" }} role="img" aria-label={`ROI preview crop${previewCropDims ? ` (${previewCropDims.w} by ${previewCropDims.h} pixels)` : ""}`} />
              <canvas ref={previewOverlayRef} width={Math.round(previewCanvasDims.w * DPR)} height={Math.round(previewCanvasDims.h * DPR)} style={{ position: "absolute", top: 0, left: 0, width: previewCanvasDims.w, height: previewCanvasDims.h, pointerEvents: "none" }} aria-hidden="true" />
              {!hideView && (
                <Box onMouseDown={handleMainResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 28, height: 28, cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.4 : 0.95, pointerEvents: lockView ? "none" : "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.border} 50%)`, "&:hover": { opacity: lockView ? 0.4 : 1 } }} />
              )}
            </Box>
            {/* All-ROI Stats - one row per ROI, same style as main stats bar */}
            {!hideStats && showStats && allRoiStats.length > 0 && (
              <Box sx={{ mt: `${SPACING.XS}px`, display: "flex", flexDirection: "column", gap: 0.5, width: previewCanvasDims.w }}>
                {allRoiStats.map((stats, i) => {
                  if (!stats) return null;
                  const color = roiItems[i]?.color || ROI_COLORS[i % ROI_COLORS.length];
                  const isSelected = i === roiSelectedIdx;
                  return (
                    <Box key={i} sx={{ px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", border: isSelected ? `1px solid ${color}` : `1px solid transparent` }}>
                      <Box sx={{ width: 8, height: 8, bgcolor: color, borderRadius: "50%", flexShrink: 0 }} />
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color }}>{formatNumber(stats.mean)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color }}>{formatNumber(stats.min)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color }}>{formatNumber(stats.max)}</Box></Typography>
                      <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color }}>{formatNumber(stats.std)}</Box></Typography>
                    </Box>
                  );
                })}
              </Box>
            )}
          </Box>
        )}

        {/* FFT Panel - same size as main image, canvas-aligned with spacer */}
        {effectiveShowFft && (
          <Box sx={{ width: canvasW }}>
            {/* Spacer - matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Controls row - matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              {roiFftActive && fftCropDims ? (
                <Typography sx={{ ...typography.label, color: themeColors.accentGreen }}>
                  ROI FFT ({fftCropDims.cropWidth}&times;{fftCropDims.cropHeight})
                </Typography>
              ) : <Box />}
              {!hideView && (
                <Button size="small" sx={compactButton} disabled={lockView || !fftNeedsReset} onClick={handleFftReset} aria-label="Reset FFT zoom and pan">Reset</Button>
              )}
            </Stack>
            {/* FFT Canvas - same size as main image */}
            <Box
              ref={fftContainerRef}
              sx={{ ...container.imageBox, width: canvasW, height: canvasH, cursor: lockView ? "default" : "grab" }}
              onMouseDown={lockView ? undefined : handleFftMouseDown}
              onMouseMove={lockView ? undefined : handleFftMouseMove}
              onMouseUp={lockView ? undefined : handleFftMouseUp}
              onMouseLeave={() => { fftClickStartRef.current = null; setIsFftDragging(false); setFftPanStart(null); }}
              onWheel={lockView ? undefined : handleFftWheel}
              onDoubleClick={lockView ? undefined : handleFftReset}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: smooth ? "auto" : "pixelated" }} role="img" aria-label={roiFftActive && fftCropDims ? `FFT power spectrum of ROI crop (${fftCropDims.cropWidth} by ${fftCropDims.cropHeight} pixels)` : "FFT power spectrum of current frame"} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} aria-hidden="true" />
            </Box>
            {/* FFT Statistics bar */}
            {showStats && !hideStats && (
              <Box sx={{ mt: 0.5, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, flexWrap: "wrap", opacity: lockStats ? 0.7 : 1 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.mean)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.min)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.max)}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats.std)}</Box></Typography>
                {fftClickInfo && (
                  <>
                    <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                    <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                      {fftClickInfo.dSpacing != null ? (
                        <>d = <Box component="span" sx={{ color: themeColors.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} Å`}</Box>{" | |g| = "}<Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.spatialFreq!.toFixed(4)} Å⁻¹</Box></>
                      ) : (
                        <>dist = <Box component="span" sx={{ color: themeColors.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                      )}
                    </Typography>
                  </>
                )}
              </Box>
            )}
            {/* FFT Controls - two rows with histogram on right (like Show4DSTEM) */}
            {showControls && <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: canvasW, boxSizing: "border-box" }}>
              {/* Left: two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center", opacity: lockDisplay ? 0.5 : 1, pointerEvents: lockDisplay ? "none" : "auto" }}>
                {/* Row 1: Scale + Auto */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Scale:</Typography>
                  <Select disabled={lockDisplay} value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT intensity scale (linear or logarithmic)" }}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Auto:</Typography>
                  <Switch checked={fftAuto} onChange={(e) => { if (!lockDisplay) setFftAuto(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle automatic FFT contrast" }} />
                  {roiFftActive && fftCropDims && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Win:</Typography>
                      <Switch checked={fftWindow} onChange={(e) => { if (!lockDisplay) setFftWindow(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle Hann windowing before FFT" }} />
                    </>
                  )}
                </Box>
                {/* Row 2: Color + Colorbar */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Color:</Typography>
                  <Select disabled={lockDisplay} value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT colormap" }}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.textMuted }}>Colorbar:</Typography>
                  <Switch checked={fftShowColorbar} onChange={(e) => { if (!lockDisplay) setFftShowColorbar(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle FFT colorbar overlay" }} />
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              {!hideHistogram && (
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
                  <Histogram
                    data={fftHistogramData}
                    vminPct={fftVminPct}
                    vmaxPct={fftVmaxPct}
                    onRangeChange={(min, max) => { if (!lockHistogram) { setFftVminPct(min); setFftVmaxPct(max); } }}
                    width={110}
                    height={58}
                    theme={themeInfo.theme}
                    dataMin={fftDataRange.min}
                    dataMax={fftDataRange.max}
                  />
                </Box>
              )}
            </Box>}
          </Box>
        )}
      </Stack>

    </Box>
  );
}

// anywidget v0.9+ deprecates `export render` in favor of `export default { render }`.
const render = createRender(Show3D);
export default { render };
