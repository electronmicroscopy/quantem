/**
 * Show2D - Static 2D image viewer with gallery support.
 * 
 * Features:
 * - Single image or gallery mode with configurable columns
 * - Scroll to zoom, double-click to reset
 * - WebGPU-accelerated FFT with default 3x zoom
 * - Equal-sized FFT and histogram panels
 * - Click to select image in gallery mode
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Switch from "@mui/material/Switch";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Tooltip from "@mui/material/Tooltip";
import { useTheme } from "../theme";
import { drawScaleBarHiDPI, drawColorbar, roundToNiceValue, exportFigure, canvasToPDF } from "../figure";
import JSZip from "jszip";
import { extractFloat32, formatNumber, downloadBlob } from "../format";
import { computeHistogramFromBytes, findDataRange, applyLogScale, percentileClip, sliderRange, computeStats } from "../stats";

function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content}
      arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 280, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography component="span" sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}>ⓘ</Typography>
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

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};
import { getWebGPUFFT, WebGPUFFT, fft2d, fft2dAsync, fftshift, computeMagnitude, autoEnhanceFFT, nextPow2, applyHannWindow2D, getGPUInfo } from "../fft";
import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse, GPUColormapEngine, getGPUColormapEngine, getGPUMaxBufferSize } from "../colormaps";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 20;

const DPR = window.devicePixelRatio || 1;

interface HistogramProps {
  data: Float32Array | null;
  precomputedBins?: number[] | null;  // GPU-computed bins bypass computeHistogramFromBytes
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

function Histogram({ data, precomputedBins, vminPct, vmaxPct, onRangeChange, width = 110, height = 40, theme = "dark", dataMin = 0, dataMax = 1 }: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const cpuBins = React.useMemo(() => precomputedBins ? null : computeHistogramFromBytes(data), [data, precomputedBins]);
  const bins = precomputedBins || cpuBins || new Array(256).fill(0);
  const isDark = theme === "dark";
  const colors = isDark ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" } : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" };

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
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
      ctx.fillStyle = (i >= vminBin && i <= vmaxBin) ? colors.barActive : colors.barInactive;
      ctx.fillRect(i * barWidth + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas ref={canvasRef} style={{ width, height, border: `1px solid ${colors.border}` }} />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => { const [newMin, newMax] = v as number[]; onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)); }}
        min={0} max={100} size="small" valueLabelDisplay="auto"
        valueLabelFormat={(pct) => { const val = dataMin + (pct / 100) * (dataMax - dataMin); return val >= 1000 ? val.toExponential(1) : val.toFixed(1); }}
        sx={{ width, py: 0, "& .MuiSlider-thumb": { width: 8, height: 8 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 }, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vminPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography><Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{(() => { const v = dataMin + (vmaxPct / 100) * (dataMax - dataMin); return v >= 1000 ? v.toExponential(1) : v.toFixed(1); })()}</Typography></Box>
    </Box>
  );
}

// ============================================================================
// Line profile sampling (bilinear interpolation along line)
// ============================================================================
function sampleLineProfile(data: Float32Array, w: number, h: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
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
// FFT peak finder (snap to Bragg spot with sub-pixel centroid refinement)
// ============================================================================
function findFFTPeak(mag: Float32Array, width: number, height: number, col: number, row: number, radius: number): { row: number; col: number } {
  // Find brightest pixel in search window
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
  // Sub-pixel refinement via weighted centroid in 3×3 window
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

// ============================================================================
// Types
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };

// ============================================================================
// Constants
// ============================================================================
const SINGLE_IMAGE_TARGET = 500;
const GALLERY_IMAGE_TARGET = 300;
const DEFAULT_FFT_ZOOM = 2;
const PROFILE_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
type ROIItem = { row: number; col: number; shape: string; radius: number; radius_inner: number; width: number; height: number; color: string; line_width: number; highlight: boolean };
const ROI_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
const RESIZE_HIT_AREA_PX = 10;

function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  shape: "circle" | "square" | "rectangle" | "annular",
  radius: number, w: number, h: number,
  activeColor: string, inactiveColor: string,
  active: boolean = false, innerRadius: number = 0
): void {
  const strokeColor = active ? activeColor : inactiveColor;
  ctx.strokeStyle = strokeColor;
  if (shape === "circle") {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - w / 2, y - h / 2, w, h);
  } else if (shape === "annular") {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = active ? "#0ff" : inactiveColor;
    ctx.beginPath(); ctx.arc(x, y, innerRadius, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = (active ? activeColor : inactiveColor) + "15";
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.arc(x, y, innerRadius, 0, Math.PI * 2, true); ctx.fill();
    ctx.strokeStyle = strokeColor;
  }
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5);
    ctx.stroke();
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
// Main Component
// ============================================================================
// Show4DSTEM-style UI constants
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
};
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
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
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};
const switchStyles = {
  small: { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } },
};
const sliderStyles = {
  small: { py: 0, "& .MuiSlider-thumb": { width: 10, height: 10 }, "& .MuiSlider-rail": { height: 2 }, "& .MuiSlider-track": { height: 2 } },
};

function Show2D() {
  // Theme
  const { themeInfo, colors: tc } = useTheme();
  const themeColors = {
    ...tc,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
  };

  const themedSelect = {
    fontSize: 10,
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

  // Model state
  const [nImages] = useModelState<number>("n_images");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [displayBinFactor] = useModelState<number>("_display_bin_factor");
  const [, setGpuMaxBufferMB] = useModelState<number>("_gpu_max_buffer_mb");
  const [widgetVersion] = useModelState<string>("widget_version");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [ncols] = useModelState<number>("ncols");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [traitVmins] = useModelState<(number | null)[] | null>("vmins");
  const [traitVmaxs] = useModelState<(number | null)[] | null>("vmaxs");
  const [zoomRowTrait] = useModelState<number | null>("zoom_row");
  const [zoomColTrait] = useModelState<number | null>("zoom_col");
  const [diffMode, setDiffMode] = useModelState<boolean>("diff_mode");
  const [diffReference] = useModelState<number>("diff_reference");
  // Align removed — diff = A − B (no shift). Drift correction happens upstream.
  const alignDy = 0;
  const alignDx = 0;

  // Customization
  const [canvasSizeTrait] = useModelState<number>("size");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const imageRenderingStyle = smooth ? "auto" : "pixelated";

  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  // UI visibility
  const [showControls] = useModelState<boolean>("show_controls");
  const [showStats] = useModelState<boolean>("show_stats");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");

  // Analysis Panels (FFT + Histogram)
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");

  // Selection
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");

  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiList, setRoiList] = useModelState<ROIItem[]>("roi_list");
  const [roiSelectedIdx, setRoiSelectedIdx] = useModelState<number>("roi_selected_idx");
  const [imageRotations, setImageRotations] = useModelState<number[]>("image_rotations");
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false);
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const resizeAspectRef = React.useRef<number | null>(null);
  const [newRoiShape, setNewRoiShape] = React.useState<"circle" | "square" | "rectangle" | "annular">("square");
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const selectedRoi = roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0) ? roiList[roiSelectedIdx] : null;

  const effectiveShowFft = showFft;

  const updateSelectedRoi = (updates: Partial<ROIItem>) => {
    if (roiSelectedIdx < 0 || !roiList) return;
    const newList = [...roiList];
    newList[roiSelectedIdx] = { ...newList[roiSelectedIdx], ...updates };
    setRoiList(newList);
  };

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const imageContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const fftContainerRefs = React.useRef<(HTMLDivElement | null)[]>([]);
  const singleFftContainerRef = React.useRef<HTMLDivElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [canvasReady, setCanvasReady] = React.useState(0);  // Trigger re-render when refs attached

  // Zoom/Pan state - per-image when not linked, shared when linked
  const [initialZoom] = useModelState<number>("initial_zoom");
  const [linkPan, setLinkPan] = useModelState<boolean>("link_pan");
  const [imgHeight] = useModelState<number>("height");
  const [imgWidth] = useModelState<number>("width");
  // Note: pan derived from zoom_row/zoom_col is applied via a useEffect AFTER canvasW/canvasH
  // are computed (see "Initial pan from zoom_row/zoom_col" effect below).
  const initialZoomState: ZoomState = React.useMemo(
    () => ({ zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, initialZoom || 1)), panX: 0, panY: 0 }),
    [initialZoom]
  );
  void linkPan; void setLinkPan; void imgWidth; void imgHeight;
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedZoomState, setLinkedZoomState] = React.useState<ZoomState>(initialZoomState);
  const [linkedZoom, setLinkedZoom] = useModelState<boolean>("link_zoom");
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Helper to get zoom state for an image. zoom and pan link independently:
  //   zoom from linkedZoomState if linkedZoom else per-image
  //   pan  from linkedZoomState if linkPan  else per-image
  const getZoomState = React.useCallback((idx: number): ZoomState => {
    const per = zoomStates.get(idx) || initialZoomState;
    return {
      zoom: linkedZoom ? linkedZoomState.zoom : per.zoom,
      panX: linkPan ? linkedZoomState.panX : per.panX,
      panY: linkPan ? linkedZoomState.panY : per.panY,
    };
  }, [linkedZoom, linkPan, linkedZoomState, zoomStates, initialZoomState]);

  // Helper to set zoom state for an image. zoom and pan honored independently:
  //   zoom: writes to linkedZoomState if linkedZoom, else per-image
  //   pan:  writes to linkedZoomState if linkPan, else per-image
  const setZoomState = React.useCallback((idx: number, state: ZoomState) => {
    if (linkedZoom || linkPan) {
      setLinkedZoomState(prev => ({
        zoom: linkedZoom ? state.zoom : prev.zoom,
        panX: linkPan ? state.panX : prev.panX,
        panY: linkPan ? state.panY : prev.panY,
      }));
    }
    if (!linkedZoom || !linkPan) {
      setZoomStates(prev => {
        const m = new Map(prev);
        const cur = m.get(idx) || initialZoomState;
        m.set(idx, {
          zoom: linkedZoom ? cur.zoom : state.zoom,
          panX: linkPan ? cur.panX : state.panX,
          panY: linkPan ? cur.panY : state.panY,
        });
        return m;
      });
    }
  }, [linkedZoom, linkPan, initialZoomState]);

  // FFT zoom/pan state (single mode)
  const [fftZoom, setFftZoom] = React.useState(DEFAULT_FFT_ZOOM);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [isDraggingFftPan, setIsDraggingFftPan] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Histogram state — per-image contrast ranges (gallery) or single (one image)
  const [linkedContrast, setLinkedContrast] = useModelState<boolean>("link_contrast");
  const [linkedContrastState, setLinkedContrastState] = React.useState<{ vminPct: number; vmaxPct: number }>({ vminPct: 0, vmaxPct: 100 });
  const [contrastStates, setContrastStates] = React.useState<Map<number, { vminPct: number; vmaxPct: number }>>(new Map());
  // Ref mirror for fast slider path (bypass React effect batching)
  const contrastRef = React.useRef<{ linked: { vminPct: number; vmaxPct: number }; perImage: Map<number, { vminPct: number; vmaxPct: number }> }>({ linked: { vminPct: 0, vmaxPct: 100 }, perImage: new Map() });
  const sliderRafRef = React.useRef(0);
  const getContrastState = React.useCallback((idx: number) => {
    if (linkedContrast) return linkedContrastState;
    return contrastStates.get(idx) || { vminPct: 0, vmaxPct: 100 };
  }, [linkedContrast, linkedContrastState, contrastStates]);
  const setContrastState = React.useCallback((idx: number, state: { vminPct: number; vmaxPct: number }) => {
    // Update ref immediately (for fast rAF render)
    if (linkedContrast) {
      contrastRef.current.linked = state;
      setLinkedContrastState(state);
    } else {
      contrastRef.current.perImage.set(idx, state);
      setContrastStates(prev => new Map(prev).set(idx, state));
    }
    // Fast path: direct GPU render via rAF, bypassing React effect batching
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current && engine.slotCount >= nImages) {
      cancelAnimationFrame(sliderRafRef.current);
      sliderRafRef.current = requestAnimationFrame(() => {
        const cachedRanges = dataRangesRef.current;
        if (cachedRanges.length === 0) return;
        const lut = COLORMAPS[cmapRef.current] || COLORMAPS.inferno;
        engine.uploadLUT(cmapRef.current, lut);
        const indices = Array.from({ length: nImages }, (_, i) => i);
        const ranges: { vmin: number; vmax: number }[] = [];
        for (let i = 0; i < nImages; i++) {
          const cs = linkedContrast ? contrastRef.current.linked : (contrastRef.current.perImage.get(i) || { vminPct: 0, vmaxPct: 100 });
          let cr = cachedRanges[i];
          if (!cr || cr.min === cr.max) {
            if (rawDataRef.current && rawDataRef.current[i]) cr = findDataRange(rawDataRef.current[i]);
          }
          cr = cr || { min: 0, max: 1 };
          if (cs.vminPct > 0 || cs.vmaxPct < 100) {
            ranges.push(sliderRange(cr.min, cr.max, cs.vminPct, cs.vmaxPct));
          } else {
            ranges.push({ vmin: cr.min, vmax: cr.max });
          }
        }
        const ls = logScaleRef.current ?? false;
        const bitmaps = engine.renderSlotsToImageBitmap(indices, ranges, ls);
        if (bitmaps && bitmaps[0]) {
          for (let i = 0; i < bitmaps.length; i++) {
            const offscreen = mainOffscreensRef.current[i];
            if (offscreen && bitmaps[i]) offscreen.getContext("2d")?.drawImage(bitmaps[i], 0, 0);
          }
          setOffscreenVersion(v => v + 1);
        }
      });
    }
  }, [linkedContrast, nImages]);
  // Convenience accessors for active image
  const activeContrastIdx = nImages > 1 ? selectedIdx : 0;
  const imageVminPct = getContrastState(activeContrastIdx).vminPct;
  const imageVmaxPct = getContrastState(activeContrastIdx).vmaxPct;

  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageHistogramBins, setImageHistogramBins] = React.useState<number[] | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // FFT display state (single mode)
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataRange, setFftDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftScaleMode, setFftScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftSmooth, setFftSmooth] = React.useState(true);
  const [fftLinkedZoom, setFftLinkedZoom] = React.useState(false);
  const [fftLinkPan, setFftLinkPan] = React.useState(false);
  const [fftLinkedContrast, setFftLinkedContrast] = React.useState(true);
  // Per-image FFT contrast (used when fftLinkedContrast=false)
  const [fftContrastStates, setFftContrastStates] = React.useState<Map<number, { vminPct: number; vmaxPct: number }>>(new Map());
  const fftContrastFor = React.useCallback((idx: number) => {
    if (fftLinkedContrast) return { vminPct: fftVminPct, vmaxPct: fftVmaxPct };
    return fftContrastStates.get(idx) || { vminPct: 0, vmaxPct: 100 };
  }, [fftLinkedContrast, fftVminPct, fftVmaxPct, fftContrastStates]);
  const setFftContrastFor = React.useCallback((idx: number, val: { vminPct: number; vmaxPct: number }) => {
    if (fftLinkedContrast) {
      setFftVminPct(val.vminPct);
      setFftVmaxPct(val.vmaxPct);
    } else {
      setFftContrastStates(prev => new Map(prev).set(idx, val));
    }
  }, [fftLinkedContrast]);
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);
  const [fftShowColorbar, setFftShowColorbar] = React.useState(false);

  // FFT loading state — shown as a pulsing overlay while FFT computes
  const [fftComputing, setFftComputing] = React.useState(false);
  const [fftProgress, setFftProgress] = React.useState("");

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number } | null>(null);

  // Colorbar state (single image mode only)
  const [showColorbar, setShowColorbar] = React.useState(false);

  // Inset magnifier state
  const [showLens, setShowLens] = React.useState(false);
  const [lensPos, setLensPos] = React.useState<{ row: number; col: number } | null>(null);
  const [lensMag, setLensMag] = React.useState(4);       // magnification 2×–8×
  const [lensDisplaySize, setLensDisplaySize] = React.useState(128); // CSS px 64–256
  const [lensAnchor, setLensAnchor] = React.useState<{ x: number; y: number } | null>(null); // custom position (CSS px from top-left of canvas)
  const [isDraggingLens, setIsDraggingLens] = React.useState(false);
  const [isResizingLens, setIsResizingLens] = React.useState(false);
  const [isHoveringLensEdge, setIsHoveringLensEdge] = React.useState(false);
  const lensDragStartRef = React.useRef<{ mx: number; my: number; ax: number; ay: number } | null>(null);
  const lensResizeStartRef = React.useRef<{ my: number; startSize: number } | null>(null);
  const lensCanvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // FFT d-spacing measurement
  const [fftClickInfo, setFftClickInfo] = React.useState<{
    row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  } | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);

  // Line profile state
  const [profileActive, setProfileActive] = React.useState(false);
  const [profileLine, setProfileLine] = useModelState<{ row: number; col: number }[]>("profile_line");
  const [profileDataAll, setProfileDataAll] = React.useState<(Float32Array | null)[]>([]);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileBaseImageRef = React.useRef<ImageData | null>(null);
  const profileLayoutRef = React.useRef<{ padLeft: number; plotW: number; padTop: number; plotH: number; gMin: number; gMax: number; totalDist: number; xUnit: string } | null>(null);

  // Sync profile points from model state
  const profilePoints = profileLine || [];
  const setProfilePoints = (pts: { row: number; col: number }[]) => setProfileLine(pts);

  // Distance measurement state (JS-only, not persisted)
  const [measureActive, setMeasureActive] = React.useState(false);
  const [measurePoints, setMeasurePoints] = React.useState<{row: number; col: number}[]>([]);

  // FFT zoom/pan state (gallery mode — per-image or linked)
  const [galleryFftStates, setGalleryFftStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedFftZoomState, setLinkedFftZoomState] = React.useState<ZoomState>({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
  const [fftPanningIdx, setFftPanningIdx] = React.useState<number | null>(null);
  const getGalleryFftState = React.useCallback((idx: number) => {
    const per = galleryFftStates.get(idx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
    return {
      zoom: fftLinkedZoom ? linkedFftZoomState.zoom : per.zoom,
      panX: fftLinkPan ? linkedFftZoomState.panX : per.panX,
      panY: fftLinkPan ? linkedFftZoomState.panY : per.panY,
    };
  }, [fftLinkedZoom, fftLinkPan, linkedFftZoomState, galleryFftStates]);
  const setGalleryFftState = React.useCallback((idx: number, state: ZoomState) => {
    if (fftLinkedZoom || fftLinkPan) {
      setLinkedFftZoomState(prev => ({
        zoom: fftLinkedZoom ? state.zoom : prev.zoom,
        panX: fftLinkPan ? state.panX : prev.panX,
        panY: fftLinkPan ? state.panY : prev.panY,
      }));
    }
    if (!fftLinkedZoom || !fftLinkPan) {
      setGalleryFftStates(prev => {
        const cur = prev.get(idx) || { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 };
        const next = new Map(prev);
        next.set(idx, {
          zoom: fftLinkedZoom ? cur.zoom : state.zoom,
          panX: fftLinkPan ? cur.panX : state.panX,
          panY: fftLinkPan ? cur.panY : state.panY,
        });
        return next;
      });
    }
  }, [fftLinkedZoom, fftLinkPan]);

  // Resizable state (gallery starts smaller)
  const [canvasSize, setCanvasSize] = React.useState(nImages > 1 ? GALLERY_IMAGE_TARGET : SINGLE_IMAGE_TARGET);

  // Sync initial sizes from traits
  React.useEffect(() => {
    if (canvasSizeTrait > 0) setCanvasSize(canvasSizeTrait);
  }, [canvasSizeTrait]);

  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number, y: number, size: number } | null>(null);

  // Profile height resize
  const [profileHeight, setProfileHeight] = React.useState(76);
  const [isResizingProfile, setIsResizingProfile] = React.useState(false);
  const [profileResizeStart, setProfileResizeStart] = React.useState<{ y: number; height: number } | null>(null);

  // WebGPU FFT
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const gpuReadyRef = React.useRef(false);
  const rawDataRef = React.useRef<Float32Array[] | null>(null);
  const diffCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const diffFftCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const diffFftMagRef = React.useRef<Float32Array | null>(null);

  // WebGPU colormap engine — uses refs (not state) to avoid re-triggering
  // effects when GPU initializes. Effects check refs opportunistically:
  // on first render they use CPU, on subsequent renders (data/slider change)
  // they use GPU if available. No double computation.
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const gpuCmapReadyRef = React.useRef(false);

  // Cached offscreen canvases for main image rendering (avoids per-zoom/pan recompute)
  const mainOffscreensRef = React.useRef<HTMLCanvasElement[]>([]);
  const mainImgDatasRef = React.useRef<ImageData[]>([]);
  const logBufferRef = React.useRef<Float32Array | null>(null);
  const colorbarVminRef = React.useRef(0);
  const colorbarVmaxRef = React.useRef(1);
  const [offscreenVersion, setOffscreenVersion] = React.useState(0);

  // Truthful first-render signal: flipped ONCE after the first colormap pass has
  // actually painted.  Python side observes `_js_rendered` and prints the real
  // end-to-end wall clock.  Two rAFs ensure the browser has composited before we
  // fire, so the printed time reflects "user can see the widget," not "data arrived."
  const [, setJsRendered] = useModelState<boolean>("_js_rendered");
  const firstRenderFiredRef = React.useRef(false);
  React.useEffect(() => {
    if (firstRenderFiredRef.current) return;
    if (offscreenVersion === 0) return;
    firstRenderFiredRef.current = true;
    requestAnimationFrame(() => requestAnimationFrame(() => setJsRendered(true)));
  }, [offscreenVersion, setJsRendered]);

  // Inline FFT refs for gallery mode
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftOffscreensRef = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftMagCacheGalleryRef = React.useRef<(Float32Array | null)[]>([]);
  const galleryFftDimsRef = React.useRef<{ w: number; h: number } | null>(null);
  const [galleryFftMagVersion, setGalleryFftMagVersion] = React.useState(0);

  // Cached FFT magnitude for single image mode (avoids recomputing on zoom/pan)
  const fftMagCacheRef = React.useRef<Float32Array | null>(null);
  const [fftMagVersion, setFftMagVersion] = React.useState(0);
  // Generation counter for FFT — coalesces rapid ROI drag events to ≤1 FFT/frame
  const fftGenRef = React.useRef(0);

  // Cached FFT offscreen canvas for single mode (avoids reprocessing on zoom/pan)
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  // Caches transformed magnitude + range + stats so contrast slider drag
  // doesn't re-run log/power/findDataRange/autoEnhance on every tick.
  const fftPipelineRef = React.useRef<{
    magnitude: Float32Array;
    displayMin: number;
    displayMax: number;
    magVersion: number;
    scaleMode: string;
    fftAuto: boolean;
  } | null>(null);
  const [fftOffscreenVersion, setFftOffscreenVersion] = React.useState(0);

  // ROI FFT state: when ROI + FFT are both active, compute FFT of cropped ROI region
  const [fftCropDims, setFftCropDims] = React.useState<{ cropWidth: number; cropHeight: number; fftWidth: number; fftHeight: number } | null>(null);

  // Layout calculations
  const isGallery = nImages > 1;
  const showDiffPanel = diffMode && nImages >= 2;
  const diffPanelCount = showDiffPanel ? Math.max(0, nImages - 1) : 0;
  const effectiveNcols = Math.min(ncols, nImages) + diffPanelCount;
  const diffOtherIndices = React.useMemo(
    () => Array.from({ length: nImages }, (_, i) => i).filter(i => i !== diffReference),
    [nImages, diffReference]
  );
  const displayScale = canvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);

  // Initial pan from zoom_row/zoom_col — runs once after first render with valid canvas dims.
  // panX/panY computed so target image (zoomRow, zoomCol) lands at canvas center after transform:
  //   ctx.translate(cx+panX, cy+panY) ⋅ scale(zoom) ⋅ translate(-cx,-cy)
  //   target screen = cx + panX + zoom * (target_canvas - cx) = cx
  //   ⟹ panX = zoom * (cx - target_canvas) = zoom * canvasW * (0.5 - col/width)
  const initialPanAppliedRef = React.useRef(false);
  React.useEffect(() => {
    if (initialPanAppliedRef.current) return;
    if (zoomRowTrait == null && zoomColTrait == null) return;
    if (canvasW <= 0 || canvasH <= 0 || width <= 0 || height <= 0) return;
    const z = initialZoomState.zoom;
    const panX = zoomColTrait != null ? z * canvasW * (0.5 - zoomColTrait / width) : 0;
    const panY = zoomRowTrait != null ? z * canvasH * (0.5 - zoomRowTrait / height) : 0;
    setLinkedZoomState({ zoom: z, panX, panY });
    setZoomStates(prev => {
      const m = new Map(prev);
      for (let i = 0; i < nImages; i++) m.set(i, { zoom: z, panX, panY });
      return m;
    });
    initialPanAppliedRef.current = true;
  }, [zoomRowTrait, zoomColTrait, canvasW, canvasH, width, height, nImages, initialZoomState.zoom]);
  const floatsPerImage = width * height;
  const galleryGridWidth = isGallery ? effectiveNcols * canvasW + (effectiveNcols - 1) * 8 : canvasW;
  const profileCanvasWidth = galleryGridWidth;

  // ROI FFT active: both ROI and FFT on, with a selected ROI
  const roiFftActive = effectiveShowFft && roiActive && roiSelectedIdx >= 0 && roiSelectedIdx < (roiList?.length ?? 0);

  // Stable key for ROI geometry — only changes when the selected ROI's geometry changes,
  // not when other ROIs move or roiList gets a new reference from unrelated edits.
  // Shared by both ROI FFT and preview panel to avoid redundant recomputes.
  const selectedRoiKey = React.useMemo(() => {
    if (!roiList || roiSelectedIdx < 0 || roiSelectedIdx >= roiList.length) return "";
    const r = roiList[roiSelectedIdx];
    return `${r.row},${r.col},${r.radius},${r.radius_inner},${r.width},${r.height},${r.shape}`;
  }, [roiList, roiSelectedIdx]);
  const roiFftKey = roiFftActive ? selectedRoiKey : "";

  // Extract raw float32 bytes and parse into Float32Arrays
  const allFloats = React.useMemo(() => extractFloat32(frameBytes), [frameBytes]);

  // Initialize WebGPU FFT + colormap engine on mount.
  // Sets refs (not state) — no effect re-triggers on GPU init.
  // Effects pick up GPU on their next natural re-run (data/slider change).
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        gpuReadyRef.current = true;
        const info = getGPUInfo();
        console.log(`[Show2D] WebGPU FFT initialized — ${info || "GPU"}`);
      } else {
        console.log("[Show2D] WebGPU unavailable — using CPU Worker fallback");
      }
    });
    getGPUColormapEngine().then(engine => {
      if (engine) {
        gpuCmapRef.current = engine;
        gpuCmapReadyRef.current = true;
        console.log("[Show2D] WebGPU colormap engine initialized");
        // Report GPU memory to Python for auto-bin budget
        getGPUMaxBufferSize().then(bytes => {
          if (bytes > 0) setGpuMaxBufferMB(Math.floor(bytes / (1024 * 1024)));
        });
        // Upload data if already parsed (GPU init may be slower than data arrival).
        // Do NOT call setState — that would re-trigger effects and cause double
        // computation. Instead, upload data and do a warm-up render via rAF.
        // This compiles the GPU pipeline in the background so the first user
        // interaction is fast (~100ms instead of ~750ms cold start).
        if (rawDataRef.current && rawDataRef.current.length > 0) {
          const nImg = rawDataRef.current.length;
          for (let i = 0; i < nImg; i++) {
            const d = rawDataRef.current[i];
            if (d) engine.uploadData(i, d, width, height);
          }
          const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
          engine.uploadLUT(cmap, lut);
          gpuDataVersionRef.current++;
          // Warm-up: render once to compile GPU pipeline + fill canvases.
          // Uses full data range (no slider adjustment) for the initial frame.
          requestAnimationFrame(async () => {
            const offscreens = mainOffscreensRef.current;
            const imgDatas = mainImgDatasRef.current;
            if (offscreens.length === 0 || imgDatas.length === 0) return;
            const cachedRanges = dataRangesRef.current;
            if (cachedRanges.length === 0) return;
            const indices = Array.from({ length: nImg }, (_, i) => i);
            const ranges = cachedRanges.map(r => ({ vmin: r.min, vmax: r.max }));
            const ofs = indices.map(i => offscreens[i] || null);
            const ids = indices.map(i => imgDatas[i] || null);
            const logSc = logScaleRef.current ?? false;
            await engine.renderSlots(indices, ranges, ofs, ids, logSc);
            setOffscreenVersion(v => v + 1);
          });
        }
      }
    });
  }, []);

  const [dataVersion, setDataVersion] = React.useState(0);

  // Keep inline FFT ref arrays in sync with nImages
  React.useEffect(() => {
    fftCanvasRefs.current = fftCanvasRefs.current.slice(0, nImages);
    fftOffscreensRef.current = fftOffscreensRef.current.slice(0, nImages);
  }, [nImages]);

  // FFT of diff (n=2 only). Computes A − B in JS at full image resolution from rawDataRef,
  // feeds to FFT pipeline. Recomputes when raw data changes.
  React.useEffect(() => {
    if (!effectiveShowFft || !showDiffPanel || nImages !== 2) return;
    const raw = rawDataRef.current;
    if (!raw || raw.length < 2 || !raw[0] || !raw[1]) return;
    const a = raw[0], b = raw[1];
    const bytes = new Float32Array(width * height);
    for (let i = 0; i < bytes.length; i++) bytes[i] = a[i] - b[i];
    const canvas = diffFftCanvasRef.current;
    if (!canvas) return;
    const fftW = nextPow2(width), fftH = nextPow2(height);
    const real = new Float32Array(fftW * fftH);
    const imag = new Float32Array(fftW * fftH);
    const src = new Float32Array(bytes);
    if (fftWindow) applyHannWindow2D(src, width, height);
    const padR = Math.floor((fftH - height) / 2), padC = Math.floor((fftW - width) / 2);
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) real[(r + padR) * fftW + c + padC] = src[r * width + c];
    }
    let cancelled = false;
    (async () => {
      // WebGPU primary (matches main + gallery FFT paths). CPU worker fallback
      // for browsers without WebGPU (Safari <17, FF behind flag).
      const result = (gpuFFTRef.current && gpuReadyRef.current)
        ? await gpuFFTRef.current.fft2D(real, imag, fftW, fftH, false)
        : await fft2dAsync(real, imag, fftW, fftH, false);
      if (cancelled) return;
      const mag = computeMagnitude(result.real, result.imag);
      fftshift(mag, fftW, fftH);
      diffFftMagRef.current = mag;
      const { min, max } = autoEnhanceFFT(mag, fftW, fftH);
      const off = renderToOffscreen(mag, fftW, fftH, COLORMAPS[fftColormap] || COLORMAPS.inferno, min, max);
      if (!off) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = fftSmooth;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(off, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
    })();
    return () => { cancelled = true; };
  }, [effectiveShowFft, showDiffPanel, nImages, dataVersion, width, height, fftWindow, fftColormap, canvasW, canvasH, fftSmooth]);

  // Diff panels render — DYNAMIC. One per non-reference image: image[ref] − image[i].
  // Computed at canvas resolution from raw float data, re-running on zoom/pan/align change.
  // For n=2: alignDy/dx applied to non-ref image. For n>2: no align (per-pair align not yet supported).
  React.useEffect(() => {
    if (!showDiffPanel) return;
    const raw = rawDataRef.current;
    if (!raw || raw.length < 2) return;
    const ref = diffReference;
    const a = raw[ref];
    if (!a) return;
    diffOtherIndices.forEach((otherIdx, slot) => {
      renderDiffPanel(slot, a, raw[otherIdx], otherIdx);
    });
    // forEach inlines below — extracted as effect helper.
    function renderDiffPanel(slot: number, refData: Float32Array, otherData: Float32Array | undefined, otherIdx: number) {
    if (!otherData) return;
    const canvas = diffCanvasRefs.current[slot];
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const zs0 = getZoomState(ref);
    const zs1 = getZoomState(otherIdx);
    const useAlign = nImages === 2;
    const adY = useAlign ? alignDy : 0;
    const adX = useAlign ? alignDx : 0;
    const a = refData, b = otherData;
    const cw = canvasW, ch = canvasH;
    const cx = cw / 2, cy = ch / 2;
    const sx = width / cw, sy = height / ch;
    const diff = new Float32Array(cw * ch);
    let mn = Infinity, mx = -Infinity;
    // Smooth: bilinear (slower, sub-pixel correct). !Smooth: nearest neighbor (faster, pixelated).
    const Hm1 = height - 1, Wm1 = width - 1;
    const a_panX = zs0.panX, a_panY = zs0.panY, a_zoom = zs0.zoom;
    const b_panX = zs1.panX, b_panY = zs1.panY, b_zoom = zs1.zoom;
    if (smooth) {
      for (let y = 0; y < ch; y++) {
        const ayu = (y - cy - a_panY) / a_zoom + cy;
        const byu = (y - cy - b_panY) / b_zoom + cy;
        const aRowF = ayu * sy;
        const bRowF = byu * sy - adY;
        const aR0 = aRowF | 0, bR0 = bRowF | 0;
        const aFr = aRowF - aR0, bFr = bRowF - bR0;
        const aRowOOB = aR0 < 0 || aR0 >= Hm1;
        const bRowOOB = bR0 < 0 || bR0 >= Hm1;
        const aRowOff = aR0 * width;
        const bRowOff = bR0 * width;
        const rowOff = y * cw;
        for (let x = 0; x < cw; x++) {
          const axu = (x - cx - a_panX) / a_zoom + cx;
          const bxu = (x - cx - b_panX) / b_zoom + cx;
          const aColF = axu * sx;
          const bColF = bxu * sx - adX;
          const aC0 = aColF | 0, bC0 = bColF | 0;
          let v = 0;
          if (!aRowOOB && !bRowOOB && aC0 >= 0 && aC0 < Wm1 && bC0 >= 0 && bC0 < Wm1) {
            const aFc = aColF - aC0, bFc = bColF - bC0;
            const ai = aRowOff + aC0;
            const bi = bRowOff + bC0;
            const aV = (a[ai] * (1 - aFc) + a[ai + 1] * aFc) * (1 - aFr) +
                       (a[ai + width] * (1 - aFc) + a[ai + width + 1] * aFc) * aFr;
            const bV = (b[bi] * (1 - bFc) + b[bi + 1] * bFc) * (1 - bFr) +
                       (b[bi + width] * (1 - bFc) + b[bi + width + 1] * bFc) * bFr;
            v = aV - bV;
          }
          diff[rowOff + x] = v;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
    } else {
      for (let y = 0; y < ch; y++) {
        const ayu = (y - cy - a_panY) / a_zoom + cy;
        const byu = (y - cy - b_panY) / b_zoom + cy;
        const aRow = (ayu * sy + 0.5) | 0;
        const bRow = (byu * sy - adY + 0.5) | 0;
        const aRowOK = aRow >= 0 && aRow < height;
        const bRowOK = bRow >= 0 && bRow < height;
        const aRowOff = aRow * width;
        const bRowOff = bRow * width;
        const rowOff = y * cw;
        for (let x = 0; x < cw; x++) {
          const axu = (x - cx - a_panX) / a_zoom + cx;
          const bxu = (x - cx - b_panX) / b_zoom + cx;
          const aCol = (axu * sx + 0.5) | 0;
          const bCol = (bxu * sx - adX + 0.5) | 0;
          let v = 0;
          if (aRowOK && bRowOK && aCol >= 0 && aCol < width && bCol >= 0 && bCol < width) {
            v = a[aRowOff + aCol] - b[bRowOff + bCol];
          }
          diff[rowOff + x] = v;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
    }
    const sym = Math.max(Math.abs(mn), Math.abs(mx));
    // Diff is signed-around-zero — use diverging cmap (RdBu) if user picked a sequential one.
    const sequentialCmaps = new Set(["inferno", "viridis", "plasma", "magma", "hot", "gray", "turbo"]);
    const diffCmap = sequentialCmaps.has(cmap) ? "RdBu" : cmap;
    const off = renderToOffscreen(diff, cw, ch, COLORMAPS[diffCmap] || COLORMAPS.RdBu, -sym, sym);
    if (!off) return;
    ctx.imageSmoothingEnabled = smooth;
    if (smooth) ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0);
    }
  }, [showDiffPanel, diffOtherIndices, diffReference, nImages, dataVersion, width, height, cmap, smooth, canvasW, canvasH,
      alignDy, alignDx, getZoomState, linkedZoom, linkPan, linkedZoomState, zoomStates]);

  React.useEffect(() => {
    if (!allFloats || allFloats.length === 0) return;
    const dataArrays: Float32Array[] = [];
    for (let i = 0; i < nImages; i++) {
      const start = i * floatsPerImage;
      const imageData = allFloats.subarray(start, start + floatsPerImage);
      dataArrays.push(new Float32Array(imageData));
    }
    rawDataRef.current = dataArrays;
    // Upload to GPU colormap engine if available (ref check, no state trigger)
    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current) {
      for (let i = 0; i < dataArrays.length; i++) engine.uploadData(i, dataArrays[i], width, height);
      gpuDataVersionRef.current++;
    }
    setDataVersion(v => v + 1);
  }, [allFloats, nImages, floatsPerImage]);

  // Initialize reusable offscreen canvases (one per image, resized when dimensions change)
  React.useEffect(() => {
    if (width <= 0 || height <= 0 || nImages <= 0) return;
    const canvases: HTMLCanvasElement[] = [];
    const imgDatas: ImageData[] = [];
    for (let i = 0; i < nImages; i++) {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      canvases.push(canvas);
      imgDatas.push(canvas.getContext("2d")!.createImageData(width, height));
    }
    mainOffscreensRef.current = canvases;
    mainImgDatasRef.current = imgDatas;
    logBufferRef.current = new Float32Array(width * height);
  }, [width, height, nImages]);

  // Compute histogram data for the displayed image (reflects log scale)
  // GPU path: uses persistent per-slot histogram buffers — no CPU data scan
  // CPU fallback: computeHistogramFromBytes (before GPU ready)
  React.useEffect(() => {
    if (!rawDataRef.current) return;
    const idx = nImages > 1 ? selectedIdx : 0;
    const raw = rawDataRef.current[idx];
    if (!raw) return;

    // Use cached ranges (no CPU findDataRange scan)
    const cachedRaw = rawRangesRef.current[idx];
    const rawRange = cachedRaw || findDataRange(raw); // fallback if cache miss
    const range = logScale
      ? { min: Math.log1p(Math.max(rawRange.min, 0)), max: Math.log1p(Math.max(rawRange.max, 0)) }
      : rawRange;
    setImageDataRange(range);

    const engine = gpuCmapRef.current;
    if (engine && gpuCmapReadyRef.current && engine.slotCount > idx) {
      // GPU histogram — single image, persistent buffers
      engine.computeHistogramWithRange(idx, range.min, range.max, logScale).then(bins => {
        setImageHistogramBins(bins);
        setImageHistogramData(null);
      });
    } else {
      // CPU fallback (before GPU ready)
      const d = logScale ? applyLogScale(raw) : raw;
      setImageHistogramBins(null);
      setImageHistogramData(d);
    }
  }, [allFloats, nImages, floatsPerImage, logScale, selectedIdx]);

  // Prevent page scroll when scrolling on canvases (must use native listener with passive: false)
  // In gallery mode, only block scroll on the selected image (or all if linkedZoom)
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const elements: (HTMLElement | null)[] = isGallery
      ? (linkedZoom
          ? [
              ...imageContainerRefs.current,
              ...(effectiveShowFft ? fftContainerRefs.current : []),
            ]
          : [
              imageContainerRefs.current[selectedIdx],
              ...(effectiveShowFft ? [fftContainerRefs.current[selectedIdx]] : []),
            ])
      : [
          imageContainerRefs.current[0],
          ...(effectiveShowFft ? [singleFftContainerRef.current] : []),
        ];
    elements.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => elements.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [canvasReady, effectiveShowFft, isGallery, selectedIdx, linkedZoom]);

  const gpuDataVersionRef = React.useRef(0);
  // Generation counter for colormap — coalesces rapid slider events to ≤1 render per frame
  // Cached per-image data ranges — only recomputed when data or logScale changes, NOT on slider drag
  const dataRangesRef = React.useRef<{ min: number; max: number }[]>([]);
  // Cached log-transformed data — avoids 12×16M log1p calls per slider tick
  const logDataCacheRef = React.useRef<Float32Array[]>([]);
  // Ref mirrors for async GPU callbacks (avoid stale closures)
  const logScaleRef = React.useRef(logScale);
  logScaleRef.current = logScale;
  const cmapRef = React.useRef(cmap);
  cmapRef.current = cmap;
  // Auto-contrast cache: GPU-computed percentile ranges per image
  const autoContrastCacheRef = React.useRef<{ vmin: number; vmax: number }[]>([]);

  // Cache per-image data ranges (raw AND log) on data change only.
  // Log ranges are derived mathematically: log1p(rawMin), log1p(rawMax).
  // NO applyLogScale here — GPU shader handles log1p per pixel.
  // Log toggle is now free: just pick the right cached ranges.
  const rawRangesRef = React.useRef<{ min: number; max: number }[]>([]);
  React.useEffect(() => {
    if (!rawDataRef.current || rawDataRef.current.length === 0) return;
    const engine = gpuCmapRef.current;
    const nImg = rawDataRef.current.length;

    if (engine && gpuCmapReadyRef.current && engine.slotCount >= nImg) {
      // GPU path: batch compute min/max on GPU (async, updates refs when done)
      const indices = Array.from({ length: nImg }, (_, i) => i);
      engine.computeRangeBatch(indices).then(rawRanges => {
        rawRangesRef.current = rawRanges;
        const logRanges = rawRanges.map(r => ({
          min: Math.log1p(Math.max(r.min, 0)),
          max: Math.log1p(Math.max(r.max, 0)),
        }));
        dataRangesRef.current = logScaleRef.current ? logRanges : rawRanges;
      });
    } else {
      // CPU fallback: scan each image for min/max
      const rawRanges: { min: number; max: number }[] = [];
      for (let i = 0; i < nImg; i++) {
        const rawData = rawDataRef.current[i];
        if (!rawData) { rawRanges.push({ min: 0, max: 1 }); continue; }
        rawRanges.push(findDataRange(rawData));
      }
      rawRangesRef.current = rawRanges;
      const logRanges = rawRanges.map(r => ({
        min: Math.log1p(Math.max(r.min, 0)),
        max: Math.log1p(Math.max(r.max, 0)),
      }));
      dataRangesRef.current = logScale ? logRanges : rawRanges;
    }
    logDataCacheRef.current = rawDataRef.current.slice();
  }, [dataVersion]);

  // When logScale toggles, just swap cached ranges (no data scan)
  React.useEffect(() => {
    if (rawRangesRef.current.length === 0) return;
    const logRanges = rawRangesRef.current.map(r => ({
      min: Math.log1p(Math.max(r.min, 0)),
      max: Math.log1p(Math.max(r.max, 0)),
    }));
    dataRangesRef.current = logScale ? logRanges : rawRangesRef.current;
  }, [logScale]);

  // GPU auto-contrast: batch-compute percentile ranges from GPU histograms.
  // One GPU submission for all images. Caches results for synchronous use in render.
  React.useEffect(() => {
    if (!autoContrast) { autoContrastCacheRef.current = []; return; }
    const engine = gpuCmapRef.current;
    if (!engine || !gpuCmapReadyRef.current || !rawDataRef.current) return;
    const cachedRanges = dataRangesRef.current;
    if (cachedRanges.length === 0) return;
    const ls = logScale;
    const nImg = Math.min(rawDataRef.current.length, engine.slotCount);
    if (nImg === 0) return;

    (async () => {
      const indices = Array.from({ length: nImg }, (_, i) => i);
      const histRanges = indices.map(i => cachedRanges[i] || { min: 0, max: 1 });
      const allBins = await engine.computeHistogramBatch(indices, histRanges, ls);

      const pLow = 2, pHigh = 98;
      const acRanges: { vmin: number; vmax: number }[] = [];
      for (let k = 0; k < allBins.length; k++) {
        const bins = allBins[k];
        const cr = histRanges[k];
        // Percentile from normalized histogram CDF
        let sum = 0;
        for (let b = 0; b < 256; b++) sum += bins[b];
        let binLow = 0, binHigh = 255;
        const targetLow = sum * pLow / 100;
        const targetHigh = sum * pHigh / 100;
        let running = 0;
        for (let b = 0; b < 256; b++) {
          running += bins[b];
          if (running >= targetLow && binLow === 0) binLow = b;
          if (running >= targetHigh) { binHigh = b; break; }
        }
        const range = cr.max - cr.min;
        acRanges.push({ vmin: cr.min + (binLow / 255) * range, vmax: cr.min + (binHigh / 255) * range });
      }
      autoContrastCacheRef.current = acRanges;
      console.log(`[Show2D] GPU auto-contrast: ${nImg} images, ${allBins.length} histograms`);
      setOffscreenVersion(v => v + 1);
    })();
  }, [autoContrast, dataVersion, logScale]);

  // -------------------------------------------------------------------------
  // Data effect: normalize + colormap → reusable offscreen canvases
  // GPU path: runs compute shader for all images in one submission
  // CPU fallback: per-image applyColormap loop
  // (does NOT depend on zoom/pan — avoids recomputing 16M pixels on every pan/zoom)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!dataVersion || !rawDataRef.current || rawDataRef.current.length === 0) return;
    if (mainOffscreensRef.current.length === 0 || mainImgDatasRef.current.length === 0) return;

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    // Compute per-image vmin/vmax from CACHED data ranges (no findDataRange per tick).
    // dataRangesRef is precomputed when data or logScale changes.
    const cachedRanges = dataRangesRef.current;
    const hasAbsoluteRange = traitVmin != null && traitVmax != null;
    const ranges: { vmin: number; vmax: number }[] = [];
    for (let i = 0; i < nImages; i++) {
      let vmin: number, vmax: number;
      const cs = linkedContrast ? linkedContrastState : (contrastStates.get(i) || { vminPct: 0, vmaxPct: 100 });

      // Per-image absolute range (vmins/vmaxs) takes precedence over scalar (vmin/vmax)
      const perI_min = traitVmins && traitVmins[i] != null ? traitVmins[i] : null;
      const perI_max = traitVmaxs && traitVmaxs[i] != null ? traitVmaxs[i] : null;
      const hasPerImage = perI_min != null && perI_max != null;
      const isDiffSlot = false;
      const diffSym = 0;

      let rangeMin: number, rangeMax: number;
      if (isDiffSlot) {
        rangeMin = -diffSym;
        rangeMax = diffSym;
      } else if (hasPerImage) {
        rangeMin = logScale ? Math.log1p(Math.max(perI_min!, 0)) : perI_min!;
        rangeMax = logScale ? Math.log1p(Math.max(perI_max!, 0)) : perI_max!;
      } else if (hasAbsoluteRange) {
        rangeMin = logScale ? Math.log1p(Math.max(traitVmin!, 0)) : traitVmin!;
        rangeMax = logScale ? Math.log1p(Math.max(traitVmax!, 0)) : traitVmax!;
      } else {
        // GPU range compute is async — when cache missing OR collapsed (min==max from race),
        // sync findDataRange on raw data to ensure non-degenerate range.
        let cached = cachedRanges[i];
        if (!cached || cached.min === cached.max) {
          if (rawDataRef.current && rawDataRef.current[i]) {
            cached = findDataRange(rawDataRef.current[i]);
          }
        }
        cached = cached || { min: 0, max: 1 };
        rangeMin = cached.min;
        rangeMax = cached.max;
      }

      if (!hasAbsoluteRange && !hasPerImage && autoContrast) {
        // Auto-contrast: use GPU-precomputed percentile ranges.
        // If GPU cache not ready yet, use full data range as placeholder
        // (GPU auto-contrast effect will fire async and trigger re-render).
        const acCache = autoContrastCacheRef.current[i];
        if (acCache) {
          vmin = acCache.vmin; vmax = acCache.vmax;
        } else {
          vmin = rangeMin; vmax = rangeMax;
        }
      } else if (rangeMin !== rangeMax && (cs.vminPct > 0 || cs.vmaxPct < 100)) {
        ({ vmin, vmax } = sliderRange(rangeMin, rangeMax, cs.vminPct, cs.vmaxPct));
      } else {
        vmin = rangeMin; vmax = rangeMax;
      }
      ranges.push({ vmin, vmax });
    }

    // Cache first image's vmin/vmax for colorbar/lens
    if (ranges.length > 0) {
      colorbarVminRef.current = ranges[0].vmin;
      colorbarVmaxRef.current = ranges[0].vmax;
    }

    // GPU colormap — first-class citizen.
    // Try zero-copy path (OffscreenCanvas → ImageBitmap, no mapAsync).
    // Falls back to renderSlots (mapAsync + putImageData) if zero-copy fails.
    const engine = gpuCmapRef.current;
    const gpuReady = engine && gpuCmapReadyRef.current && engine.slotCount >= nImages;
    if (gpuReady) {
      engine!.uploadLUT(cmap, lut);
      const capturedRanges = ranges.slice();
      const capturedLogScale = logScale;
      const capturedNImages = nImages;
      requestAnimationFrame(async () => {
        const indices = Array.from({ length: capturedNImages }, (_, i) => i);

        // Zero-copy path: GPU → OffscreenCanvas → ImageBitmap → drawImage
        const bitmaps = engine!.renderSlotsToImageBitmap(indices, capturedRanges, capturedLogScale);
        if (bitmaps && bitmaps.length > 0 && bitmaps[0]) {
          for (let i = 0; i < bitmaps.length; i++) {
            const offscreen = mainOffscreensRef.current[i];
            if (!offscreen || !bitmaps[i]) continue;
            const ctx = offscreen.getContext("2d");
            if (ctx) ctx.drawImage(bitmaps[i], 0, 0);
          }
          setOffscreenVersion(v => v + 1);
          return;
        }

        // Fallback: renderSlots (mapAsync + copy to ImageData)
        const offscreens = indices.map(i => mainOffscreensRef.current[i] || null);
        const imgDatas = indices.map(i => mainImgDatasRef.current[i] || null);
        const rendered = await engine!.renderSlots(indices, capturedRanges, offscreens, imgDatas, capturedLogScale);
        if (rendered === 0) {
          for (let i = 0; i < capturedNImages; i++) {
            const offscreen = mainOffscreensRef.current[i];
            const imgData = mainImgDatasRef.current[i];
            if (!offscreen || !imgData) continue;
            const raw = rawDataRef.current?.[i];
            if (!raw) continue;
            const processed = capturedLogScale ? applyLogScale(raw) : raw;
            renderToOffscreenReuse(processed, lut, capturedRanges[i].vmin, capturedRanges[i].vmax, offscreen, imgData);
          }
        }
        setOffscreenVersion(v => v + 1);
      });
    } else {
      // CPU fallback: initial render or no WebGPU
      // CPU must do log transform itself (GPU shader would handle it)
      for (let i = 0; i < nImages; i++) {
        const offscreen = mainOffscreensRef.current[i];
        const imgData = mainImgDatasRef.current[i];
        if (!offscreen || !imgData) continue;
        const raw = rawDataRef.current?.[i];
        if (!raw) continue;
        const processed = logScale ? applyLogScale(raw) : raw;
        renderToOffscreenReuse(processed, lut, ranges[i].vmin, ranges[i].vmax, offscreen, imgData);
      }
      setOffscreenVersion(v => v + 1);
    }
  }, [dataVersion, nImages, width, height, cmap, logScale, autoContrast, linkedContrast, linkedContrastState, contrastStates, traitVmin, traitVmax, traitVmins, traitVmaxs, diffMode]);

  // -------------------------------------------------------------------------
  // Draw effect: zoom/pan changes — cheap, just drawImage from cached offscreens
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    if (mainOffscreensRef.current.length === 0) return;

    for (let i = 0; i < nImages; i++) {
      const canvas = canvasRefs.current[i];
      const offscreen = mainOffscreensRef.current[i];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      ctx.imageSmoothingEnabled = smooth;
      if (smooth) ctx.imageSmoothingQuality = "high";
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const zs = getZoomState(i);
      const { zoom, panX, panY } = zs;

      if (zoom !== 1 || panX !== 0 || panY !== 0) {
        ctx.save();
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        ctx.translate(cx + panX, cy + panY);
        ctx.scale(zoom, zoom);
        ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
      }
    }
  }, [offscreenVersion, nImages, width, height, displayScale, canvasW, canvasH, canvasReady, linkedZoom, linkedZoomState, zoomStates, smooth]);

  // -------------------------------------------------------------------------
  // Render Overlays (scale bar, colorbar, zoom indicator)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let i = 0; i < nImages; i++) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;

      if (scaleBarVisible) {
        const zs = getZoomState(i);
        const unit = pixelSize > 0 ? pixelUnit : "px";
        const pxSize = pixelSize > 0 ? pixelSize : 1;
        drawScaleBarHiDPI(overlay, DPR, zs.zoom, pxSize, unit, width);
      } else {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }

      // Colorbar (single image mode only) — uses cached vmin/vmax from data effect
      if (showColorbar && !isGallery) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        const cssW = overlay.width / DPR;
        const cssH = overlay.height / DPR;
        const vmin = colorbarVminRef.current;
        const vmax = colorbarVmaxRef.current;

        ctx.save();
        ctx.scale(DPR, DPR);
        drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, logScale);
        ctx.restore();
      }

      // ROI overlay — draw all ROIs
      if (roiActive && roiList && roiList.length > 0) {
        const zs = getZoomState(i);
        const { zoom, panX, panY } = zs;
        const cx = canvasW / 2;
        const cy = canvasH / 2;

        // Highlight mask: dim everything outside highlighted ROIs
        const highlightedRois = roiList.filter(r => r.highlight);
        if (highlightedRois.length > 0) {
          ctx.save();
          ctx.scale(DPR, DPR);
          ctx.fillStyle = "rgba(0,0,0,0.6)";
          ctx.fillRect(0, 0, canvasW, canvasH);
          ctx.globalCompositeOperation = "destination-out";
          for (const roi of highlightedRois) {
            const sx = (roi.col * displayScale - cx) * zoom + cx + panX;
            const sy = (roi.row * displayScale - cy) * zoom + cy + panY;
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
              // Re-darken inner ring
              ctx.globalCompositeOperation = "source-over";
              ctx.fillStyle = "rgba(0,0,0,0.6)";
              const sir = roi.radius_inner * displayScale * zoom;
              ctx.beginPath(); ctx.arc(sx, sy, sir, 0, Math.PI * 2); ctx.fill();
              ctx.globalCompositeOperation = "destination-out";
            }
          }
          ctx.restore();
        }

        ctx.save();
        ctx.scale(DPR, DPR);
        for (let ri = 0; ri < roiList.length; ri++) {
          const roi = roiList[ri];
          const isSelected = ri === roiSelectedIdx;
          const screenX = (roi.col * displayScale - cx) * zoom + cx + panX;
          const screenY = (roi.row * displayScale - cy) * zoom + cy + panY;
          const screenRadius = roi.radius * displayScale * zoom;
          const screenW = roi.width * displayScale * zoom;
          const screenH = roi.height * displayScale * zoom;
          const screenRadiusInner = roi.radius_inner * displayScale * zoom;
          const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
          ctx.lineWidth = roi.line_width || 2;
          drawROI(ctx, screenX, screenY, shape, screenRadius, screenW, screenH, roi.color || ROI_COLORS[ri % ROI_COLORS.length], roi.color || ROI_COLORS[ri % ROI_COLORS.length], isSelected && isDraggingROI, screenRadiusInner);
          if (isSelected) {
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1;
            if (shape === "circle" || shape === "annular") {
              ctx.beginPath(); ctx.arc(screenX, screenY, screenRadius + 3, 0, Math.PI * 2); ctx.stroke();
            } else if (shape === "square") {
              ctx.strokeRect(screenX - screenRadius - 3, screenY - screenRadius - 3, (screenRadius + 3) * 2, (screenRadius + 3) * 2);
            } else if (shape === "rectangle") {
              ctx.strokeRect(screenX - screenW / 2 - 3, screenY - screenH / 2 - 3, screenW + 6, screenH + 6);
            }
            ctx.setLineDash([]);
          }
        }
        ctx.restore();
      }

      // Line profile overlay
      if (profileActive && profilePoints.length > 0) {
        const zs = getZoomState(i);
        const { zoom, panX, panY } = zs;
        ctx.save();
        ctx.scale(DPR, DPR);

        // Transform image coords to screen coords
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toScreenX = (ix: number) => (ix * displayScale - cx) * zoom + cx + panX;
        const toScreenY = (iy: number) => (iy * displayScale - cy) * zoom + cy + panY;

        // Draw point A
        const ax = toScreenX(profilePoints[0].col);
        const ay = toScreenY(profilePoints[0].row);
        ctx.fillStyle = themeColors.accent;
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw line and point B if complete
        if (profilePoints.length === 2) {
          const bx = toScreenX(profilePoints[1].col);
          const by = toScreenY(profilePoints[1].row);

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

        ctx.restore();
      }

      // Distance measurement overlay
      if (measureActive && measurePoints.length >= 1) {
        const zs = getZoomState(i);
        const { zoom, panX, panY } = zs;
        ctx.save();
        ctx.scale(DPR, DPR);
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        const toSX = (ix: number) => (ix * displayScale - cx) * zoom + cx + panX;
        const toSY = (iy: number) => (iy * displayScale - cy) * zoom + cy + panY;

        ctx.shadowColor = "rgba(0,0,0,0.6)";
        ctx.shadowBlur = 3;

        // Endpoint A
        const ax = toSX(measurePoints[0].col);
        const ay = toSY(measurePoints[0].row);
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();

        if (measurePoints.length === 2) {
          const bx = toSX(measurePoints[1].col);
          const by = toSY(measurePoints[1].row);

          // Solid white line (distinct from profile's dashed accent line)
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();

          // Endpoint B
          ctx.beginPath();
          ctx.arc(bx, by, 4, 0, Math.PI * 2);
          ctx.fill();

          // Distance label
          const dc = measurePoints[1].col - measurePoints[0].col;
          const dr = measurePoints[1].row - measurePoints[0].row;
          const distPx = Math.sqrt(dc * dc + dr * dr);
          let label: string;
          if (pixelSize > 0) {
            const distA = distPx * pixelSize;
            label = distA >= 10 ? `${(distA / 10).toFixed(2)} nm` : `${distA.toFixed(2)} Å`;
          } else {
            label = `${distPx.toFixed(1)} px`;
          }

          const mx = (ax + bx) / 2;
          const my = (ay + by) / 2;
          ctx.font = "bold 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillStyle = "#fff";
          ctx.fillText(label, mx, my - 8);
        }

        ctx.shadowBlur = 0;
        ctx.restore();
      }
    }
  }, [nImages, pixelSize, scaleBarVisible, selectedIdx, isGallery, canvasW, canvasH, width, displayScale, linkedZoom, linkedZoomState, zoomStates, dataVersion, showColorbar, cmap, offscreenVersion, logScale, profileActive, profilePoints, roiActive, roiList, roiSelectedIdx, isDraggingROI, themeColors, measureActive, measurePoints]);

  // -------------------------------------------------------------------------
  // Inset magnifier (lens) — renders magnified region at cursor in bottom-left
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const lensCanvas = lensCanvasRef.current;
    if (lensCanvas) {
      const lctx = lensCanvas.getContext("2d");
      if (lctx) lctx.clearRect(0, 0, lensCanvas.width, lensCanvas.height);
    }
    if (!showLens || isGallery || !lensPos || !rawDataRef.current?.[0]) return;
    if (!lensCanvas) return;
    const ctx = lensCanvas.getContext("2d");
    if (!ctx) return;

    const raw = rawDataRef.current[0];
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    // Use cached vmin/vmax from data effect (avoids full-image applyLogScale + findDataRange)
    const vmin = colorbarVminRef.current;
    const vmax = colorbarVmaxRef.current;

    // Extract region around cursor — regionSize = displaySize / magnification
    const regionSize = Math.max(4, Math.round(lensDisplaySize / lensMag));
    const lensSize = lensDisplaySize;
    const margin = 12;
    const half = Math.floor(regionSize / 2);
    const r0 = lensPos.row - half;
    const c0 = lensPos.col - half;

    // Create small offscreen canvas for the region
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
          // Apply log scale inline per-pixel (only for the small region, not full image)
          const rawVal = raw[sr * width + sc];
          const val = logScale ? Math.log1p(rawVal) : rawVal;
          const t = Math.max(0, Math.min(1, (val - vmin) / range));
          const li = Math.round(t * 255);
          imgData.data[idx] = lut[li * 3]; imgData.data[idx + 1] = lut[li * 3 + 1]; imgData.data[idx + 2] = lut[li * 3 + 2]; imgData.data[idx + 3] = 255;
        }
      }
    }
    rctx.putImageData(imgData, 0, 0);

    // Draw lens inset on overlay — use custom anchor or default bottom-left
    ctx.save();
    ctx.scale(DPR, DPR);
    const lx = lensAnchor ? lensAnchor.x : margin;
    const ly = lensAnchor ? lensAnchor.y : canvasH - lensSize - margin - 20;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(regionCanvas, lx, ly, lensSize, lensSize);
    ctx.strokeStyle = themeColors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, lensSize, lensSize);
    // Crosshair at center
    const cx = lx + lensSize / 2;
    const cy = ly + lensSize / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    // Magnification label
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "10px monospace";
    ctx.fillText(`${lensMag}×`, lx + 4, ly + lensSize - 4);
    ctx.restore();
  }, [showLens, lensPos, isGallery, cmap, logScale, offscreenVersion, width, height, canvasH, themeColors, lensMag, lensDisplaySize, lensAnchor]);

  // -------------------------------------------------------------------------
  // Auto-compute profile when profile_line is set (e.g. from Python)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (profilePoints.length === 2 && rawDataRef.current) {
      const p0 = profilePoints[0], p1 = profilePoints[1];
      const allProfiles: (Float32Array | null)[] = [];
      for (let i = 0; i < rawDataRef.current.length; i++) {
        const raw = rawDataRef.current[i];
        allProfiles.push(raw ? sampleLineProfile(raw, width, height, p0.row, p0.col, p1.row, p1.col) : null);
      }
      setProfileDataAll(allProfiles);
      if (!profileActive) setProfileActive(true);
    }
  }, [profilePoints, dataVersion, profileActive]);

  // -------------------------------------------------------------------------
  // Render sparkline for line profile
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = profileCanvasWidth;
    const cssH = profileHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const isDark = themeInfo.theme === "dark";
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f0f0f0";
    ctx.fillRect(0, 0, cssW, cssH);

    const hasData = profileDataAll.some(d => d && d.length >= 2);
    if (!hasData) {
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

    // Find global min/max across all profiles
    let gMin = Infinity, gMax = -Infinity;
    for (const d of profileDataAll) {
      if (!d) continue;
      for (let i = 0; i < d.length; i++) {
        if (d[i] < gMin) gMin = d[i];
        if (d[i] > gMax) gMax = d[i];
      }
    }
    const range = gMax - gMin || 1;

    // Draw each profile
    const colors = profileDataAll.length === 1 ? [themeColors.accent] : PROFILE_COLORS;
    for (let pIdx = 0; pIdx < profileDataAll.length; pIdx++) {
      const d = profileDataAll[pIdx];
      if (!d || d.length < 2) continue;
      ctx.strokeStyle = colors[pIdx % colors.length];
      ctx.lineWidth = pIdx === selectedIdx || profileDataAll.length === 1 ? 1.5 : 1;
      ctx.globalAlpha = pIdx === selectedIdx || profileDataAll.length === 1 ? 1 : 0.5;
      ctx.beginPath();
      for (let i = 0; i < d.length; i++) {
        const x = padLeft + (i / (d.length - 1)) * plotW;
        const y = padTop + plotH - ((d[i] - gMin) / range) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Compute total distance for x-axis
    const firstProfile = profileDataAll.find(d => d);
    let totalDist = (firstProfile?.length ?? 2) - 1;
    let xUnit = "px";
    if (profilePoints.length === 2) {
      const dx = profilePoints[1].col - profilePoints[0].col;
      const dy = profilePoints[1].row - profilePoints[0].row;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (pixelSize > 0) {
        totalDist = distPx * pixelSize;
        xUnit = pixelUnit;
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

    // Draw y-axis min/max labels
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

    // Legend (gallery mode with multiple images)
    if (profileDataAll.length > 1) {
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.font = "9px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      let legendX = cssW - 4;
      for (let pIdx = profileDataAll.length - 1; pIdx >= 0; pIdx--) {
        if (!profileDataAll[pIdx]) continue;
        const label = labels?.[pIdx] || `#${pIdx + 1}`;
        const color = colors[pIdx % colors.length];
        const textW = ctx.measureText(label).width;
        ctx.globalAlpha = pIdx === selectedIdx ? 1 : 0.5;
        ctx.fillStyle = color;
        ctx.fillRect(legendX - textW - 10, 2, 6, 6);
        ctx.fillStyle = isDark ? "#aaa" : "#555";
        ctx.fillText(label, legendX, 1);
        legendX -= textW + 16;
      }
      ctx.globalAlpha = 1;
    }

    // Save base rendering + layout for hover overlay
    profileBaseImageRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    profileLayoutRef.current = { padLeft, plotW, padTop, plotH, gMin, gMax, totalDist, xUnit };
  }, [profileDataAll, themeInfo.theme, themeColors.accent, profilePoints, pixelSize, selectedIdx, labels, profileCanvasWidth, profileHeight]);

  // Profile hover handler — draws crosshair + value readout
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

    // Restore base image
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

    // Dot on each profile line + collect values
    const colors = profileDataAll.length === 1 ? [themeColors.accent] : PROFILE_COLORS;
    const activeIdx = isGallery ? selectedIdx : 0;
    let displayVal: number | null = null;
    for (let pIdx = 0; pIdx < profileDataAll.length; pIdx++) {
      const d = profileDataAll[pIdx];
      if (!d || d.length < 2) continue;
      const dataIdx = Math.min(d.length - 1, Math.max(0, Math.round(frac * (d.length - 1))));
      const val = d[dataIdx];
      const y = padTop + plotH - ((val - gMin) / range) * plotH;
      ctx.fillStyle = colors[pIdx % colors.length];
      ctx.globalAlpha = pIdx === activeIdx || profileDataAll.length === 1 ? 1 : 0.5;
      ctx.beginPath();
      ctx.arc(cssX, y, 3, 0, Math.PI * 2);
      ctx.fill();
      if (pIdx === activeIdx || profileDataAll.length === 1) displayVal = val;
    }
    ctx.globalAlpha = 1;

    // Value readout label
    if (displayVal !== null) {
      const dist = frac * totalDist;
      const label = `${formatNumber(displayVal)}  @  ${dist.toFixed(1)} ${xUnit}`;
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
  }, [profileDataAll, themeInfo.theme, themeColors.accent, isGallery, selectedIdx]);

  const handleProfileMouseLeave = React.useCallback(() => {
    const canvas = profileCanvasRef.current;
    const base = profileBaseImageRef.current;
    if (!canvas || !base) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(base, 0, 0);
  }, []);

  // -------------------------------------------------------------------------
  // Compute FFT magnitude (cached — only recomputes when data changes)
  // Supports ROI-scoped FFT: when ROI is active with a selected ROI, compute
  // FFT of the cropped region instead of the full image.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || isGallery || !rawDataRef.current) return;
    if (!rawDataRef.current[selectedIdx]) return;
    // Generation counter: coalesces rapid ROI drag events so at most one
    // FFT runs per animation frame. The rAF yield lets the browser paint
    // the ROI position update before the (potentially blocking) FFT runs.
    const gen = ++fftGenRef.current;

    const doCompute = async () => {
      // Yield to next animation frame — browser paints updated ROI first,
      // and stale requests (from earlier drag events) are discarded below.
      await new Promise<void>(r => requestAnimationFrame(() => r()));
      if (gen !== fftGenRef.current) return;

      // Wait for WebGPU init if it's still in flight — avoids first-call CPU race.
      if (!gpuReadyRef.current) {
        try {
          const fft = await getWebGPUFFT();
          if (fft) { gpuFFTRef.current = fft; gpuReadyRef.current = true; }
        } catch (_e) { /* fall to CPU */ }
        if (gen !== fftGenRef.current) return;
      }
      const backend = gpuFFTRef.current && gpuReadyRef.current ? "WebGPU" : "CPU Worker";
      setFftComputing(true);
      setFftProgress(`Computing FFT… (${backend})`);
      const t0 = performance.now();
      const data = rawDataRef.current![selectedIdx];
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

      const tCrop = performance.now();
      const real = inputData.slice();
      const imag = new Float32Array(inputData.length);

      if (gpuFFTRef.current && gpuReadyRef.current) {
        const result = await gpuFFTRef.current.fft2D(real, imag, fftW, fftH, false);
        if (gen !== fftGenRef.current) return;
        const tGpu = performance.now();
        fftshift(result.real, fftW, fftH);
        fftshift(result.imag, fftW, fftH);
        fftMagCacheRef.current = computeMagnitude(result.real, result.imag);
        console.log(`[Show2D FFT] GPU ${fftW}×${fftH}: crop=${(tCrop-t0).toFixed(1)}ms gpu=${(tGpu-tCrop).toFixed(1)}ms post=${(performance.now()-tGpu).toFixed(1)}ms`);
      } else {
        // CPU fallback: run in Web Worker to avoid blocking the main thread
        const result = await fft2dAsync(real, imag, fftW, fftH, false);
        if (gen !== fftGenRef.current) return;
        fftMagCacheRef.current = result.magnitude;
        console.log(`[Show2D FFT] Worker ${fftW}×${fftH}: crop=${(tCrop-t0).toFixed(1)}ms worker=${(performance.now()-tCrop).toFixed(1)}ms`);
      }
      // Track FFT dimensions when they differ from image dimensions (ROI crop or non-pow2 padding)
      if (origCropW > 0) {
        setFftCropDims({ cropWidth: origCropW, cropHeight: origCropH, fftWidth: fftW, fftHeight: fftH });
      } else if (fftW !== width || fftH !== height) {
        setFftCropDims({ cropWidth: width, cropHeight: height, fftWidth: fftW, fftHeight: fftH });
      } else {
        setFftCropDims(null);
      }
      setFftMagVersion(v => v + 1);
      setFftComputing(false);
      setFftProgress("");
    };

    doCompute();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveShowFft, isGallery, selectedIdx, width, height, dataVersion, roiFftKey, fftWindow]);

  // Clear FFT measurement when image, FFT state, or ROI changes
  React.useEffect(() => { setFftClickInfo(null); }, [selectedIdx, effectiveShowFft, roiFftActive, roiSelectedIdx]);

  // -------------------------------------------------------------------------
  // FFT data effect: normalize + colormap → cached offscreen canvas
  // (does NOT depend on fftZoom/fftPanX/fftPanY — avoids reprocessing on zoom/pan)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || isGallery || !fftMagCacheRef.current) return;

    const fftMag = fftMagCacheRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Use crop dimensions when ROI FFT is active
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;

    // Heavy steps (log/power transform, range, stats, histogram-data copy) only
    // when source magnitude OR scale-mode changed — NOT on every contrast slider tick.
    // Cached values live in fftPipelineRef for cheap re-renders.
    const sourceChanged = (
      fftPipelineRef.current?.magVersion !== fftMagVersion ||
      fftPipelineRef.current?.scaleMode !== fftScaleMode ||
      fftPipelineRef.current?.fftAuto !== fftAuto
    );
    if (sourceChanged) {
      const magnitude = new Float32Array(fftMag.length);
      for (let i = 0; i < fftMag.length; i++) {
        if (fftScaleMode === "log") magnitude[i] = Math.log1p(fftMag[i]);
        else if (fftScaleMode === "power") magnitude[i] = Math.pow(fftMag[i], 0.5);
        else magnitude[i] = fftMag[i];
      }
      let displayMin: number, displayMax: number;
      if (fftAuto) ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, fftW, fftH));
      else ({ min: displayMin, max: displayMax } = findDataRange(magnitude));
      const { mean, std } = computeStats(magnitude);
      setFftStats([mean, displayMin, displayMax, std]);
      setFftHistogramData(magnitude);  // no .slice() — magnitude is fresh
      setFftDataRange({ min: displayMin, max: displayMax });
      fftPipelineRef.current = { magnitude, displayMin, displayMax, magVersion: fftMagVersion, scaleMode: fftScaleMode, fftAuto };
    }

    const cache = fftPipelineRef.current!;
    const { vmin, vmax } = sliderRange(cache.displayMin, cache.displayMax, fftVminPct, fftVmaxPct);

    // GPU colormap path for FFT — uses dedicated slot at index nImages.
    // Uploads magnitude only when source changed; contrast/cmap drag triggers cheap re-render.
    const engine = gpuCmapRef.current;
    const fftSlot = nImages;  // dedicate slot just past main image slots
    if (engine && gpuCmapReadyRef.current) {
      try {
        if (sourceChanged) engine.uploadData(fftSlot, cache.magnitude, fftW, fftH);
        engine.uploadLUT(fftColormap, lut);
        const bitmaps = engine.renderSlotsToImageBitmap([fftSlot], [{ vmin, vmax }], false);
        if (bitmaps && bitmaps[0]) {
          const oc = fftOffscreenRef.current && fftOffscreenRef.current.width === fftW && fftOffscreenRef.current.height === fftH
            ? fftOffscreenRef.current
            : Object.assign(document.createElement("canvas"), { width: fftW, height: fftH });
          const ctx = oc.getContext("2d");
          if (ctx) {
            ctx.drawImage(bitmaps[0], 0, 0);
            fftOffscreenRef.current = oc;
            setFftOffscreenVersion(v => v + 1);
            return;
          }
        }
      } catch (_e) { /* fall through to CPU */ }
    }
    // CPU fallback
    const offscreen = renderToOffscreen(cache.magnitude, fftW, fftH, lut, vmin, vmax);
    if (!offscreen) return;
    fftOffscreenRef.current = offscreen;
    setFftOffscreenVersion(v => v + 1);
  }, [effectiveShowFft, isGallery, fftMagVersion, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftAuto, width, height, fftCropDims, nImages]);

  // -------------------------------------------------------------------------
  // FFT draw effect: cheap drawImage from cached offscreen (zoom/pan changes)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || isGallery || !fftCanvasRef.current || !fftOffscreenRef.current) return;

    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const offscreen = fftOffscreenRef.current;
    const fftW = offscreen.width;
    const fftH = offscreen.height;

    // Use bilinear smoothing when FFT is smaller than canvas (avoids blocky upscaling)
    ctx.imageSmoothingEnabled = fftSmooth || (fftW < canvasW || fftH < canvasH);
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.save();

    const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;

    ctx.translate(centerOffsetX, centerOffsetY);
    ctx.scale(fftZoom, fftZoom);
    // Stretch cropped FFT to fill the full canvas (no layout change during drag)
    ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
    ctx.restore();
  }, [effectiveShowFft, isGallery, fftOffscreenVersion, canvasW, canvasH, fftZoom, fftPanX, fftPanY, fftSmooth]);

  // -------------------------------------------------------------------------
  // Render FFT overlay (scale bar + colorbar + d-spacing marker)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const overlay = fftOverlayRef.current;
    if (!overlay || !effectiveShowFft || isGallery) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Use crop dimensions for reciprocal-space calculations
    const fftW = fftCropDims?.fftWidth ?? width;

    // FFT colorbar
    if (fftShowColorbar && fftDataRange.min !== fftDataRange.max) {
      const { vmin, vmax } = sliderRange(fftDataRange.min, fftDataRange.max, fftVminPct, fftVmaxPct);
      const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
      ctx.save();
      ctx.scale(DPR, DPR);
      const cssW = overlay.width / DPR;
      const cssH = overlay.height / DPR;
      drawColorbar(ctx, cssW, cssH, lut, vmin, vmax, fftScaleMode === "log");
      ctx.restore();
    }

    // D-spacing crosshair marker — use crop dims for coordinate mapping
    const fftH = fftCropDims?.fftHeight ?? height;
    if (fftClickInfo) {
      ctx.save();
      ctx.scale(DPR, DPR);
      const centerOffsetX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
      const centerOffsetY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
      const screenX = centerOffsetX + fftZoom * (fftClickInfo.col / fftW * canvasW);
      const screenY = centerOffsetY + fftZoom * (fftClickInfo.row / fftH * canvasH);
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
  }, [effectiveShowFft, isGallery, fftClickInfo, canvasW, canvasH, fftZoom, fftPanX, fftPanY, width, height, pixelSize, fftDataRange, fftVminPct, fftVmaxPct, fftColormap, fftScaleMode, fftShowColorbar, fftCropDims]);

  // -------------------------------------------------------------------------
  // Compute FFT magnitudes for gallery mode (cache raw magnitudes)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!effectiveShowFft || !isGallery || !rawDataRef.current) return;
    if (rawDataRef.current.length === 0) return;
    let cancelled = false;

    const computeAllFFTs = async () => {
      // Wait for WebGPU init if it's still in flight — avoids first-call CPU race.
      if (!gpuReadyRef.current) {
        try {
          const fft = await getWebGPUFFT();
          if (fft) { gpuFFTRef.current = fft; gpuReadyRef.current = true; }
        } catch (_e) { /* fall to CPU */ }
        if (cancelled) return;
      }
      // Initialize cache; preserve existing entries (only recompute missing)
      if (fftMagCacheGalleryRef.current.length !== nImages) {
        fftMagCacheGalleryRef.current = new Array(nImages).fill(null);
      }
      setFftComputing(true);
      const useGPU = !!(gpuFFTRef.current && gpuReadyRef.current);
      const backend = useGPU ? "WebGPU" : "CPU Worker";
      setFftProgress(`FFT (${backend})`);
      await new Promise<void>(r => requestAnimationFrame(() => r()));
      if (cancelled) { setFftComputing(false); return; }

      const useRoiCrop = roiFftActive && roiList && roiSelectedIdx >= 0 && roiSelectedIdx < roiList.length;
      const roi = useRoiCrop ? roiList[roiSelectedIdx] : null;
      const t0 = performance.now();

      // Helper: prep one image for FFT (crop, pad, window)
      const prepOne = (idx: number): { real: Float32Array; imag: Float32Array; w: number; h: number } | null => {
        const data = rawDataRef.current![idx];
        if (!data) return null;
        let inputData = data;
        let curW = width, curH = height;
        if (roi) {
          const crop = cropROIRegion(data, width, height, roi);
          if (crop) {
            if (fftWindow) applyHannWindow2D(crop.cropped, crop.cropW, crop.cropH);
            const padW = nextPow2(crop.cropW), padH = nextPow2(crop.cropH);
            const padded = new Float32Array(padW * padH);
            for (let y = 0; y < crop.cropH; y++)
              for (let x = 0; x < crop.cropW; x++)
                padded[y * padW + x] = crop.cropped[y * crop.cropW + x];
            inputData = padded; curW = padW; curH = padH;
          }
        } else {
          const padW = nextPow2(curW), padH = nextPow2(curH);
          if (padW !== curW || padH !== curH) {
            const padded = new Float32Array(padW * padH);
            for (let y = 0; y < curH; y++)
              for (let x = 0; x < curW; x++)
                padded[y * padW + x] = inputData[y * curW + x];
            inputData = padded; curW = padW; curH = padH;
          }
        }
        return { real: inputData.slice(), imag: new Float32Array(inputData.length), w: curW, h: curH };
      };

      // ── Prep all images ──
      const inputs: { real: Float32Array; imag: Float32Array }[] = [];
      let fftW = width, fftH = height;
      for (let idx = 0; idx < nImages; idx++) {
        const input = prepOne(idx);
        if (input) {
          fftW = input.w; fftH = input.h;
          inputs.push({ real: input.real, imag: input.imag });
        } else {
          inputs.push({ real: new Float32Array(0), imag: new Float32Array(0) });
        }
      }
      galleryFftDimsRef.current = { w: fftW, h: fftH };
      const tPrep = performance.now() - t0;
      if (cancelled) { setFftComputing(false); return; }

      // ── Batched progressive FFT: batch BATCH_SIZE at a time, display after each batch ──
      const BATCH_SIZE = 4;
      const tFFT0 = performance.now();
      for (let batchStart = 0; batchStart < nImages; batchStart += BATCH_SIZE) {
        if (cancelled) { setFftComputing(false); return; }
        const batchEnd = Math.min(batchStart + BATCH_SIZE, nImages);
        const batchInputs = inputs.slice(batchStart, batchEnd).filter(inp => inp.real.length > 0);
        setFftProgress(`FFT ${batchStart + 1}–${batchEnd}/${nImages} (${backend})`);

        if (useGPU && batchInputs.length > 1) {
          // GPU batch: one submission for BATCH_SIZE images
          const batchResults = await gpuFFTRef.current!.fft2DBatch(batchInputs, fftW, fftH);
          if (cancelled) { setFftComputing(false); return; }
          let ri = 0;
          for (let idx = batchStart; idx < batchEnd; idx++) {
            if (inputs[idx].real.length === 0) continue;
            fftshift(batchResults[ri].real, fftW, fftH);
            fftshift(batchResults[ri].imag, fftW, fftH);
            fftMagCacheGalleryRef.current[idx] = computeMagnitude(batchResults[ri].real, batchResults[ri].imag);
            ri++;
          }
        } else {
          // CPU or single image
          for (let idx = batchStart; idx < batchEnd; idx++) {
            if (inputs[idx].real.length === 0) continue;
            if (cancelled) { setFftComputing(false); return; }
            const { real, imag } = inputs[idx];
            if (useGPU) {
              const result = await gpuFFTRef.current!.fft2D(real, imag, fftW, fftH, false);
              fftshift(result.real, fftW, fftH);
              fftshift(result.imag, fftW, fftH);
              fftMagCacheGalleryRef.current[idx] = computeMagnitude(result.real, result.imag);
            } else {
              fft2d(real, imag, fftW, fftH, false);
              fftshift(real, fftW, fftH);
              fftshift(imag, fftW, fftH);
              fftMagCacheGalleryRef.current[idx] = computeMagnitude(real, imag);
            }
          }
        }
        // Show this batch immediately (progressive top-to-bottom)
        setGalleryFftMagVersion(v => v + 1);
        // Yield to let the browser paint the batch
        await new Promise<void>(r => requestAnimationFrame(() => r()));
      }
      const tFFT = performance.now() - tFFT0;
      const tTotal = performance.now() - t0;
      if (!cancelled) {
        console.log(`[Show2D FFT] Gallery ${nImages}×${fftW}×${fftH}: prep=${tPrep.toFixed(0)}ms fft=${tFFT.toFixed(0)}ms total=${tTotal.toFixed(0)}ms (${backend} batch=${BATCH_SIZE})`);
      }
      setFftComputing(false);
      setFftProgress("");
    };

    computeAllFFTs();

    return () => { cancelled = true; setFftComputing(false); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveShowFft, isGallery, nImages, width, height, dataVersion, roiFftKey, fftWindow]);

  // Gallery FFT data effect: normalize + colormap → cached offscreen canvases
  // (does NOT depend on gallery zoom/pan states)
  const [galleryFftOffscreenVersion, setGalleryFftOffscreenVersion] = React.useState(0);
  React.useEffect(() => {
    if (!effectiveShowFft || !isGallery) return;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const fftW = galleryFftDimsRef.current?.w ?? width;
    const fftH = galleryFftDimsRef.current?.h ?? height;

    for (let idx = 0; idx < nImages; idx++) {
      const magnitude = fftMagCacheGalleryRef.current[idx];
      if (!magnitude) continue;

      // Apply scale transform (same logic as single mode)
      let displayData: Float32Array;
      let displayMin: number, displayMax: number;
      if (fftScaleMode === "log") {
        displayData = applyLogScale(magnitude);
      } else if (fftScaleMode === "power") {
        displayData = new Float32Array(magnitude.length);
        for (let j = 0; j < magnitude.length; j++) displayData[j] = Math.sqrt(magnitude[j]);
      } else {
        displayData = magnitude;
      }
      if (fftAuto) {
        ({ min: displayMin, max: displayMax } = autoEnhanceFFT(magnitude, fftW, fftH));
        if (fftScaleMode === "log") { displayMin = Math.log1p(displayMin); displayMax = Math.log1p(displayMax); }
        else if (fftScaleMode === "power") { displayMin = Math.sqrt(displayMin); displayMax = Math.sqrt(displayMax); }
      } else {
        ({ min: displayMin, max: displayMax } = findDataRange(displayData));
      }
      const fc = fftContrastFor(idx);
      const { vmin, vmax } = sliderRange(displayMin, displayMax, fc.vminPct, fc.vmaxPct);

      const offscreen = renderToOffscreen(displayData, fftW, fftH, lut, vmin, vmax);
      if (!offscreen) continue;
      fftOffscreensRef.current[idx] = offscreen;
    }

    // Update FFT histogram from selected image
    const selMag = fftMagCacheGalleryRef.current[selectedIdx];
    if (selMag) {
      let histData: Float32Array;
      if (fftScaleMode === "log") histData = applyLogScale(selMag);
      else if (fftScaleMode === "power") { histData = new Float32Array(selMag.length); for (let j = 0; j < selMag.length; j++) histData[j] = Math.sqrt(selMag[j]); }
      else histData = selMag;
      setFftHistogramData(histData);
      setFftDataRange(findDataRange(histData));
    }
    setGalleryFftOffscreenVersion(v => v + 1);
  }, [effectiveShowFft, isGallery, nImages, width, height, galleryFftMagVersion, fftColormap, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct, selectedIdx, fftLinkedContrast, fftContrastStates]);

  // Gallery FFT draw effect: cheap drawImage from cached offscreens (zoom/pan changes)
  React.useLayoutEffect(() => {
    if (!effectiveShowFft || !isGallery) return;
    const fftW = galleryFftDimsRef.current?.w ?? width;
    const fftH = galleryFftDimsRef.current?.h ?? height;

    for (let idx = 0; idx < nImages; idx++) {
      const offscreen = fftOffscreensRef.current[idx];
      const canvas = fftCanvasRefs.current[idx];
      if (!offscreen || !canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      const { zoom, panX, panY } = getGalleryFftState(idx);
      ctx.imageSmoothingEnabled = fftSmooth;
      ctx.clearRect(0, 0, canvasW, canvasH);
      ctx.save();
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      ctx.translate(cx + panX, cy + panY);
      ctx.scale(zoom, zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, fftW, fftH, 0, 0, canvasW, canvasH);
      ctx.restore();
    }
  }, [effectiveShowFft, isGallery, nImages, canvasW, canvasH, width, height, galleryFftOffscreenVersion, galleryFftStates, fftLinkedZoom, linkedFftZoomState, fftSmooth]);

  // -------------------------------------------------------------------------
  // Mouse Handlers for Zoom/Pan
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, idx: number) => {
    // In gallery mode, only allow zoom on the selected image (unless linked)
    if (isGallery && idx !== selectedIdx && !linkedZoom) return;
    e.preventDefault(); // Prevent page scroll when zooming

    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    
    // Get current zoom state
    const zs = getZoomState(idx);
    
    // Mouse position relative to canvas (in canvas pixel coordinates)
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    // Canvas center
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    // Mouse position relative to the current view (accounting for pan and zoom)
    // The transformation is: translate(cx + panX, cy + panY) -> scale(zoom) -> translate(-cx, -cy)
    // So a point on screen at (screenX, screenY) maps to image space as:
    // imageX = (screenX - cx - panX) / zoom + cx
    const mouseImageX = (mouseCanvasX - cx - zs.panX) / zs.zoom + cx;
    const mouseImageY = (mouseCanvasY - cy - zs.panY) / zs.zoom + cy;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * zoomFactor));
    
    // Calculate new pan to keep the mouse position fixed on the same image point
    // After zoom: screenX = (imageX - cx) * newZoom + cx + newPanX
    // We want screenX to stay at mouseCanvasX, so:
    // newPanX = mouseCanvasX - (imageX - cx) * newZoom - cx
    const newPanX = mouseCanvasX - (mouseImageX - cx) * newZoom - cx;
    const newPanY = mouseCanvasY - (mouseImageY - cy) * newZoom - cy;

    setZoomState(idx, { zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handleDoubleClick = (idx: number) => {
    setZoomState(idx, initialZoomState);
  };

  // Reset view (zoom/pan only — preserves profile, FFT state, etc.)
  const handleResetAll = () => {
    setZoomStates(new Map());
    setLinkedZoomState(initialZoomState);
    setGalleryFftStates(new Map());
    setLinkedFftZoomState({ zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 });
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
  };

  // FFT zoom/pan — cursor-anchored zoom matching FFT's own canvas transform.
  // FFT render: translate(centerOffsetX, centerOffsetY) → scale(zoom) where
  //   centerOffsetX = (canvasW - canvasW*zoom)/2 + panX
  // Solving for image-space u in [0,1]:
  //   u = (screenX - centerOffsetX) / (zoom * canvasW)
  // After zoom change, keep screenX of mouse at u:
  //   newPanX = mouseX - (canvasW - canvasW*newZoom)/2 - newZoom*u*canvasW
  const handleFftWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = fftCanvasRef.current;
    if (!canvas) {
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      setFftZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor)));
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cw = canvas.width, ch = canvas.height;
    const cOffX = (cw - cw * fftZoom) / 2 + fftPanX;
    const cOffY = (ch - ch * fftZoom) / 2 + fftPanY;
    const u = (mouseX - cOffX) / (fftZoom * cw);
    const v = (mouseY - cOffY) / (fftZoom * ch);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor));
    const newPanX = mouseX - (cw - cw * newZoom) / 2 - newZoom * u * cw;
    const newPanY = mouseY - (ch - ch * newZoom) / 2 - newZoom * v * ch;
    setFftZoom(newZoom);
    setFftPanX(newPanX);
    setFftPanY(newPanY);
  };

  const handleFftDoubleClick = () => {
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
    setFftClickInfo(null);
  };

  // Convert FFT canvas mouse position to FFT image pixel coordinates
  const fftScreenToImg = (e: React.MouseEvent): { col: number; row: number } | null => {
    const canvas = fftCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const cOffX = (canvasW - canvasW * fftZoom) / 2 + fftPanX;
    const cOffY = (canvasH - canvasH * fftZoom) / 2 + fftPanY;
    const fftW = fftCropDims?.fftWidth ?? width;
    const fftH = fftCropDims?.fftHeight ?? height;
    const imgCol = ((mouseX - cOffX) / fftZoom) / canvasW * fftW;
    const imgRow = ((mouseY - cOffY) / fftZoom) / canvasH * fftH;
    if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
      return { col: imgCol, row: imgRow };
    }
    return null;
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY };
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingFftPan || !fftPanStart) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    setFftPanX(fftPanStart.pX + dx);
    setFftPanY(fftPanStart.pY + dy);
  };

  const handleFftMouseUp = (e: React.MouseEvent) => {
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
          // Snap to nearest Bragg spot (local max in FFT magnitude)
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
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  const handleFftMouseLeave = () => {
    fftClickStartRef.current = null;
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  // Gallery FFT zoom/pan handlers (only selected image's FFT responds)
  const handleGalleryFftWheel = (e: React.WheelEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx && !fftLinkedZoom) return;
    e.preventDefault(); // Prevent page scroll when zooming FFT
    const zs = getGalleryFftState(idx);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setGalleryFftState(idx, { ...zs, zoom: Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * zoomFactor)) });
  };

  const handleGalleryFftMouseDown = (e: React.MouseEvent, idx: number) => {
    if (isGallery && idx !== selectedIdx) {
      setSelectedIdx(idx);
      return; // Select first, don't start panning
    }
    const zs = getGalleryFftState(idx);
    setFftPanningIdx(idx);
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleGalleryFftMouseMove = (e: React.MouseEvent, idx: number) => {
    if (!isDraggingFftPan || !fftPanStart || fftPanningIdx !== idx) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    const zs = getGalleryFftState(idx);
    setGalleryFftState(idx, { ...zs, panX: fftPanStart.pX + dx, panY: fftPanStart.pY + dy });
  };

  const handleGalleryFftMouseUp = () => {
    setIsDraggingFftPan(false);
    setFftPanStart(null);
    setFftPanningIdx(null);
  };

  // Track which image is being panned
  const [panningIdx, setPanningIdx] = React.useState<number | null>(null);
  const clickStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const [draggingProfileEndpoint, setDraggingProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isDraggingProfileLine, setIsDraggingProfileLine] = React.useState(false);
  const [hoveredProfileEndpoint, setHoveredProfileEndpoint] = React.useState<0 | 1 | null>(null);
  const [isHoveringProfileLine, setIsHoveringProfileLine] = React.useState(false);
  const profileDragStartRef = React.useRef<{ row: number; col: number; p0: { row: number; col: number }; p1: { row: number; col: number } } | null>(null);

  const screenToImg = (e: React.MouseEvent, idx: number): { imgCol: number; imgRow: number } => {
    const canvas = canvasRefs.current[idx];
    if (!canvas) return { imgCol: 0, imgRow: 0 };
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zs = getZoomState(idx);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    return {
      imgCol: ((mouseCanvasX - cx - zs.panX) / zs.zoom + cx) / displayScale,
      imgRow: ((mouseCanvasY - cy - zs.panY) / zs.zoom + cy) / displayScale,
    };
  };

  const updateAllProfileData = (p0: { row: number; col: number }, p1: { row: number; col: number }) => {
    if (!rawDataRef.current) return;
    const allProfiles: (Float32Array | null)[] = [];
    for (let j = 0; j < rawDataRef.current.length; j++) {
      const raw = rawDataRef.current[j];
      allProfiles.push(raw ? sampleLineProfile(raw, width, height, p0.row, p0.col, p1.row, p1.col) : null);
    }
    setProfileDataAll(allProfiles);
  };

  const updateROI = (e: React.MouseEvent, idx: number) => {
    const { imgCol, imgRow } = screenToImg(e, idx);
    updateSelectedRoi({ col: Math.max(0, Math.min(width - 1, Math.floor(imgCol))), row: Math.max(0, Math.min(height - 1, Math.floor(imgRow))) });
  };

  const hitTestROI = (imgCol: number, imgRow: number): number => {
    if (!roiActive || !roiList) return -1;
    for (let ri = roiList.length - 1; ri >= 0; ri--) {
      const roi = roiList[ri];
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
  };

  const getHitArea = () => {
    const zoom = (getZoomState(selectedIdx)).zoom;
    return RESIZE_HIT_AREA_PX / (displayScale * zoom);
  };

  const isNearEdge = (imgCol: number, imgRow: number, roi: ROIItem): boolean => {
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
  };

  const isNearResizeHandle = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !selectedRoi) return false;
    return isNearEdge(imgCol, imgRow, selectedRoi);
  };

  const isNearAnyEdge = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !roiList) return false;
    return roiList.some(roi => isNearEdge(imgCol, imgRow, roi));
  };

  const isNearResizeHandleInner = (imgCol: number, imgRow: number): boolean => {
    if (!roiActive || !selectedRoi || selectedRoi.shape !== "annular") return false;
    const hitArea = getHitArea();
    const dist = Math.sqrt((imgCol - selectedRoi.col) ** 2 + (imgRow - selectedRoi.row) ** 2);
    return Math.abs(dist - selectedRoi.radius_inner) < hitArea;
  };

  const handleMouseDown = (e: React.MouseEvent, idx: number) => {
    const zs = getZoomState(idx);
    if (isGallery && idx !== selectedIdx) {
      setSelectedIdx(idx);
      // Continue to pan setup so click-drag on unselected panel pans immediately
      // (no double-click required to select first then drag).
    }
    // Check if click is on the lens inset — edge = resize, interior = drag
    if (showLens && !isGallery && idx === 0) {
      const canvas = canvasRefs.current[0];
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const margin = 12;
        const lx = lensAnchor ? lensAnchor.x : margin;
        const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
        if (cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize) {
          const edgeHit = 8;
          const nearEdge = cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit || cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit;
          if (nearEdge) {
            setIsResizingLens(true);
            lensResizeStartRef.current = { my: e.clientY, startSize: lensDisplaySize };
          } else {
            setIsDraggingLens(true);
            lensDragStartRef.current = { mx: e.clientX, my: e.clientY, ax: lx, ay: ly };
          }
          e.preventDefault();
          return;
        }
      }
    }
    clickStartRef.current = { x: e.clientX, y: e.clientY };
    if (profileActive) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      if (profilePoints.length === 2) {
        const p0 = profilePoints[0];
        const p1 = profilePoints[1];
        const hitRadius = 10 / (displayScale * zs.zoom);
        const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
        const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
        if (d0 <= hitRadius || d1 <= hitRadius) {
          setDraggingProfileEndpoint(d0 <= d1 ? 0 : 1);
          setIsDraggingPan(false);
          setPanStart(null);
          setPanningIdx(null);
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
          setPanningIdx(null);
          return;
        }
      }
      setIsDraggingPan(true);
      setPanningIdx(idx);
      setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
      return;
    }
    if (roiActive) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      // Check resize handles on selected ROI first
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
      // Check edge of any ROI — auto-select and start resize
      if (roiList) {
        for (let ri = 0; ri < roiList.length; ri++) {
          if (isNearEdge(imgCol, imgRow, roiList[ri])) {
            e.preventDefault();
            const roi = roiList[ri];
            resizeAspectRef.current = roi && (roi.shape === "rectangle") && roi.width > 0 && roi.height > 0 ? roi.width / roi.height : null;
            setRoiSelectedIdx(ri);
            setIsDraggingResize(true);
            return;
          }
        }
      }
      // Hit-test existing ROIs (click inside to select + drag)
      const hitIdx = hitTestROI(imgCol, imgRow);
      if (hitIdx >= 0) {
        setRoiSelectedIdx(hitIdx);
        setIsDraggingROI(true);
        return;
      }
      // Click on empty space — deselect and allow panning
      setRoiSelectedIdx(-1);
    }
    // Start panning (works in both ROI-active and normal modes)
    {
      setIsDraggingPan(true);
      setPanningIdx(idx);
      setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent, idx: number) => {
    // Fast path: during pan drag, skip all cursor/hover/lens work — just update pan
    if (isDraggingPan && panStart && panningIdx !== null) {
      const canvas = canvasRefs.current[idx];
      if (!canvas || idx !== panningIdx) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      const zs = getZoomState(idx);
      setZoomState(idx, { ...zs, panX: panStart.pX + dx, panY: panStart.pY + dy });
      return;
    }

    // Cursor readout: convert screen position to image pixel coordinates
    const canvas = canvasRefs.current[idx];
    if (canvas && rawDataRef.current) {
      const rect = canvas.getBoundingClientRect();
      const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
      const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
      const zs = getZoomState(idx);
      const cx = canvasW / 2;
      const cy = canvasH / 2;
      const imageCanvasX = (mouseCanvasX - cx - zs.panX) / zs.zoom + cx;
      const imageCanvasY = (mouseCanvasY - cy - zs.panY) / zs.zoom + cy;
      const imgX = Math.floor(imageCanvasX / displayScale);
      const imgY = Math.floor(imageCanvasY / displayScale);
      if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
        const rawData = rawDataRef.current[idx];
        if (rawData) setCursorInfo({ row: imgY, col: imgX, value: rawData[imgY * width + imgX] });
        if (showLens && !isGallery) setLensPos({ row: imgY, col: imgX });
      } else {
        setCursorInfo(null);
        // Don't clear lensPos — lens stays at last position when toggle is on
      }
    }

    // Lens drag
    if (isDraggingLens && lensDragStartRef.current) {
      const dx = e.clientX - lensDragStartRef.current.mx;
      const dy = e.clientY - lensDragStartRef.current.my;
      setLensAnchor({ x: lensDragStartRef.current.ax + dx, y: lensDragStartRef.current.ay + dy });
      return;
    }
    // Lens resize drag
    if (isResizingLens && lensResizeStartRef.current) {
      const dy = e.clientY - lensResizeStartRef.current.my;
      setLensDisplaySize(Math.max(64, Math.min(256, lensResizeStartRef.current.startSize + dy)));
      return;
    }

    if (profileActive && profilePoints.length === 2) {
      const { imgCol, imgRow } = screenToImg(e, idx);
      const p0 = profilePoints[0];
      const p1 = profilePoints[1];
      const activeZoom = linkedZoom ? linkedZoomState.zoom : (zoomStates.get(idx) || initialZoomState).zoom;
      const hitRadius = 10 / (displayScale * activeZoom);
      const d0 = Math.sqrt((imgCol - p0.col) ** 2 + (imgRow - p0.row) ** 2);
      const d1 = Math.sqrt((imgCol - p1.col) ** 2 + (imgRow - p1.row) ** 2);
      if (draggingProfileEndpoint !== null) {
        const clampedRow = Math.max(0, Math.min(height - 1, imgRow));
        const clampedCol = Math.max(0, Math.min(width - 1, imgCol));
        const next = [
          draggingProfileEndpoint === 0 ? { row: clampedRow, col: clampedCol } : profilePoints[0],
          draggingProfileEndpoint === 1 ? { row: clampedRow, col: clampedCol } : profilePoints[1],
        ];
        setProfilePoints(next);
        updateAllProfileData(next[0], next[1]);
        return;
      }
      if (isDraggingProfileLine && profileDragStartRef.current) {
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
        setProfilePoints(next);
        updateAllProfileData(next[0], next[1]);
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

    // ROI resize drag (inner annular ring)
    if (isDraggingResizeInner && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e, idx);
      const newR = Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
      updateSelectedRoi({ radius_inner: Math.max(1, Math.min(selectedRoi.radius - 1, Math.round(newR))) });
      return;
    }
    // ROI resize drag (outer)
    if (isDraggingResize && selectedRoi) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e, idx);
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
        const newR = shape === "square" ? Math.max(Math.abs(ic - selectedRoi.col), Math.abs(ir - selectedRoi.row)) : Math.sqrt((ic - selectedRoi.col) ** 2 + (ir - selectedRoi.row) ** 2);
        const minR = shape === "annular" ? selectedRoi.radius_inner + 1 : 1;
        updateSelectedRoi({ radius: Math.max(minR, Math.round(newR)) });
      }
      return;
    }
    // ROI drag (move center)
    if (isDraggingROI) {
      updateROI(e, idx);
      return;
    }
    // Lens edge hover detection
    if (showLens && !isGallery && canvas) {
      const rect = canvas.getBoundingClientRect();
      const cssX = e.clientX - rect.left;
      const cssY = e.clientY - rect.top;
      const margin = 12;
      const lx = lensAnchor ? lensAnchor.x : margin;
      const ly = lensAnchor ? lensAnchor.y : canvasH - lensDisplaySize - margin - 20;
      const inside = cssX >= lx && cssX <= lx + lensDisplaySize && cssY >= ly && cssY <= ly + lensDisplaySize;
      const edgeHit = 8;
      const nearEdge = inside && (cssX - lx < edgeHit || lx + lensDisplaySize - cssX < edgeHit || cssY - ly < edgeHit || ly + lensDisplaySize - cssY < edgeHit);
      setIsHoveringLensEdge(nearEdge);
    } else {
      setIsHoveringLensEdge(false);
    }
    // Hover detection for resize handles (show cursor on any ROI edge)
    if (roiActive && !isDraggingPan) {
      const { imgCol: ic, imgRow: ir } = screenToImg(e, idx);
      setIsHoveringResizeInner(isNearResizeHandleInner(ic, ir));
      setIsHoveringResize(isNearAnyEdge(ic, ir));
    }

    // Panning
    if (!isDraggingPan || !panStart || panningIdx === null) return;
    if (idx !== panningIdx) return;
    if (!canvas) return;
    const rect2 = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect2.width;
    const scaleY = canvas.height / rect2.height;
    const dx = (e.clientX - panStart.x) * scaleX;
    const dy = (e.clientY - panStart.y) * scaleY;

    const zs = getZoomState(idx);
    setZoomState(idx, { ...zs, panX: panStart.pX + dx, panY: panStart.pY + dy });
  };

  const handleMouseUp = (e: React.MouseEvent, idx: number) => {
    if (isDraggingLens) {
      setIsDraggingLens(false);
      lensDragStartRef.current = null;
      return;
    }
    if (isResizingLens) {
      setIsResizingLens(false);
      lensResizeStartRef.current = null;
      return;
    }
    if (draggingProfileEndpoint !== null || isDraggingProfileLine) {
      setDraggingProfileEndpoint(null);
      setIsDraggingProfileLine(false);
      profileDragStartRef.current = null;
      clickStartRef.current = null;
      setIsDraggingROI(false);
      setIsDraggingResize(false);
      setIsDraggingResizeInner(false);
      setIsDraggingPan(false);
      setPanStart(null);
      setPanningIdx(null);
      setHoveredProfileEndpoint(null);
      setIsHoveringProfileLine(false);
      return;
    }
    // Detect click (vs drag) for profile mode
    if (profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        // It's a click — compute image coordinates
        const canvas = canvasRefs.current[idx];
        if (canvas && rawDataRef.current) {
          const rect = canvas.getBoundingClientRect();
          const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const zs = getZoomState(idx);
          const cx = canvasW / 2;
          const cy = canvasH / 2;
          const imgX = ((mouseCanvasX - cx - zs.panX) / zs.zoom + cx) / displayScale;
          const imgY = ((mouseCanvasY - cy - zs.panY) / zs.zoom + cy) / displayScale;
          if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
            const pt = { row: imgY, col: imgX };
            if (profilePoints.length === 0 || profilePoints.length === 2) {
              // Start new line
              setProfilePoints([pt]);
              setProfileDataAll([]);
            } else {
              // Complete the line
              const p0 = profilePoints[0];
              setProfilePoints([p0, pt]);
              updateAllProfileData(p0, pt);
            }
          }
        }
      }
    }
    // Detect click for measurement mode (only when profile is not active)
    if (measureActive && !profileActive && clickStartRef.current) {
      const dx = e.clientX - clickStartRef.current.x;
      const dy = e.clientY - clickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = canvasRefs.current[idx];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const zs = getZoomState(idx);
          const cx = canvasW / 2;
          const cy = canvasH / 2;
          const imgX = ((mouseCanvasX - cx - zs.panX) / zs.zoom + cx) / displayScale;
          const imgY = ((mouseCanvasY - cy - zs.panY) / zs.zoom + cy) / displayScale;
          if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
            const pt = { row: imgY, col: imgX };
            if (measurePoints.length < 2) {
              setMeasurePoints([...measurePoints, pt]);
            } else {
              setMeasurePoints([pt]);
            }
          }
        }
      }
    }
    clickStartRef.current = null;
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    profileDragStartRef.current = null;
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
  };

  const handleMouseLeave = (idx: number) => {
    setCursorInfo(null);
    // Don't clear lensPos — lens stays at last position when toggle is on
    setIsDraggingLens(false);
    setIsResizingLens(false);
    lensDragStartRef.current = null;
    lensResizeStartRef.current = null;
    setIsHoveringLensEdge(false);
    setIsDraggingROI(false);
    setIsDraggingResize(false);
    setIsDraggingResizeInner(false);
    setDraggingProfileEndpoint(null);
    setIsDraggingProfileLine(false);
    setHoveredProfileEndpoint(null);
    setIsHoveringProfileLine(false);
    profileDragStartRef.current = null;
    setIsHoveringResize(false);
    setIsHoveringResizeInner(false);
    if (panningIdx === idx) {
      setIsDraggingPan(false);
      setPanStart(null);
      setPanningIdx(null);
    }
  };

  // -------------------------------------------------------------------------
  // Copy to clipboard handler
  const handleCopy = React.useCallback(async () => {
    const canvas = canvasRefs.current[isGallery ? selectedIdx : 0];
    if (!canvas) return;
    try {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/png"));
      if (!blob) return;
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    } catch {
      // Fallback: download if clipboard API unavailable
      canvas.toBlob((b) => { if (b) downloadBlob(b, `show2d_${labels?.[selectedIdx] || "image"}.png`); }, "image/png");
    }
  }, [isGallery, selectedIdx, labels]);

  // Export publication-quality figure with scale bar, colorbar, annotations
  const handleExportFigure = React.useCallback((withScaleBar: boolean, withColorbar: boolean) => {
    setExportAnchor(null);
    const idx = isGallery ? selectedIdx : 0;
    const rawData = rawDataRef.current?.[idx];
    if (!rawData) return;

    const processed = logScale ? applyLogScale(rawData) : rawData;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    const hasAbsRange = traitVmin != null && traitVmax != null;
    const rMin = hasAbsRange ? (logScale ? Math.log1p(Math.max(traitVmin!, 0)) : traitVmin!) : imageDataRange.min;
    const rMax = hasAbsRange ? (logScale ? Math.log1p(Math.max(traitVmax!, 0)) : traitVmax!) : imageDataRange.max;
    if (rMin !== rMax && (imageVminPct > 0 || imageVmaxPct < 100)) {
      ({ vmin, vmax } = sliderRange(rMin, rMax, imageVminPct, imageVmaxPct));
    } else if (!hasAbsRange && autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      vmin = rMin;
      vmax = rMax;
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    const figCanvas = exportFigure({
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: pixelSize > 0 ? pixelSize : undefined,
      showColorbar: withColorbar,
      showScaleBar: withScaleBar && pixelSize > 0,
      drawAnnotations: (ctx) => {
        // ROI highlight mask
        if (roiActive && roiList) {
          const hlRois = roiList.filter(r => r.highlight);
          if (hlRois.length > 0) {
            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            ctx.fillRect(0, 0, width, height);
            ctx.globalCompositeOperation = "destination-out";
            for (const roi of hlRois) {
              ctx.fillStyle = "rgba(0,0,0,1)";
              const shape = roi.shape || "circle";
              if (shape === "circle") { ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill(); }
              else if (shape === "square") { ctx.fillRect(roi.col - roi.radius, roi.row - roi.radius, roi.radius * 2, roi.radius * 2); }
              else if (shape === "rectangle") { ctx.fillRect(roi.col - roi.width / 2, roi.row - roi.height / 2, roi.width, roi.height); }
              else if (shape === "annular") {
                ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill();
                ctx.globalCompositeOperation = "source-over";
                ctx.fillStyle = "rgba(0,0,0,0.6)";
                ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius_inner, 0, Math.PI * 2); ctx.fill();
                ctx.globalCompositeOperation = "destination-out";
              }
            }
            ctx.restore();
          }
          // ROI outlines
          for (const roi of roiList) {
            const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
            ctx.lineWidth = roi.line_width || 2;
            drawROI(ctx, roi.col, roi.row, shape, roi.radius, roi.width, roi.height, roi.color, roi.color, false, roi.radius_inner);
          }
        }
        // Profile line
        if (profileActive && profilePoints.length === 2) {
          ctx.strokeStyle = "#4fc3f7";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(profilePoints[0].col, profilePoints[0].row);
          ctx.lineTo(profilePoints[1].col, profilePoints[1].row);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "#4fc3f7";
          ctx.beginPath();
          ctx.arc(profilePoints[0].col, profilePoints[0].row, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(profilePoints[1].col, profilePoints[1].row, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      },
    });

    canvasToPDF(figCanvas).then((blob) => downloadBlob(blob, `show2d_figure_${labels?.[selectedIdx] || "image"}.pdf`));
  }, [isGallery, selectedIdx, labels, width, height, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, pixelSize, title, roiActive, roiList, profileActive, profilePoints]);

  // Export all variants (PNG + PDF) as zip
  const handleExportAll = React.useCallback(async () => {
    setExportAnchor(null);
    const idx = isGallery ? selectedIdx : 0;
    const rawData = rawDataRef.current?.[idx];
    if (!rawData) return;

    const processed = logScale ? applyLogScale(rawData) : rawData;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    let vmin: number, vmax: number;
    const hasAbsRange2 = traitVmin != null && traitVmax != null;
    const rMin2 = hasAbsRange2 ? (logScale ? Math.log1p(Math.max(traitVmin!, 0)) : traitVmin!) : imageDataRange.min;
    const rMax2 = hasAbsRange2 ? (logScale ? Math.log1p(Math.max(traitVmax!, 0)) : traitVmax!) : imageDataRange.max;
    if (rMin2 !== rMax2 && (imageVminPct > 0 || imageVmaxPct < 100)) {
      ({ vmin, vmax } = sliderRange(rMin2, rMax2, imageVminPct, imageVmaxPct));
    } else if (!hasAbsRange2 && autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      vmin = rMin2;
      vmax = rMax2;
    }

    const offscreen = renderToOffscreen(processed, width, height, lut, vmin, vmax);
    if (!offscreen) return;

    const drawAnnotations = (ctx: CanvasRenderingContext2D) => {
      if (roiActive && roiList) {
        const hlRois = roiList.filter(r => r.highlight);
        if (hlRois.length > 0) {
          ctx.save();
          ctx.fillStyle = "rgba(0,0,0,0.6)";
          ctx.fillRect(0, 0, width, height);
          ctx.globalCompositeOperation = "destination-out";
          for (const roi of hlRois) {
            ctx.fillStyle = "rgba(0,0,0,1)";
            const shape = roi.shape || "circle";
            if (shape === "circle") { ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill(); }
            else if (shape === "square") { ctx.fillRect(roi.col - roi.radius, roi.row - roi.radius, roi.radius * 2, roi.radius * 2); }
            else if (shape === "rectangle") { ctx.fillRect(roi.col - roi.width / 2, roi.row - roi.height / 2, roi.width, roi.height); }
            else if (shape === "annular") {
              ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius, 0, Math.PI * 2); ctx.fill();
              ctx.globalCompositeOperation = "source-over";
              ctx.fillStyle = "rgba(0,0,0,0.6)";
              ctx.beginPath(); ctx.arc(roi.col, roi.row, roi.radius_inner, 0, Math.PI * 2); ctx.fill();
              ctx.globalCompositeOperation = "destination-out";
            }
          }
          ctx.restore();
          for (const roi of roiList) {
            const shape = (roi.shape || "circle") as "circle" | "square" | "rectangle" | "annular";
            ctx.lineWidth = roi.line_width || 2;
            drawROI(ctx, roi.col, roi.row, shape, roi.radius, roi.width, roi.height, roi.color, roi.color, false, roi.radius_inner);
          }
        }
      }
      if (profileActive && profilePoints.length === 2) {
        ctx.strokeStyle = "#4fc3f7";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(profilePoints[0].col, profilePoints[0].row);
        ctx.lineTo(profilePoints[1].col, profilePoints[1].row);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#4fc3f7";
        ctx.beginPath(); ctx.arc(profilePoints[0].col, profilePoints[0].row, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(profilePoints[1].col, profilePoints[1].row, 3, 0, Math.PI * 2); ctx.fill();
      }
    };

    const hasScale = pixelSize > 0;
    const baseOpts = {
      imageCanvas: offscreen,
      title: title || undefined,
      lut,
      vmin,
      vmax,
      logScale,
      pixelSize: hasScale ? pixelSize : undefined,
      drawAnnotations,
    };

    const variants: { name: string; showScaleBar: boolean; showColorbar: boolean }[] = [
      { name: "figure", showScaleBar: false, showColorbar: false },
      { name: "figure_scalebar", showScaleBar: true, showColorbar: false },
      { name: "figure_scalebar_colorbar", showScaleBar: true, showColorbar: true },
    ];

    const zip = new JSZip();
    const prefix = `show2d_${labels?.[selectedIdx] || "image"}`;
    const metadata = {
      metadata_version: "1.0",
      widget_name: "Show2D",
      widget_version: widgetVersion || "unknown",
      exported_at: new Date().toISOString(),
      format: "zip",
      export_kind: "figure_variants",
      selected_idx: idx,
      image_shape: { rows: height, cols: width },
      display: {
        cmap,
        log_scale: logScale,
        auto_contrast: autoContrast,
        vmin_pct: imageVminPct,
        vmax_pct: imageVmaxPct,
      },
      variants,
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));

    for (const v of variants) {
      const figCanvas = exportFigure({ ...baseOpts, showScaleBar: v.showScaleBar && hasScale, showColorbar: v.showColorbar });
      const pngBlob = await new Promise<Blob>((resolve) => figCanvas.toBlob((b) => resolve(b!), "image/png"));
      zip.file(`${prefix}_${v.name}.png`, pngBlob);
      const pdfBlob = await canvasToPDF(figCanvas);
      zip.file(`${prefix}_${v.name}.pdf`, pdfBlob);
    }

    const blob = await zip.generateAsync({ type: "blob" });
    downloadBlob(blob, `${prefix}_all.zip`);
  }, [isGallery, selectedIdx, labels, width, height, cmap, logScale, autoContrast, imageDataRange, imageVminPct, imageVmaxPct, pixelSize, title, roiActive, roiList, profileActive, profilePoints, widgetVersion]);

  // Resize Handlers
  // -------------------------------------------------------------------------
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : canvasSize;

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(200, resizeStart.size + delta);
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
      setResizeStart(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, resizeStart]);

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

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Number keys 1-9 select gallery images (avoids arrow key conflicts with Jupyter)
    if (isGallery && e.key >= "1" && e.key <= "9") {
      const idx = parseInt(e.key) - 1;
      if (idx < nImages) { e.preventDefault(); setSelectedIdx(idx); }
      return;
    }
    switch (e.key) {
      case "ArrowLeft":
        if (isGallery) { e.preventDefault(); setSelectedIdx(Math.max(0, selectedIdx - 1)); }
        break;
      case "ArrowRight":
        if (isGallery) { e.preventDefault(); setSelectedIdx(Math.min(nImages - 1, selectedIdx + 1)); }
        break;
      case "r":
      case "R":
        handleResetAll();
        break;
      case "m":
      case "M":
        if (measureActive) {
          setMeasureActive(false);
          setMeasurePoints([]);
        } else {
          setMeasureActive(true);
          setMeasurePoints([]);
        }
        break;
      case "Escape":
        if (measureActive) {
          setMeasureActive(false);
          setMeasurePoints([]);
        }
        break;
      case "]":
        {
          e.preventDefault();
          const rIdx = isGallery ? selectedIdx : 0;
          const rots = [...(imageRotations || [])];
          while (rots.length <= rIdx) rots.push(0);
          rots[rIdx] = (rots[rIdx] + 3) % 4;
          setImageRotations(rots);
        }
        break;
      case "[":
        {
          e.preventDefault();
          const rIdx2 = isGallery ? selectedIdx : 0;
          const rots2 = [...(imageRotations || [])];
          while (rots2.length <= rIdx2) rots2.push(0);
          rots2[rIdx2] = (rots2[rIdx2] + 1) % 4;
          setImageRotations(rots2);
        }
        break;
      case "Delete":
      case "Backspace":
        if (roiActive && roiSelectedIdx >= 0 && roiList && roiSelectedIdx < roiList.length) {
          e.preventDefault();
          const newList = roiList.filter((_, i) => i !== roiSelectedIdx);
          setRoiList(newList);
          setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1);
        }
        break;
    }
  };

  // -------------------------------------------------------------------------
  // Render (Show3D-style layout)
  // -------------------------------------------------------------------------
  const needsReset = getZoomState(isGallery ? selectedIdx : 0).zoom !== 1 || getZoomState(isGallery ? selectedIdx : 0).panX !== 0 || getZoomState(isGallery ? selectedIdx : 0).panY !== 0;
  const statsIdx = isGallery ? selectedIdx : 0;

  // Calibrated cursor position - unit is whatever the user passed via sampling/units.
  const calibratedUnit = pixelSize > 0 ? pixelUnit : "";
  const calibratedFactor = pixelSize;

  return (
    <Box className="show2d-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ p: 2, bgcolor: themeColors.bg, color: themeColors.text, width: "fit-content", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", "& canvas": { display: "block" } }}>
      <Stack direction="row" spacing={`${SPACING.LG}px`} alignItems="flex-start">
        {/* Main panel */}
        <Box sx={{ width: galleryGridWidth, maxWidth: galleryGridWidth }}>
          {/* Title row */}
          <Typography variant="caption" sx={{ ...typography.label, color: themeColors.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
            {title || (isGallery ? "Gallery" : "Image")}
            {displayBinFactor > 1 && (
              <Box component="span" sx={{ ml: 0.5, px: 0.5, py: 0, fontSize: 9, fontWeight: 600, borderRadius: "3px", backgroundColor: themeColors.accent + "22", color: themeColors.accent, border: `1px solid ${themeColors.accent}44` }}>
                {displayBinFactor}× binned
              </Box>
            )}
            {(() => { const rk = (imageRotations?.[isGallery ? selectedIdx : 0] ?? 0) % 4; return rk !== 0 ? (
              <Box
                component="span"
                onClick={() => {
                  const ri = isGallery ? selectedIdx : 0;
                  const rots = [...(imageRotations || [])];
                  while (rots.length <= ri) rots.push(0);
                  rots[ri] = (rots[ri] + 3) % 4;
                  setImageRotations(rots);
                }}
                sx={{ ml: 0.5, color: themeColors.accent, cursor: "pointer", fontSize: "inherit", "&:hover": { opacity: 0.7 } }}
              >
                ({rk * 90}°)
              </Box>
            ) : null; })()}
            <InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT: Show power spectrum (Fourier transform) alongside image.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Profile: Click two points on image to draw a line intensity profile.</Typography>
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>ROI: Region of Interest — click to place, drag to move.</Typography>
              {!isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Lens: Magnifier inset that follows the cursor.</Typography>}
              <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto: Percentile-based contrast (2nd–98th percentile). FFT Auto masks DC + clips to 99.9th.</Typography>
              {isGallery && <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Link Zoom / Contrast: Sync zoom or histogram range across all gallery images.</Typography>}
              <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
              <KeyboardShortcuts items={isGallery ? [["← / →", "Prev / Next image"], ["1 – 9", "Select image"], ["] / [", "Rotate CW / CCW 90°"], ["Del / ⌫", "Delete selected ROI"], ["M", "Measure distance"], ["Esc", "Exit measure"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]] : [["] / [", "Rotate CW / CCW 90°"], ["Del / ⌫", "Delete selected ROI"], ["M", "Measure distance"], ["Esc", "Exit measure"], ["R", "Reset zoom"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
            </Box>} theme={themeInfo.theme} />
          </Typography>
          {/* Controls row: Profile, ROI, Lens, FFT, Export, Reset, Copy */}
          <Stack direction="row" alignItems="center" spacing={`${SPACING.SM}px`} sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            {(
              <>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Profile:</Typography>
                <Switch
                  checked={profileActive}
                 
                  onChange={(e) => {
                    const on = e.target.checked;
                    setProfileActive(on);
                    if (on) {
                      setRoiActive(false);
                    } else {
                      setProfilePoints([]);
                      setProfileDataAll([]);
                      setHoveredProfileEndpoint(null);
                      setIsHoveringProfileLine(false);
                    }
                  }}
                  size="small"
                  sx={switchStyles.small}
                />
              </>
            )}
            {!isGallery && (
              <>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
                <Switch
                  checked={roiActive}
                 
                  onChange={(e) => {
                    const on = e.target.checked;
                    setRoiActive(on);
                    if (on) {
                      setProfileActive(false);
                      setProfilePoints([]);
                      setProfileDataAll([]);
                      setHoveredProfileEndpoint(null);
                      setIsHoveringProfileLine(false);
                    } else {
                      setRoiSelectedIdx(-1);
                    }
                  }}
                  size="small"
                  sx={switchStyles.small}
                />
              </>
            )}
            {(
              <>
                {!isGallery && (
                  <>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Lens:</Typography>
                    <Switch
                      checked={showLens}
                      onChange={() => {
                        if (!showLens) {
                          setShowLens(true);
                          setLensPos({ row: Math.floor(height / 2), col: Math.floor(width / 2) });
                        } else {
                          setShowLens(false);
                          setLensPos(null);
                        }
                      }}
                     
                      size="small"
                      sx={switchStyles.small}
                    />
                  </>
                )}
                <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
                <Switch
                  checked={showFft}
                  onChange={(e) => {
                    const on = e.target.checked;
                    if (on && width * height > 2048 * 2048) {
                      console.warn(`Show2D: FFT on ${width}×${height} image (${(width * height / 1e6).toFixed(1)}M pixels) may be slow`);
                    }
                    setShowFft(on);
                  }}
                 
                  size="small"
                  sx={switchStyles.small}
                />
                {nImages === 2 && (
                  <>
                    <Typography sx={{ ...typography.label, fontSize: 10 }} title="Show A − B as a third panel. Use w.align() first to cancel drift.">Diff:</Typography>
                    <Switch checked={diffMode} onChange={() => { setDiffMode(!diffMode); }} size="small" sx={switchStyles.small} />
                  </>
                )}
              </>
            )}
            <Box sx={{ flex: 1 }} />
            {(
              <Button size="small" sx={compactButton} disabled={!needsReset} onClick={handleResetAll}>Reset</Button>
            )}
            {(
              <>
                <Button size="small" sx={{ ...compactButton, color: themeColors.accent }} onClick={(e) => { setExportAnchor(e.currentTarget); }}>Export</Button>
                <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)} anchorOrigin={{ vertical: "bottom", horizontal: "left" }} transformOrigin={{ vertical: "top", horizontal: "left" }} sx={{ zIndex: 9999 }}>
                  <MenuItem onClick={() => handleExportFigure(true, true)} sx={{ fontSize: 12 }}>PDF + scalebar + colorbar</MenuItem>
                  <MenuItem onClick={() => handleExportFigure(true, false)} sx={{ fontSize: 12 }}>PDF + scalebar</MenuItem>
                  <MenuItem onClick={() => handleExportFigure(false, false)} sx={{ fontSize: 12 }}>PDF</MenuItem>
                  <MenuItem onClick={handleExportAll} sx={{ fontSize: 12 }}>All (PNG + PDF)</MenuItem>
                </Menu>
                <Button size="small" sx={compactButton} onClick={handleCopy}>Copy</Button>
              </>
            )}
          </Stack>

          {isGallery ? (
            /* Gallery mode */
            <Box sx={{ display: "grid", gridTemplateColumns: `repeat(${effectiveNcols}, ${canvasW}px)`, gap: 1 }}>
              {Array.from({ length: nImages }).map((_, i) => (
                <Box key={i} sx={{ cursor: i === selectedIdx ? ((isDraggingResize || isDraggingResizeInner || isHoveringResize || isHoveringResizeInner) ? "nwse-resize" : isDraggingROI ? "move" : (draggingProfileEndpoint !== null || isDraggingProfileLine) ? "grabbing" : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine)) ? "grab" : (profileActive || roiActive || measureActive) ? "crosshair" : "grab") : ("pointer") }}>
                  <Box
                    ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[i] = el; }}
                    sx={{ position: "relative", bgcolor: "#000", border: `2px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}`, borderRadius: 0, width: canvasW, height: canvasH }}
                    onMouseDown={(e) => handleMouseDown(e, i)}
                    onMouseMove={(e) => handleMouseMove(e, i)}
                    onMouseUp={(e) => handleMouseUp(e, i)}
                    onMouseLeave={() => handleMouseLeave(i)}
                    onWheel={(i === selectedIdx || linkedZoom) ? (e) => handleWheel(e, i) : undefined}
                    onDoubleClick={() => handleDoubleClick(i)}
                  >
                    <canvas
                      ref={(el) => { if (el && canvasRefs.current[i] !== el) { canvasRefs.current[i] = el; setCanvasReady(c => c + 1); } }}
                      width={canvasW} height={canvasH}
                      style={{ width: canvasW, height: canvasH, imageRendering: imageRenderingStyle }}
                    />
                    <canvas
                      ref={(el) => { overlayRefs.current[i] = el; }}
                      width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                      style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
                    />
                    {(
                      <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, pointerEvents: "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
                    )}
                  </Box>
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted, textAlign: "center", mt: 0.25 }}>
                    {labels?.[i] || `Image ${i + 1}`}
                    {(imageRotations?.[i] ?? 0) % 4 !== 0 && (
                      <Box
                        component="span"
                        onClick={(e: React.MouseEvent) => {
                          e.stopPropagation();
                          const rots = [...(imageRotations || [])];
                          while (rots.length <= i) rots.push(0);
                          rots[i] = (rots[i] + 3) % 4;
                          setImageRotations(rots);
                        }}
                        sx={{ ml: 0.5, color: themeColors.accent, cursor: "pointer", "&:hover": { opacity: 0.7 } }}
                      >
                        ({(imageRotations[i] % 4) * 90}°)
                      </Box>
                    )}
                  </Typography>
                  {effectiveShowFft && (
                    <Box
                      ref={(el: HTMLDivElement | null) => { fftContainerRefs.current[i] = el; }}
                      sx={{ mt: 0.5, position: "relative", border: `2px solid ${i === selectedIdx ? themeColors.accent : themeColors.border}`, borderRadius: 0, bgcolor: "#000", cursor: "grab" }}
                      onWheel={(i === selectedIdx || fftLinkedZoom) ? (e) => handleGalleryFftWheel(e, i) : undefined}
                      onDoubleClick={() => setGalleryFftState(i, { zoom: DEFAULT_FFT_ZOOM, panX: 0, panY: 0 })}
                      onMouseDown={(e) => handleGalleryFftMouseDown(e, i)}
                      onMouseMove={(e) => handleGalleryFftMouseMove(e, i)}
                      onMouseUp={handleGalleryFftMouseUp}
                      onMouseLeave={handleGalleryFftMouseUp}
                    >
                      <canvas
                        ref={(el) => { fftCanvasRefs.current[i] = el; }}
                        width={canvasW} height={canvasH}
                        style={{ width: canvasW, height: canvasH, imageRendering: imageRenderingStyle, display: "block" }}
                      />
                      {fftComputing && !fftMagCacheGalleryRef.current[i] && (
                        <Box sx={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.6)", pointerEvents: "none" }}>
                          <Typography sx={{ fontSize: 10, color: "#aaa", fontFamily: "monospace", "@keyframes pulse": { "0%,100%": { opacity: 0.4 }, "50%": { opacity: 1 } }, animation: "pulse 1.2s ease-in-out infinite" }}>FFT…</Typography>
                        </Box>
                      )}
                    </Box>
                  )}
                </Box>
              ))}
              {showDiffPanel && diffOtherIndices.map((otherIdx, slot) => (
                <Box key={`diff_${slot}`}>
                  <Box sx={{ position: "relative", bgcolor: "#000", border: `2px solid ${themeColors.border}`, borderRadius: 0, width: canvasW, height: canvasH }}>
                    <canvas
                      ref={(el) => { diffCanvasRefs.current[slot] = el; }}
                      width={canvasW} height={canvasH}
                      style={{ width: canvasW, height: canvasH, imageRendering: imageRenderingStyle }}
                    />
                  </Box>
                  <Typography sx={{ fontSize: 10, color: themeColors.textMuted, textAlign: "center", mt: 0.25 }}>
                    {nImages === 2 ? "Diff (A − B)" : `Diff (#${diffReference + 1} − #${otherIdx + 1})`}
                  </Typography>
                  {/* FFT of diff (n=2 only) */}
                  {effectiveShowFft && nImages === 2 && slot === 0 && (
                    <Box sx={{ mt: 0.5, position: "relative", border: `2px solid ${themeColors.border}`, borderRadius: 0, bgcolor: "#000" }}>
                      <canvas
                        ref={(el) => { diffFftCanvasRef.current = el; }}
                        width={canvasW} height={canvasH}
                        style={{ width: canvasW, height: canvasH, imageRendering: imageRenderingStyle, display: "block" }}
                      />
                    </Box>
                  )}
                </Box>
              ))}
            </Box>
          ) : (
            /* Single image mode */
            <Box
              ref={(el: HTMLDivElement | null) => { imageContainerRefs.current[0] = el; }}
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, width: canvasW, height: canvasH, cursor: isHoveringLensEdge ? "nwse-resize" : isDraggingROI ? "move" : (isDraggingResize || isDraggingResizeInner || isHoveringResize || isHoveringResizeInner) ? "nwse-resize" : (draggingProfileEndpoint !== null || isDraggingProfileLine) ? "grabbing" : (profileActive && (hoveredProfileEndpoint !== null || isHoveringProfileLine)) ? "grab" : (profileActive || roiActive || measureActive) ? "crosshair" : "grab" }}
              onMouseDown={(e) => handleMouseDown(e, 0)}
              onMouseMove={(e) => handleMouseMove(e, 0)}
              onMouseUp={(e) => handleMouseUp(e, 0)}
              onMouseLeave={() => handleMouseLeave(0)}
              onWheel={(e) => handleWheel(e, 0)}
              onDoubleClick={() => handleDoubleClick(0)}
            >
              <canvas
                ref={(el) => { if (el && canvasRefs.current[0] !== el) { canvasRefs.current[0] = el; setCanvasReady(c => c + 1); } }}
                width={canvasW} height={canvasH}
                style={{ width: canvasW, height: canvasH, imageRendering: imageRenderingStyle }}
              />
              <canvas
                ref={(el) => { overlayRefs.current[0] = el; }}
                width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
              />
              <canvas
                ref={lensCanvasRef}
                width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)}
                style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }}
              />
              {cursorInfo && (
                <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                  <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                    ({cursorInfo.row}, {cursorInfo.col}){pixelSize > 0 ? ` = (${(cursorInfo.row * calibratedFactor).toFixed(1)}, ${(cursorInfo.col * calibratedFactor).toFixed(1)} ${calibratedUnit})` : ""} {formatNumber(cursorInfo.value)}
                  </Typography>
                </Box>
              )}
              {(
                <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, pointerEvents: "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
              )}
            </Box>
          )}

          {/* Stats bar - right below canvas (Show3D style) */}
          {showStats && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2, alignItems: "center", boxSizing: "border-box", overflow: "hidden", whiteSpace: "nowrap", opacity: 1 }}>
              {isGallery && (
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>{labels?.[statsIdx] || `#${statsIdx + 1}`}</Typography>
              )}
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMean?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMin?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsMax?.[statsIdx] ?? 0)}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(statsStd?.[statsIdx] ?? 0)}</Box></Typography>
              {measureActive && (
                <>
                  <Box sx={{ borderLeft: `1px solid ${themeColors.border}`, height: 14 }} />
                  <Typography sx={{ fontSize: 11, color: "#fff", fontWeight: "bold" }}>Measuring</Typography>
                </>
              )}
            </Box>
          )}

          {/* Gallery FFT Controls - below gallery grid */}
          {effectiveShowFft && isGallery && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "flex-start" }}>
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT Scale:</Typography>
                  <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                  </Select>
                  {roiFftActive && fftCropDims && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Win:</Typography>
                      <Switch checked={fftWindow} onChange={(e) => { setFftWindow(e.target.checked); }} size="small" sx={switchStyles.small} />
                    </>
                  )}
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                    {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                  </Select>
                </Box>
                {/* FFT Row 2: Auto + Smooth + Link Zoom/Pan/Contrast (mirrors main image Row 2) */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                  <Switch checked={fftAuto} onChange={(e) => { setFftAuto(e.target.checked); }} size="small" sx={switchStyles.small} />
                  <Typography sx={{ ...typography.label, fontSize: 10 }} title="CSS bilinear interpolation on the FFT canvas.">Smooth:</Typography>
                  <Switch checked={fftSmooth} onChange={(e) => { setFftSmooth(e.target.checked); }} size="small" sx={switchStyles.small} />
                  {isGallery && (
                    <>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Link:</Typography>
                      <Typography sx={{ ...typography.label, fontSize: 10 }} title="Zoom together across FFT panels (FFT-only, independent of main image link).">Zoom</Typography>
                      <Switch checked={fftLinkedZoom} onChange={() => { setFftLinkedZoom(!fftLinkedZoom); }} size="small" sx={switchStyles.small} />
                      <Typography sx={{ ...typography.label, fontSize: 10 }} title="Pan FFT panels together (FFT-only).">Pan</Typography>
                      <Switch checked={fftLinkPan} onChange={() => { setFftLinkPan(!fftLinkPan); }} size="small" sx={switchStyles.small} />
                      <Typography sx={{ ...typography.label, fontSize: 10 }} title="Share FFT contrast slider across panels (FFT-only).">Contrast</Typography>
                      <Switch checked={fftLinkedContrast} onChange={() => { setFftLinkedContrast(!fftLinkedContrast); }} size="small" sx={switchStyles.small} />
                    </>
                  )}
                </Box>
              </Box>
              {(
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: 1, pointerEvents: "auto" }}>
                  {fftHistogramData && (
                    !fftLinkedContrast && isGallery ? (
                      <Box sx={{ display: "grid", gridTemplateColumns: `repeat(${effectiveNcols}, 110px)`, gap: "15px" }}>
                        {Array.from({ length: nImages }).map((_, i) => {
                          const fc = fftContrastFor(i);
                          const mag = fftMagCacheGalleryRef.current[i];
                          let perData: Float32Array | null = null;
                          if (mag) {
                            if (fftScaleMode === "log") perData = applyLogScale(mag);
                            else if (fftScaleMode === "power") {
                              perData = new Float32Array(mag.length);
                              for (let j = 0; j < mag.length; j++) perData[j] = Math.sqrt(mag[j]);
                            } else perData = mag;
                          }
                          const dr = perData ? findDataRange(perData) : fftDataRange;
                          return (
                            <Histogram
                              key={i}
                              data={perData || fftHistogramData}
                              vminPct={fc.vminPct} vmaxPct={fc.vmaxPct}
                              onRangeChange={(min, max) => { setFftContrastFor(i, { vminPct: min, vmaxPct: max }); }}
                              width={110} height={58}
                              theme={themeInfo.theme === "dark" ? "dark" : "light"}
                              dataMin={dr.min} dataMax={dr.max}
                            />
                          );
                        })}
                      </Box>
                    ) : (() => {
                      const fc = fftContrastFor(selectedIdx);
                      return (
                        <Histogram
                          data={fftHistogramData}
                          vminPct={fc.vminPct}
                          vmaxPct={fc.vmaxPct}
                          onRangeChange={(min, max) => { setFftContrastFor(selectedIdx, { vminPct: min, vmaxPct: max }); }}
                          width={110} height={58}
                          theme={themeInfo.theme === "dark" ? "dark" : "light"}
                          dataMin={fftDataRange.min} dataMax={fftDataRange.max}
                        />
                      );
                    })()
                  )}
                </Box>
              )}
            </Box>
          )}

          {/* Line profile sparkline — always reserve space when profile is active */}
          {profileActive && (
            <Box sx={{ mt: `${SPACING.XS}px`, maxWidth: profileCanvasWidth, boxSizing: "border-box" }}>
              <canvas
                ref={profileCanvasRef}
                onMouseMove={handleProfileMouseMove}
                onMouseLeave={handleProfileMouseLeave}
                style={{ width: profileCanvasWidth, height: profileHeight, display: "block", border: `1px solid ${themeColors.border}`, borderBottom: "none", cursor: "crosshair" }}
              />
              <div
                onMouseDown={(e) => {
                  e.preventDefault();
                  setIsResizingProfile(true);
                  setProfileResizeStart({ y: e.clientY, height: profileHeight });
                }}
                style={{ width: profileCanvasWidth, height: 4, cursor: "ns-resize", borderLeft: `1px solid ${themeColors.border}`, borderRight: `1px solid ${themeColors.border}`, borderBottom: `1px solid ${themeColors.border}`, background: `linear-gradient(to bottom, ${themeColors.border}, transparent)`, opacity: 1, pointerEvents: "auto" }}
              />
            </Box>
          )}

          {/* Controls: two rows left + histogram right, ROI below */}
          {showControls && (
            <Box sx={{ mt: (effectiveShowFft && isGallery) ? `${SPACING.XS}px` : `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, boxSizing: "border-box" }}>
              {/* Top: control rows + histogram side by side */}
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "flex-start" }}>
                  {/* Row 1: Scale + Color */}
                  {(
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                      <Select value={logScale ? "log" : "linear"} onChange={(e) => setLogScale(e.target.value === "log")} size="small" sx={{ ...themedSelect, minWidth: 45 }} MenuProps={themedMenuProps}>
                        <MenuItem value="linear">Lin</MenuItem>
                        <MenuItem value="log">Log</MenuItem>
                      </Select>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                      <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 60 }}>
                        {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                      </Select>
                      {!isGallery && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Colorbar:</Typography>
                          <Switch checked={showColorbar} onChange={() => { setShowColorbar(!showColorbar); }} size="small" sx={switchStyles.small} />
                        </>
                      )}
                    </Box>
                  )}
                  {/* Row 2: Auto + Lens settings + Link Zoom (gallery) + zoom indicator */}
                  {(
                    <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                      <Switch checked={autoContrast} onChange={() => { setAutoContrast(!autoContrast); }} size="small" sx={switchStyles.small} />
                      <Typography sx={{ ...typography.label, fontSize: 10 }} title="CSS bilinear interpolation. Same data, browser smooths visually — useful when upscaling small images on a large canvas.">Smooth:</Typography>
                      <Switch checked={smooth} onChange={() => { setSmooth(!smooth); }} size="small" sx={switchStyles.small} />
                      {!isGallery && showLens && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Lens {lensMag}×</Typography>
                          <Slider value={lensMag} min={2} max={8} step={1} onChange={(_, v) => setLensMag(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>{lensDisplaySize}px</Typography>
                          <Slider value={lensDisplaySize} min={64} max={256} step={16} onChange={(_, v) => setLensDisplaySize(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35 }} />
                        </>
                      )}
                      {isGallery && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Link:</Typography>
                          <Typography sx={{ ...typography.label, fontSize: 10 }} title="Zoom together across panels.">Zoom</Typography>
                          <Switch checked={linkedZoom} onChange={() => { setLinkedZoom(!linkedZoom); }} size="small" sx={switchStyles.small} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }} title="Pan together (independent of zoom).">Pan</Typography>
                          <Switch checked={linkPan} onChange={() => { setLinkPan(!linkPan); }} size="small" sx={switchStyles.small} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }} title="Share contrast slider across panels.">Contrast</Typography>
                          <Switch checked={linkedContrast} onChange={() => { setLinkedContrast(!linkedContrast); }} size="small" sx={switchStyles.small} />
                        </>
                      )}
                      {getZoomState(isGallery ? selectedIdx : 0).zoom !== 1 && (
                        <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{getZoomState(isGallery ? selectedIdx : 0).zoom.toFixed(1)}x</Typography>
                      )}
                    </Box>
                  )}
                </Box>
                {/* Right: histograms. Unlinked + gallery → grid matching gallery layout
                    (same effectiveNcols × rows). Linked or single image → one histogram. */}
                {(imageHistogramData || imageHistogramBins || (isGallery && !linkedContrast && rawDataRef.current)) && (
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start", gap: 0.5, opacity: 1, pointerEvents: "auto" }}>
                    {(!linkedContrast && isGallery && rawDataRef.current) ? (
                      <Box sx={{ display: "grid", gridTemplateColumns: `repeat(${effectiveNcols}, 110px)`, gap: "15px" }}>
                        {Array.from({ length: nImages }).map((_, i) => {
                          const cs = contrastStates.get(i) || { vminPct: 0, vmaxPct: 100 };
                          const raw = rawDataRef.current?.[i] || null;
                          return (
                            <Histogram key={i} data={raw} vminPct={cs.vminPct} vmaxPct={cs.vmaxPct}
                              onRangeChange={(min, max) => { setContrastState(i, { vminPct: min, vmaxPct: max }); }}
                              width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"}
                              dataMin={dataRangesRef.current[i]?.min ?? imageDataRange.min}
                              dataMax={dataRangesRef.current[i]?.max ?? imageDataRange.max} />
                          );
                        })}
                      </Box>
                    ) : (
                      <Histogram data={imageHistogramData} precomputedBins={imageHistogramBins} vminPct={imageVminPct} vmaxPct={imageVmaxPct} onRangeChange={(min, max) => { setContrastState(activeContrastIdx, { vminPct: min, vmaxPct: max }); }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={traitVmin != null && traitVmax != null ? (logScale ? Math.log1p(Math.max(traitVmin, 0)) : traitVmin) : imageDataRange.min} dataMax={traitVmin != null && traitVmax != null ? (logScale ? Math.log1p(Math.max(traitVmax, 0)) : traitVmax) : imageDataRange.max} />
                    )}
                  </Box>
                )}
              </Box>
              {/* ROI Section (own box, below control rows) */}
              {roiActive && (
                <Box sx={{ border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, px: 1, py: 0.5, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, opacity: 1, pointerEvents: "auto" }}>
                  {/* ROI: shape + ADD + CLEAR */}
                  <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
                    <Select
                      size="small"
                      value={newRoiShape}
                      onChange={(e) => setNewRoiShape(e.target.value as "circle" | "square" | "rectangle" | "annular")}
                      MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                    >
                      {(["square", "rectangle", "circle", "annular"] as const).map((s) => (<MenuItem key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</MenuItem>))}
                    </Select>
                    <Button size="small" sx={compactButton} onClick={() => {
                      const defR = Math.max(10, Math.round(Math.min(width, height) * 0.05));
                      const newRoi: ROIItem = { row: Math.floor(height / 2), col: Math.floor(width / 2), shape: newRoiShape, radius: defR, radius_inner: Math.max(5, Math.round(defR * 0.5)), width: defR * 2, height: defR * 2, color: ROI_COLORS[(roiList?.length ?? 0) % ROI_COLORS.length], line_width: 2, highlight: false };
                      const newList = [...(roiList || []), newRoi];
                      setRoiList(newList);
                      setRoiSelectedIdx(newList.length - 1);
                    }}>ADD</Button>
                    <Box sx={{ flex: 1 }} />
                    <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 24, color: "#ef5350" }} disabled={!roiList?.length} onClick={() => { setRoiList([]); setRoiSelectedIdx(-1); }}>CLEAR</Button>
                  </Box>
                  {/* Selected ROI details */}
                  {selectedRoi && (
                    <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: `${SPACING.SM}px`, borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                      <Typography sx={{ ...typography.label, fontSize: 10, color: selectedRoi.color }}>#{roiSelectedIdx + 1}/{roiList?.length ?? 0}</Typography>
                      <Select
                        size="small"
                        value={selectedRoi.shape || "circle"}
                        onChange={(e) => updateSelectedRoi({ shape: e.target.value })}
                        MenuProps={themedMenuProps} sx={{ ...themedSelect, minWidth: 85, fontSize: 10 }}
                      >
                        {(["square", "rectangle", "circle", "annular"] as const).map((s) => (<MenuItem key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</MenuItem>))}
                      </Select>
                      {selectedRoi.shape === "rectangle" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>W</Typography>
                          <Slider value={selectedRoi.width} min={5} max={width} onChange={(_, v) => updateSelectedRoi({ width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>H</Typography>
                          <Slider value={selectedRoi.height} min={5} max={height} onChange={(_, v) => updateSelectedRoi({ height: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                        </>
                      )}
                      {selectedRoi.shape === "annular" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Inner</Typography>
                          <Slider value={selectedRoi.radius_inner} min={1} max={selectedRoi.radius - 1} onChange={(_, v) => updateSelectedRoi({ radius_inner: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Outer</Typography>
                          <Slider value={selectedRoi.radius} min={selectedRoi.radius_inner + 1} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 40 }} />
                        </>
                      )}
                      {selectedRoi.shape !== "rectangle" && selectedRoi.shape !== "annular" && (
                        <>
                          <Typography sx={{ ...typography.label, fontSize: 10 }}>Size</Typography>
                          <Slider value={selectedRoi.radius} min={5} max={Math.max(width, height)} onChange={(_, v) => updateSelectedRoi({ radius: v as number })} size="small" sx={{ ...sliderStyles.small, width: 50 }} />
                        </>
                      )}
                      <Box sx={{ display: "flex", gap: "2px" }}>
                        {ROI_COLORS.map(c => (
                          <Box key={c} onClick={() => updateSelectedRoi({ color: c })} sx={{ width: 12, height: 12, bgcolor: c, cursor: "pointer", border: c === selectedRoi.color ? `2px solid ${themeColors.text}` : "1px solid transparent", "&:hover": { opacity: 0.8 } }} />
                        ))}
                      </Box>
                      <Typography sx={{ ...typography.label, fontSize: 10 }}>Border</Typography>
                      <Slider value={selectedRoi.line_width} min={1} max={6} step={1} onChange={(_, v) => updateSelectedRoi({ line_width: v as number })} size="small" sx={{ ...sliderStyles.small, width: 30 }} />
                      <Box
                        onClick={() => updateSelectedRoi({ highlight: !selectedRoi.highlight })}
                        sx={{ cursor: "pointer", fontSize: 10, color: selectedRoi.highlight ? themeColors.accentGreen : themeColors.textMuted, "&:hover": { opacity: 0.8 } }}
                        title="Focus (dim outside)"
                      >{selectedRoi.highlight ? "\u25C9 Focus" : "\u25CB Focus"}</Box>
                      <Button size="small" sx={{ ...compactButton, fontSize: 9, minWidth: 20, color: "#ef5350" }} onClick={() => {
                        const newList = roiList!.filter((_, j) => j !== roiSelectedIdx);
                        setRoiList(newList);
                        setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1);
                      }}>&times;</Button>
                    </Box>
                  )}
                  {/* ROI list */}
                  {roiList && roiList.length > 0 && (
                    <Box sx={{ display: "flex", flexDirection: "column", borderTop: `1px solid ${themeColors.border}`, pt: `${SPACING.XS}px` }}>
                      {roiList.map((roi, i) => {
                        const c = roi.color || ROI_COLORS[i % ROI_COLORS.length];
                        const isSelected = i === roiSelectedIdx;
                        const shapeLabel = roi.shape === "rectangle" ? `${roi.width}×${roi.height}` : roi.shape === "annular" ? `r${roi.radius_inner}-${roi.radius}` : `r${roi.radius}`;
                        return (
                          <Box key={i} onClick={() => setRoiSelectedIdx(i)} sx={{ display: "flex", alignItems: "center", gap: "3px", lineHeight: 1.6, cursor: "pointer", "&:hover .roi-delete": { opacity: 1 } }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: roi.shape === "square" || roi.shape === "rectangle" ? 0 : "50%", bgcolor: c, border: isSelected ? "2px solid #fff" : "1px solid transparent", flexShrink: 0 }} />
                            <Typography component="span" sx={{ fontSize: 10, fontFamily: "monospace", color: isSelected ? themeColors.text : themeColors.textMuted, fontWeight: isSelected ? "bold" : "normal" }}>
                              <Box component="span" sx={{ color: c }}>{i + 1}</Box>{" "}
                              {roi.shape} ({roi.row}, {roi.col}) {shapeLabel}
                            </Typography>
                            <Box
                              onClick={(e) => { e.stopPropagation(); const newList = roiList.map((r, j) => ({ ...r, highlight: j === i ? !r.highlight : false })); setRoiList(newList); }}
                              sx={{ cursor: "pointer", fontSize: 10, color: roi.highlight ? themeColors.accentGreen : themeColors.textMuted, lineHeight: 1, opacity: roi.highlight ? 1 : 0.5, "&:hover": { opacity: 1 } }}
                              title="Focus (dim outside)"
                            >{roi.highlight ? "\u25C9" : "\u25CB"}</Box>
                            <Box
                              className="roi-delete"
                              onClick={(e) => { e.stopPropagation(); const newList = roiList.filter((_, j) => j !== i); setRoiList(newList); setRoiSelectedIdx(newList.length > 0 ? Math.min(roiSelectedIdx, newList.length - 1) : -1); }}
                              sx={{ opacity: 0, cursor: "pointer", fontSize: 10, color: themeColors.textMuted, ml: 0.5, lineHeight: 1, "&:hover": { color: "#f44336" } }}
                            >&times;</Box>
                          </Box>
                        );
                      })}
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* FFT Panel - canvas + stats (single mode only) */}
        {effectiveShowFft && !isGallery && (
          <Box sx={{ width: canvasW }}>
            {/* Spacer — matches main panel title row height for canvas alignment */}
            <Box sx={{ mb: `${SPACING.XS}px`, height: 16 }} />
            {/* Controls row — matches main panel controls row height */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              {fftComputing ? (
                <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: themeColors.textMuted, "@keyframes pulse": { "0%,100%": { opacity: 0.4 }, "50%": { opacity: 1 } }, animation: "pulse 1.2s ease-in-out infinite" }}>
                  {fftProgress || "Computing FFT…"}</Typography>
              ) : roiFftActive && fftCropDims ? (
                <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: themeColors.accentGreen }}>
                  ROI FFT ({fftCropDims.cropWidth}&times;{fftCropDims.cropHeight})
                </Typography>
              ) : <Box />}
              {(
                <Button size="small" sx={compactButton} disabled={(fftZoom === DEFAULT_FFT_ZOOM && fftPanX === 0 && fftPanY === 0)} onClick={handleFftDoubleClick}>Reset</Button>
              )}
            </Stack>
            <Box
              ref={singleFftContainerRef}
              sx={{ position: "relative", bgcolor: "#000", border: `1px solid ${themeColors.border}`, cursor: "crosshair", width: canvasW, height: canvasH }}
              onWheel={handleFftWheel}
              onDoubleClick={handleFftDoubleClick}
              onMouseDown={handleFftMouseDown}
              onMouseMove={handleFftMouseMove}
              onMouseUp={handleFftMouseUp}
              onMouseLeave={handleFftMouseLeave}
            >
              <canvas ref={fftCanvasRef} width={canvasW} height={canvasH} style={{ width: canvasW, height: canvasH, imageRendering: imageRenderingStyle }} />
              <canvas ref={fftOverlayRef} width={Math.round(canvasW * DPR)} height={Math.round(canvasH * DPR)} style={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, pointerEvents: "none" }} />
              {fftComputing && (
                <Box sx={{ position: "absolute", top: 0, left: 0, width: canvasW, height: canvasH, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.6)", pointerEvents: "none" }}>
                  <Typography sx={{ fontSize: 11, color: "#aaa", fontFamily: "monospace", "@keyframes pulse": { "0%,100%": { opacity: 0.4 }, "50%": { opacity: 1 } }, animation: "pulse 1.2s ease-in-out infinite" }}>
                    {fftProgress || "Computing FFT…"}
                  </Typography>
                </Box>
              )}
              {(
                <Box onMouseDown={handleCanvasResizeStart} sx={{ position: "absolute", bottom: 0, right: 0, width: 16, height: 16, cursor: "nwse-resize", opacity: 0.6, pointerEvents: "auto", background: `linear-gradient(135deg, transparent 50%, ${themeColors.accent} 50%)`, borderRadius: "0 0 4px 0", "&:hover": { opacity: 1 } }} />
              )}
            </Box>
            {/* FFT Stats Bar */}
            {fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatNumber(fftStats[3])}</Box></Typography>
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
            {/* FFT Controls - two rows + histogram (matching main panel layout) */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, width: canvasW, boxSizing: "border-box" }}>
              <Box sx={{ display: "flex", gap: `${SPACING.SM}px` }}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "flex-start" }}>
                  {/* Row 1: Scale + Color + Colorbar */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                    <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={themedMenuProps}>
                      <MenuItem value="linear">Lin</MenuItem>
                      <MenuItem value="log">Log</MenuItem>
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                    <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={themedMenuProps}>
                      {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                    </Select>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Colorbar:</Typography>
                    <Switch checked={fftShowColorbar} onChange={(e) => { setFftShowColorbar(e.target.checked); }} size="small" sx={switchStyles.small} />
                  </Box>
                  {/* Row 2: Auto + zoom indicator */}
                  <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, opacity: 1, pointerEvents: "auto" }}>
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:</Typography>
                    <Switch checked={fftAuto} onChange={(e) => { setFftAuto(e.target.checked); }} size="small" sx={switchStyles.small} />
                    {fftCropDims && (
                      <>
                        <Typography sx={{ ...typography.label, fontSize: 10 }}>Win:</Typography>
                        <Switch checked={fftWindow} onChange={(e) => { setFftWindow(e.target.checked); }} size="small" sx={switchStyles.small} />
                      </>
                    )}
                    {fftZoom !== DEFAULT_FFT_ZOOM && (
                      <Typography sx={{ ...typography.label, fontSize: 10, color: themeColors.accent, fontWeight: "bold" }}>{fftZoom.toFixed(1)}x</Typography>
                    )}
                  </Box>
                </Box>
                {/* Right: FFT Histogram */}
                {(
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center", opacity: 1, pointerEvents: "auto" }}>
                    {fftHistogramData && (
                      <Histogram data={fftHistogramData} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme === "dark" ? "dark" : "light"} dataMin={fftDataRange.min} dataMax={fftDataRange.max} />
                    )}
                  </Box>
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Stack>
    </Box>
  );
}

export const render = createRender(Show2D);
