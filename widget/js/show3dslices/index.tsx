/// <reference types="@webgpu/types" />
/**
 * Show3DSlices - Orthogonal slice viewer for 3D volumetric data.
 *
 * Top plus arbitrary-angle vertical slice panels with synchronized sliders and a 3D orientation view.
 * All slicing done in JS from raw float32 volume data for instant response.
 *
 * Ptycho-focused single-object workflow; tomography/comparison flows belong in
 * Show3DVolume.
 */
import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import Tooltip from "@mui/material/Tooltip";
import Select from "@mui/material/Select";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import StopIcon from "@mui/icons-material/Stop";
import { useTheme } from "../theme";
import { VolumeRenderer, CameraState, DEFAULT_CAMERA } from "../webgpu-volume";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar } from "../figure";
import { downloadBlob, extractBytes, extractFloat32, formatNumber } from "../format";
import { findDataRange, applyLogScale, percentileClip, sliderRange, computeHistogramFromBytes } from "../stats";

const MAX_PLAYBACK_FPS = 30;

// ============================================================================
// Style tokens (inlined - matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const PLANE_KEYS = ["xy", "oblique"] as const;
const PLANE_LABELS = ["Top", "Side"] as const;
const PLANE_COLORS = ["#4d80ff", "#4dff66"] as const;
const OBLIQUE_PROFILE_EDGE_INSET = 1;
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
const planeToggleButtonSx = {
  minWidth: 30,
  height: 18,
  px: 0.7,
  py: 0.1,
  fontSize: 10,
  lineHeight: 1,
  color: "primary.main",
  borderColor: "divider",
  textTransform: "none",
  letterSpacing: 0,
  "&.Mui-selected": {
    color: "primary.contrastText",
    bgcolor: "primary.main",
    "&:hover": { bgcolor: "primary.dark" },
  },
  "&:hover": { bgcolor: "action.hover" },
} as const;
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
const typographyLabel = {
  fontSize: 10,
  textTransform: "none" as const,
  letterSpacing: 0,
};
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

// ============================================================================
// Inlined utilities (mirrors Show3D - keep widgets self-contained)
// ============================================================================
const signedLog1p = (x: number): number => x >= 0 ? Math.log1p(x) : -Math.log1p(-x);

type Show3DSlicesWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show3DSlicesFileHandle = {
  createWritable: () => Promise<Show3DSlicesWritableFile>;
};

type Show3DSlicesSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show3DSlicesWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show3DSlicesSavePickerOptions) => Promise<Show3DSlicesFileHandle>;
};

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

function extractXY(vol: Float32Array, nx: number, ny: number, nz: number, z: number): Float32Array {
  if (z < 0 || z >= nz) return new Float32Array(ny * nx);
  const start = z * ny * nx;
  return vol.subarray(start, start + ny * nx);
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function pointToSegmentDistance(
  col: number,
  row: number,
  col0: number,
  row0: number,
  col1: number,
  row1: number,
): number {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const lenSq = dc * dc + dr * dr;
  if (lenSq <= 1e-12) return Math.hypot(col - col0, row - row0);
  const tRaw = ((col - col0) * dc + (row - row0) * dr) / lenSq;
  const t = clampNumber(tRaw, 0, 1);
  const projCol = col0 + t * dc;
  const projRow = row0 + t * dr;
  return Math.hypot(col - projCol, row - projRow);
}

function obliqueLineEndpoints(
  nx: number,
  ny: number,
  cx: number,
  cy: number,
  angleDeg: number,
): [{ x: number; y: number }, { x: number; y: number }] {
  const theta = (angleDeg * Math.PI) / 180;
  const dx = Math.cos(theta);
  const dy = Math.sin(theta);
  const candidates: number[] = [];
  const maxX = Math.max(0, nx - 1);
  const maxY = Math.max(0, ny - 1);
  if (Math.abs(dx) > 1e-8) {
    const t0 = (0 - cx) / dx;
    const y0 = cy + t0 * dy;
    if (y0 >= 0 && y0 <= maxY) candidates.push(t0);
    const t1 = (maxX - cx) / dx;
    const y1 = cy + t1 * dy;
    if (y1 >= 0 && y1 <= maxY) candidates.push(t1);
  }
  if (Math.abs(dy) > 1e-8) {
    const t0 = (0 - cy) / dy;
    const x0 = cx + t0 * dx;
    if (x0 >= 0 && x0 <= maxX) candidates.push(t0);
    const t1 = (maxY - cy) / dy;
    const x1 = cx + t1 * dx;
    if (x1 >= 0 && x1 <= maxX) candidates.push(t1);
  }
  if (candidates.length < 2) {
    return [
      { x: clampNumber(cx, 0, maxX), y: clampNumber(cy, 0, maxY) },
      { x: clampNumber(cx, 0, maxX), y: clampNumber(cy, 0, maxY) },
    ];
  }
  const minT = Math.min(...candidates);
  const maxT = Math.max(...candidates);
  return [
    { x: clampNumber(cx + minT * dx, 0, maxX), y: clampNumber(cy + minT * dy, 0, maxY) },
    { x: clampNumber(cx + maxT * dx, 0, maxX), y: clampNumber(cy + maxT * dy, 0, maxY) },
  ];
}

function obliqueNormal(angleDeg: number): { x: number; y: number } {
  const theta = (angleDeg * Math.PI) / 180;
  return { x: -Math.sin(theta), y: Math.cos(theta) };
}

function obliqueSegmentOffsetBounds(
  nx: number,
  ny: number,
  angleDeg: number,
  start: { x: number; y: number },
  stop: { x: number; y: number },
  inset: number = 0,
): [number, number] {
  const normal = obliqueNormal(angleDeg);
  const points = [start, stop];
  const xMin = Math.min(inset, Math.max(0, nx - 1));
  const yMin = Math.min(inset, Math.max(0, ny - 1));
  const xMax = Math.max(xMin, Math.max(1, nx) - 1 - inset);
  const yMax = Math.max(yMin, Math.max(1, ny) - 1 - inset);
  let minDelta = -Infinity;
  let maxDelta = Infinity;
  for (const point of points) {
    if (Math.abs(normal.x) > 1e-8) {
      const d0 = (xMin - point.x) / normal.x;
      const d1 = (xMax - point.x) / normal.x;
      minDelta = Math.max(minDelta, Math.min(d0, d1));
      maxDelta = Math.min(maxDelta, Math.max(d0, d1));
    }
    if (Math.abs(normal.y) > 1e-8) {
      const d0 = (yMin - point.y) / normal.y;
      const d1 = (yMax - point.y) / normal.y;
      minDelta = Math.max(minDelta, Math.min(d0, d1));
      maxDelta = Math.min(maxDelta, Math.max(d0, d1));
    }
  }
  if (!Number.isFinite(minDelta) || !Number.isFinite(maxDelta) || minDelta > maxDelta) return [0, 0];
  return [Math.ceil(minDelta), Math.floor(maxDelta)];
}

function obliqueCenterOffset(
  nx: number,
  ny: number,
  angleDeg: number,
  start: { x: number; y: number },
  stop: { x: number; y: number },
): number {
  const normal = obliqueNormal(angleDeg);
  const ox = (Math.max(1, nx) - 1) / 2;
  const oy = (Math.max(1, ny) - 1) / 2;
  const cx = (start.x + stop.x) / 2;
  const cy = (start.y + stop.y) / 2;
  return (cx - ox) * normal.x + (cy - oy) * normal.y;
}

function profilePointFromAny(value: unknown): { x: number; y: number } | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const row = Number(record.row);
  const col = Number(record.col);
  if (!Number.isFinite(row) || !Number.isFinite(col)) return null;
  return { x: col, y: row };
}

function profileLinePayload(start: { x: number; y: number }, stop: { x: number; y: number }): { row: number; col: number }[] {
  return [
    { row: start.y, col: start.x },
    { row: stop.y, col: stop.x },
  ];
}

function clampPointToImage(point: { x: number; y: number }, nx: number, ny: number, inset: number = 0): { x: number; y: number } {
  const xInset = Math.min(inset, Math.max(0, nx - 1) / 2);
  const yInset = Math.min(inset, Math.max(0, ny - 1) / 2);
  return {
    x: clampNumber(point.x, xInset, Math.max(xInset, Math.max(0, nx - 1) - xInset)),
    y: clampNumber(point.y, yInset, Math.max(yInset, Math.max(0, ny - 1) - yInset)),
  };
}

function segmentWidth(start: { x: number; y: number }, stop: { x: number; y: number }): number {
  return Math.max(1, Math.ceil(Math.hypot(stop.x - start.x, stop.y - start.y)) + 1);
}

function sampleVolumeBilinear(
  vol: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  z: number,
  x: number,
  y: number,
): number {
  if (z < 0 || z >= nz || x < 0 || y < 0 || x > nx - 1 || y > ny - 1) return Number.NaN;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(nx - 1, x0 + 1);
  const y1 = Math.min(ny - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;
  const base = z * ny * nx;
  const v00 = vol[base + y0 * nx + x0];
  const v10 = vol[base + y0 * nx + x1];
  const v01 = vol[base + y1 * nx + x0];
  const v11 = vol[base + y1 * nx + x1];
  return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

function extractOblique(
  vol: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  start: { x: number; y: number },
  stop: { x: number; y: number },
): Float32Array {
  const width = segmentWidth(start, stop);
  const out = new Float32Array(nz * width);
  const denom = Math.max(1, width - 1);
  for (let z = 0; z < nz; z++) {
    for (let col = 0; col < width; col++) {
      const t = col / denom;
      const x = start.x + (stop.x - start.x) * t;
      const y = start.y + (stop.y - start.y) * t;
      const value = sampleVolumeBilinear(vol, nx, ny, nz, z, x, y);
      out[z * width + col] = Number.isFinite(value) ? value : 0;
    }
  }
  return out;
}

function extractVolumeFloat32(
  dataView: DataView | ArrayBuffer | Uint8Array,
  offline: boolean,
  offlineMin: number,
  offlineMax: number,
  nx: number,
  ny: number,
  nz: number,
): Float32Array | null {
  if (!offline) return extractFloat32(dataView);
  const bytes = extractBytes(dataView);
  const count = Math.max(0, Math.floor(nx) * Math.floor(ny) * Math.floor(nz));
  if (bytes.length === 0 || count === 0) return null;
  const out = new Float32Array(count);
  const usable = Math.min(count, bytes.length);
  const lo = Number.isFinite(offlineMin) ? offlineMin : 0;
  const hi = Number.isFinite(offlineMax) ? offlineMax : lo;
  const scale = hi > lo ? (hi - lo) / 255.0 : 0;
  for (let i = 0; i < usable; i++) out[i] = bytes[i] * scale + lo;
  if (usable < count) out.fill(lo, usable);
  return out;
}

function makeExportFilename(title: string, nz: number, ny: number, nx: number, mode: string): string {
  let slug = (title || "show3dslices")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  while (slug.includes("__")) slug = slug.replace(/__/g, "_");
  if (!slug) slug = "show3dslices";
  const suffix = mode === "quantized" ? "quantized" : "exact";
  return `${slug}_${nz}x${ny}x${nx}_${suffix}.html`;
}

function formatSavedBytes(bytes: number): string {
  const mb = Math.max(0, bytes) / (1024 * 1024);
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(2)} MB`;
}

function isAbortLikeError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function reverseLut(lut: Uint8Array): Uint8Array {
  const out = new Uint8Array(lut.length);
  const n = lut.length / 3;
  for (let i = 0; i < n; i++) {
    const src = (n - 1 - i) * 3;
    const dst = i * 3;
    out[dst + 0] = lut[src + 0];
    out[dst + 1] = lut[src + 1];
    out[dst + 2] = lut[src + 2];
  }
  return out;
}

function maybeFlip(data: Float32Array, flip: boolean): Float32Array {
  if (!flip) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = -data[i];
  return out;
}

function makeHistogramSample(data: Float32Array | null, target = 1_000_000): Float32Array | null {
  if (!data || data.length === 0) return null;
  if (data.length <= target) return data;
  const stride = Math.ceil(data.length / target);
  const out = new Float32Array(Math.ceil(data.length / stride));
  for (let src = 0, dst = 0; src < data.length; src += stride, dst++) out[dst] = data[src];
  return out;
}

function transformDisplaySample(data: Float32Array | null, logScale: boolean, flip: boolean): Float32Array | null {
  if (!data) return null;
  if (!logScale && !flip) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = logScale ? signedLog1p(data[i]) : data[i];
    out[i] = flip ? -v : v;
  }
  return out;
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
// Inlined components (Histogram + InfoTooltip + KeyboardShortcuts)
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
  onRangeCommit?: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
  pinBinsToRange?: boolean;
  ariaHidden?: boolean;
}

function Histogram({
  data, vminPct, vmaxPct, onRangeChange, onRangeCommit,
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
  const [liveRange, setLiveRange] = React.useState<[number, number]>([vminPct, vmaxPct]);
  React.useEffect(() => { setLiveRange([vminPct, vmaxPct]); }, [vminPct, vmaxPct]);
  const [liveVminPct, liveVmaxPct] = liveRange;
  const colors = React.useMemo(() => theme === "dark"
    ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
    : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" },
  [theme]);
  const normalizeRange = (value: number[]): [number, number] => {
    const [newMin, newMax] = value;
    return [Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1)];
  };
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
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
    const vminBin = Math.floor((liveVminPct / 100) * displayBins);
    const vmaxBin = Math.floor((liveVmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, liveVminPct, liveVmaxPct, width, height, colors]);
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
        value={liveRange}
        onChange={(_, v) => {
          const next = normalizeRange(v as number[]);
          setLiveRange(next);
          onRangeChange(next[0], next[1]);
        }}
        onChangeCommitted={(_, v) => {
          const next = normalizeRange(v as number[]);
          setLiveRange(next);
          (onRangeCommit ?? onRangeChange)(next[0], next[1]);
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
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(liveVminPct)}</Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(liveVmaxPct)}</Typography>
      </Box>
    </Box>
  );
}

interface LiveNumberSliderProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onLiveChange: (value: number) => void;
  onCommit: (value: number) => void;
  size?: "small" | "medium";
  valueLabelDisplay?: "auto" | "on" | "off";
  sx?: React.ComponentProps<typeof Slider>["sx"];
  ariaLabel: string;
}

const LiveNumberSlider = React.memo(function LiveNumberSlider({
  value, min, max, step, onLiveChange, onCommit, size = "small", valueLabelDisplay = "auto", sx, ariaLabel,
}: LiveNumberSliderProps) {
  const [liveValue, setLiveValue] = React.useState(value);
  React.useEffect(() => { setLiveValue(value); }, [value]);
  return (
    <Slider
      value={liveValue}
      min={min}
      max={max}
      step={step}
      onChange={(_, v) => {
        const next = v as number;
        setLiveValue(next);
        onLiveChange(next);
      }}
      onChangeCommitted={(_, v) => {
        const next = v as number;
        setLiveValue(next);
        onCommit(next);
      }}
      size={size}
      valueLabelDisplay={valueLabelDisplay}
      sx={sx}
      aria-label={ariaLabel}
    />
  );
});

const controlLabel = { ...typography.label, ...typographyLabel };
const clickableControlLabel = {
  ...controlLabel,
  cursor: "pointer",
  userSelect: "none",
} as const;

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const HTML_EXPORT_OVERHEAD_BYTES = 700_000;

function formatEstimatedHtmlSize(payloadBytes: number): string {
  const htmlBytes = Math.max(0, payloadBytes) * 4 / 3 + HTML_EXPORT_OVERHEAD_BYTES;
  const mb = htmlBytes / (1024 * 1024);
  if (mb >= 100) return `~${Math.round(mb)} MB`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB`;
  return `~${mb.toFixed(2)} MB`;
}

const container = {
  // overflowX:auto so panels stay reachable via horizontal scroll on narrow
  // viewport instead of being silently clipped past the cell edge.
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", overflowX: "auto", overflowY: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse, createGPUColormapEngine, GPUColormapEngine } from "../colormaps";

import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, nextPow2, computeMagnitude, autoEnhanceFFT, applyHannWindow2D } from "../fft";

// ============================================================================
// Zoom constants (matching Show3D)
// ============================================================================
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 30;

// ============================================================================
// Constants
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const DEFAULT_FFT_ZOOM: ZoomState = { zoom: 2, panX: 0, panY: 0 };
const CANVAS_TARGET = 480;
const AXES = ["xy", "oblique"] as const;
const PANEL_NAMES = ["XY", "Oblique"] as const;
// Show3DSlices opens in the same orientation as the main top slice panel:
// x/columns left-to-right and y/rows top-to-bottom.
const SHOW3DSLICES_DEFAULT_CAMERA: CameraState = {
  ...DEFAULT_CAMERA,
  yaw: Math.PI,
  pitch: 0,
  roll: Math.PI,
};
const VOLUME_VIEW_PRESETS = [
  { value: "xy", label: "Top", description: "top (XY) view" },
  { value: "side", label: "Side", description: "oblique vertical plane view" },
] as const;
const DPR = window.devicePixelRatio || 1;

interface Show3DSlicesPerfCounters {
  widget: "Show3DSlices";
  dims: string;
  startedAt: number;
  lastUpdated: number;
  renderedFrames: number;
  visualFrames: number;
  directPaintFrames: number;
  playbackFrames: number;
  sliderFrames: number;
  contrastFrames: number;
  volumeFrames: number;
  zStretchFrames: number;
  zoomFrames: number;
  lastRenderMs: number;
  avgRenderMs: number;
  maxRenderMs: number;
  frameIntervalAvgMs: number;
  maxFrameIntervalMs: number;
  currentFps: number;
  minRecentFps: number;
  overBudgetFrames: number;
  lastPath: string;
  lastAction: string;
  lastAxis: number;
  lastIndex: number;
  gpuResident: boolean;
}

declare global {
  interface Window {
    __quantemShow3DSlicesPerf?: Show3DSlicesPerfCounters;
  }
}

// ============================================================================
// Main Component
// ============================================================================
const FFT_SNAP_RADIUS = 5;

function Show3DSlices() {
  // Theme detection (offline HTML exports force a light/white background)
  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const { themeInfo, colors: baseColors } = useTheme(offlineForTheme);
  const tc = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };

  const themedSelect = {
    ...controlPanel.select,
    bgcolor: tc.controlBg,
    color: tc.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: tc.controlBg, color: tc.text, border: `1px solid ${tc.border}` } },
  };

  // Model state
  const [nx] = useModelState<number>("nx");
  const [ny] = useModelState<number>("ny");
  const [nz] = useModelState<number>("nz");
  const [volumeBytes] = useModelState<DataView>("volume_bytes");
  const [offline] = useModelState<boolean>("offline");
  const [offlineMin] = useModelState<number>("_offline_min");
  const [offlineMax] = useModelState<number>("_offline_max");
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [sliceX, setSliceX] = useModelState<number>("slice_x");
  const [sliceY, setSliceY] = useModelState<number>("slice_y");
  const [sliceZ, setSliceZ] = useModelState<number>("slice_z");
  const [obliqueAngle, setObliqueAngle] = useModelState<number>("oblique_angle");
  const [obliqueProfileLine, setObliqueProfileLine] = useModelState<{ row: number; col: number }[]>("oblique_profile_line");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [showControls] = useModelState<boolean>("show_controls");
  const [showCrosshair] = useModelState<boolean>("show_crosshair");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [orthographic, setOrthographic] = useModelState<boolean>("orthographic");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [flip, setFlip] = useModelState<boolean>("flip");
  // No disabled_tools / hidden_tools traits in new monorepo Show3DSlices.
  const [dimLabels] = useModelState<string[]>("dim_labels");
  const [pixelSize] = useModelState<number>("pixel_size");
  // Per-axis sampling [pz, py, px] for anisotropic data; falls back to [pixelSize]*3.
  const [pixelSizeAxes] = useModelState<number[]>("pixel_size_axes");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [modelZStretch, setModelZStretch] = useModelState<number>("z_stretch");
  const [zStretch, setZStretch] = React.useState(modelZStretch);
  const pendingZStretchRef = React.useRef(modelZStretch);
  const zStretchLiveDirtyRef = React.useRef(false);
  const zStretchRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    zStretchLiveDirtyRef.current = false;
    pendingZStretchRef.current = modelZStretch;
    setZStretch(modelZStretch);
  }, [modelZStretch]);
  React.useEffect(() => {
    return () => {
      if (zStretchRafRef.current != null) cancelAnimationFrame(zStretchRafRef.current);
    };
  }, []);

  // Initialize WebGPU FFT
  React.useEffect(() => {
    let disposed = false;
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
    // Colormap engine: volume-resident GPU slice + colormap (no CPU per-scrub work).
    createGPUColormapEngine().then(engine => {
      if (disposed) { engine?.destroy(); return; }
      if (engine) { gpuCmapRef.current = engine; setCmapReady(true); }
    });
    return () => { disposed = true; gpuCmapRef.current?.destroy(); gpuCmapRef.current = null; volUploadedKeyRef.current = null; };
  }, []);

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const uiRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const imageBoxRefs = React.useRef<(HTMLDivElement | null)[]>([null, null, null]);

  // FFT state
  const [fftColormap, setFftColormap] = useModelState<string>("fft_colormap");
  const [fftLogScale, setFftLogScale] = useModelState<boolean>("fft_log_scale");
  const [fftAuto, setFftAuto] = useModelState<boolean>("fft_auto");
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [fftZooms, setFftZooms] = React.useState<ZoomState[]>([DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM]);
  const [fftDragAxis, setFftDragAxis] = React.useState<number | null>(null);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // FFT d-spacing measurement
  type FftClickInfo = {
    axis: number; row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  };
  const [fftClickInfo, setFftClickInfo] = React.useState<FftClickInfo | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number; axis: number } | null>(null);
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOverlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftImgDataRefs = React.useRef<(ImageData | null)[]>([null, null, null]);
  const fftMagCacheRefs = React.useRef<(Float32Array | null)[]>([null, null, null]);
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
  const [cmapReady, setCmapReady] = React.useState(false);
  const volUploadedKeyRef = React.useRef<Float32Array | null>(null);
  const gpuVolReadyRef = React.useRef(false);
  const perfRef = React.useRef<Show3DSlicesPerfCounters | null>(null);
  const recordPerfRef = React.useRef<(
    action: string,
    renderMs: number,
    axis?: number,
    index?: number,
    gpuResident?: boolean,
  ) => void>(() => {});
  recordPerfRef.current = (
    action: string,
    renderMs: number,
    axis = -1,
    index = -1,
    gpuResident = gpuVolReadyRef.current,
  ) => {
    const now = performance.now();
    const dims = `${nz}x${ny}x${nx}`;
    let p = perfRef.current;
    if (!p || p.dims !== dims) {
      p = {
        widget: "Show3DSlices",
        dims,
        startedAt: now,
        lastUpdated: 0,
        renderedFrames: 0,
        visualFrames: 0,
        directPaintFrames: 0,
        playbackFrames: 0,
        sliderFrames: 0,
        contrastFrames: 0,
        volumeFrames: 0,
        zStretchFrames: 0,
        zoomFrames: 0,
        lastRenderMs: 0,
        avgRenderMs: 0,
        maxRenderMs: 0,
        frameIntervalAvgMs: 0,
        maxFrameIntervalMs: 0,
        currentFps: 0,
        minRecentFps: 0,
        overBudgetFrames: 0,
        lastPath: "",
        lastAction: "",
        lastAxis: -1,
        lastIndex: -1,
        gpuResident,
      };
      perfRef.current = p;
      window.__quantemShow3DSlicesPerf = p;
    }

    p.renderedFrames += 1;
    p.directPaintFrames += action === "slider" || action === "playback" || action === "contrast" || action === "loop" || action === "stop" || action === "volumeSlice" ? 1 : 0;
    p.sliderFrames += action === "slider" ? 1 : 0;
    p.playbackFrames += action === "playback" ? 1 : 0;
    p.contrastFrames += action === "contrast" ? 1 : 0;
    p.volumeFrames += action === "volume" || action === "volumeWheel" || action === "volumeDrag" ? 1 : 0;
    p.zStretchFrames += action === "zStretch" ? 1 : 0;
    p.zoomFrames += action === "zoom" || action === "pan" ? 1 : 0;
    p.lastRenderMs = renderMs;
    p.avgRenderMs = p.renderedFrames === 1 ? renderMs : p.avgRenderMs * 0.9 + renderMs * 0.1;
    p.maxRenderMs = Math.max(p.maxRenderMs, renderMs);
    p.lastPath = gpuResident ? "webgpu-resident" : "fallback";
    p.lastAction = action;
    p.lastAxis = axis;
    p.lastIndex = index;
    p.gpuResident = gpuResident;

    const prevUpdated = p.lastUpdated;
    const dt = prevUpdated > 0 ? now - prevUpdated : 0;
    // Several GPU passes can happen inside one requestAnimationFrame (for
    // example slice paint plus 3D plane overlay). Count those as one visual
    // frame for FPS, otherwise the displayed FPS is inflated.
    if (prevUpdated === 0 || dt >= 4) {
      p.lastUpdated = now;
      p.visualFrames += 1;
    }
    if (prevUpdated > 0 && dt >= 4) {
      p.frameIntervalAvgMs = p.frameIntervalAvgMs === 0 ? dt : p.frameIntervalAvgMs * 0.9 + dt * 0.1;
      p.maxFrameIntervalMs = Math.max(p.maxFrameIntervalMs, dt);
      p.currentFps = p.frameIntervalAvgMs > 0 ? 1000 / p.frameIntervalAvgMs : 0;
      if (p.currentFps > 0) p.minRecentFps = p.minRecentFps === 0 ? p.currentFps : Math.min(p.minRecentFps, p.currentFps);
      if (dt > 1000 / 60) p.overBudgetFrames += 1;
    }
    window.__quantemShow3DSlicesPerf = p;
  };
  // Live params snapshot for direct-paint (slider handler bypasses React).
  const paintParamsRef = React.useRef<{
    cmap: string; logScale: boolean; flip: boolean; autoContrast: boolean;
    imageVminPct: number; imageVmaxPct: number; imageDataRange: { min: number; max: number };
    traitVmin: number | null; traitVmax: number | null;
    zooms: { zoom: number; panX: number; panY: number }[]; canvasSizes: { w: number; h: number }[]; smooth: boolean;
  } | null>(null);
  const fftComputeGenerationRef = React.useRef(0);
  const [gpuReady, setGpuReady] = React.useState(false);
  // Counter to trigger FFT redraw after async compute finishes
  const [fftVersion, setFftVersion] = React.useState(0);

  // Zoom/pan per axis
  const [zooms, setZooms] = React.useState<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  const [dragAxis, setDragAxis] = React.useState<number | null>(null);
  const [dragStart, setDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  // rAF bypass: keep live zoom in ref during drag, sync to React state on mouseup.
  // Only sync ref from state when NOT dragging - otherwise an unrelated re-render
  // (playback tick, cursor update) would clobber in-flight pan values.
  const liveZoomsRef = React.useRef<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  const liveZoomDirtyRef = React.useRef(false);
  if (dragAxis === null && !liveZoomDirtyRef.current) liveZoomsRef.current = zooms;
  const zoomRafRef = React.useRef<number>(0);
  const zoomCommitTimeoutRef = React.useRef<number | null>(null);
  const liveFftZoomsRef = React.useRef<ZoomState[]>([DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM]);
  const liveFftZoomDirtyRef = React.useRef(false);
  if (fftDragAxis === null && !liveFftZoomDirtyRef.current) liveFftZoomsRef.current = fftZooms;
  const fftZoomRafRef = React.useRef<number>(0);
  const fftZoomCommitTimeoutRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    return () => {
      if (zoomCommitTimeoutRef.current != null) window.clearTimeout(zoomCommitTimeoutRef.current);
      if (fftZoomCommitTimeoutRef.current != null) window.clearTimeout(fftZoomCommitTimeoutRef.current);
    };
  }, []);

  // Canvas resize (matching Show2D pattern)
  const [canvasTarget, setCanvasTarget] = React.useState(CANVAS_TARGET);
  const [sideCanvasTarget, setSideCanvasTarget] = React.useState(CANVAS_TARGET);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number; target: "primary" | "side" } | null>(null);

  // Playback state (synced with Python)
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [playAxis, setPlayAxis] = useModelState<number>("play_axis");
  const playbackAxis = playAxis === 0 || playAxis === 3 ? playAxis : 1;
  React.useEffect(() => {
    if (playAxis !== playbackAxis) setPlayAxis(playbackAxis);
  }, [playAxis, playbackAxis, setPlayAxis]);
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [modelFps, setModelFps] = useModelState<number>("fps");
  const [fps, setFps] = React.useState(() => Math.max(1, Math.min(MAX_PLAYBACK_FPS, modelFps)));
  const fpsRef = React.useRef(Math.max(1, Math.min(MAX_PLAYBACK_FPS, modelFps)));
  React.useEffect(() => {
    const capped = Math.max(1, Math.min(MAX_PLAYBACK_FPS, modelFps));
    fpsRef.current = capped;
    setFps(capped);
  }, [modelFps]);
  const [loop, setLoop] = useModelState<boolean>("loop");
  const playRafRef = React.useRef<number | null>(null);
  const lastPlayTsRef = React.useRef<number | null>(null);
  const playAccumulatorRef = React.useRef(0);
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const bounceDirRef = React.useRef<1 | -1>(1);
  const [loopStarts, setLoopStarts] = React.useState([0, 0, 0]);
  const [loopEnds, setLoopEnds] = React.useState([-1, -1, -1]);
  const loopStartsRef = React.useRef(loopStarts);
  const loopEndsRef = React.useRef(loopEnds);
  const pendingLoopRangeRef = React.useRef<{ starts: number[]; ends: number[] } | null>(null);
  const loopRangeRafRef = React.useRef<number | null>(null);
  React.useEffect(() => { loopStartsRef.current = loopStarts; }, [loopStarts]);
  React.useEffect(() => { loopEndsRef.current = loopEnds; }, [loopEnds]);
  React.useEffect(() => () => {
    if (loopRangeRafRef.current != null) cancelAnimationFrame(loopRangeRafRef.current);
  }, []);
  const fastTrackSliceRef = React.useRef<((axis: number, value: number) => void) | null>(null);
  const commitSliceValuesRef = React.useRef<() => void>(() => {});
  const pausePlaybackForEdit = React.useCallback(() => {
    if (playRafRef.current != null) {
      cancelAnimationFrame(playRafRef.current);
      playRafRef.current = null;
    }
    lastPlayTsRef.current = null;
    playAccumulatorRef.current = 0;
    if (playing) setPlaying(false);
  }, [playing, setPlaying]);

  // 3D volume renderer state
  const volumeCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const volumeRendererRef = React.useRef<VolumeRenderer | null>(null);
  const [camera, setCamera] = React.useState<CameraState>(SHOW3DSLICES_DEFAULT_CAMERA);
  const [volumeDrag, setVolumeDrag] = React.useState<{
    button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number;
  } | null>(null);
  const [webgpuSupported, setWebgpuSupported] = React.useState(true);
  const [rendererReady, setRendererReady] = React.useState(0);
  const [volumeCanvasSize, setVolumeCanvasSize] = React.useState(CANVAS_TARGET);
  const [volumeResizing, setVolumeResizing] = React.useState(false);
  const volumeResizeStartRef = React.useRef<{ x: number; y: number; size: number } | null>(null);
  const [showSlicePlanes, setShowSlicePlanes] = useModelState<boolean | undefined>("show_slice_planes");
  const [planeVisibility, setPlaneVisibility] = useModelState<boolean[] | undefined>("plane_visibility");
  const normalizedPlaneVisibility = PLANE_KEYS.map((_, i) => Boolean(planeVisibility?.[i] ?? showSlicePlanes ?? true));
  const visiblePlanes = PLANE_KEYS.filter((_, i) => normalizedPlaneVisibility[i]);
  const slicePlaneMask = normalizedPlaneVisibility.reduce((mask, visible, i) => (
    visible ? mask | (1 << i) : mask
  ), 0);
  const anySlicePlaneVisible = slicePlaneMask !== 0;

  // Histogram state
  const [imageVminPct, setImageVminPct] = useModelState<number>("image_vmin_pct");
  const [imageVmaxPct, setImageVmaxPct] = useModelState<number>("image_vmax_pct");
  const manualImageRangeBeforeAutoRef = React.useRef<{ min: number; max: number } | null>(null);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);

  // Volume opacity for the 3D context renderer.
  const [opacityA, setOpacityA] = useModelState<number>("volume_opacity");
  // Slice plane opacity in 3D renderer
  const [slicePlaneOpacity, setSlicePlaneOpacity] = useModelState<number>("slice_plane_opacity");
  const pendingVolumeControlsRef = React.useRef({ opacity: opacityA, slicePlaneOpacity });
  const volumeControlsRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    return () => {
      if (volumeControlsRafRef.current != null) cancelAnimationFrame(volumeControlsRafRef.current);
    };
  }, []);

  // Cached offscreen canvases for slice rendering (avoids recomputing colormap on zoom/pan)
  const sliceOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  // Reusable ImageData per axis to avoid GC churn (allocated once per dimension change)
  const sliceImgDataRefs = React.useRef<(ImageData | null)[]>([null, null, null]);

  // Colorbar state
  const [showColorbar, setShowColorbar] = useModelState<boolean>("show_colorbar");
  const [exportMenuAnchor, setExportMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const pendingExportRef = React.useRef<{
    id: string;
    filename: string;
    mode: string;
    handle: Show3DSlicesFileHandle | null;
  } | null>(null);

  const effectiveShowFft = showFft;

  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) {
      setExportBusy(true);
    } else if (!pendingExportRef.current) {
      setExportBusy(false);
    }
  }, [exportStatus]);

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; view: string } | null>(null);
  const [obliqueHoverTarget, setObliqueHoverTarget] = React.useState<"endpoint" | "line" | null>(null);
  const cursorInfoRef = React.useRef<typeof cursorInfo>(null);
  const pendingCursorInfoRef = React.useRef<typeof cursorInfo>(null);
  const cursorRafRef = React.useRef<number | null>(null);
  const setCursorInfoThrottled = (next: typeof cursorInfo) => {
    pendingCursorInfoRef.current = next;
    if (cursorRafRef.current != null) return;
    cursorRafRef.current = requestAnimationFrame(() => {
      cursorRafRef.current = null;
      const pending = pendingCursorInfoRef.current;
      const prev = cursorInfoRef.current;
      const same = prev === pending || (!!prev && !!pending &&
        prev.row === pending.row && prev.col === pending.col && prev.view === pending.view && prev.value === pending.value);
      if (!same) {
        cursorInfoRef.current = pending;
        setCursorInfo(pending);
      }
    });
  };
  React.useEffect(() => () => {
    if (cursorRafRef.current != null) cancelAnimationFrame(cursorRafRef.current);
  }, []);

  // Parse volume data. Live notebooks receive exact float32 bytes; offline
  // reports receive uint8 bytes plus global min/max metadata to reduce HTML size.
  const allFloats = React.useMemo(
    () => extractVolumeFloat32(volumeBytes, offline, offlineMin, offlineMax, nx, ny, nz),
    [volumeBytes, offline, offlineMin, offlineMax, nx, ny, nz],
  );
  const obliqueSegment = React.useMemo(() => {
    const startFromState = profilePointFromAny(obliqueProfileLine?.[0]);
    const stopFromState = profilePointFromAny(obliqueProfileLine?.[1]);
    if (startFromState && stopFromState) {
      return {
        start: clampPointToImage(startFromState, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
        stop: clampPointToImage(stopFromState, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
        explicit: true,
      };
    }
    const [start, stop] = obliqueLineEndpoints(nx, ny, sliceX, sliceY, obliqueAngle);
    return {
      start: clampPointToImage(start, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
      stop: clampPointToImage(stop, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
      explicit: false,
    };
  }, [obliqueProfileLine, nx, ny, sliceX, sliceY, obliqueAngle]);
  // SYNCHRONOUS data range (useMemo, not useState+effect). If this lands a frame
  // late, the first render uses the default {0,1} range so a value-based contrast
  // (vmin/vmax) converts to the wrong percent -> secondary planes paint with the
  // wrong contrast ("blue") until a scrub recomputes. Inline makes frame 1 correct.
  const imageDataRange = React.useMemo(
    () => (allFloats && allFloats.length > 0 ? findDataRange(allFloats) : { min: 0, max: 1 }),
    [allFloats],
  );
  const voxelCount = Math.max(0, Math.floor(nx) * Math.floor(ny) * Math.floor(nz));
  const exactExportSize = formatEstimatedHtmlSize(voxelCount * 4);
  const quantizedExportSize = formatEstimatedHtmlSize(voxelCount);
  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportMenuAnchor(event.currentTarget);
  };
  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };
  const handleExportSelect = async (mode: string) => {
    setExportMenuAnchor(null);
    if (mode !== "exact" && mode !== "quantized") return;
    const filename = makeExportFilename(title, nz, ny, nx, mode);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as Show3DSlicesWindow).showSaveFilePicker;
    let handle: Show3DSlicesFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [{ description: "Standalone HTML", accept: { "text/html": [".html"] } }],
        });
      } catch (err) {
        if (isAbortLikeError(err)) {
          setExportBusy(false);
          setLocalExportStatus("Export canceled");
          return;
        }
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }
    pendingExportRef.current = { id, filename, mode, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode, id, filename, download: true }));
  };

  React.useEffect(() => {
    const pending = pendingExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    let canceled = false;
    const save = async () => {
      const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes
        : bytes.slice();
      const filename = exportPayloadFilename || pending.filename;
      const blob = new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" });
      try {
        if (pending.handle) {
          setLocalExportStatus(`Saving ${filename}...`);
          const writable = await pending.handle.createWritable();
          await writable.write(blob);
          await writable.close();
        } else {
          downloadBlob(blob, filename);
        }
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(bytes.byteLength)})`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      } catch (err) {
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // Slice dimensions: [xy: ny x nx], [oblique: nz x diagonal]
  const sliceDims = React.useMemo<[number, number][]>(
    () => [[ny, nx], [nz, segmentWidth(obliqueSegment.start, obliqueSegment.stop)]],
    [ny, nx, nz, obliqueSegment],
  );

  // Canvas sizes. For depth panels, keep the Z scale independent from the
  // oblique profile length; otherwise shortening the profile would secretly
  // magnify Z before the explicit z_stretch slider is applied.
  // smooth=true → CSS bilinear (auto); smooth=false → nearest-neighbor (pixelated).
  // Overlay canvases (crosshair, scale bar, colorbar, FFT scale bar) use displayH
  // for their pixel buffer to avoid distortion under CSS stretch.
  const canvasSizes = React.useMemo(() => sliceDims.map(([h, w], a) => {
    const isDepth = a > 0;
    const target = isDepth ? sideCanvasTarget : canvasTarget;
    const baseW = isDepth ? target : Math.round(w * (target / Math.max(w, h)));
    const baseH = isDepth ? Math.max(1, h) : Math.round(h * (target / Math.max(w, h)));
    const scaleX = baseW / Math.max(1, w);
    const scaleY = baseH / Math.max(1, h);
    const displayH = isDepth ? Math.min(target, Math.round(baseH * Math.max(1, zStretch))) : baseH;
    return { w: baseW, h: baseH, displayH, scale: scaleX, scaleX, scaleY };
  }), [sliceDims, sideCanvasTarget, canvasTarget, zStretch]);
  const dataPointToCanvas = React.useCallback((axis: number, x: number, y: number): { x: number; y: number } => {
    const { w: cw, h: ch, displayH: dh, scaleX, scaleY } = canvasSizes[axis];
    const stretchY = dh / ch;
    const zs = liveZoomsRef.current[axis];
    const cx = cw / 2, cy = dh / 2;
    let canvasX = x * scaleX;
    let canvasY = y * scaleY * stretchY;
    if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
      canvasX = (canvasX - cx) * zs.zoom + cx + zs.panX;
      canvasY = (canvasY - cy) * zs.zoom + cy + zs.panY * stretchY;
    }
    return { x: canvasX, y: canvasY };
  }, [canvasSizes]);
  const rasterCanvasSizes = React.useMemo(() => sliceDims.map(([h, w], a) => {
    const isDepth = a > 0;
    const target = isDepth ? sideCanvasTarget : canvasTarget;
    const baseW = isDepth ? target : Math.round(w * (target / Math.max(w, h)));
    const baseH = isDepth ? Math.max(1, h) : Math.round(h * (target / Math.max(w, h)));
    const scaleX = baseW / Math.max(1, w);
    const scaleY = baseH / Math.max(1, h);
    return {
      w: baseW,
      h: baseH,
      scale: scaleX,
      scaleX,
      scaleY,
    };
  }), [sliceDims, sideCanvasTarget, canvasTarget]);

  // Pre-allocate reusable offscreen canvases + ImageData per axis (avoids GC churn)
  React.useEffect(() => {
    for (let a = 0; a < sliceDims.length; a++) {
      const [h, w] = sliceDims[a];
      // Check if existing offscreen matches dimensions
      const existing = sliceOffscreenRefs.current[a];
      if (!existing || existing.width !== w || existing.height !== h) {
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        sliceOffscreenRefs.current[a] = c;
        sliceImgDataRefs.current[a] = new ImageData(w, h);
      }
    }
  }, [sliceDims]);

  // Prevent page scroll on canvases
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    fftCanvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      canvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
      fftCanvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
    };
  }, [allFloats, effectiveShowFft]);

  // Keep the exact full volume resident on the GPU. Hot display toggles (flip,
  // log, auto) must not allocate or upload a transformed 45M-voxel volume.
  const volumeFloats = allFloats;
  const histogramSample = React.useMemo(() => makeHistogramSample(allFloats), [allFloats]);
  const displayHistogramSample = React.useMemo(
    () => transformDisplaySample(histogramSample, logScale, false),
    [histogramSample, logScale],
  );

  // Compute UI histogram and auto-contrast from a deterministic sample. The
  // rendered slice pixels still come from the exact full-resolution GPU volume.
  React.useEffect(() => {
    if (!displayHistogramSample || displayHistogramSample.length === 0) return;
    setImageHistogramData(displayHistogramSample);
  }, [displayHistogramSample]);

  const displayDataRange = React.useMemo(() => {
    return resolveDisplayBounds(
      imageDataRange.min,
      imageDataRange.max,
      traitVmin,
      traitVmax,
      logScale,
    );
  }, [imageDataRange, traitVmin, traitVmax, logScale]);
  const renderRangeForFlip = (range: { vmin: number; vmax: number }) => (
    flip ? { vmin: -range.vmax, vmax: -range.vmin } : range
  );

  const handleAutoContrastChange = (on: boolean) => {
    if (on) {
      manualImageRangeBeforeAutoRef.current = { min: imageVminPct, max: imageVmaxPct };
    }
    setAutoContrast(on);
    if (on && imageHistogramData) {
      const { vmin: pmin, vmax: pmax } = percentileClip(imageHistogramData, 2, 98);
      const span = displayDataRange.max - displayDataRange.min;
      if (span > 0) {
        setImageVminPct(Math.max(0, Math.min(100, ((pmin - displayDataRange.min) / span) * 100)));
        setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - displayDataRange.min) / span) * 100)));
      }
    } else {
      const restore = manualImageRangeBeforeAutoRef.current;
      if (restore) {
        setImageVminPct(restore.min);
        setImageVmaxPct(restore.max);
        manualImageRangeBeforeAutoRef.current = null;
      } else {
        setImageVminPct(0);
        setImageVmaxPct(100);
      }
    }
  };

  // Initial-mount Auto snap: when autoContrast is true from Python and histogram data
  // just loaded with default 0/100 slider, snap thumbs to 2/98 percentile so user sees
  // the actual range being rendered.
  React.useEffect(() => {
    if (!autoContrast || !imageHistogramData) return;
    if (imageVminPct !== 0 || imageVmaxPct !== 100) return;  // user already moved
    const { vmin: pmin, vmax: pmax } = percentileClip(imageHistogramData, 2, 98);
    const span = displayDataRange.max - displayDataRange.min;
    if (span > 0) {
      setImageVminPct(Math.max(0, Math.min(100, ((pmin - displayDataRange.min) / span) * 100)));
      setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - displayDataRange.min) / span) * 100)));
    }
  }, [autoContrast, imageHistogramData, displayDataRange]);


  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // -------------------------------------------------------------------------
  // 3D Volume Renderer - init, upload, render
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas) return;
    if (!VolumeRenderer.isSupported()) { setWebgpuSupported(false); return; }
    let disposed = false;
    VolumeRenderer.create(canvas).then(renderer => {
      if (disposed) { renderer.dispose(); return; }
      volumeRendererRef.current = renderer;
      setRendererReady(n => n + 1);
    }).catch(() => { setWebgpuSupported(false); });
    return () => { disposed = true; volumeRendererRef.current?.dispose(); volumeRendererRef.current = null; };
  }, []);


  // Upload volume data
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    renderer.uploadVolume(volumeFloats, nx, ny, nz);
  }, [volumeFloats, nx, ny, nz, rendererReady]);

  // Upload colormap. When flip, reverse the LUT entry order so the 3D volume
  // inverts contrast the same way slice panels do (slices negate the data and
  // swap vmin/vmax, equivalent to reversing the colormap lookup). LUT is
  // 256 RGB triplets (768 bytes); reverse per-entry, not per-byte.
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    renderer.uploadColormap(flip ? reverseLut(lut) : lut);
  }, [cmap, rendererReady, flip]);

  // Render 3D volume
  // Map slider %s + optional traitVmin/Vmax to the texture's [0,1] normalized space.
  // The 3D context texture is uploaded from raw data only once; log/flip are
  // hot display toggles handled by the exact slice shader and LUT reversal, not
  // by re-uploading a transformed volume.
  const volumeTextureRangeForPercent = (minPct: number, maxPct: number) => {
    const span = imageDataRange.max - imageDataRange.min;
    if (span <= 0) return { vmin: 0, vmax: 1 };
    const subMinData = imageDataRange.min + span * (minPct / 100);
    const subMaxData = imageDataRange.min + span * (maxPct / 100);
    const subMin = (subMinData - imageDataRange.min) / span;
    const subMax = (subMaxData - imageDataRange.min) / span;
    return { vmin: subMin, vmax: subMax };
  };
  const volTexRange = volumeTextureRangeForPercent(imageVminPct, imageVmaxPct);
  // Keep live slice positions separate from committed model traits. Slider drag
  // updates these refs every frame; model traits sync only on release.
  const liveSliceParamsRef = React.useRef({ sliceX, sliceY, sliceZ });
  const committedSliceParamsRef = React.useRef({ sliceX, sliceY, sliceZ });
  const committedSliceParams = committedSliceParamsRef.current;
  if (
    committedSliceParams.sliceX !== sliceX ||
    committedSliceParams.sliceY !== sliceY ||
    committedSliceParams.sliceZ !== sliceZ
  ) {
    const next = { sliceX, sliceY, sliceZ };
    committedSliceParamsRef.current = next;
    liveSliceParamsRef.current = next;
  }

  // Keep render params in ref for direct rAF rendering (bypasses React during drag)
  const volumeRenderParamsRef = React.useRef({
    ...liveSliceParamsRef.current, nx, ny, nz,
    opacity: opacityA, brightness: 1.0, slicePlaneMask, slicePlaneOpacity,
    obliqueAngleDeg: obliqueAngle,
    obliqueStartX: obliqueSegment.start.x,
    obliqueStartY: obliqueSegment.start.y,
    obliqueEndX: obliqueSegment.stop.x,
    obliqueEndY: obliqueSegment.stop.y,
    vmin: volTexRange.vmin, vmax: volTexRange.vmax,
  });
  volumeRenderParamsRef.current = {
    ...liveSliceParamsRef.current, nx, ny, nz,
    opacity: opacityA, brightness: 1.0, slicePlaneMask, slicePlaneOpacity,
    obliqueAngleDeg: obliqueAngle,
    obliqueStartX: obliqueSegment.start.x,
    obliqueStartY: obliqueSegment.start.y,
    obliqueEndX: obliqueSegment.stop.x,
    obliqueEndY: obliqueSegment.stop.y,
    vmin: volTexRange.vmin, vmax: volTexRange.vmax,
  };
  const bgColorRef = React.useRef<[number, number, number]>([0, 0, 0]);
  React.useEffect(() => {
    const r = parseInt(tc.bg.slice(1, 3), 16) / 255;
    const g = parseInt(tc.bg.slice(3, 5), 16) / 255;
    const b = parseInt(tc.bg.slice(5, 7), 16) / 255;
    bgColorRef.current = [r, g, b];
  }, [tc.bg]);

  // Render 3D volume (non-interactive: triggered by React state changes)
  React.useEffect(() => {
    if (volumeDrag) return; // Skip during drag - rAF handles it directly
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    renderer.render(volumeRenderParamsRef.current, camera, bgColorRef.current, undefined, undefined, zStretch, orthographic);
  }, [volumeFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, nx, ny, nz, cmap, camera, volumeCanvasSize, tc.bg, slicePlaneMask, slicePlaneOpacity, volumeDrag, rendererReady, volTexRange, opacityA, zStretch, orthographic, flip]);

  // Prevent scroll on volume canvas
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas || !webgpuSupported) return;
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvas.addEventListener("wheel", preventDefault, { passive: false });
    return () => canvas.removeEventListener("wheel", preventDefault);
  }, [webgpuSupported]);

  // -------------------------------------------------------------------------
  // 3D Volume mouse handlers - document-level listeners for robust drag
  // -------------------------------------------------------------------------
  const volumeRafRef = React.useRef<number>(0);
  const liveCameraRef = React.useRef<CameraState>(camera);
  // Live z_stretch ref for rAF drag path - keeps latest value without re-binding closure.
  const zStretchRef = React.useRef(zStretch);
  if (!zStretchLiveDirtyRef.current) zStretchRef.current = zStretch;
  const applyDepthPanelHeight = (value: number) => {
    for (let axis = 1; axis < sliceDims.length; axis++) {
      const base = rasterCanvasSizes[axis];
      if (!base) continue;
      const displayH = Math.min(sideCanvasTarget, Math.round(base.h * Math.max(1, value)));
      const height = `${displayH}px`;
      const box = imageBoxRefs.current[axis];
      const canvas = canvasRefs.current[axis];
      const overlay = overlayRefs.current[axis];
      const ui = uiRefs.current[axis];
      if (box) box.style.height = height;
      if (canvas) canvas.style.height = height;
      if (overlay) overlay.style.height = height;
      if (ui) ui.style.height = height;
    }
  };
  React.useEffect(() => { applyDepthPanelHeight(zStretch); }, [zStretch, rasterCanvasSizes, sideCanvasTarget]);
  const handleZStretchChange = (value: number) => {
    zStretchLiveDirtyRef.current = true;
    pendingZStretchRef.current = value;
    zStretchRef.current = value;
    if (zStretchRafRef.current != null) return;
    zStretchRafRef.current = requestAnimationFrame(() => {
      zStretchRafRef.current = null;
      const next = pendingZStretchRef.current;
      applyDepthPanelHeight(next);
      const renderer = volumeRendererRef.current;
      if (renderer && volumeFloats && volumeFloats.length > 0) {
        const t0 = performance.now();
        renderer.render(volumeRenderParamsRef.current, liveCameraRef.current, bgColorRef.current, undefined, undefined, next, orthographic);
        recordPerfRef.current("zStretch", performance.now() - t0, -1, -1, true);
      }
    });
  };
  const handleZStretchCommit = (value: number) => {
    if (zStretchRafRef.current != null) {
      cancelAnimationFrame(zStretchRafRef.current);
      zStretchRafRef.current = null;
    }
    pendingZStretchRef.current = value;
    zStretchRef.current = value;
    applyDepthPanelHeight(value);
    zStretchLiveDirtyRef.current = false;
    setZStretch(value);
    setModelZStretch(value);
  };
  const handleVolumeControlChange = (key: "opacity" | "slicePlaneOpacity", value: number) => {
    pendingVolumeControlsRef.current = { ...pendingVolumeControlsRef.current, [key]: value };
    if (volumeControlsRafRef.current != null) return;
    volumeControlsRafRef.current = requestAnimationFrame(() => {
      volumeControlsRafRef.current = null;
      const next = pendingVolumeControlsRef.current;
      volumeRenderParamsRef.current = {
        ...volumeRenderParamsRef.current,
        opacity: next.opacity,
        slicePlaneOpacity: next.slicePlaneOpacity,
      };
      const renderer = volumeRendererRef.current;
      if (renderer && volumeFloats && volumeFloats.length > 0) {
        const t0 = performance.now();
        renderer.render(
          volumeRenderParamsRef.current,
          liveCameraRef.current,
          bgColorRef.current,
          undefined,
          undefined,
          zStretchRef.current,
          orthographic,
        );
        recordPerfRef.current("volume", performance.now() - t0, -1, -1, true);
      }
    });
  };
  const handleVolumeControlCommit = (key: "opacity" | "slicePlaneOpacity", value: number) => {
    pendingVolumeControlsRef.current = { ...pendingVolumeControlsRef.current, [key]: value };
    const next = pendingVolumeControlsRef.current;
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      opacity: next.opacity,
      slicePlaneOpacity: next.slicePlaneOpacity,
    };
    setOpacityA(next.opacity);
    setSlicePlaneOpacity(next.slicePlaneOpacity);
  };
  const handlePlaneVisibilityChange = (_event: React.MouseEvent<HTMLElement>, nextPlanes: string[]) => {
    const nextVisibility = PLANE_KEYS.map((key) => nextPlanes.includes(key));
    const nextMask = nextVisibility.reduce((mask, visible, i) => (
      visible ? mask | (1 << i) : mask
    ), 0);
    setPlaneVisibility(nextVisibility);
    setShowSlicePlanes(nextMask !== 0);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      slicePlaneMask: nextMask,
    };
    const renderer = volumeRendererRef.current;
    if (renderer && volumeFloats && volumeFloats.length > 0) {
      const t0 = performance.now();
      renderer.render(
        volumeRenderParamsRef.current,
        liveCameraRef.current,
        bgColorRef.current,
        undefined,
        undefined,
        zStretchRef.current,
        orthographic,
      );
      recordPerfRef.current("planeVisibility", performance.now() - t0, -1, -1, true);
    }
  };

  const handleObliqueAngleChange = (_event: Event, value: number | number[]) => {
    const nextAngle = Array.isArray(value) ? value[0] : value;
    const center = {
      x: (obliqueSegment.start.x + obliqueSegment.stop.x) / 2,
      y: (obliqueSegment.start.y + obliqueSegment.stop.y) / 2,
    };
    const length = Math.max(1, Math.hypot(
      obliqueSegment.stop.x - obliqueSegment.start.x,
      obliqueSegment.stop.y - obliqueSegment.start.y,
    ));
    const theta = (nextAngle * Math.PI) / 180;
    const halfDx = Math.cos(theta) * length / 2;
    const halfDy = Math.sin(theta) * length / 2;
    const start = clampPointToImage({ x: center.x - halfDx, y: center.y - halfDy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage({ x: center.x + halfDx, y: center.y + halfDy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    setObliqueAngle(nextAngle);
    setObliqueProfileLine(profileLinePayload(start, stop));
    setObliquePositionBounds(null);
    updateObliqueCenter((start.x + stop.x) / 2, (start.y + stop.y) / 2, nextAngle);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      obliqueAngleDeg: nextAngle,
      obliqueStartX: start.x,
      obliqueStartY: start.y,
      obliqueEndX: stop.x,
      obliqueEndY: stop.y,
    };
    const renderer = volumeRendererRef.current;
    if (renderer && volumeFloats && volumeFloats.length > 0) {
      const t0 = performance.now();
      renderer.render(
        volumeRenderParamsRef.current,
        liveCameraRef.current,
        bgColorRef.current,
        undefined,
        undefined,
        zStretchRef.current,
        orthographic,
      );
      recordPerfRef.current("obliqueAngle", performance.now() - t0, -1, -1, true);
    }
  };
  if (!volumeDrag) liveCameraRef.current = camera;
  const volumeDragDataRef = React.useRef<{ button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number } | null>(null);

  const handleVolumeMouseDown = (e: React.MouseEvent) => {
    const dragData = {
      button: e.button, x: e.clientX, y: e.clientY,
      yaw: camera.yaw, pitch: camera.pitch, panX: camera.panX, panY: camera.panY,
    };
    volumeDragDataRef.current = dragData;
    setVolumeDrag(dragData);
    e.preventDefault();
  };

  React.useEffect(() => {
    if (!volumeDrag) return;
    const onMove = (e: MouseEvent) => {
      const drag = volumeDragDataRef.current;
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      let next: CameraState;
      if (drag.button === 0 && !e.shiftKey) {
        next = {
          ...liveCameraRef.current,
          yaw: drag.yaw + dx * 0.005,
          pitch: Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, drag.pitch - dy * 0.005)),
        };
      } else {
        const sens = 0.003 * liveCameraRef.current.distance;
        next = {
          ...liveCameraRef.current,
          panX: drag.panX + dx * sens,
          panY: drag.panY - dy * sens,
        };
      }
      liveCameraRef.current = next;
      if (!volumeRafRef.current) {
        volumeRafRef.current = requestAnimationFrame(() => {
          volumeRafRef.current = 0;
          const cam = liveCameraRef.current;
          const params = volumeRenderParamsRef.current;
          const bg = bgColorRef.current;
          const rendererA = volumeRendererRef.current;
          if (rendererA) {
            const t0 = performance.now();
            rendererA.render(params, cam, bg, undefined, undefined, zStretchRef.current, orthographic);
            recordPerfRef.current("volumeDrag", performance.now() - t0, -1, -1, true);
          }
        });
      }
    };
    const onUp = () => {
      setCamera(liveCameraRef.current);
      setVolumeDrag(null);
      volumeDragDataRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [volumeDrag, orthographic]);

  const handleVolumeWheel = (e: React.WheelEvent) => {
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const next = { ...liveCameraRef.current, distance: Math.max(0.5, Math.min(10, liveCameraRef.current.distance * factor)) };
    liveCameraRef.current = next;
    const renderer = volumeRendererRef.current;
    if (renderer) {
      const t0 = performance.now();
      renderer.render(volumeRenderParamsRef.current, next, bgColorRef.current, undefined, undefined, zStretchRef.current, orthographic);
      recordPerfRef.current("volumeWheel", performance.now() - t0, -1, -1, true);
    }
    setCamera(next);
  };

  const handleVolumeDoubleClick = () => setCamera(SHOW3DSLICES_DEFAULT_CAMERA);

  const setVolumeView = (view: "xy" | "side") => {
    const distance = liveCameraRef.current.distance || camera.distance || SHOW3DSLICES_DEFAULT_CAMERA.distance;
    // Match the 2D slice panels rather than mathematical world-up:
    // Top: x right, row/y down. Side: x right, z down.
    const presets: Record<"xy" | "side", Pick<CameraState, "yaw" | "pitch" | "roll">> = {
      xy: { yaw: Math.PI, pitch: 0, roll: Math.PI },
      side: { yaw: 0, pitch: Math.PI * 0.49, roll: 0 },
    };
    const next = { ...SHOW3DSLICES_DEFAULT_CAMERA, ...presets[view], distance, panX: 0, panY: 0 };
    liveCameraRef.current = next;
    setCamera(next);
  };

  const rollVolumeView = (direction: -1 | 1) => {
    const current = liveCameraRef.current;
    const next = { ...current, roll: (current.roll ?? 0) + direction * Math.PI / 2 };
    liveCameraRef.current = next;
    setCamera(next);
  };

  // -------------------------------------------------------------------------
  // 3D Volume canvas resize
  // -------------------------------------------------------------------------
  const volumeResizeRafRef = React.useRef(0);

  const handleVolumeResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation(); e.preventDefault();
    setVolumeResizing(true);
    volumeResizeStartRef.current = { x: e.clientX, y: e.clientY, size: volumeCanvasSize };
  };

  React.useEffect(() => {
    if (!volumeResizing) return;
    const onMove = (e: MouseEvent) => {
      const start = volumeResizeStartRef.current;
      if (!start) return;
      const delta = Math.max(e.clientX - start.x, e.clientY - start.y);
      const newSize = Math.max(300, Math.min(800, start.size + delta));
      // Throttle canvas resize to rAF for smooth drag
      if (!volumeResizeRafRef.current) {
        volumeResizeRafRef.current = requestAnimationFrame(() => {
          volumeResizeRafRef.current = 0;
          setVolumeCanvasSize(newSize);
        });
      }
    };
    const onUp = () => {
      if (volumeResizeRafRef.current) { cancelAnimationFrame(volumeResizeRafRef.current); volumeResizeRafRef.current = 0; }
      setVolumeResizing(false);
      volumeResizeStartRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [volumeResizing]);

  const cameraChanged = camera.yaw !== SHOW3DSLICES_DEFAULT_CAMERA.yaw || camera.pitch !== SHOW3DSLICES_DEFAULT_CAMERA.pitch || (camera.roll ?? 0) !== (SHOW3DSLICES_DEFAULT_CAMERA.roll ?? 0) || camera.distance !== SHOW3DSLICES_DEFAULT_CAMERA.distance || camera.panX !== SHOW3DSLICES_DEFAULT_CAMERA.panX || camera.panY !== SHOW3DSLICES_DEFAULT_CAMERA.panY;

  // Reset Zoom is intentionally narrow: slice and FFT zoom/pan only. Camera,
  // contrast, colormap, and playback loop state have their own controls/state.
  const anyZoomDirty = zooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0)
    || fftZooms.some(z => z.zoom !== DEFAULT_FFT_ZOOM.zoom || z.panX !== DEFAULT_FFT_ZOOM.panX || z.panY !== DEFAULT_FFT_ZOOM.panY);

  // -------------------------------------------------------------------------
  // Build colormapped offscreen canvases (expensive: log scale, percentile, colormap LUT)
  // Per-panel: XY depends on sliceZ; oblique depends on the XY center and angle.
  // Excludes zoom/pan so dragging only triggers the cheap redraw below.
  // useLayoutEffect so offscreens are ready before the draw useLayoutEffect runs.
  // -------------------------------------------------------------------------
  const prevCacheRef = React.useRef<{
    sliceX: number; sliceY: number; sliceZ: number;
    cmap: string; logScale: boolean; autoContrast: boolean;
    imageVminPct: number; imageVmaxPct: number;
    imageRangeMin: number; imageRangeMax: number;
    allFloats: Float32Array | null;
    nx: number; ny: number; nz: number;
    traitVmin: number | null; traitVmax: number | null;
    flip: boolean;
  }>({ sliceX: -1, sliceY: -1, sliceZ: -1, cmap: "", logScale: false, autoContrast: false, imageVminPct: -1, imageVmaxPct: -1, imageRangeMin: Number.NaN, imageRangeMax: Number.NaN, allFloats: null, nx: 0, ny: 0, nz: 0, traitVmin: null, traitVmax: null, flip: false });

  React.useLayoutEffect(() => {
    if (!allFloats || allFloats.length === 0) return;

    const prev = prevCacheRef.current;
    const globalChanged = allFloats !== prev.allFloats || cmap !== prev.cmap ||
      logScale !== prev.logScale || autoContrast !== prev.autoContrast ||
      imageVminPct !== prev.imageVminPct || imageVmaxPct !== prev.imageVmaxPct ||
      displayDataRange.min !== prev.imageRangeMin || displayDataRange.max !== prev.imageRangeMax ||
      traitVmin !== prev.traitVmin || traitVmax !== prev.traitVmax ||
      flip !== prev.flip ||
      nx !== prev.nx || ny !== prev.ny || nz !== prev.nz;
    const axisChanged = [
      globalChanged || sliceZ !== prev.sliceZ,
      true,
    ];

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const extractors = [
      () => extractXY(allFloats, nx, ny, nz, sliceZ),
      () => extractOblique(allFloats, nx, ny, nz, obliqueSegment.start, obliqueSegment.stop),
    ];
    // GPU path: upload the whole volume ONCE; each scrub only slices + colormaps on
    // the GPU (no CPU extract / re-upload), so scrubbing stays buffer-smooth even on
    // a 1688x1688x16 volume. CPU path is the fallback (no engine / volume too big).
    const engine = gpuCmapRef.current;
    let gpuVolReady = false;
    if (cmapReady && engine && allFloats) {
      if (volUploadedKeyRef.current !== allFloats) {
        gpuVolReady = engine.uploadVolume(allFloats, nx, ny, nz);
        volUploadedKeyRef.current = gpuVolReady ? allFloats : null;
      } else {
        gpuVolReady = true;
      }
      if (gpuVolReady) engine.uploadLUT(cmap, lut);
    }
    gpuVolReadyRef.current = gpuVolReady;
    for (let a = 0; a < sliceDims.length; a++) {
      if (!axisChanged[a]) continue;
      const [sliceH, sliceW] = sliceDims[a];
      const hasTraitRange = traitVmin != null || traitVmax != null;
      const rMin = displayDataRange.min;
      const rMax = displayDataRange.max;
      let vmin: number, vmax: number;
      if (gpuVolReady && engine && a === 0) {
        // Stack-wide range on the GPU path: no per-slice CPU percentile scan, so
        // contrast stays consistent across slices and scrubbing never touches the CPU.
        if (imageVminPct > 0 || imageVmaxPct < 100) {
          ({ vmin, vmax } = sliderRange(rMin, rMax, imageVminPct, imageVmaxPct));
        } else {
          vmin = rMin; vmax = rMax;
        }
        ({ vmin, vmax } = renderRangeForFlip({ vmin, vmax }));
        // Always cache the native slice raster. The displayed panel may be
        // smaller, but zoom/pan must reveal source pixels instead of magnifying
        // a display-resolution scrub proxy.
        const bitmap = engine.renderVolumeSliceToImageBitmap(0, sliceZ, { vmin, vmax }, logScale, flip);
        if (bitmap) {
          let offscreen = sliceOffscreenRefs.current[a];
          if (!offscreen || offscreen.width !== bitmap.width || offscreen.height !== bitmap.height) {
            offscreen = document.createElement("canvas");
            offscreen.width = bitmap.width; offscreen.height = bitmap.height;
            sliceOffscreenRefs.current[a] = offscreen;
            sliceImgDataRefs.current[a] = null;
          }
          const octx = offscreen.getContext("2d");
          if (octx) { octx.clearRect(0, 0, offscreen.width, offscreen.height); octx.drawImage(bitmap, 0, 0); }
          bitmap.close();
          continue;
        }
      }
      // CPU fallback
      const processed = maybeFlip(logScale ? applyLogScale(extractors[a]()) : extractors[a](), flip);
      if (!hasTraitRange && autoContrast) {
        ({ vmin, vmax } = percentileClip(processed, 2, 98));
      } else if (imageVminPct > 0 || imageVmaxPct < 100) {
        ({ vmin, vmax } = sliderRange(rMin, rMax, imageVminPct, imageVmaxPct));
      } else {
        vmin = rMin; vmax = rMax;
      }
      ({ vmin, vmax } = renderRangeForFlip({ vmin, vmax }));
      const offscreen = sliceOffscreenRefs.current[a];
      const imgData = sliceImgDataRefs.current[a];
      if (offscreen && imgData && offscreen.width === sliceW && offscreen.height === sliceH) {
        renderToOffscreenReuse(processed, lut, vmin, vmax, offscreen, imgData);
      } else {
        sliceOffscreenRefs.current[a] = renderToOffscreen(processed, sliceW, sliceH, lut, vmin, vmax);
      }
    }
    prevCacheRef.current = { sliceX, sliceY, sliceZ, cmap, logScale, autoContrast, imageVminPct, imageVmaxPct, imageRangeMin: displayDataRange.min, imageRangeMax: displayDataRange.max, allFloats, nx, ny, nz, traitVmin, traitVmax, flip };
  }, [allFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, nx, ny, nz, cmap, logScale, autoContrast, sliceDims, imageVminPct, imageVmaxPct, displayDataRange, traitVmin, traitVmax, flip, cmapReady]);

  // Snapshot of everything direct-paint needs, refreshed every render so the
  // slider handler (which fires faster than React commits) reads current values.
  React.useEffect(() => {
    paintParamsRef.current = {
      cmap, logScale, flip, autoContrast, imageVminPct, imageVmaxPct, imageDataRange: displayDataRange,
      traitVmin, traitVmax, zooms, canvasSizes, smooth,
    };
  });

  // DIRECT PAINT (Show3D's 60fps-at-4k trick): paint ONE plane straight to its
  // visible canvas via the resident-volume GPU slice path, bypassing React. The
  // slice sliders are anywidget model traits (slice_x/y/z) whose setter does a
  // comm round-trip (model.set + save_changes) that React BATCHES during a drag,
  // so the render effect keyed on them doesn't fire per drag-frame -> the lag.
  // The slider onChange calls this for an INSTANT image, then sets the trait for
  // crosshair/title/state to catch up. The shader samples the float32 resident
  // volume and area-averages every source pixel covered by the displayed pixel.
  const directPaintPlane = React.useCallback((axis: number, idx: number, action = "slider"): boolean => {
    const t0 = performance.now();
    const engine = gpuCmapRef.current;
    const p = paintParamsRef.current;
    if (axis !== 0) return false;
    if (!engine || !gpuVolReadyRef.current || !p) return false;
    const canvas = canvasRefs.current[axis];
    if (!canvas) return false;
    const cs = p.canvasSizes[axis]; const zs = p.zooms[axis];
    if (!cs) return false;
    const rMin = p.imageDataRange.min;
    const rMax = p.imageDataRange.max;
    let vmin: number, vmax: number;
    if (p.imageVminPct > 0 || p.imageVmaxPct < 100) {
      ({ vmin, vmax } = sliderRange(rMin, rMax, p.imageVminPct, p.imageVmaxPct));
    } else { vmin = rMin; vmax = rMax; }
    ({ vmin, vmax } = p.flip ? { vmin: -vmax, vmax: -vmin } : { vmin, vmax });
    engine.uploadLUT(p.cmap, COLORMAPS[p.cmap] || COLORMAPS.inferno);
    const cw = cs.w, ch = cs.h;
    const bitmap = engine.renderVolumeSliceToImageBitmap(
      axis,
      idx,
      { vmin, vmax },
      p.logScale,
      p.flip,
      undefined,
      { zoom: zs?.zoom || 1, panX: zs?.panX || 0, panY: zs?.panY || 0, canvasW: cw, canvasH: ch },
    );
    if (!bitmap) return false;
    const ctx = canvas.getContext("2d");
    if (!ctx) { bitmap.close(); return false; }
    ctx.imageSmoothingEnabled = p.smooth;
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, cw, ch);
    bitmap.close();
    recordPerfRef.current(action, performance.now() - t0, axis, idx, true);
    return true;
  }, []);

  const renderVolumePlanesLive = React.useCallback((action = "volumeSlice") => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    const params = { ...volumeRenderParamsRef.current, ...liveSliceParamsRef.current };
    volumeRenderParamsRef.current = params;
    const t0 = performance.now();
    renderer.render(
      params,
      liveCameraRef.current,
      bgColorRef.current,
      1,
      32,
      zStretchRef.current,
      orthographic,
    );
    recordPerfRef.current(action, performance.now() - t0, -1, -1, true);
  }, [orthographic, volumeFloats]);

  // -------------------------------------------------------------------------
  // Redraw slices with zoom/pan (cheap: just drawImage from cached offscreen)
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    for (let a = 0; a < sliceDims.length; a++) {
      const canvas = canvasRefs.current[a];
      const offscreen = sliceOffscreenRefs.current[a];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      // Source rect = the offscreen's ACTUAL size. The GPU path renders at display
      // resolution so the offscreen may be smaller than the full slice; reading
      // sliceDims here would sample a partly-empty buffer.
      const srcW = offscreen.width, srcH = offscreen.height;
      const { w: cw, h: ch } = canvasSizes[a];
      ctx.imageSmoothingEnabled = smooth;
      ctx.clearRect(0, 0, cw, ch);
      const zs = zooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY);
        ctx.scale(zs.zoom, zs.zoom);
        ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, srcW, srcH, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, srcW, srcH, 0, 0, cw, ch);
      }
    }
  }, [allFloats, sliceX, sliceY, sliceZ, obliqueAngle, nx, ny, nz, cmap, logScale, autoContrast, zooms, sliceDims, canvasSizes, imageVminPct, imageVmaxPct, smooth, flip]);

  // -------------------------------------------------------------------------
  // Render crosshair lines for the orthogonal slice intersections.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!allFloats) return;
    const crossPositions: [number, number][] = [
      [sliceX, sliceY],
      [(sliceDims[1]?.[1] ?? 1) / 2, sliceZ],
    ];
    for (let a = 0; a < sliceDims.length; a++) {
      const overlay = overlayRefs.current[a];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch, displayH: dh, scaleX, scaleY } = canvasSizes[a];
      const stretchY = dh / ch;
      ctx.clearRect(0, 0, cw, dh);
      if (a === 0) {
        const { start, stop } = obliqueSegment;
        const startCanvas = dataPointToCanvas(a, start.x, start.y);
        const stopCanvas = dataPointToCanvas(a, stop.x, stop.y);
        ctx.save();
        ctx.strokeStyle = tc.accent;
        ctx.fillStyle = tc.accent;
        ctx.lineWidth = obliqueHoverTarget === "line" || obliqueHandleDragRef.current?.mode === "line" ? 2.5 : 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(startCanvas.x, startCanvas.y);
        ctx.lineTo(stopCanvas.x, stopCanvas.y);
        ctx.stroke();
        ctx.setLineDash([]);
        const drawHandle = (point: { x: number; y: number }) => {
          const activeEndpoint = obliqueHoverTarget === "endpoint" || obliqueHandleDragRef.current?.mode === "endpoint";
          const radius = activeEndpoint ? 6 : 4;
          const x = clampNumber(point.x, radius + 2, cw - radius - 2);
          const y = clampNumber(point.y, radius + 2, dh - radius - 2);
          if (activeEndpoint) {
            ctx.save();
            ctx.strokeStyle = tc.bg;
            ctx.lineWidth = 2;
          }
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fill();
          if (activeEndpoint) {
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x - radius - 4, y);
            ctx.lineTo(x - radius - 1, y);
            ctx.moveTo(x + radius + 1, y);
            ctx.lineTo(x + radius + 4, y);
            ctx.moveTo(x, y - radius - 4);
            ctx.lineTo(x, y - radius - 1);
            ctx.moveTo(x, y + radius + 1);
            ctx.lineTo(x, y + radius + 4);
            ctx.stroke();
            ctx.restore();
          }
        };
        drawHandle(startCanvas);
        drawHandle(stopCanvas);
        ctx.restore();
      }
      if (!showCrosshair) continue;
      const zs = zooms[a];
      const [dataX, dataY] = crossPositions[a];
      const cx = cw / 2, cy = dh / 2;
      let canvasX = dataX * scaleX;
      let canvasY = dataY * scaleY * stretchY;
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        canvasX = (canvasX - cx) * zs.zoom + cx + zs.panX;
        canvasY = (canvasY - cy) * zs.zoom + cy + zs.panY * stretchY;
      }
      ctx.strokeStyle = tc.accentYellow + "80";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(canvasX, 0); ctx.lineTo(canvasX, dh); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, canvasY); ctx.lineTo(cw, canvasY); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [allFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, obliqueHoverTarget, zooms, showCrosshair, tc, canvasSizes, sliceDims, nx, ny, dataPointToCanvas]);

  // -------------------------------------------------------------------------
  // Scale bar (HiDPI UI overlay)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let a = 0; a < sliceDims.length; a++) {
      const uiCanvas = uiRefs.current[a];
      if (!uiCanvas) continue;
      const { w: cw, displayH: dh } = canvasSizes[a];
      uiCanvas.width = Math.round(cw * DPR);
      uiCanvas.height = Math.round(dh * DPR);
      const uiCtx = uiCanvas.getContext("2d");
      if (!uiCtx) continue;
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      if (scaleBarVisible) {
        const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
        const theta = (obliqueAngle * Math.PI) / 180;
        const obliquePx = axes
          ? Math.hypot(Math.cos(theta) * axes[2], Math.sin(theta) * axes[1])
          : (pixelSize || 0);
        const pxSize = a === 0 ? (axes ? axes[2] : (pixelSize || 0)) : obliquePx;
        const sliceW = sliceDims[a][1];
        const unit = pxSize > 0 ? "Å" : "px";
        const size = pxSize > 0 ? pxSize : 1;
        drawScaleBarHiDPI(uiCanvas, DPR, zooms[a].zoom, size, unit, sliceW);
      }

      if (showColorbar) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        const baseMin = displayDataRange.min;
        const baseMax = displayDataRange.max;
        const { vmin, vmax } = sliderRange(baseMin, baseMax, imageVminPct, imageVmaxPct);
        const cssW = uiCanvas.width / DPR;
        const cssH = uiCanvas.height / DPR;
        uiCtx.save();
        uiCtx.scale(DPR, DPR);
        drawColorbar(uiCtx, cssW, cssH, lut, vmin, vmax, logScale);
        uiCtx.restore();
      }
    }
  }, [pixelSize, pixelSizeAxes, scaleBarVisible, zooms, canvasSizes, sliceDims, showColorbar, cmap, displayDataRange, imageVminPct, imageVmaxPct, obliqueAngle, themeInfo.theme]);

  // -------------------------------------------------------------------------
  // FFT computation and caching (per-axis: only recompute changed axes)
  // -------------------------------------------------------------------------
  const prevFFTCacheRef = React.useRef<{
    sliceX: number; sliceY: number; sliceZ: number;
    allFloats: Float32Array | null;
    fftColormap: string; fftLogScale: boolean; fftAuto: boolean; fftWindow: boolean; gpuReady: boolean;
    effectiveShowFft: boolean;
  }>({ sliceX: -1, sliceY: -1, sliceZ: -1, allFloats: null, fftColormap: "", fftLogScale: false, fftAuto: false, fftWindow: false, gpuReady: false, effectiveShowFft: false });

  React.useEffect(() => {
    if (!effectiveShowFft || !allFloats || allFloats.length === 0) {
      // Release FFT caches when toggling off (each is up to 64 MB per axis).
      if (prevFFTCacheRef.current.effectiveShowFft && !effectiveShowFft) {
        for (let a = 0; a < sliceDims.length; a++) {
          fftMagCacheRefs.current[a] = null;
          fftOffscreenRefs.current[a] = null;
          fftImgDataRefs.current[a] = null;
        }
        prevFFTCacheRef.current.effectiveShowFft = false;
      }
      return;
    }

    const prevFFT = prevFFTCacheRef.current;
    const globalFFTChanged = allFloats !== prevFFT.allFloats || fftColormap !== prevFFT.fftColormap ||
      fftLogScale !== prevFFT.fftLogScale || fftAuto !== prevFFT.fftAuto ||
      fftWindow !== prevFFT.fftWindow ||
      gpuReady !== prevFFT.gpuReady || !prevFFT.effectiveShowFft;
    const fftAxisChanged = [
      globalFFTChanged || sliceZ !== prevFFT.sliceZ,
      true,
    ];

    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const generation = ++fftComputeGenerationRef.current;
    let cancelled = false;

    const computeFFTsForVolume = async (
      floats: Float32Array,
      magCache: React.MutableRefObject<(Float32Array | null)[]>,
      offscreenCache: React.MutableRefObject<(HTMLCanvasElement | null)[]>,
      imgDataCache: React.MutableRefObject<(ImageData | null)[]>,
      forceAll: boolean,
    ) => {
      const extractors = [
        () => extractXY(floats, nx, ny, nz, sliceZ),
        () => extractOblique(floats, nx, ny, nz, obliqueSegment.start, obliqueSegment.stop),
      ];
      const dims = sliceDims;

      for (let a = 0; a < sliceDims.length; a++) {
        if (!forceAll && !fftAxisChanged[a]) continue;
        const extracted = extractors[a]();
        const data = fftWindow ? new Float32Array(extracted) : extracted;
        const [sliceH, sliceW] = dims[a];
        if (fftWindow) applyHannWindow2D(data, sliceW, sliceH);

        const pw = nextPow2(sliceW);
        const ph = nextPow2(sliceH);
        const paddedSize = pw * ph;
        let real: Float32Array, imag: Float32Array;

        if (gpuReady && gpuFFTRef.current) {
          const padReal = new Float32Array(paddedSize);
          const padImag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) padReal[y * pw + x] = data[y * sliceW + x];
          const result = await gpuFFTRef.current.fft2D(padReal, padImag, pw, ph, false);
          real = result.real; imag = result.imag;
        } else {
          real = new Float32Array(paddedSize);
          imag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) real[y * pw + x] = data[y * sliceW + x];
          fft2d(real, imag, pw, ph, false);
        }

        fftshift(real, pw, ph);
        fftshift(imag, pw, ph);

        const mag = computeMagnitude(real, imag);
        magCache.current[a] = mag;

        let displayMin: number, displayMax: number;
        if (fftAuto) {
          ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, pw, ph));
        } else {
          ({ min: displayMin, max: displayMax } = findDataRange(mag));
        }

        const displayData = fftLogScale ? applyLogScale(mag) : mag;
        if (fftLogScale) { displayMin = Math.log1p(displayMin); displayMax = Math.log1p(displayMax); }

        // Reuse cached offscreen if dims match - saves ~4 MB ImageData alloc per axis.
        const existingOff = offscreenCache.current[a];
        const existingImg = imgDataCache.current[a];
        if (existingOff && existingImg && existingOff.width === pw && existingOff.height === ph) {
          renderToOffscreenReuse(displayData, lut, displayMin, displayMax, existingOff, existingImg);
        } else {
          const offscreen = renderToOffscreen(displayData, pw, ph, lut, displayMin, displayMax);
          if (!offscreen) continue;
          offscreenCache.current[a] = offscreen;
          const ctx = offscreen.getContext("2d");
          imgDataCache.current[a] = ctx ? ctx.getImageData(0, 0, pw, ph) : null;
        }

        // Drawing is handled by the separate cheap redraw effect below
      }
    };

    const computeAllFFTs = async () => {
      const localMagCache = { current: fftMagCacheRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(Float32Array | null)[]>;
      const localOffscreenCache = { current: fftOffscreenRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(HTMLCanvasElement | null)[]>;
      const localImgDataCache = { current: fftImgDataRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(ImageData | null)[]>;
      await computeFFTsForVolume(allFloats, localMagCache, localOffscreenCache, localImgDataCache, false);
      if (cancelled || generation !== fftComputeGenerationRef.current) return false;
      fftMagCacheRefs.current = localMagCache.current;
      fftOffscreenRefs.current = localOffscreenCache.current;
      fftImgDataRefs.current = localImgDataCache.current;
      prevFFTCacheRef.current = { sliceX, sliceY, sliceZ, allFloats, fftColormap, fftLogScale, fftAuto, fftWindow, gpuReady, effectiveShowFft };
      return true;
    };

    // Debounce FFT compute during slider scrubbing: defer 80 ms so a 60 Hz drag
    // collapses to ~12 Hz, freeing the main thread for image redraws.
    const debounceMs = 80;
    const timeoutId = setTimeout(() => {
      if (cancelled) return;
      computeAllFFTs().then((committed) => { if (committed) setFftVersion(v => v + 1); });
    }, debounceMs);
    return () => { cancelled = true; clearTimeout(timeoutId); };
  }, [effectiveShowFft, allFloats, sliceX, sliceY, sliceZ, obliqueAngle, obliqueSegment, nx, ny, nz, sliceDims, fftColormap, fftLogScale, fftAuto, fftWindow, gpuReady]);

  // Redraw cached FFT with zoom/pan (cheap -- no recomputation)
  React.useLayoutEffect(() => {
    if (!effectiveShowFft) return;
    for (let a = 0; a < sliceDims.length; a++) {
      const canvas = fftCanvasRefs.current[a];
      const offscreen = fftOffscreenRefs.current[a];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch } = canvasSizes[a];
      const ow = offscreen.width, oh = offscreen.height;
      ctx.imageSmoothingEnabled = smooth;
      ctx.clearRect(0, 0, cw, ch);
      const zs = fftZooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
      }
    }
  }, [effectiveShowFft, fftZooms, canvasSizes, sliceDims, fftVersion, smooth]);

  // Render FFT overlays (reciprocal-space scale bars + d-spacing crosshair per axis)
  React.useEffect(() => {
    if (!effectiveShowFft) return;
    const dims = sliceDims;
    for (let a = 0; a < sliceDims.length; a++) {
      const overlay = fftOverlayRefs.current[a];
      if (!overlay) continue;
      const { w: cw, h: ch, displayH: dh } = canvasSizes[a];
      const stretchY = dh / ch;
      overlay.width = Math.round(cw * DPR);
      overlay.height = Math.round(dh * DPR);
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
      const theta = (obliqueAngle * Math.PI) / 180;
      const obliquePx = axes
        ? Math.hypot(Math.cos(theta) * axes[2], Math.sin(theta) * axes[1])
        : pixelSize;
      const realPx = a === 0 ? (axes ? axes[2] : pixelSize) : obliquePx;
      if (realPx > 0) {
        const [, sliceW] = dims[a];
        const pw = nextPow2(sliceW);
        const fftPixelSize = 1 / (pw * realPx);
        drawFFTScaleBarHiDPI(overlay, DPR, fftZooms[a].zoom, fftPixelSize, pw, "Å⁻¹");
      }

      if (fftClickInfo && fftClickInfo.axis === a) {
        const [sliceH, sliceW] = dims[a];
        const fftW = nextPow2(sliceW);
        const fftH = nextPow2(sliceH);

        ctx.save();
        ctx.scale(DPR, DPR);
        const zs = fftZooms[a];
        const cx = cw / 2, cy = dh / 2;
        const rawX = fftClickInfo.col / fftW * cw;
        const rawY = fftClickInfo.row / fftH * dh;
        const screenX = (rawX - cx) * zs.zoom + cx + zs.panX;
        const screenY = (rawY - cy) * zs.zoom + cy + zs.panY * stretchY;

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
          const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} \u00C5`;
          ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.fillStyle = "white";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(label, screenX + 10, screenY - 4);
        }
        ctx.restore();
      }
    }
  }, [effectiveShowFft, fftZooms, canvasSizes, pixelSize, pixelSizeAxes, nx, ny, nz, fftClickInfo]);

  // -------------------------------------------------------------------------
  // Playback logic (matching Show3D pattern)
  // -------------------------------------------------------------------------
  const sliceSettersRef = React.useRef<((v: number) => void)[]>([setSliceZ, setSliceY, setSliceX]);
  sliceSettersRef.current = [setSliceZ, setSliceY, setSliceX];
  const obliquePlaybackStateRef = React.useRef({ current: 0, start: 0, end: 0 });
  const effectiveLoopEnds = React.useMemo(() => loopEnds.map((end, i) => {
    const max = [nz - 1, ny - 1, nx - 1][i];
    return end < 0 ? max : Math.min(end, max);
  }), [loopEnds, nz, ny, nx]);
  React.useEffect(() => {
    if (!playing) return;
    let cancelled = false;
    let hiddenPaused = false;

    const clearPlayFrame = () => {
      if (playRafRef.current != null) {
        cancelAnimationFrame(playRafRef.current);
        playRafRef.current = null;
      }
    };

    const playbackAxes = playbackAxis === 3 ? [0, 1] : [playbackAxis];
    const axisBounds = (axis: number) => {
      if (axis === 0) return { start: loopStarts[0], end: effectiveLoopEnds[0], current: sliceValuesRef.current[0] };
      const state = obliquePlaybackStateRef.current;
      return { start: state.start, end: state.end, current: state.current };
    };
    const setPlaybackAxisFast = (axis: number, value: number) => {
      if (axis === 0) {
        if (fastTrackSliceRef.current) fastTrackSliceRef.current(0, value);
        else sliceSettersRef.current[0](value);
        sliceValuesRef.current[0] = value;
        return;
      }
      obliquePlaybackStateRef.current = { ...obliquePlaybackStateRef.current, current: value };
      updateObliqueFromNormalOffset(value);
    };

    const advanceAllAxes = (): boolean => {
      const dir = boomerang ? bounceDirRef.current : (reverse ? -1 : 1);
      let wouldHitEdge = false;
      for (const axis of playbackAxes) {
        const { start, end, current } = axisBounds(axis);
        const next = current + dir;
        if (next > end || next < start) {
          wouldHitEdge = true;
          break;
        }
      }
      if (boomerang && wouldHitEdge) {
        bounceDirRef.current = (-bounceDirRef.current) as 1 | -1;
      }
      const finalDir = boomerang ? bounceDirRef.current : dir;
      for (const axis of playbackAxes) {
        const { start, end, current } = axisBounds(axis);
        let next = current + finalDir;
        if (next > end) next = loop || boomerang ? start : end;
        else if (next < start) next = loop || boomerang ? end : start;
        setPlaybackAxisFast(axis, next);
      }
      return !loop && !boomerang && wouldHitEdge;
    };

    const advanceSingleAxis = (): boolean => {
      const axis = playbackAxis === 3 ? 0 : playbackAxis;
      const { start, end, current: prev } = axisBounds(axis);
      let next = prev;
      let hitStop = false;
      if (boomerang) {
        const candidate = prev + bounceDirRef.current;
        if (candidate > end) {
          bounceDirRef.current = -1;
          next = prev - 1 >= start ? prev - 1 : prev;
        } else if (candidate < start) {
          bounceDirRef.current = 1;
          next = prev + 1 <= end ? prev + 1 : prev;
        } else {
          next = candidate;
        }
      } else {
        next = prev + (reverse ? -1 : 1);
        if (reverse && next < start) {
          hitStop = !loop;
          next = loop ? end : start;
        } else if (!reverse && next > end) {
          hitStop = !loop;
          next = loop ? start : end;
        }
      }
      setPlaybackAxisFast(axis, next);
      return hitStop;
    };

    const advanceOnce = () => (playbackAxis === 3 ? advanceAllAxes() : advanceSingleAxis());

    const tick = (ts: number) => {
      if (cancelled) return;
      const fpsSafe = Math.max(1, Math.min(MAX_PLAYBACK_FPS, Math.round(fpsRef.current || 1)));
      const intervalMs = 1000 / fpsSafe;
      const lastTs = lastPlayTsRef.current;
      lastPlayTsRef.current = ts;
      if (lastTs != null) {
        playAccumulatorRef.current += ts - lastTs;
        if (playAccumulatorRef.current > intervalMs * 4) {
          playAccumulatorRef.current = intervalMs;
        }
      }

      let steps = 0;
      while (playAccumulatorRef.current >= intervalMs && steps < 3) {
        playAccumulatorRef.current -= intervalMs;
        steps += 1;
        if (advanceOnce()) {
          setPlaying(false);
          return;
        }
      }
      playRafRef.current = requestAnimationFrame(tick);
    };

    const startFrameLoop = () => {
      if (playRafRef.current != null) return;
      lastPlayTsRef.current = null;
      playAccumulatorRef.current = 0;
      playRafRef.current = requestAnimationFrame(tick);
    };

    startFrameLoop();

    const onVis = () => {
      if (document.hidden) {
        hiddenPaused = playRafRef.current != null;
        clearPlayFrame();
      } else if (hiddenPaused) {
        hiddenPaused = false;
        startFrameLoop();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVis);
      clearPlayFrame();
      commitSliceValuesRef.current();
    };
  }, [playing, reverse, boomerang, loop, playbackAxis, loopStarts, effectiveLoopEnds]);

  // -------------------------------------------------------------------------
  // Direct canvas draw (bypasses React state for 60fps pan during drag)
  // -------------------------------------------------------------------------
  const drawSliceDirect = (axis: number, action = "zoom") => {
    const t0 = performance.now();
    const zs = liveZoomsRef.current[axis];
    const cs = canvasSizes[axis];
    const cw = cs.w, ch = cs.h;
    const canvas = canvasRefs.current[axis];
    const offscreen = sliceOffscreenRefs.current[axis];
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, cw, ch);
    if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
      ctx.save();
      const cx = cw / 2, cy = ch / 2;
      ctx.translate(cx + zs.panX, cy + zs.panY);
      ctx.scale(zs.zoom, zs.zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, cw, ch);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, offscreen.width, offscreen.height, 0, 0, cw, ch);
    }
    recordPerfRef.current(action, performance.now() - t0, axis, -1, gpuVolReadyRef.current);
  };

  const drawFftDirect = (axis: number) => {
    const zs = liveFftZoomsRef.current[axis];
    const cs = canvasSizes[axis];
    const cw = cs.w, ch = cs.h;
    const canvas = fftCanvasRefs.current[axis];
    const offscreen = fftOffscreenRefs.current[axis];
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const ow = offscreen.width, oh = offscreen.height;
    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, cw, ch);
    if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
      ctx.save();
      const cx = cw / 2, cy = ch / 2;
      ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
    }
  };

  // -------------------------------------------------------------------------
  // Zoom/Pan handlers (matching Show3D)
  // -------------------------------------------------------------------------
  const commitLiveZoomsNow = () => {
    if (zoomCommitTimeoutRef.current != null) {
      window.clearTimeout(zoomCommitTimeoutRef.current);
      zoomCommitTimeoutRef.current = null;
    }
    liveZoomDirtyRef.current = false;
    const next = liveZoomsRef.current;
    setZooms(next);
  };
  const commitLiveZoomsSoon = () => {
    liveZoomDirtyRef.current = true;
    if (zoomCommitTimeoutRef.current != null) window.clearTimeout(zoomCommitTimeoutRef.current);
    zoomCommitTimeoutRef.current = window.setTimeout(commitLiveZoomsNow, 120);
  };
  const commitLiveFftZoomsNow = () => {
    if (fftZoomCommitTimeoutRef.current != null) {
      window.clearTimeout(fftZoomCommitTimeoutRef.current);
      fftZoomCommitTimeoutRef.current = null;
    }
    liveFftZoomDirtyRef.current = false;
    setFftZooms(liveFftZoomsRef.current);
  };
  const commitLiveFftZoomsSoon = () => {
    liveFftZoomDirtyRef.current = true;
    if (fftZoomCommitTimeoutRef.current != null) window.clearTimeout(fftZoomCommitTimeoutRef.current);
    fftZoomCommitTimeoutRef.current = window.setTimeout(commitLiveFftZoomsNow, 120);
  };
  const handleWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = canvasRefs.current[axis];
    if (!canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const zs = liveZoomsRef.current[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    const next = [...liveZoomsRef.current];
    next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY };
    liveZoomsRef.current = next;
    if (!zoomRafRef.current) {
      zoomRafRef.current = requestAnimationFrame(() => {
        zoomRafRef.current = 0;
        drawSliceDirect(axis, "zoom");
      });
    }
    commitLiveZoomsSoon();
  };

  const clickJumpTimerRef = React.useRef<number | null>(null);

  const handleDoubleClick = (axis: number) => {
    if (clickJumpTimerRef.current !== null) {
      window.clearTimeout(clickJumpTimerRef.current);
      clickJumpTimerRef.current = null;
    }
    const next = [...liveZoomsRef.current];
    next[axis] = DEFAULT_ZOOM;
    liveZoomsRef.current = next;
    commitLiveZoomsNow();
  };

  // Synchronous click-detection ref: synthetic events (CDP, automation) fire
  // mousedown→mouseup back-to-back before React commits setDragStart. The ref
  // is always current, so handleMouseUp can detect a stationary click even
  // when dragStart state hasn't been flushed yet.
  const clickStartRef = React.useRef<{ x: number; y: number; axis: number } | null>(null);
  const obliqueHandleDragRef = React.useRef<{
    mode: "endpoint";
    handle: "start" | "stop";
    opposite: { x: number; y: number };
  } | {
    mode: "line";
    angleDeg: number;
    origin: { x: number; y: number };
    start: { x: number; y: number };
    stop: { x: number; y: number };
  } | null>(null);
  const obliquePositionDragRef = React.useRef<{
    angleDeg: number;
    currentOffset: number;
    minOffset: number;
    maxOffset: number;
    start: { x: number; y: number };
    stop: { x: number; y: number };
  } | null>(null);
  const [liveObliqueOffset, setLiveObliqueOffset] = React.useState<number | null>(null);
  const [obliqueAngleBounds, setObliqueAngleBounds] = React.useState<[number, number] | null>(null);
  const [obliquePositionBounds, setObliquePositionBounds] = React.useState<[number, number] | null>(null);
  const imagePointFromEvent = (e: React.MouseEvent | MouseEvent, axis: number): { col: number; row: number } | null => {
    const canvas = canvasRefs.current?.[axis];
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const { w: cw, h: ch, scaleX, scaleY } = canvasSizes[axis];
    const zs = liveZoomsRef.current[axis];
    const cx = cw / 2, cy = ch / 2;
    const col = ((canvasX - cx - zs.panX) / zs.zoom + cx) / scaleX;
    const row = ((canvasY - cy - zs.panY) / zs.zoom + cy) / scaleY;
    return { col, row };
  };
  const updateObliqueCenter = (
    cx: number,
    cy: number,
    angleDeg = obliqueAngle,
    segment: { start: { x: number; y: number }; stop: { x: number; y: number } } = obliqueSegment,
  ) => {
    const x = clampNumber(Math.round(cx), 0, nx - 1);
    const y = clampNumber(Math.round(cy), 0, ny - 1);
    setSliceX(x);
    setSliceY(y);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      sliceX: x,
      sliceY: y,
      obliqueAngleDeg: angleDeg,
      obliqueStartX: segment.start.x,
      obliqueStartY: segment.start.y,
      obliqueEndX: segment.stop.x,
      obliqueEndY: segment.stop.y,
    };
    const renderer = volumeRendererRef.current;
    if (renderer && volumeFloats && volumeFloats.length > 0) {
      renderer.render(
        volumeRenderParamsRef.current,
        liveCameraRef.current,
        bgColorRef.current,
        undefined,
        undefined,
        zStretchRef.current,
        orthographic,
      );
    }
  };
  const updateObliqueFromEndpoints = (moving: { x: number; y: number }, opposite: { x: number; y: number }) => {
    const start = clampPointToImage(moving, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage(opposite, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const cx = clampNumber(Math.round((start.x + stop.x) / 2), 0, nx - 1);
    const cy = clampNumber(Math.round((start.y + stop.y) / 2), 0, ny - 1);
    const rawAngle = (Math.atan2(start.y - stop.y, start.x - stop.x) * 180) / Math.PI;
    const nextAngle = ((rawAngle % 180) + 180) % 180;
    setObliqueAngle(nextAngle);
    setObliqueProfileLine(profileLinePayload(start, stop));
    setObliquePositionBounds(null);
    updateObliqueCenter(cx, cy, nextAngle, { start, stop });
  };
  const updateObliqueFromNormalOffset = (offset: number) => {
    const dragBasis = obliquePositionDragRef.current;
    const angleDeg = dragBasis?.angleDeg ?? obliqueAngle;
    const baseStart = dragBasis?.start ?? obliqueSegment.start;
    const baseStop = dragBasis?.stop ?? obliqueSegment.stop;
    const currentOffset = dragBasis?.currentOffset ?? obliqueCenterOffset(nx, ny, angleDeg, baseStart, baseStop);
    const [minDelta, maxDelta] = dragBasis
      ? [dragBasis.minOffset - dragBasis.currentOffset, dragBasis.maxOffset - dragBasis.currentOffset]
      : obliqueSegmentOffsetBounds(nx, ny, angleDeg, baseStart, baseStop, OBLIQUE_PROFILE_EDGE_INSET);
    const normal = obliqueNormal(angleDeg);
    const nextOffset = clampNumber(offset, currentOffset + minDelta, currentOffset + maxDelta);
    const delta = nextOffset - currentOffset;
    const dx = normal.x * delta;
    const dy = normal.y * delta;
    const start = clampPointToImage({ x: baseStart.x + dx, y: baseStart.y + dy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    const stop = clampPointToImage({ x: baseStop.x + dx, y: baseStop.y + dy }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET);
    obliquePlaybackStateRef.current = {
      ...obliquePlaybackStateRef.current,
      current: Math.round(nextOffset),
    };
    setLiveObliqueOffset(Math.round(nextOffset));
    setObliqueProfileLine(profileLinePayload(start, stop));
    updateObliqueCenter((start.x + stop.x) / 2, (start.y + stop.y) / 2, angleDeg, { start, stop });
  };
  const updateObliqueFromLineDrag = (
    drag: {
      angleDeg: number;
      origin: { x: number; y: number };
      start: { x: number; y: number };
      stop: { x: number; y: number };
    },
    point: { col: number; row: number },
  ) => {
    const normal = obliqueNormal(drag.angleDeg);
    const delta =
      (point.col - drag.origin.x) * normal.x +
      (point.row - drag.origin.y) * normal.y;
    const currentOffset =
      obliquePositionDragRef.current?.currentOffset ??
      obliqueCenterOffset(nx, ny, drag.angleDeg, drag.start, drag.stop);
    updateObliqueFromNormalOffset(Math.round(currentOffset + delta));
  };
  const obliqueHitTargetFromEvent = (e: React.MouseEvent, axis: number): "endpoint" | "line" | null => {
    if (axis !== 0) return null;
    const canvas = canvasRefs.current?.[axis];
    const point = imagePointFromEvent(e, axis);
    if (!canvas || !point) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const { start, stop } = obliqueSegment;
    const { w: hitW, displayH: hitH, scaleX, scaleY } = canvasSizes[axis];
    const zs = liveZoomsRef.current[axis];
    const imageHitRadius = (screenPx: number) => screenPx / Math.max(1e-6, Math.min(scaleX, scaleY) * zs.zoom);
    const visualRadius = 8;
    const startCanvas = dataPointToCanvas(axis, start.x, start.y);
    const stopCanvas = dataPointToCanvas(axis, stop.x, stop.y);
    const hitStart = {
      x: clampNumber(startCanvas.x, visualRadius + 2, hitW - visualRadius - 2),
      y: clampNumber(startCanvas.y, visualRadius + 2, hitH - visualRadius - 2),
    };
    const hitStop = {
      x: clampNumber(stopCanvas.x, visualRadius + 2, hitW - visualRadius - 2),
      y: clampNumber(stopCanvas.y, visualRadius + 2, hitH - visualRadius - 2),
    };
    const handleRadius = 20 * (window.devicePixelRatio || 1);
    if (Math.hypot(mouseX - hitStart.x, mouseY - hitStart.y) <= handleRadius) return "endpoint";
    if (Math.hypot(mouseX - hitStop.x, mouseY - hitStop.y) <= handleRadius) return "endpoint";
    const lineDist = pointToSegmentDistance(point.col, point.row, start.x, start.y, stop.x, stop.y);
    return lineDist <= imageHitRadius(32 * (window.devicePixelRatio || 1)) ? "line" : null;
  };
  const handleMouseDown = (e: React.MouseEvent, axis: number) => {
    if (clickJumpTimerRef.current !== null) {
      window.clearTimeout(clickJumpTimerRef.current);
      clickJumpTimerRef.current = null;
    }
    if (axis === 0 && playing && (playbackAxis === 1 || playbackAxis === 3)) {
      pausePlaybackForEdit();
    }
    if (axis === 0) {
      const canvas = canvasRefs.current?.[axis];
      const point = imagePointFromEvent(e, axis);
      if (canvas && point) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
        const { start, stop } = obliqueSegment;
        const { w: hitW, displayH: hitH, scaleX, scaleY } = canvasSizes[axis];
        const zs = liveZoomsRef.current[axis];
        const imageHitRadius = (screenPx: number) => screenPx / Math.max(1e-6, Math.min(scaleX, scaleY) * zs.zoom);
        const visualRadius = 8;
        const startCanvas = dataPointToCanvas(axis, start.x, start.y);
        const stopCanvas = dataPointToCanvas(axis, stop.x, stop.y);
        const hitStart = {
          x: clampNumber(startCanvas.x, visualRadius + 2, hitW - visualRadius - 2),
          y: clampNumber(startCanvas.y, visualRadius + 2, hitH - visualRadius - 2),
        };
        const hitStop = {
          x: clampNumber(stopCanvas.x, visualRadius + 2, hitW - visualRadius - 2),
          y: clampNumber(stopCanvas.y, visualRadius + 2, hitH - visualRadius - 2),
        };
        const startDist = Math.hypot(mouseX - hitStart.x, mouseY - hitStart.y);
        const stopDist = Math.hypot(mouseX - hitStop.x, mouseY - hitStop.y);
        const handleRadius = 20 * (window.devicePixelRatio || 1);
        if (startDist <= handleRadius || stopDist <= handleRadius) {
          e.preventDefault();
          e.stopPropagation();
          pausePlaybackForEdit();
          obliqueHandleDragRef.current = startDist <= stopDist
            ? { mode: "endpoint", handle: "start", opposite: stop }
            : { mode: "endpoint", handle: "stop", opposite: start };
          clickStartRef.current = null;
          setDragAxis(null);
          setDragStart(null);
          return;
        }
        const lineDist = pointToSegmentDistance(point.col, point.row, start.x, start.y, stop.x, stop.y);
        if (lineDist <= imageHitRadius(32 * (window.devicePixelRatio || 1))) {
          e.preventDefault();
          e.stopPropagation();
          pausePlaybackForEdit();
          const currentOffset = obliqueCenterOffset(nx, ny, obliqueAngle, start, stop);
          const [minDelta, maxDelta] = obliqueSegmentOffsetBounds(nx, ny, obliqueAngle, start, stop, OBLIQUE_PROFILE_EDGE_INSET);
          obliquePositionDragRef.current = {
            angleDeg: obliqueAngle,
            currentOffset,
            minOffset: Math.ceil(currentOffset + minDelta),
            maxOffset: Math.floor(currentOffset + maxDelta),
            start: { ...start },
            stop: { ...stop },
          };
          setLiveObliqueOffset(Math.round(currentOffset));
          obliqueHandleDragRef.current = {
            mode: "line",
            angleDeg: obliqueAngle,
            origin: { x: point.col, y: point.row },
            start,
            stop,
          };
          clickStartRef.current = null;
          setDragAxis(null);
          setDragStart(null);
          return;
        }
      }
    }
    const zs = liveZoomsRef.current[axis];
    setDragAxis(axis);
    setDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
    clickStartRef.current = { x: e.clientX, y: e.clientY, axis };
    liveZoomDirtyRef.current = true;
  };
  React.useEffect(() => () => {
    if (clickJumpTimerRef.current !== null) window.clearTimeout(clickJumpTimerRef.current);
  }, []);

  const handleMouseMove = (e: React.MouseEvent, axis: number) => {
    if (axis === 0 && obliqueHandleDragRef.current) {
      setObliqueHoverTarget(obliqueHandleDragRef.current.mode === "endpoint" ? "endpoint" : "line");
      const point = imagePointFromEvent(e, axis);
      if (!point) return;
      const drag = obliqueHandleDragRef.current;
      if (drag.mode === "endpoint") {
        updateObliqueFromEndpoints(
          clampPointToImage({ x: point.col, y: point.row }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
          drag.opposite,
        );
      } else {
        updateObliqueFromLineDrag(drag, point);
      }
      return;
    }
    if (axis === 0) {
      setObliqueHoverTarget(obliqueHitTargetFromEvent(e, axis));
    } else if (obliqueHoverTarget !== null) {
      setObliqueHoverTarget(null);
    }
    if (dragAxis === axis && dragStart) {
      const canvas = canvasRefs.current?.[axis];
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dx = (e.clientX - dragStart.x) * (canvas.width / rect.width);
      const dy = (e.clientY - dragStart.y) * (canvas.height / rect.height);
      const newZoom = { ...liveZoomsRef.current[axis], panX: dragStart.pX + dx, panY: dragStart.pY + dy };
      const next = [...liveZoomsRef.current]; next[axis] = newZoom;
      liveZoomsRef.current = next;
      if (!zoomRafRef.current) {
        zoomRafRef.current = requestAnimationFrame(() => {
          zoomRafRef.current = 0;
          drawSliceDirect(axis, "pan");
        });
      }
      return;
    }
    const cursorCanvas = canvasRefs.current?.[axis];
    if (!cursorCanvas || !allFloats || allFloats.length === 0) return;
    const rect = cursorCanvas.getBoundingClientRect();
    const canvasX = (e.clientX - rect.left) * (cursorCanvas.width / rect.width);
    const canvasY = (e.clientY - rect.top) * (cursorCanvas.height / rect.height);
    const { w: cw, h: ch, scaleX, scaleY } = canvasSizes[axis];
    const zs = liveZoomsRef.current[axis];
    const cx = cw / 2, cy = ch / 2;
    let imgCol: number, imgRow: number;
    if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
      imgCol = ((canvasX - cx - zs.panX) / zs.zoom + cx) / scaleX;
      imgRow = ((canvasY - cy - zs.panY) / zs.zoom + cy) / scaleY;
    } else {
      imgCol = canvasX / scaleX;
      imgRow = canvasY / scaleY;
    }
    const pixelCol = Math.floor(imgCol);
    const pixelRow = Math.floor(imgRow);
    const [sliceH, sliceW] = sliceDims[axis];
    if (pixelCol < 0 || pixelCol >= sliceW || pixelRow < 0 || pixelRow >= sliceH) {
      setCursorInfoThrottled(null);
      return;
    }
    // 3D voxel lookup. XY is a Z slice; oblique is a vertical plane through
    // the current XY center, rotated about Z.
    let value: number;
    if (axis === 0) {
      value = allFloats[sliceZ * ny * nx + pixelRow * nx + pixelCol];
    } else {
      const denom = Math.max(1, sliceW - 1);
      const t = pixelCol / denom;
      const x = obliqueSegment.start.x + (obliqueSegment.stop.x - obliqueSegment.start.x) * t;
      const y = obliqueSegment.start.y + (obliqueSegment.stop.y - obliqueSegment.start.y) * t;
      value = sampleVolumeBilinear(allFloats, nx, ny, nz, pixelRow, x, y);
    }
    setCursorInfoThrottled({
      row: pixelRow,
      col: pixelCol,
      value: Number.isFinite(value) ? value : Number.NaN,
      view: PANEL_NAMES[axis] ?? "Slice",
    });
  };

  React.useEffect(() => {
    const handleDocumentMove = (e: MouseEvent) => {
      const drag = obliqueHandleDragRef.current;
      if (!drag) return;
      const point = imagePointFromEvent(e, 0);
      if (!point) return;
      if (drag.mode === "endpoint") {
        updateObliqueFromEndpoints(
          clampPointToImage({ x: point.col, y: point.row }, nx, ny, OBLIQUE_PROFILE_EDGE_INSET),
          drag.opposite,
        );
      } else {
        updateObliqueFromLineDrag(drag, point);
      }
    };
    const handleDocumentUp = () => {
      obliqueHandleDragRef.current = null;
      setObliqueHoverTarget(null);
      endObliquePositionDrag();
    };
    document.addEventListener("mousemove", handleDocumentMove);
    document.addEventListener("mouseup", handleDocumentUp);
    return () => {
      document.removeEventListener("mousemove", handleDocumentMove);
      document.removeEventListener("mouseup", handleDocumentUp);
    };
  });

  // Stationary click on a slice panel = jump-to-voxel. Convert the click's
  // canvas-pixel position into image-pixel coords (same math as handleMouseMove
  // cursor readout), then set the matching volume indices.
  const handleMouseUp = (e?: React.MouseEvent, axis?: number, refs?: React.RefObject<(HTMLCanvasElement | null)[]>) => {
    if (zoomRafRef.current) { cancelAnimationFrame(zoomRafRef.current); zoomRafRef.current = 0; }
    commitLiveZoomsNow();
    const wasDraggingObliqueHandle = obliqueHandleDragRef.current !== null;
    obliqueHandleDragRef.current = null;
    setObliqueHoverTarget(null);
    endObliquePositionDrag();
    const click = clickStartRef.current;
    if (!wasDraggingObliqueHandle && e && axis !== undefined && refs && click && click.axis === axis) {
      const moved = Math.abs(e.clientX - click.x) + Math.abs(e.clientY - click.y);
      if (moved < 4) {
        const canvas = refs.current?.[axis];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const { w: cw, h: ch, scaleX, scaleY } = canvasSizes[axis];
          const zs = liveZoomsRef.current[axis];
          const cx = cw / 2, cy = ch / 2;
          const imgCol = ((canvasX - cx - zs.panX) / zs.zoom + cx) / scaleX;
          const imgRow = ((canvasY - cy - zs.panY) / zs.zoom + cy) / scaleY;
          const pixelCol = Math.floor(imgCol), pixelRow = Math.floor(imgRow);
          const [sliceH, sliceW] = sliceDims[axis];
          if (pixelCol >= 0 && pixelCol < sliceW && pixelRow >= 0 && pixelRow < sliceH) {
            if (clickJumpTimerRef.current !== null) {
              window.clearTimeout(clickJumpTimerRef.current);
            }
            clickJumpTimerRef.current = window.setTimeout(() => {
              if (axis === 0) {
                setSliceY(pixelRow);
                setSliceX(pixelCol);
              } else {
                const denom = Math.max(1, sliceW - 1);
                const t = pixelCol / denom;
                const x = obliqueSegment.start.x + (obliqueSegment.stop.x - obliqueSegment.start.x) * t;
                const y = obliqueSegment.start.y + (obliqueSegment.stop.y - obliqueSegment.start.y) * t;
                const nextX = Math.round(x);
                const nextY = Math.round(y);
                if (nextX >= 0 && nextX < nx && nextY >= 0 && nextY < ny) {
                  setSliceZ(pixelRow);
                  setSliceX(nextX);
                  setSliceY(nextY);
                }
              }
              clickJumpTimerRef.current = null;
            }, 220);
          }
        }
      }
    }
    clickStartRef.current = null;
    setDragAxis(null); setDragStart(null);
  };
  // Don't kill the drag when the cursor briefly leaves the panel - users routinely
  // drag past the edge while panning. Only clear the cursor readout overlay.
  const handleMouseLeave = () => {
    setCursorInfoThrottled(null);
    if (!obliqueHandleDragRef.current) setObliqueHoverTarget(null);
  };

  // Global mouseup ensures drag ends even if the user releases the mouse outside
  // any slice or FFT canvas (e.g. they drag onto the volume panel and let go).
  // Without this the dragAxis state stays pinned and the next mouseMove on ANY
  // panel pans it - very confusing.
  React.useEffect(() => {
    if (dragAxis === null && fftDragAxis === null) return;
    const onUp = () => {
      if (zoomRafRef.current) { cancelAnimationFrame(zoomRafRef.current); zoomRafRef.current = 0; }
      if (fftZoomRafRef.current) { cancelAnimationFrame(fftZoomRafRef.current); fftZoomRafRef.current = 0; }
      commitLiveZoomsNow();
      commitLiveFftZoomsNow();
      setDragAxis(null); setDragStart(null);
      setFftDragAxis(null); setFftDragStart(null);
      fftClickStartRef.current = null;
    };
    document.addEventListener("mouseup", onUp);
    return () => document.removeEventListener("mouseup", onUp);
  }, [dragAxis, fftDragAxis]);

  const handleResetSlices = () => {
    const resetZooms = [DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM];
    const resetFftZooms = [DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM];
    liveZoomsRef.current = resetZooms;
    liveFftZoomsRef.current = resetFftZooms;
    liveZoomDirtyRef.current = false;
    liveFftZoomDirtyRef.current = false;
    setZooms(resetZooms);
    setFftZooms(resetFftZooms);
    setFftClickInfo(null);
  };

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  // Arrow Left/Right  : prev/next active transport axis (slice or plane)
  // Arrow Up/Down     : decrease/increase oblique angle
  // Home / End        : first / last on active transport axis
  // Space             : play/pause
  // r / R             : reset slice/FFT zoom and pan
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Keep native keyboard behavior for sliders/selects/buttons.
    if (shouldIgnoreWidgetShortcut(e.target)) return;
    const activeAxis = playbackAxis === 3 ? 0 : playbackAxis;
    const advanceTransportAxis = (axis: number, delta: number) => {
      e.preventDefault();
      if (axis === 0) {
        setSliceZ(Math.max(0, Math.min(nz - 1, sliceZ + delta)));
        return;
      }
      const state = obliquePlaybackStateRef.current;
      updateObliqueFromNormalOffset(clampNumber(state.current + delta, state.start, state.end));
    };
    const jumpTransportAxis = (axis: number, toEnd: boolean) => {
      e.preventDefault();
      if (axis === 0) {
        setSliceZ(toEnd ? nz - 1 : 0);
        return;
      }
      const state = obliquePlaybackStateRef.current;
      updateObliqueFromNormalOffset(toEnd ? state.end : state.start);
    };
    switch (e.key) {
      case " ":
        e.preventDefault();
        setPlaying(!playing);
        break;
      case "ArrowLeft":
        advanceTransportAxis(activeAxis, -1);
        break;
      case "ArrowRight":
        advanceTransportAxis(activeAxis, 1);
        break;
      case "ArrowUp":
        e.preventDefault();
        updateObliqueAngleWithinBounds(Math.round(obliqueAngle) - 1);
        break;
      case "ArrowDown":
        e.preventDefault();
        updateObliqueAngleWithinBounds(Math.round(obliqueAngle) + 1);
        break;
      case "Home":
        jumpTransportAxis(activeAxis, false);
        break;
      case "End":
        jumpTransportAxis(activeAxis, true);
        break;
      case "r":
      case "R":
        // Only handle 'r' when no modifier so we don't shadow Ctrl+R / Cmd+R reload.
        if (!e.ctrlKey && !e.metaKey && !e.altKey) {
          e.preventDefault();
          handleResetSlices();
        }
        break;
    }
  };

  // -------------------------------------------------------------------------
  // FFT Zoom/Pan handlers
  // -------------------------------------------------------------------------
  const handleFftWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const zs = liveFftZoomsRef.current[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    const next = [...liveFftZoomsRef.current];
    next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY };
    liveFftZoomsRef.current = next;
    if (!fftZoomRafRef.current) {
      fftZoomRafRef.current = requestAnimationFrame(() => {
        fftZoomRafRef.current = 0;
        drawFftDirect(axis);
      });
    }
    commitLiveFftZoomsSoon();
  };

  const handleFftDoubleClick = (axis: number) => {
    const next = [...liveFftZoomsRef.current];
    next[axis] = DEFAULT_FFT_ZOOM;
    liveFftZoomsRef.current = next;
    commitLiveFftZoomsNow();
  };

  const handleFftMouseDown = (e: React.MouseEvent, axis: number) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY, axis };
    const zs = liveFftZoomsRef.current[axis];
    setFftDragAxis(axis);
    setFftDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
    liveFftZoomDirtyRef.current = true;
  };

  const handleFftMouseMove = (e: React.MouseEvent, axis: number) => {
    if (fftDragAxis !== axis || !fftDragStart) return;
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - fftDragStart.x) * (canvas.width / rect.width);
    const dy = (e.clientY - fftDragStart.y) * (canvas.height / rect.height);
    const newZoom = { ...liveFftZoomsRef.current[axis], panX: fftDragStart.pX + dx, panY: fftDragStart.pY + dy };
    const next = [...liveFftZoomsRef.current]; next[axis] = newZoom;
    liveFftZoomsRef.current = next;
    if (!fftZoomRafRef.current) {
      fftZoomRafRef.current = requestAnimationFrame(() => {
        fftZoomRafRef.current = 0;
        drawFftDirect(axis);
      });
    }
  };

  const handleFftMouseUp = (e: React.MouseEvent, axis: number) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current && fftClickStartRef.current.axis === axis) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = fftCanvasRefs.current[axis];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const { w: cw, h: ch } = canvasSizes[axis];
          const zs = liveFftZoomsRef.current[axis];

          // Determine FFT dimensions for this panel.
          const [sliceH, sliceW] = sliceDims[axis];
          const fftW = nextPow2(sliceW);
          const fftH = nextPow2(sliceH);

          const mouseX = (e.clientX - rect.left) * (cw / rect.width);
          const mouseY = (e.clientY - rect.top) * (ch / rect.height);
          const cx = cw / 2, cy = ch / 2;
          const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
          const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
          let imgCol = imgX / cw * fftW;
          let imgRow = imgY / ch * fftH;

          const cachedMag = fftMagCacheRefs.current[axis];
          if (cachedMag && imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            const snapped = findFFTPeak(cachedMag, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }

          if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            const dcCol = imgCol - fftW / 2;
            const dcRow = imgRow - fftH / 2;
            const distPx = Math.sqrt(dcCol * dcCol + dcRow * dcRow);
            if (distPx < 1) {
              setFftClickInfo(null);
            } else {
              let spatialFreq: number | null = null;
              let dSpacing: number | null = null;
              const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
              const theta = (obliqueAngle * Math.PI) / 180;
              const obliqueSpacing = axes
                ? Math.hypot(Math.cos(theta) * axes[2], Math.sin(theta) * axes[1])
                : pixelSize;
              const rowSpacing = axis === 0 ? (axes ? axes[1] : pixelSize) : (axes ? axes[0] : pixelSize);
              const colSpacing = axis === 0 ? (axes ? axes[2] : pixelSize) : obliqueSpacing;
              if (rowSpacing > 0 && colSpacing > 0) {
                const paddedW = fftW;
                const paddedH = fftH;
                const freqC = dcCol / paddedW / colSpacing;
                const freqR = dcRow / paddedH / rowSpacing;
                spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
                dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
              }
              setFftClickInfo({ axis, row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
            }
          }
        }
      }
    }
    fftClickStartRef.current = null;
    if (fftZoomRafRef.current) { cancelAnimationFrame(fftZoomRafRef.current); fftZoomRafRef.current = 0; }
    commitLiveFftZoomsNow();
    setFftDragAxis(null);
    setFftDragStart(null);
  };

  const handleFftResetAxis = (a: number) => {
    const next = [...liveFftZoomsRef.current];
    next[a] = DEFAULT_FFT_ZOOM;
    liveFftZoomsRef.current = next;
    commitLiveFftZoomsNow();
    if (fftClickInfo && fftClickInfo.axis === a) setFftClickInfo(null);
  };

  const fftNeedsResetAxis = (a: number) => {
    const z = fftZooms[a];
    return z.zoom !== DEFAULT_FFT_ZOOM.zoom || z.panX !== DEFAULT_FFT_ZOOM.panX || z.panY !== DEFAULT_FFT_ZOOM.panY;
  };

  // -------------------------------------------------------------------------
  // Canvas resize (matching Show2D)
  // -------------------------------------------------------------------------
  const handleResizeStart = (e: React.MouseEvent, axis: number = 0) => {
    e.stopPropagation();
    e.preventDefault();
    const target = axis > 0 ? "side" : "primary";
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: target === "side" ? sideCanvasTarget : canvasTarget, target });
  };

  React.useEffect(() => {
    if (!isResizing || !resizeStart) return;
    let rafId = 0;
    let latestSize = resizeStart.size;
    const handleMouseMove = (e: MouseEvent) => {
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(300, resizeStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          if (resizeStart.target === "side") setSideCanvasTarget(latestSize);
          else setCanvasTarget(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      if (resizeStart?.target === "side") setSideCanvasTarget(latestSize);
      else setCanvasTarget(latestSize);
      setIsResizing(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, resizeStart]);

  // -------------------------------------------------------------------------
  // Labels and setters
  // -------------------------------------------------------------------------
  // Default mirrors Python's dim_labels default ["slice", "row", "col"]: axis
  // 0 is the slice (multislice depth), axis 1 is row, axis 2 is col. Fallback
  // fires only when the trait is briefly undefined (initial mount race).
  const dl = dimLabels || ["slice", "row", "col"];
  const sliceValues = [sliceZ, sliceY, sliceX];
  // Mirror of slice values for playback intervals to read between renders.
  // The interval's `sliceValuesRef.current[a] = next` writes are load-bearing
  // at high fps (>~20): React batches setSliceZ/Y/X so two ticks can fire
  // before the next render reassigns this ref to the new [sliceZ,sliceY,sliceX].
  // Without the mutation the second tick reads the stale value and computes the
  // same `next`, freezing playback.
  const sliceValuesRef = React.useRef(sliceValues);
  if (!playing) sliceValuesRef.current = sliceValues;
  const sliceMaxes = [nz - 1, ny - 1, nx - 1];
  // Live thumb mirror: updates per drag-frame so the thumb tracks, WITHOUT touching
  // the model traits (whose change re-runs the heavy render/layout/crosshair effects).
  // Those effects key on sliceX/Y/Z, so during a drag (traits unchanged) they don't
  // run - only the slider JSX re-renders + directPaintPlane paints the GPU image.
  const [liveSlider, setLiveSlider] = React.useState<number[]>([sliceZ, sliceY, sliceX]);
  const liveSliderRef = React.useRef<number[]>([sliceZ, sliceY, sliceX]);
  const pendingPaintRef = React.useRef<Map<number, number>>(new Map());
  const pendingPaintSourceRef = React.useRef<Map<number, string>>(new Map());
  const sliderPaintRafRef = React.useRef<number | null>(null);
  const pendingContrastRangeRef = React.useRef<[number, number] | null>(null);
  const contrastPaintRafRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const next = [sliceZ, sliceY, sliceX];
    liveSliderRef.current = next;
    setLiveSlider(next);
  }, [sliceZ, sliceY, sliceX]);
  React.useEffect(() => {
    return () => {
      if (sliderPaintRafRef.current != null) cancelAnimationFrame(sliderPaintRafRef.current);
      if (contrastPaintRafRef.current != null) cancelAnimationFrame(contrastPaintRafRef.current);
    };
  }, []);
  // DURING DRAG: only direct-paint (GPU, off React) - do NOT set the model trait,
  // which would re-render the whole component per drag-frame (the 39->stuck cap).
  // ON RELEASE (onChangeCommitted): set the trait once so crosshair/title/state sync.
  const paintAndTrackRef = React.useRef<((axis: number, v: number, source?: string) => void) | null>(null);
  paintAndTrackRef.current = (axis: number, v: number, source = "slider") => {
    if (liveSliderRef.current[axis] === v) return;
    const next = [...liveSliderRef.current];
    next[axis] = v;
    liveSliderRef.current = next;
    sliceValuesRef.current = next;
    liveSliceParamsRef.current = { sliceZ: next[0], sliceY: next[1], sliceX: next[2] };
    volumeRenderParamsRef.current = { ...volumeRenderParamsRef.current, ...liveSliceParamsRef.current };
    pendingPaintRef.current.set(axis, v);
    pendingPaintSourceRef.current.set(axis, source);
    if (sliderPaintRafRef.current != null) return;
    sliderPaintRafRef.current = requestAnimationFrame(() => {
      sliderPaintRafRef.current = null;
      const pending = pendingPaintRef.current;
      pendingPaintRef.current = new Map();
      const pendingSources = pendingPaintSourceRef.current;
      pendingPaintSourceRef.current = new Map();
      for (const [pendingAxis, pendingValue] of pending) directPaintPlane(pendingAxis, pendingValue, pendingSources.get(pendingAxis) || "slider");
      renderVolumePlanesLive("volumeSlice");
      setLiveSlider(liveSliderRef.current);
    });
  };
  fastTrackSliceRef.current = (axis: number, value: number) => {
    paintAndTrackRef.current?.(axis, value, "playback");
  };
  const paintContrastRange = (min: number, max: number) => {
    pendingContrastRangeRef.current = [min, max];
    const p = paintParamsRef.current;
    if (p) paintParamsRef.current = { ...p, imageVminPct: min, imageVmaxPct: max };
    const nextVolRange = volumeTextureRangeForPercent(min, max);
    volumeRenderParamsRef.current = {
      ...volumeRenderParamsRef.current,
      vmin: nextVolRange.vmin,
      vmax: nextVolRange.vmax,
    };
    if (contrastPaintRafRef.current != null) return;
    contrastPaintRafRef.current = requestAnimationFrame(() => {
      contrastPaintRafRef.current = null;
      const pending = pendingContrastRangeRef.current;
      pendingContrastRangeRef.current = null;
      if (!pending) return;
      const [pendingMin, pendingMax] = pending;
      const current = paintParamsRef.current;
      if (current) paintParamsRef.current = { ...current, imageVminPct: pendingMin, imageVmaxPct: pendingMax };
      const pendingVolRange = volumeTextureRangeForPercent(pendingMin, pendingMax);
      volumeRenderParamsRef.current = {
        ...volumeRenderParamsRef.current,
        vmin: pendingVolRange.vmin,
        vmax: pendingVolRange.vmax,
      };
      const slices = liveSliderRef.current;
      for (let a = 0; a < sliceDims.length; a++) directPaintPlane(a, slices[a], "contrast");
      renderVolumePlanesLive("contrast");
    });
  };
  commitSliceValuesRef.current = () => {
    const [z, y, x] = sliceValuesRef.current;
    if (sliceZ !== z) setSliceZ(z);
    if (sliceY !== y) setSliceY(y);
    if (sliceX !== x) setSliceX(x);
  };
  const stopPlaybackAndRewind = () => {
    setPlaying(false);
    const axes = playbackAxis === 3 ? [0, 1] : [playbackAxis];
    const next = [...sliceValuesRef.current];
    for (const axis of axes) {
      if (axis === 0) {
        const start = Math.max(0, Math.min(loopStarts[0], sliceMaxes[0]));
        next[0] = start;
        paintAndTrackRef.current?.(0, start, "stop");
        sliceSettersRef.current[0](start);
      } else {
        const start = obliquePlaybackStateRef.current.start;
        obliquePlaybackStateRef.current = { ...obliquePlaybackStateRef.current, current: start };
        updateObliqueFromNormalOffset(start);
      }
    }
    sliceValuesRef.current = next;
  };
  const sliceSetters = [
    (_: Event, v: number | number[]) => paintAndTrackRef.current!(0, v as number, "slider"),
    (_: Event, v: number | number[]) => paintAndTrackRef.current!(1, v as number, "slider"),
    (_: Event, v: number | number[]) => paintAndTrackRef.current!(2, v as number, "slider"),
  ];
  const sliceCommitters = [
    (_: unknown, v: number | number[]) => setSliceZ(v as number),
    (_: unknown, v: number | number[]) => setSliceY(v as number),
    (_: unknown, v: number | number[]) => setSliceX(v as number),
  ];
  const loopSliderValues = (axis: number) => {
    return [loopStarts[axis], liveSlider[axis], effectiveLoopEnds[axis]];
  };
  const handleLoopSliderChange = (axis: number, vals: number[]) => {
    paintAndTrackRef.current?.(axis, vals[1], "loop");
    if (vals[0] === loopStartsRef.current[axis] && vals[2] === loopEndsRef.current[axis]) return;
    const nextStarts = [...loopStartsRef.current];
    const nextEnds = [...loopEndsRef.current];
    nextStarts[axis] = vals[0];
    nextEnds[axis] = vals[2];
    loopStartsRef.current = nextStarts;
    loopEndsRef.current = nextEnds;
    pendingLoopRangeRef.current = { starts: nextStarts, ends: nextEnds };
    if (loopRangeRafRef.current == null) {
      loopRangeRafRef.current = requestAnimationFrame(() => {
        loopRangeRafRef.current = null;
        const pending = pendingLoopRangeRef.current;
        pendingLoopRangeRef.current = null;
        if (!pending) return;
        setLoopStarts(pending.starts);
        setLoopEnds(pending.ends);
      });
    }
  };
  const handleLoopSliderCommit = (axis: number, vals: number[]) => {
    if (loopRangeRafRef.current != null) {
      cancelAnimationFrame(loopRangeRafRef.current);
      loopRangeRafRef.current = null;
    }
    const startsChanged = vals[0] !== loopStartsRef.current[axis];
    const endsChanged = vals[2] !== loopEndsRef.current[axis];
    pendingLoopRangeRef.current = null;
    if (startsChanged || endsChanged) {
      const nextStarts = [...loopStartsRef.current];
      const nextEnds = [...loopEndsRef.current];
      nextStarts[axis] = vals[0];
      nextEnds[axis] = vals[2];
      loopStartsRef.current = nextStarts;
      loopEndsRef.current = nextEnds;
      setLoopStarts(nextStarts);
      setLoopEnds(nextEnds);
    }
    [setSliceZ, setSliceY, setSliceX][axis](vals[1]);
  };
  const handleLoopSliderPointerDownCapture = (axis: number, event: React.PointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    const target = event.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const max = sliceMaxes[axis];
    const valueFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      return Math.max(0, Math.min(max, Math.round(pct * max)));
    };
    const moveCurrent = (clientX: number, commit: boolean) => {
      const next = valueFromClientX(clientX);
      paintAndTrackRef.current?.(axis, next, "loop");
      if (commit) [setSliceZ, setSliceY, setSliceX][axis](next);
    };
    event.preventDefault();
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    moveCurrent(event.clientX, false);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX, false);
    };
    const onUp = (ev: PointerEvent) => {
      ev.preventDefault();
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX, true);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  // Over-clip detection: user dragged hist thumbs past data peak → image goes black.
  // Compute effective vmin/vmax in data units, compare against 1st/99th percentile of histogram.
  // If vmin > 99% of data OR vmax < 1% of data, no visible content.
  const imageClipBounds = React.useMemo(() => {
    if (!imageHistogramData || imageHistogramData.length === 0) return null;
    return percentileClip(imageHistogramData, 1, 99);
  }, [imageHistogramData]);
  const isOverClipped = (() => {
    if (autoContrast) return false;
    if (imageVminPct <= 0 && imageVmaxPct >= 100) return false;
    if (!imageClipBounds) return false;
    const span = displayDataRange.max - displayDataRange.min;
    if (span <= 0) return false;
    const vmin = displayDataRange.min + (imageVminPct / 100) * span;
    const vmax = displayDataRange.min + (imageVmaxPct / 100) * span;
    return vmin >= imageClipBounds.vmax || vmax <= imageClipBounds.vmin;
  })();

  // Thin-Z layout: depth axis much smaller than lateral. Show the top panel
  // beside the single oblique depth panel.
  const thinZ = nz < Math.min(nx, ny) / 4;
  const panelTotalW = (canvasSizes[0]?.w ?? CANVAS_TARGET) + (canvasSizes[1]?.w ?? 0) + SPACING.SM;
  const primaryPanelW = canvasSizes[0]?.w ?? CANVAS_TARGET;
  const compactControlsW = Math.min(primaryPanelW, CANVAS_TARGET);
  const obliqueAngleSliderMin = 0;
  const obliqueAngleSliderMax = 179;
  const [rawObliqueAngleMinBound, rawObliqueAngleMaxBound] = obliqueAngleBounds ?? [
    obliqueAngleSliderMin,
    obliqueAngleSliderMax,
  ];
  const obliqueAngleMinBound = clampNumber(rawObliqueAngleMinBound, obliqueAngleSliderMin, obliqueAngleSliderMax);
  const obliqueAngleMaxBound = clampNumber(rawObliqueAngleMaxBound, obliqueAngleMinBound, obliqueAngleSliderMax);
  const boundedObliqueAngle = clampNumber(obliqueAngle, obliqueAngleMinBound, obliqueAngleMaxBound);
  const obliqueAngleSliderValues = [
    obliqueAngleMinBound,
    boundedObliqueAngle,
    obliqueAngleMaxBound,
  ];
  const updateObliqueAngleWithinBounds = (angle: number, minBound = obliqueAngleMinBound, maxBound = obliqueAngleMaxBound) => {
    const nextAngle = clampNumber(angle, minBound, maxBound);
    if (Math.round(nextAngle) !== Math.round(obliqueAngle)) handleObliqueAngleChange(new Event("change"), nextAngle);
  };
  const handleObliqueAngleSliderChange = (vals: number[]) => {
    setObliqueAngleBounds([vals[0], vals[2]]);
    updateObliqueAngleWithinBounds(vals[1], vals[0], vals[2]);
  };
  const handleObliqueAnglePointerDownCapture = (event: React.PointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    pausePlaybackForEdit();
    const target = event.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const valueFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      const full = obliqueAngleSliderMin + pct * (obliqueAngleSliderMax - obliqueAngleSliderMin);
      return Math.round(clampNumber(full, obliqueAngleMinBound, obliqueAngleMaxBound));
    };
    const moveCurrent = (clientX: number) => updateObliqueAngleWithinBounds(valueFromClientX(clientX));
    event.preventDefault();
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    moveCurrent(event.clientX);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX);
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  const obliqueCurrentOffset = obliqueCenterOffset(nx, ny, obliqueAngle, obliqueSegment.start, obliqueSegment.stop);
  const [obliqueDeltaMin, obliqueDeltaMax] = obliqueSegmentOffsetBounds(
    nx,
    ny,
    obliqueAngle,
    obliqueSegment.start,
    obliqueSegment.stop,
    OBLIQUE_PROFILE_EDGE_INSET,
  );
  const obliqueOffsetMin = Math.ceil(obliqueCurrentOffset + obliqueDeltaMin);
  const obliqueOffsetMax = Math.floor(obliqueCurrentOffset + obliqueDeltaMax);
  const obliqueOffset = liveObliqueOffset ?? clampNumber(Math.round(obliqueCurrentOffset), obliqueOffsetMin, obliqueOffsetMax);
  const beginObliquePositionDrag = () => {
    pausePlaybackForEdit();
    const currentOffset = obliqueCenterOffset(nx, ny, obliqueAngle, obliqueSegment.start, obliqueSegment.stop);
    const [minDelta, maxDelta] = obliqueSegmentOffsetBounds(nx, ny, obliqueAngle, obliqueSegment.start, obliqueSegment.stop, OBLIQUE_PROFILE_EDGE_INSET);
    obliquePositionDragRef.current = {
      angleDeg: obliqueAngle,
      currentOffset,
      minOffset: Math.ceil(currentOffset + minDelta),
      maxOffset: Math.floor(currentOffset + maxDelta),
      start: { ...obliqueSegment.start },
      stop: { ...obliqueSegment.stop },
    };
    setLiveObliqueOffset(Math.round(currentOffset));
  };
  const endObliquePositionDrag = () => {
    obliquePositionDragRef.current = null;
    setLiveObliqueOffset(null);
  };
  const obliquePositionSliderMin = obliquePositionDragRef.current?.minOffset ?? obliqueOffsetMin;
  const obliquePositionSliderMax = obliquePositionDragRef.current?.maxOffset ?? obliqueOffsetMax;
  const [rawObliquePositionMinBound, rawObliquePositionMaxBound] = obliquePositionBounds ?? [
    obliquePositionSliderMin,
    obliquePositionSliderMax,
  ];
  const obliquePositionMinBound = clampNumber(
    rawObliquePositionMinBound,
    obliquePositionSliderMin,
    obliquePositionSliderMax,
  );
  const obliquePositionMaxBound = clampNumber(
    rawObliquePositionMaxBound,
    obliquePositionMinBound,
    obliquePositionSliderMax,
  );
  const boundedObliqueOffset = clampNumber(obliqueOffset, obliquePositionMinBound, obliquePositionMaxBound);
  const obliquePositionSliderValues = [
    obliquePositionMinBound,
    boundedObliqueOffset,
    obliquePositionMaxBound,
  ];
  obliquePlaybackStateRef.current = {
    current: boundedObliqueOffset,
    start: obliquePositionMinBound,
    end: obliquePositionMaxBound,
  };
  const slicePanelCursor = (axis: number) => {
    if (axis === 0) {
      if (obliqueHandleDragRef.current || dragAxis === axis) return "grabbing";
      if (obliqueHoverTarget === "endpoint" || obliqueHoverTarget === "line") return "grab";
    }
    return dragAxis === axis ? "grabbing" : "crosshair";
  };
  const handleObliquePositionChange = (vals: number[]) => {
    setObliquePositionBounds([vals[0], vals[2]]);
    updateObliqueFromNormalOffset(clampNumber(vals[1], vals[0], vals[2]));
  };
  const handleObliquePositionCommit = (vals: number[]) => {
    setObliquePositionBounds([vals[0], vals[2]]);
    updateObliqueFromNormalOffset(clampNumber(vals[1], vals[0], vals[2]));
    endObliquePositionDrag();
  };
  const handleObliquePositionPointerDownCapture = (event: React.PointerEvent<HTMLSpanElement>) => {
    beginObliquePositionDrag();
    if (event.button !== 0) return;
    const target = event.target as HTMLElement;
    if (target.closest(".MuiSlider-thumb")) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const valueFromClientX = (clientX: number) => {
      const pct = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
      const full = obliquePositionSliderMin + pct * (obliquePositionSliderMax - obliquePositionSliderMin);
      return Math.round(clampNumber(full, obliquePositionMinBound, obliquePositionMaxBound));
    };
    const moveCurrent = (clientX: number, commit: boolean) => {
      updateObliqueFromNormalOffset(valueFromClientX(clientX));
      if (commit) endObliquePositionDrag();
    };
    event.preventDefault();
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    moveCurrent(event.clientX, false);
    const onMove = (ev: PointerEvent) => {
      ev.preventDefault();
      moveCurrent(ev.clientX, false);
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      moveCurrent(ev.clientX, true);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  const controlRowHeight = 28;
  const denseControlRow = {
    ...controlRow,
    minHeight: controlRowHeight,
    py: 0.25,
    boxSizing: "border-box" as const,
  };
  const panelControlRow = {
    ...denseControlRow,
    border: `1px solid ${tc.border}`,
    bgcolor: tc.controlBg,
    boxSizing: "border-box" as const,
  };
  const inlineVolumeControlRow = {
    ...denseControlRow,
    px: 0,
    py: 0,
    minHeight: 22,
    width: volumeCanvasSize,
    maxWidth: volumeCanvasSize,
    flexWrap: "nowrap" as const,
    overflow: "hidden",
  };
  const denseSelect = {
    ...themedSelect,
    height: 22,
    fontSize: 10,
    "& .MuiSelect-select": { py: 0.25, px: 1 },
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <Box className="show3dslices-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...container.root, bgcolor: tc.bg, color: tc.text, outline: "none", "&:focus": { outline: "2px solid #0af", outlineOffset: 2 }, "& canvas": { display: "block" } }}>
      {/* 3D volume on the LEFT, slice toolbar + projected slice panels on the RIGHT.
          Side-by-side layout keeps the whole widget within a 13" laptop viewport. */}
      <Box sx={{ display: "flex", flexDirection: "row", alignItems: "flex-start", gap: `${SPACING.SM}px` }}>
      {/* 3D Volume Renderer (left column) */}
      <Box sx={{ mb: 0, flexShrink: 0 }}>
        {/* Title row */}
        <Typography variant="caption" sx={{ ...typography.label, color: tc.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
          {title || "Volume 3D"}<InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT shows the power spectrum below each slice.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto uses percentile-based contrast (2nd-98th percentile). FFT Auto masks DC + clips to 99.9th.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Colorbar displays a colorbar overlay on each slice canvas.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop repeats playback. Drag end markers on slider for loop range.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce alternates forward and reverse playback.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Planes toggles the Top and angled vertical slice planes in the 3D volume view.</Typography>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
            <KeyboardShortcuts items={[["Space", "Play / Pause"], ["← / →", "Active axis -/+"], ["↑ / ↓", "Angle -/+"], ["Home / End", "First / Last on active axis"], ["R", "Reset zoom"], ["Click panel", "Jump to voxel"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
          </Box>} theme={themeInfo.theme} />
          {/* ControlCustomizer dropped in new monorepo */}
        </Typography>
        {webgpuSupported ? (
          <Stack direction="row" spacing={`${SPACING.SM}px`}>
            {/* Volume A */}
            <Box>
              <Box sx={{ ...inlineVolumeControlRow, mb: `${SPACING.XS}px` }}>
                <Typography sx={{ ...controlLabel }}>Planes</Typography>
                <ToggleButtonGroup
                  size="small"
                  value={visiblePlanes}
                  onChange={handlePlaneVisibilityChange}
                  aria-label="Slice plane visibility"
                  sx={{ height: 18, "& .MuiToggleButtonGroup-grouped": { m: 0 } }}
                >
                  {PLANE_KEYS.map((key, i) => (
                    <ToggleButton
                      key={key}
                      value={key}
                      aria-label={`${PLANE_LABELS[i]} plane`}
                      sx={planeToggleButtonSx}
                    >
                      {PLANE_LABELS[i]}
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>
                <Typography sx={{ ...controlLabel }}>Ortho</Typography>
                <Switch checked={orthographic} onChange={(e) => setOrthographic(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle orthographic 3D projection" }} />
                {anySlicePlaneVisible && (
                  <>
                    <Typography sx={{ ...controlLabel }}>Opacity</Typography>
                    <LiveNumberSlider value={slicePlaneOpacity} min={0.05} max={1} step={0.05} onLiveChange={(v) => handleVolumeControlChange("slicePlaneOpacity", v)} onCommit={(v) => handleVolumeControlCommit("slicePlaneOpacity", v)} sx={{ ...sliderStyles.small, width: 50 }} ariaLabel="Slice plane opacity" />
                  </>
                )}
                <Typography sx={{ ...controlLabel }}>Vol Strength</Typography>
                <LiveNumberSlider value={opacityA} min={0} max={1} step={0.05} onLiveChange={(v) => handleVolumeControlChange("opacity", v)} onCommit={(v) => handleVolumeControlCommit("opacity", v)} sx={{ ...sliderStyles.small, width: 50 }} ariaLabel="Volume strength" />
              </Box>
              <Box
                sx={{
                  ...container.imageBox,
                  border: `1px solid ${tc.border}`,
                  width: volumeCanvasSize,
                  height: volumeCanvasSize,
                  cursor: volumeDrag ? "grabbing" : "grab",
                }}
                onMouseDown={handleVolumeMouseDown}
                onWheel={handleVolumeWheel}
                onDoubleClick={handleVolumeDoubleClick}
                onContextMenu={(e) => e.preventDefault()}
              >
                <canvas
                  ref={volumeCanvasRef}
                  style={{ width: volumeCanvasSize, height: volumeCanvasSize, display: "block" }}
                  role="img"
                  aria-label={`3D volume rendering${title ? `: ${title}` : ""} (${nx} by ${ny} by ${nz} voxels). Drag to rotate, wheel to zoom.`}
                />
                {cameraChanged && (
                  <Button
                    size="small"
                    sx={{ ...compactButton, position: "absolute", top: 4, right: 4, minWidth: 0, px: 0.75, bgcolor: "rgba(255,255,255,0.75)", "&:hover": { bgcolor: "rgba(255,255,255,0.9)" } }}
                    onClick={(e) => { e.stopPropagation(); setCamera(SHOW3DSLICES_DEFAULT_CAMERA); }}
                    aria-label="Reset 3D camera view"
                    title="Reset 3D camera view"
                  >
                    Reset View
                  </Button>
                )}
                <Box
                  onMouseDown={handleVolumeResizeStart}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                    cursor: "nwse-resize", opacity: 0.4,
                    background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                    "&:hover": { opacity: 1 },
                  }}
                />
              </Box>
              <Box sx={{ ...inlineVolumeControlRow, mt: 0 }}>
                <Typography sx={{ ...controlLabel }} title="Align the 3D camera to a slice plane.">View</Typography>
                {VOLUME_VIEW_PRESETS.map(({ value, label, description }) => (
                  <Button
                    key={value}
                    size="small"
                    sx={{ ...compactButton, minWidth: label === "Top" ? 28 : 30, px: 0.5 }}
                    onClick={() => setVolumeView(value)}
                    aria-label={`Set 3D view to ${description}`}
                    title={`Set 3D view to ${description}`}
                  >
                    {label}
                  </Button>
                ))}
                <Button
                  size="small"
                  sx={{ ...compactButton, minWidth: 28, px: 0.5, fontSize: 13 }}
                  onClick={() => rollVolumeView(1)}
                  aria-label="Roll 3D camera view counterclockwise 90 degrees"
                  title="Roll view counterclockwise 90 degrees"
                >
                  ↺90
                </Button>
                <Button
                  size="small"
                  sx={{ ...compactButton, minWidth: 28, px: 0.5, fontSize: 13 }}
                  onClick={() => rollVolumeView(-1)}
                  aria-label="Roll 3D camera view clockwise 90 degrees"
                  title="Roll view clockwise 90 degrees"
                >
                  ↻90
                </Button>
              </Box>
            </Box>
          </Stack>
        ) : (
          <Box sx={{
            ...container.imageBox, width: volumeCanvasSize, height: 80,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Typography sx={{ ...typography.label, color: tc.textMuted, px: 2, textAlign: "center" }}>
              WebGPU not available. 3D volume rendering requires a WebGPU capable browser.
            </Typography>
          </Box>
        )}
      </Box>
      {/* Right column: slice toolbar + projected slice panels (grouped so they
          sit beside the 3D volume rather than below it). */}
      <Box sx={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
      {/* Slice toolbar: compact row above the side column. */}
      <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, mt: 0, mb: 0, minHeight: 18, justifyContent: "flex-end", width: panelTotalW, maxWidth: panelTotalW, boxSizing: "border-box" }}>
        <Typography sx={{ ...controlLabel }}>FFT</Typography>
        <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle FFT power spectrum panels" }} />
        {exportEnabled && (
          <>
          <Button
            size="small"
            sx={compactButton}
            disabled={exportBusy}
            onClick={handleExportMenuOpen}
            aria-label="Export standalone HTML"
            aria-controls={exportMenuAnchor ? "show3dslices-export-menu" : undefined}
            aria-expanded={exportMenuAnchor ? "true" : undefined}
            aria-haspopup="menu"
            title={localExportStatus || exportStatus || "Export standalone HTML with a save dialog"}
          >
            {exportBusy ? "Exporting" : "Export"}
          </Button>
          <Menu
            id="show3dslices-export-menu"
            anchorEl={exportMenuAnchor}
            open={Boolean(exportMenuAnchor)}
            onClose={handleExportMenuClose}
            MenuListProps={{ "aria-label": "Export standalone HTML options" }}
            {...themedMenuProps}
          >
            <MenuItem onClick={() => handleExportSelect("exact")}>HTML exact float32 ({exactExportSize})</MenuItem>
            <MenuItem onClick={() => handleExportSelect("quantized")}>HTML quantized uint8 ({quantizedExportSize})</MenuItem>
          </Menu>
          </>
        )}
        {exportEnabled && (localExportStatus || exportStatus) && (
          <Typography
            sx={{
              ...controlLabel,
              maxWidth: 120,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              color: (localExportStatus || exportStatus).startsWith("Export failed") ? "#d32f2f" : tc.textMuted,
            }}
            title={localExportStatus || exportStatus}
          >
            {localExportStatus || exportStatus}
          </Typography>
        )}
        <Button
          size="small"
          sx={compactButton}
          disabled={!anyZoomDirty}
          onClick={handleResetSlices}
          title="Reset slice and FFT zoom/pan only"
          aria-label="Reset slice and FFT zoom/pan"
        >
          Reset Zoom
        </Button>
      </Box>
      {(() => {
        const panels = AXES.map((_, a) => {
          const { w: cw, h: ch, displayH: dh } = canvasSizes[a];
          const panelName = PANEL_NAMES[a] ?? "Slice";
          return (
            <Box key={a} sx={{ minWidth: cw, gridArea: `a${a}` }}>
              {/* Canvas with plane-colored border. dh = displayH (stretched for depth panels). */}
              <Box
                ref={(el: HTMLDivElement | null) => { imageBoxRefs.current[a] = el; }}
                sx={{ ...container.imageBox, width: cw, height: dh, cursor: slicePanelCursor(a), borderColor: PLANE_COLORS[a] }}
                onMouseDown={(e) => handleMouseDown(e, a)}
                onMouseMove={(e) => handleMouseMove(e, a)}
                onMouseUp={(e) => handleMouseUp(e, a, canvasRefs)}
                onMouseLeave={handleMouseLeave}
                onWheel={(e) => handleWheel(e, a)}
                onDoubleClick={() => handleDoubleClick(a)}
              >
                <canvas
                  ref={(el) => { canvasRefs.current[a] = el; }}
                  width={cw}
                  height={ch}
                  style={{ width: cw, height: dh, imageRendering: smooth ? "auto" : "pixelated" }}
                  role="img"
                  aria-label={a === 0
                    ? `XY slice ${sliceZ + 1} of ${nz} along ${dl[0]} axis${title ? `: ${title}` : ""} (${cw} by ${ch} pixels)`
                    : `Oblique vertical slice at ${obliqueAngle.toFixed(1)} degrees, position ${Math.round(obliqueCurrentOffset)}${title ? `: ${title}` : ""} (${cw} by ${ch} pixels)`}
                />
                <canvas
                  ref={(el) => { overlayRefs.current[a] = el; }}
                  width={cw}
                  height={dh}
                  style={{ position: "absolute", top: 0, left: 0, width: cw, height: dh, pointerEvents: "none" }}
                  aria-hidden="true"
                />
                <canvas
                  ref={(el) => { uiRefs.current[a] = el; }}
                  width={Math.round(cw * DPR)}
                  height={Math.round(dh * DPR)}
                  style={{ position: "absolute", top: 0, left: 0, width: cw, height: dh, pointerEvents: "none" }}
                  aria-hidden="true"
                />
                {/* Cursor readout overlay */}
                {cursorInfo && cursorInfo.view === panelName && (
                  <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                    <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                      ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                    </Typography>
                  </Box>
                )}
                {/* Over-clip warning: image is mostly black because histogram thumbs sit outside data range */}
                {isOverClipped && a === 0 && (
                  <Box sx={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", bgcolor: "rgba(255, 180, 0, 0.85)", color: "#000", px: 1, py: 0.5, fontSize: 11, fontWeight: "bold", borderRadius: 0.5, textAlign: "center", lineHeight: 1.3, pointerEvents: "none", maxWidth: cw - 20 }}>
                    No data visible<br/>
                    <span style={{ fontSize: 9, fontWeight: "normal" }}>Adjust contrast range or enable Auto</span>
                  </Box>
                )}
                {/* Resize handle */}
                <Box
                  onMouseDown={(e) => handleResizeStart(e, a)}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                    cursor: "nwse-resize", opacity: 0.4,
                    background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                    "&:hover": { opacity: 1 },
                  }}
                />
              </Box>
              {/* FFT canvas (inline, below stats) */}
              {effectiveShowFft && (
                <Box sx={{ mt: `${SPACING.SM}px` }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 20 }}>
                    <Stack direction="row" alignItems="center" sx={{ overflow: "hidden" }}>
                      <Typography variant="caption" sx={{ ...typography.label, fontSize: 10, flexShrink: 0 }}>
                        {`FFT ${a === 0 ? `${dl[1]}${dl[2]}` : `oblique ${obliqueAngle.toFixed(1)}°`} ${gpuReady ? "" : " (CPU fallback)"}`}
                      </Typography>
                      {fftClickInfo && fftClickInfo.axis === a && (
                        <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted, ml: 1, whiteSpace: "nowrap" }}>
                          {fftClickInfo.dSpacing != null ? (
                            <>d=<Box component="span" sx={{ color: tc.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}</Box>{" |g|="}<Box component="span" sx={{ color: tc.accent }}>{`${fftClickInfo.spatialFreq!.toFixed(4)} \u00C5\u207B\u00B9`}</Box></>
                          ) : (
                            <>dist=<Box component="span" sx={{ color: tc.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                          )}
                        </Typography>
                      )}
                    </Stack>
                    <Button size="small" sx={compactButton} disabled={!fftNeedsResetAxis(a)} onClick={() => handleFftResetAxis(a)} aria-label={`Reset ${panelName} FFT zoom and pan`}>Reset</Button>
                  </Stack>
                  <Box
                    sx={{ ...container.imageBox, width: cw, height: dh, cursor: "grab", borderColor: PLANE_COLORS[a] }}
                    onMouseDown={(e) => handleFftMouseDown(e, a)}
                    onMouseMove={(e) => handleFftMouseMove(e, a)}
                    onMouseUp={(e) => handleFftMouseUp(e, a)}
                    onMouseLeave={() => { fftClickStartRef.current = null; setFftDragAxis(null); setFftDragStart(null); }}
                    onWheel={(e) => handleFftWheel(e, a)}
                    onDoubleClick={() => handleFftDoubleClick(a)}
                  >
                    <canvas
                      ref={(el) => { fftCanvasRefs.current[a] = el; }}
                      width={cw}
                      height={ch}
                      style={{ width: cw, height: dh, imageRendering: smooth ? "auto" : "pixelated" }}
                      role="img"
                      aria-label={`FFT power spectrum of ${panelName} slice (reciprocal space, ${cw} by ${ch} pixels)`}
                    />
                    <canvas
                      ref={(el) => { fftOverlayRefs.current[a] = el; }}
                      width={Math.round(cw * DPR)}
                      height={Math.round(dh * DPR)}
                      style={{ position: "absolute", top: 0, left: 0, width: cw, height: dh, pointerEvents: "none" }}
                      aria-hidden="true"
                    />
                  </Box>
                </Box>
              )}
              <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, width: cw, maxWidth: cw, boxSizing: "border-box", ...(a === 1 ? { flexDirection: "column", alignItems: "stretch", gap: `${SPACING.XS}px` } : {}) }}>
                {a === 1 ? (
                  <>
                    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 18 }}>
                      <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0, minWidth: 42 }}>Angle</Typography>
                      <Slider
                        value={obliqueAngleSliderValues}
                        min={obliqueAngleSliderMin}
                        max={obliqueAngleSliderMax}
                        step={1}
                        onPointerDownCapture={handleObliqueAnglePointerDownCapture}
                        onChange={(_, v) => handleObliqueAngleSliderChange(v as number[])}
                        onChangeCommitted={(_, v) => handleObliqueAngleSliderChange(v as number[])}
                        disableSwap
                        size="small"
                        sx={{
                          ...sliderStyles.small,
                          flex: 1,
                          minWidth: 40,
                          "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: tc.textMuted },
                          "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
                          "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: tc.textMuted },
                          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
                        }}
                        aria-label={`Oblique plane angle ${Math.round(boundedObliqueAngle)} degrees within ${obliqueAngleMinBound} to ${obliqueAngleMaxBound}`}
                        valueLabelDisplay="off"
                        valueLabelFormat={(v) => `${v as number}°`}
                      />
                      <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 36, textAlign: "right", flexShrink: 0 }}>
                        {Math.round(boundedObliqueAngle)}°
                      </Typography>
                    </Box>
                    <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, minHeight: 18 }}>
                      <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0, minWidth: 42 }}>Position</Typography>
                      <Slider
                        value={obliquePositionSliderValues}
                        min={obliquePositionSliderMin}
                        max={obliquePositionSliderMax}
                        step={1}
                        onPointerDownCapture={handleObliquePositionPointerDownCapture}
                        onChange={(_, v) => handleObliquePositionChange(v as number[])}
                        onChangeCommitted={(_, v) => {
                          handleObliquePositionCommit(v as number[]);
                        }}
                        disableSwap
                        size="small"
                        sx={{
                          ...sliderStyles.small,
                          flex: 1,
                          minWidth: 40,
                          "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: tc.textMuted },
                          "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
                          "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: tc.textMuted },
                          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
                        }}
                        aria-label={`Oblique plane position ${boundedObliqueOffset} within ${obliquePositionMinBound} to ${obliquePositionMaxBound}`}
                        valueLabelDisplay="off"
                        valueLabelFormat={(v) => `${v as number}`}
                      />
                      <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 36, textAlign: "right", flexShrink: 0 }}>
                        {boundedObliqueOffset}
                      </Typography>
                    </Box>
                  </>
                ) : (
                  <>
                <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>{dl[0]}</Typography>
                {loop ? (
                  <Slider
                    value={loopSliderValues(a)}
                    onPointerDownCapture={(e) => handleLoopSliderPointerDownCapture(a, e)}
                    onChange={(_, v) => {
                      handleLoopSliderChange(a, v as number[]);
                    }}
                    onChangeCommitted={(_, v) => {
                      handleLoopSliderCommit(a, v as number[]);
                    }}
                    disableSwap
                    min={0}
                    max={sliceMaxes[a]}
                    size="small"
                    valueLabelDisplay="off"
                    sx={{
                      ...sliderStyles.small,
                      flex: 1,
                      minWidth: 40,
                      "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
                      "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
                    }}
                    aria-label={`Loop range and current ${dl[a]} slice (${liveSlider[a] + 1} of ${sliceMaxes[a] + 1}, loop ${loopStarts[a] + 1} to ${effectiveLoopEnds[a] + 1})`}
                    valueLabelFormat={(v) => `${v as number}`}
                  />
                ) : (
                  <Slider
                    value={liveSlider[a]}
                    min={0}
                    max={sliceMaxes[a]}
                    onChange={sliceSetters[a]}
                    onChangeCommitted={sliceCommitters[a]}
                    size="small"
                    sx={{ ...sliderStyles.small, flex: 1, minWidth: 40 }}
                    aria-label={`${dl[a]} slice ${liveSlider[a] + 1} of ${sliceMaxes[a] + 1}`}
                    valueLabelDisplay="off"
                    valueLabelFormat={(v) => `${v as number}`}
                  />
                )}
                {a === 0 && (
                  <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                    {liveSlider[a]}/{sliceMaxes[a]}
                  </Typography>
                )}
                  </>
                )}
              </Box>
            </Box>
          );
        });
        return (
          <Box sx={{ display: "flex", alignItems: "flex-start", gap: `${SPACING.SM}px`, justifyContent: "flex-start" }}>
            {panels}
          </Box>
        );
      })()}
      </Box> {/* end right column (toolbar + slices) */}
      </Box> {/* end side-by-side row (3D volume + slices) */}
      {/* FFT controls row */}
      {effectiveShowFft && (
        <Box sx={{ ...panelControlRow, mt: `${SPACING.SM}px`, width: primaryPanelW, maxWidth: primaryPanelW, flexWrap: "wrap" }}>
          <Typography sx={{ ...controlLabel }}>FFT Scale</Typography>
          <Select value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...denseSelect, minWidth: 45 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT intensity scale (linear or logarithmic)" }}>
            <MenuItem value="linear">Lin</MenuItem>
            <MenuItem value="log">Log</MenuItem>
          </Select>
          <Typography sx={{ ...controlLabel }}>FFT Color</Typography>
          <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...denseSelect, minWidth: 60 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT colormap" }}>
            {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
          </Select>
          <Typography sx={{ ...controlLabel }}>FFT Auto</Typography>
          <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle automatic FFT contrast" }} />
          <Typography sx={{ ...controlLabel }} title="Apply a Hann window before zero-padding each slice FFT to reduce edge leakage.">Window</Typography>
          <Switch checked={!!fftWindow} onChange={(e) => setFftWindow(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle Hann window before FFT" }} />
        </Box>
      )}
      {/* Controls row with histogram anchored to the slice panel columns. */}
      {showControls && (() => {
        const histogramW = 110;
        const histogramH = controlRowHeight * 2 + SPACING.XS;
        return (
        <Box sx={{
          mt: `${SPACING.SM}px`,
          display: "flex",
          gap: `${SPACING.SM}px`,
          alignItems: "flex-start",
          width: "fit-content",
          maxWidth: panelTotalW,
          boxSizing: "border-box",
        }}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, justifyContent: "flex-start", minWidth: 0 }}>
            <Box sx={{ ...panelControlRow, width: compactControlsW, maxWidth: compactControlsW, flexWrap: "wrap" }}>
              <Typography sx={{ ...controlLabel }}>Color</Typography>
              <Select size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...denseSelect, minWidth: 60 }} inputProps={{ "aria-label": "Image colormap" }}>
                {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
              </Select>
              <Typography sx={{ ...controlLabel }}>Colorbar</Typography>
              <Switch checked={showColorbar} onChange={(e) => setShowColorbar(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle colorbar overlay" }} />
              <Typography sx={{ ...controlLabel }} title="CSS bilinear interpolation on image canvas. Off = pixelated.">Smooth</Typography>
              <Switch checked={smooth} onChange={(e) => setSmooth(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle bilinear smoothing" }} />
            </Box>
            <Box sx={{ ...panelControlRow, width: compactControlsW, maxWidth: compactControlsW, flexWrap: "wrap" }}>
              {thinZ && (
                <>
                  <Typography sx={{ ...controlLabel }} title="Depth-axis display height multiplier (1-50x). CSS-only stretch; data unchanged. Useful when nz << nxy (e.g. multislice ptycho).">Z stretch</Typography>
                  <LiveNumberSlider value={zStretch} min={1} max={50} step={0.5} onLiveChange={handleZStretchChange} onCommit={handleZStretchCommit} sx={{ ...sliderStyles.small, width: 80, mr: 1, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }} ariaLabel="Depth axis display stretch multiplier" />
                </>
              )}
              <Typography sx={clickableControlLabel} title="Negate displayed values. Useful when phase sign is inverted." onClick={() => setFlip(!flip)}>Flip</Typography>
              <Switch checked={flip} onChange={(e) => setFlip(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Flip (negate) displayed values" }} />
              <Typography sx={{ ...controlLabel }} title="Log scale (signed log1p). Useful for high-dynamic-range volumes.">Log</Typography>
              <Switch checked={logScale} onChange={(e) => setLogScale(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle log scale (signed log1p) display" }} />
              <Typography sx={{ ...controlLabel }}>Auto</Typography>
              <Switch checked={autoContrast} onChange={(e) => handleAutoContrastChange(e.target.checked)} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle automatic percentile-based contrast" }} />
            </Box>
          </Box>
          <Box sx={{ display: "flex", flexDirection: "row", gap: `${SPACING.SM}px`, alignItems: "flex-start", justifyContent: "flex-start" }}>
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start" }}>
              <Histogram
                data={imageHistogramData}
                vminPct={imageVminPct}
                vmaxPct={imageVmaxPct}
                onRangeChange={(min, max) => {
                  paintContrastRange(min, max);
                }}
                onRangeCommit={(min, max) => {
                  // User drag overrides Auto. Commit once on release so dragging stays local.
                  if (autoContrast) {
                    manualImageRangeBeforeAutoRef.current = null;
                    setAutoContrast(false);
                  }
                  setImageVminPct(min);
                  setImageVmaxPct(max);
                }}
                width={histogramW}
                height={histogramH}
                theme={themeInfo.theme === "dark" ? "dark" : "light"}
                dataMin={displayDataRange.min}
                dataMax={displayDataRange.max}
                pinBinsToRange={false}
                ariaHidden
              />
            </Box>
          </Box>
        </Box>
        );
      })()}
      {/* Playback: transport + axis selector + fps + loop + bounce */}
      <Box sx={{ ...panelControlRow, mt: `${SPACING.SM}px`, width: compactControlsW, maxWidth: compactControlsW, flexWrap: "nowrap" }}>
        <Select
          value={playbackAxis}
          onChange={(e) => { setPlaying(false); setPlayAxis(Number(e.target.value)); }}
          size="small"
          sx={{ ...denseSelect, minWidth: 40 }}
          MenuProps={themedMenuProps}
          inputProps={{ "aria-label": "Playback axis (Top, Side, or All)" }}
        >
          <MenuItem value={0}>Top</MenuItem>
          <MenuItem value={1}>Side</MenuItem>
          <MenuItem value={3}>All</MenuItem>
        </Select>
        <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
          <IconButton size="small" onClick={() => setReverse(!reverse)} sx={{ color: reverse ? tc.accent : tc.textMuted, p: 0.25 }} aria-label={reverse ? "Playback direction reverse" : "Playback direction forward"} aria-pressed={reverse} title={reverse ? "Direction: reverse" : "Direction: forward"}>
            <FastRewindIcon sx={{ fontSize: 18, transform: reverse ? "none" : "scaleX(-1)" }} />
          </IconButton>
          <IconButton size="small" onClick={() => setPlaying(!playing)} sx={{ color: tc.accent, p: 0.3 }} aria-label={playing ? "Pause playback" : "Play"} title={playing ? "Pause (Space)" : "Play (Space)"}>
            {playing ? <PauseIcon sx={{ fontSize: 20 }} /> : <PlayArrowIcon sx={{ fontSize: 20 }} />}
          </IconButton>
          <IconButton size="small" onClick={stopPlaybackAndRewind} sx={{ color: tc.textMuted, p: 0.25 }} aria-label="Stop and rewind to loop start" title="Stop">
            <StopIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Stack>
        <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>fps</Typography>
        <LiveNumberSlider
          value={fps}
          min={1}
          max={MAX_PLAYBACK_FPS}
          step={1}
          onLiveChange={(value) => {
            fpsRef.current = value;
            setFps(value);
          }}
          onCommit={(value) => {
            fpsRef.current = value;
            setFps(value);
            setModelFps(value);
          }}
          sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }}
          ariaLabel={`Playback frames per second (${Math.round(fps)})`}
        />
        <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>Loop</Typography>
        <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ ...switchStyles.small, flexShrink: 0 }} inputProps={{ "aria-label": "Toggle loop playback" }} />
        <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>Bounce</Typography>
        <Switch size="small" checked={boomerang} onChange={() => setBoomerang(!boomerang)} sx={{ ...switchStyles.small, flexShrink: 0 }} inputProps={{ "aria-label": "Toggle bounce (ping-pong) playback" }} />
      </Box>
    </Box>
  );
}

// anywidget v0.9+ deprecates `export render` in favor of `export default { render }`.
const render = createRender(Show3DSlices);
export default { render };
