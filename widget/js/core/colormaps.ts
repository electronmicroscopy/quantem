/**
 * Colormap definitions and utilities for image display.
 * Shared across Show2D, Show3D, Show4D widgets.
 */

// Control points for interpolation
export const COLORMAP_POINTS: Record<string, number[][]> = {
  inferno: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [252, 193, 57], [252, 255, 164],
  ],
  viridis: [
    [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141],
    [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 191, 112],
    [122, 209, 81], [189, 223, 38], [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
  ],
  hot: [
    [0, 0, 0], [87, 0, 0], [173, 0, 0], [255, 0, 0],
    [255, 87, 0], [255, 173, 0], [255, 255, 0], [255, 255, 128], [255, 255, 255],
  ],
  gray: [[0, 0, 0], [255, 255, 255]],
};

/** Available colormap names */
export const COLORMAP_NAMES = Object.keys(COLORMAP_POINTS);

/** Create 256-entry LUT from control points */
export function createColormapLUT(points: number[][]): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (points.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;
    const p0 = points[Math.min(idx, points.length - 1)];
    const p1 = points[Math.min(idx + 1, points.length - 1)];
    lut[i * 3] = Math.round(p0[0] + frac * (p1[0] - p0[0]));
    lut[i * 3 + 1] = Math.round(p0[1] + frac * (p1[1] - p0[1]));
    lut[i * 3 + 2] = Math.round(p0[2] + frac * (p1[2] - p0[2]));
  }
  return lut;
}

/** Pre-computed LUTs for all colormaps (flat Uint8Array, 256*3 bytes each) */
export const COLORMAPS: Record<string, Uint8Array> = Object.fromEntries(
  Object.entries(COLORMAP_POINTS).map(([name, points]) => [name, createColormapLUT(points)])
);

/** Apply colormap to a single normalized value [0,1] */
export function applyColormapValue(
  value: number,
  cmap: number[][]
): [number, number, number] {
  const n = cmap.length - 1;
  const t = Math.max(0, Math.min(1, value)) * n;
  const i = Math.min(Math.floor(t), n - 1);
  const f = t - i;
  return [
    Math.round(cmap[i][0] * (1 - f) + cmap[i + 1][0] * f),
    Math.round(cmap[i][1] * (1 - f) + cmap[i + 1][1] * f),
    Math.round(cmap[i][2] * (1 - f) + cmap[i + 1][2] * f),
  ];
}

/**
 * Apply colormap to uint8 grayscale data, returning RGBA ImageData.
 * @param data - Uint8Array of grayscale values (0-255)
 * @param width - Image width
 * @param height - Image height
 * @param cmapName - Name of colormap to use
 * @returns Uint8ClampedArray of RGBA values
 */
export function applyColormapToImage(
  data: Uint8Array,
  width: number,
  height: number,
  cmapName: string
): Uint8ClampedArray {
  const lut = COLORMAPS[cmapName] || COLORMAPS.inferno;
  const rgba = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < data.length; i++) {
    const v = Math.max(0, Math.min(255, data[i]));
    const j = i * 4;
    const lutIdx = v * 3;
    rgba[j] = lut[lutIdx];
    rgba[j + 1] = lut[lutIdx + 1];
    rgba[j + 2] = lut[lutIdx + 2];
    rgba[j + 3] = 255;
  }

  return rgba;
}
