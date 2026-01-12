/**
 * Export utilities for downloading widget canvases as images.
 * Composites multiple canvas layers and burns in overlays (scale bars, etc).
 */

/**
 * Generate a timestamped filename.
 * @param prefix - Filename prefix (e.g., "show2d", "show3d")
 * @param extension - File extension (default: "png")
 */
export function generateFilename(prefix: string, extension: string = "png"): string {
  const now = new Date();
  const timestamp = now.toISOString()
    .replace(/[:.]/g, "-")
    .slice(0, 19);
  return `${prefix}_${timestamp}.${extension}`;
}

/**
 * Composite multiple canvases into a single canvas.
 * Layers are drawn in order (first = bottom, last = top).
 * @param layers - Array of canvases to composite
 * @param width - Output width
 * @param height - Output height
 */
export function compositeCanvases(
  layers: (HTMLCanvasElement | null)[],
  width: number,
  height: number
): HTMLCanvasElement {
  const output = document.createElement("canvas");
  output.width = width;
  output.height = height;
  const ctx = output.getContext("2d");

  if (ctx) {
    // Fill with black background
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    // Draw each layer
    for (const layer of layers) {
      if (layer) {
        ctx.drawImage(layer, 0, 0, width, height);
      }
    }
  }

  return output;
}

/**
 * Download a canvas as a PNG file.
 * @param canvas - The canvas to download
 * @param filename - Output filename
 */
export function downloadCanvas(canvas: HTMLCanvasElement, filename: string): void {
  canvas.toBlob((blob) => {
    if (!blob) return;

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();

    // Cleanup
    URL.revokeObjectURL(url);
  }, "image/png");
}

/**
 * Export a widget's canvas with overlays burned in.
 * @param imageCanvas - Main image canvas
 * @param overlayCanvas - Overlay canvas (scale bar, etc)
 * @param prefix - Filename prefix
 * @param label - Optional label to append to filename
 */
export function exportWithOverlay(
  imageCanvas: HTMLCanvasElement | null,
  overlayCanvas: HTMLCanvasElement | null,
  prefix: string,
  label?: string
): void {
  if (!imageCanvas) return;

  const width = imageCanvas.width;
  const height = imageCanvas.height;

  const output = compositeCanvases([imageCanvas, overlayCanvas], width, height);

  // Generate filename with optional label
  const cleanLabel = label ? `_${label.replace(/[^a-zA-Z0-9]/g, "_")}` : "";
  const filename = generateFilename(`${prefix}${cleanLabel}`);

  downloadCanvas(output, filename);
}

/**
 * Export multiple canvases as a ZIP file (for galleries).
 * Requires JSZip to be available.
 */
export async function exportGalleryAsZip(
  canvases: { image: HTMLCanvasElement | null; overlay: HTMLCanvasElement | null; label: string }[],
  prefix: string
): Promise<void> {
  // Dynamic import to avoid bundling JSZip if not needed
  const JSZip = (await import("jszip")).default;
  const zip = new JSZip();

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

  for (let i = 0; i < canvases.length; i++) {
    const { image, overlay, label } = canvases[i];
    if (!image) continue;

    const output = compositeCanvases([image, overlay], image.width, image.height);
    const cleanLabel = label.replace(/[^a-zA-Z0-9]/g, "_");
    const filename = `${String(i + 1).padStart(3, "0")}_${cleanLabel}.png`;

    const blob = await new Promise<Blob>((resolve) => {
      output.toBlob((b) => resolve(b!), "image/png");
    });

    zip.file(filename, blob);
  }

  const zipBlob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(zipBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${prefix}_gallery_${timestamp}.zip`;
  link.click();
  URL.revokeObjectURL(url);
}
