/** Convert anywidget DataView/ArrayBuffer to Uint8Array. */
export function extractBytes(dataView: DataView | ArrayBuffer | Uint8Array): Uint8Array {
  if (dataView instanceof Uint8Array) return dataView;
  if (dataView instanceof ArrayBuffer) return new Uint8Array(dataView);
  if (dataView && "buffer" in dataView) {
    return new Uint8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength);
  }
  return new Uint8Array(0);
}

/** Extract Float32Array from anywidget DataView. Returns null if empty. */
export function extractFloat32(dataView: DataView | ArrayBuffer | Uint8Array): Float32Array | null {
  const bytes = extractBytes(dataView);
  if (bytes.length === 0) return null;
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** Download a Blob as a file. */
export function downloadBlob(blob: Blob, filename: string): void {
  const link = document.createElement("a");
  link.download = filename;
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.click();
  // Defer revocation to ensure browser has time to start the download
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

/** Download a DataView as a file (e.g. GIF/ZIP from Python). */
export function downloadDataView(dataView: DataView, filename: string, mimeType: string): void {
  const buf = new Uint8Array(dataView.buffer as ArrayBuffer, dataView.byteOffset, dataView.byteLength);
  downloadBlob(new Blob([buf as BlobPart], { type: mimeType }), filename);
}

/** Format number with exponential notation for large/small values. */
export function formatNumber(val: number, decimals: number = 2): string {
  if (val === 0) return "0";
  if (Math.abs(val) >= 1000 || Math.abs(val) < 0.01) return val.toExponential(decimals);
  return val.toFixed(decimals);
}
