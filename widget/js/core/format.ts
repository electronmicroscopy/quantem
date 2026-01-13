/**
 * Number and text formatting utilities.
 */

/**
 * Format a number for display with appropriate precision.
 * Uses exponential notation for very large or small values.
 * @param val - Value to format
 * @param decimals - Number of decimal places (default 2)
 * @returns Formatted string
 */
export function formatNumber(val: number, decimals: number = 2): string {
  if (val === 0) return "0";
  if (Math.abs(val) >= 1000 || Math.abs(val) < 0.01) {
    return val.toExponential(decimals);
  }
  return val.toFixed(decimals);
}

/**
 * Format bytes as human-readable size.
 * @param bytes - Number of bytes
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * Format time duration.
 * @param seconds - Duration in seconds
 * @returns Formatted string (e.g., "1.5 s" or "150 ms")
 */
export function formatDuration(seconds: number): string {
  if (seconds < 0.001) {
    return `${(seconds * 1e6).toFixed(0)} µs`;
  }
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(1)} ms`;
  }
  if (seconds < 60) {
    return `${seconds.toFixed(2)} s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs.toFixed(0)}s`;
}

/**
 * Clamp a value between min and max.
 */
export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}
