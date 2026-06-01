/**
 * Shared theme detection and color system for all widgets.
 * Detects JupyterLab, VS Code, Colab, Classic Jupyter, and OS preferences.
 */

import { useState, useEffect, useMemo } from "react";

// ============================================================================
// Types
// ============================================================================
export type Environment = "jupyterlab" | "vscode" | "colab" | "jupyter-classic" | "unknown";
export type Theme = "light" | "dark";

export interface ThemeInfo {
  environment: Environment;
  theme: Theme;
}

export interface ThemeColors {
  bg: string;
  bgAlt: string;
  text: string;
  textMuted: string;
  border: string;
  controlBg: string;
  accent: string;
}

// ============================================================================
// Color palettes
// ============================================================================
export const DARK_COLORS: ThemeColors = {
  bg: "#1e1e1e",
  bgAlt: "#1a1a1a",
  text: "#e0e0e0",
  textMuted: "#888",
  border: "#3a3a3a",
  controlBg: "#252525",
  accent: "#5af",
};

export const LIGHT_COLORS: ThemeColors = {
  bg: "#ffffff",
  bgAlt: "#f5f5f5",
  text: "#1e1e1e",
  textMuted: "#666",
  border: "#ccc",
  controlBg: "#f0f0f0",
  accent: "#0066cc",
};

export function getThemeColors(theme: Theme): ThemeColors {
  return theme === "dark" ? DARK_COLORS : LIGHT_COLORS;
}

// ============================================================================
// Theme detection
// ============================================================================

/** Check if a CSS color string is dark (luminance < 0.5) */
export function isColorDark(color: string): boolean {
  const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (!match) return true;
  const [, r, g, b] = match.map(Number);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
}

export function detectTheme(): ThemeInfo {
  // 1. JupyterLab - has data-jp-theme-light attribute
  const jpThemeLight = document.body.dataset.jpThemeLight;
  if (jpThemeLight !== undefined) {
    return {
      environment: "jupyterlab",
      theme: jpThemeLight === "true" ? "light" : "dark",
    };
  }

  // 2. VS Code - has vscode-* classes on body or html
  const bodyClasses = document.body.className;
  const htmlClasses = document.documentElement.className;
  if (bodyClasses.includes("vscode-") || htmlClasses.includes("vscode-")) {
    const isDark = bodyClasses.includes("vscode-dark") || htmlClasses.includes("vscode-dark");
    return {
      environment: "vscode",
      theme: isDark ? "dark" : "light",
    };
  }

  // 3. Google Colab - has specific markers
  if (document.querySelector('colab-shaded-scroller') || document.body.classList.contains('colaboratory')) {
    const bg = getComputedStyle(document.body).backgroundColor;
    return {
      environment: "colab",
      theme: isColorDark(bg) ? "dark" : "light",
    };
  }

  // 4. Classic Jupyter Notebook - has #notebook element
  if (document.getElementById('notebook')) {
    const bodyBg = getComputedStyle(document.body).backgroundColor;
    return {
      environment: "jupyter-classic",
      theme: isColorDark(bodyBg) ? "dark" : "light",
    };
  }

  // 5. Fallback: check OS preference, then computed background
  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)')?.matches;
  if (prefersDark !== undefined) {
    return {
      environment: "unknown",
      theme: prefersDark ? "dark" : "light",
    };
  }

  // Final fallback: check body background luminance
  const bg = getComputedStyle(document.body).backgroundColor;
  return {
    environment: "unknown",
    theme: isColorDark(bg) ? "dark" : "light",
  };
}

// ============================================================================
// React hook
// ============================================================================
export function useTheme(forceLight = false): { themeInfo: ThemeInfo; colors: ThemeColors } {
  const [detected, setDetected] = useState<ThemeInfo>(() => detectTheme());

  useEffect(() => {
    const mediaQuery = window.matchMedia?.('(prefers-color-scheme: dark)');
    const handleChange = () => setDetected(detectTheme());
    mediaQuery?.addEventListener?.('change', handleChange);

    const observer = new MutationObserver(() => setDetected(detectTheme()));
    observer.observe(document.body, { attributes: true, attributeFilter: ['data-jp-theme-light', 'class'] });

    return () => {
      mediaQuery?.removeEventListener?.('change', handleChange);
      observer.disconnect();
    };
  }, []);

  // Offline HTML exports always render on a light (white) background: the report
  // is a static, theme-less artifact and a forced dark page leaks the kernel's
  // dark Jupyter theme into a shared file the colleague opens in any browser.
  const themeInfo: ThemeInfo = forceLight ? { ...detected, theme: "light" } : detected;
  // Memoize by theme string so `colors` is referentially stable across renders —
  // effects/components that depend on `colors` only re-run when the theme flips.
  const colors = useMemo(() => getThemeColors(themeInfo.theme), [themeInfo.theme]);
  return { themeInfo, colors };
}
