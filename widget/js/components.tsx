/**
 * Shared styling constants and simple UI components for bobleesj.widget.
 * 
 * ARCHITECTURE NOTE: Only styling should be shared here.
 * Widget-specific logic (resize handlers, zoom handlers) should be inlined per-widget.
 */

import * as React from "react";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { colors, controlPanel, typography } from "./CONFIG";

// ============================================================================
// Switch Style Constants
// ============================================================================
export const switchStyles = {
    small: {
        '& .MuiSwitch-thumb': { width: 12, height: 12 },
        '& .MuiSwitch-switchBase': { padding: '4px' },
    },
    medium: {
        '& .MuiSwitch-thumb': { width: 14, height: 14 },
        '& .MuiSwitch-switchBase': { padding: '4px' },
    },
};

// ============================================================================
// Select MenuProps for upward dropdown (all widgets use this)
// ============================================================================
export const upwardMenuProps = {
    anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
    transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
    sx: { zIndex: 9999 },
};

// ============================================================================
// LabeledSwitch - Label + Switch combo (optional, use if needed)
// ============================================================================
interface LabeledSwitchProps {
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    size?: "small" | "medium";
}

export function LabeledSwitch({ label, checked, onChange, size = "small" }: LabeledSwitchProps) {
    return (
        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>{label}:</Typography>
            <Switch
                checked={checked}
                onChange={(e) => onChange(e.target.checked)}
                size="small"
                sx={switchStyles[size]}
            />
        </Stack>
    );
}

// ============================================================================
// LabeledSelect - Label + Select dropdown combo (optional, use if needed)
// ============================================================================
interface LabeledSelectProps<T extends string> {
    label: string;
    value: T;
    options: readonly T[] | T[];
    onChange: (value: T) => void;
    formatLabel?: (value: T) => string;
}

export function LabeledSelect<T extends string>({
    label,
    value,
    options,
    onChange,
    formatLabel,
}: LabeledSelectProps<T>) {
    return (
        <Stack direction="row" spacing={1} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>{label}:</Typography>
            <Select
                value={value}
                onChange={(e) => onChange(e.target.value as T)}
                size="small"
                sx={{ ...controlPanel.select }}
                MenuProps={upwardMenuProps}
            >
                {options.map((opt) => (
                    <MenuItem key={opt} value={opt}>
                        {formatLabel ? formatLabel(opt) : opt}
                    </MenuItem>
                ))}
            </Select>
        </Stack>
    );
}

// ============================================================================
// ScaleBar - Overlay component for canvas scale bars
// ============================================================================
interface ScaleBarProps {
    zoom: number;
    size: number;
    label?: string;
}

export function ScaleBar({ zoom, size, label = "px" }: ScaleBarProps) {
    const scaleBarPx = 50;
    const realPixels = Math.round(scaleBarPx / zoom);

    return (
        <div style={{
            position: "absolute",
            bottom: 8,
            right: 8,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            pointerEvents: "none"
        }}>
            <span style={{
                color: "#fff",
                fontSize: 9,
                textShadow: "0 0 3px #000",
                marginBottom: 2
            }}>
                {realPixels} {label}
            </span>
            <div style={{
                width: scaleBarPx,
                height: 3,
                backgroundColor: "#fff",
                boxShadow: "0 0 3px #000"
            }} />
        </div>
    );
}

// ============================================================================
// ZoomIndicator - Overlay component for zoom level display
// ============================================================================
interface ZoomIndicatorProps {
    zoom: number;
}

export function ZoomIndicator({ zoom }: ZoomIndicatorProps) {
    return (
        <span style={{
            position: "absolute",
            bottom: 8,
            left: 8,
            color: "#fff",
            fontSize: 10,
            textShadow: "0 0 3px #000",
            pointerEvents: "none"
        }}>
            {zoom.toFixed(1)}×
        </span>
    );
}

// ============================================================================
// ResetButton - Compact reset button
// ============================================================================
interface ResetButtonProps {
    onClick: () => void;
    label?: string;
}

export function ResetButton({ onClick, label = "Reset" }: ResetButtonProps) {
    return (
        <Typography
            component="span"
            onClick={onClick}
            sx={{ ...controlPanel.button }}
        >
            {label}
        </Typography>
    );
}

// ============================================================================
// ControlGroup - Wrapper for control panel groups
// ============================================================================
interface ControlGroupProps {
    children: React.ReactNode;
}

export function ControlGroup({ children }: ControlGroupProps) {
    return (
        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
            {children}
        </Stack>
    );
}

// ============================================================================
// ColormapSelect - Colormap dropdown with standard options
// ============================================================================
const COLORMAP_OPTIONS = ["inferno", "viridis", "plasma", "magma", "hot", "gray"] as const;

interface ColormapSelectProps {
    value: string;
    onChange: (value: string) => void;
}

export function ColormapSelect({ value, onChange }: ColormapSelectProps) {
    return (
        <LabeledSelect
            label="Colormap"
            value={value as typeof COLORMAP_OPTIONS[number]}
            options={COLORMAP_OPTIONS}
            onChange={onChange}
            formatLabel={(v) => v.charAt(0).toUpperCase() + v.slice(1)}
        />
    );
}
