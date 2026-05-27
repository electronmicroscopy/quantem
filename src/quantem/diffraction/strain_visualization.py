from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization.visualization_utils import ScalebarConfig, add_scalebar_to_ax


class StrainFitting(AutoSerialize):
    mask: np.ndarray | None = None
    real_space: bool = False

    strain_raw_err: Dataset2d
    strain_raw_ecc: Dataset2d
    strain_raw_erc: Dataset2d
    strain_rotation: Dataset2d

    u_ref: np.ndarray | None = None
    v_ref: np.ndarray | None = None
    u_array: np.ndarray
    v_array: np.ndarray

    ds_sampling = 1.0
    ds_units = 'pixels'
    ds_shape: tuple[int, ...]


    def __init__(
        self,
        u_array: np.ndarray,
        v_array: np.ndarray,
        ds_shape: tuple[int, ...],
        real_space: bool,

        u_ref: np.ndarray | None = None,
        v_ref: np.ndarray | None = None,
        mask: np.ndarray | None = None,

        ds_sampling: float | None = None,
        ds_units: str | None = None,
    ):
        super().__init__()
        self.u_array = u_array
        self.v_array = v_array

        self.ds_shape = ds_shape
        self.real_space = real_space

        self.ds_sampling = 1.0 if ds_sampling is None else ds_sampling
        self.ds_units = 'pixels' if ds_units is None else ds_units

        self.mask = np.ones(ds_shape[:2]) if mask is None else mask
        self.mask = (self.mask - np.min(self.mask))/np.linalg.norm(self.mask)   

        self.u_ref = u_ref if u_ref is not None else None
        self.v_ref = v_ref if v_ref is not None else None
        


    def fit_strain(
        self, 
        plot_strain = False,
        strain_mask: np.ndarray | None = None,
    ) -> "StrainFitting":
        
        if self.u_array is None:
            raise RuntimeError("u_array not available. Set u_array before calling fit_strain().")
        if self.v_array is None:
            raise RuntimeError("v_array not available. Set v_array before calling fit_strain().")

        u_fit = self.u_array
        v_fit = self.v_array

        if strain_mask is not None:
            m = np.asarray((strain_mask==1), dtype=bool)
            u_ref = np.array(
                (np.median(u_fit[m, 0]), np.median(u_fit[m, 1])),
                dtype=float,
            )
            v_ref = np.array(
                (np.median(v_fit[m, 0]), np.median(v_fit[m, 1])),
                dtype=float,
            )
        elif self.mask is None:
            u_ref = np.median(u_fit.reshape(-1, 2), axis=0)
            v_ref = np.median(v_fit.reshape(-1, 2), axis=0)
        else:
            m = np.asarray((self.mask==1), dtype=bool)
            u_ref = np.array(
                (np.median(u_fit[m, 0]), np.median(u_fit[m, 1])),
                dtype=float,
            )
            v_ref = np.array(
                (np.median(v_fit[m, 0]), np.median(v_fit[m, 1])),
                dtype=float,
            )
        if self.u_ref is not None:
            u_ref = self.u_ref
        if self.v_ref is not None:
            v_ref = self.v_ref
        self.u_ref = u_ref
        self.v_ref = v_ref
        scan_r = self.ds_shape[0]
        scan_c = self.ds_shape[1]

        Uref = np.stack((u_ref, v_ref), axis=1).astype(float)
        strain_trans = np.zeros((scan_r, scan_c, 2, 2))
        for r in range(scan_r):
            for c in range(scan_c):
                U = np.stack((u_fit[r, c, :], v_fit[r, c, :]), axis=1)
                det = np.linalg.det(U)
                if not np.isfinite(det) or abs(det) < 1e-12:
                    U_inv = np.linalg.pinv(U)
                else:
                    U_inv = np.linalg.inv(U)
                strain_trans[r, c, :, :] = Uref @ U_inv
        const = 1
        if self.real_space is False:
            const = -1

        self.strain_raw_err = Dataset2d.from_array(
            strain_trans[:, :, 0, 0] - 1,
            name="strain err",
            signal_units="fractional",
        )
        self.strain_raw_ecc = Dataset2d.from_array(
            strain_trans[:, :, 1, 1] - 1,
            name="strain ecc",
            signal_units="fractional",
        )
        self.strain_raw_erc = Dataset2d.from_array(
            strain_trans[:, :, 1, 0] * 0.5 * const + strain_trans[:, :, 0, 1] * 0.5 * const,
            name="strain erc",
            signal_units="fractional",
        )
        self.strain_rotation = Dataset2d.from_array(
            strain_trans[:, :, 1, 0] * -0.5 * const + strain_trans[:, :, 0, 1] * 0.5 * const,
            name="strain rotation",
            signal_units="fractional",
        )
        if plot_strain:
            self.plot_strain()
        return self


    def plot_strain(
        self,
        rotation_angle=20,
        strain_range_percent=(-3.0, 3.0),
        rotation_range_degrees=(-2.0, 2.0),
        plot_rotation=True,
        plot_gvecs=True,
        plot_scalebar=False,
        cmap_strain="RdBu_r",
        cmap_rotation="PiYG",
        layout="horizontal",
        figsize=None,
        **kwargs,
    ):
        if not hasattr(self, 'strain_raw_err') or self.strain_raw_err is None:
            raise RuntimeError("Call fit_strain() first.")
        if not hasattr(self, 'strain_raw_ecc') or self.strain_raw_ecc is None:
            raise RuntimeError("Call fit_strain() first.")
        if not hasattr(self, 'strain_raw_erc') or self.strain_raw_erc is None:
            raise RuntimeError("Call fit_strain() first.")
        if not hasattr(self, 'strain_rotation') or self.strain_rotation is None:
            raise RuntimeError("Call fit_strain() first.")
        
        if self.mask is None:
            self.mask = np.zeros(self.ds_shape[:2])

        if cmap_rotation is None:
            cmap_rotation = cmap_strain

        if layout not in ["horizontal", "vertical"]:
            raise ValueError("layout must be 'horizontal' or 'vertical'")

        angle = np.deg2rad(rotation_angle)
        c = np.cos(angle)
        s = np.sin(angle)

        err = self.strain_raw_err.array
        ecc = self.strain_raw_ecc.array
        erc = self.strain_raw_erc.array

        euu = err * (c * c) + 2.0 * erc * (c * s) + ecc * (s * s)
        evv = err * (s * s) - 2.0 * erc * (c * s) + ecc * (c * c)
        euv = (ecc - err) * (c * s) + erc * (c * c - s * s)

        strain_euu = self.strain_raw_err.copy()
        strain_evv = self.strain_raw_ecc.copy()
        strain_euv = self.strain_raw_erc.copy()
        strain_euu.array[...] = euu
        strain_evv.array[...] = evv
        strain_euv.array[...] = euv

        ncols = 4 if plot_rotation else 3
        is_horizontal = layout == "horizontal"
        
        if figsize is None:
            figsize = (6, 6) if is_horizontal else (6, 8)
        
        if is_horizontal:
            fig, ax = plt.subplots(1, ncols, figsize=figsize)
        else:
            fig, ax = plt.subplots(ncols, 1, figsize=figsize)

        cm_strain = plt.get_cmap(cmap_strain).copy()
        cm_strain.set_bad(color="black")
        cm_rot = plt.get_cmap(cmap_rotation).copy()
        cm_rot.set_bad(color="black")

        euu_pct = strain_euu.array * 100
        evv_pct = strain_evv.array * 100
        euv_pct = strain_euv.array * 100
        rot_deg = np.rad2deg(self.strain_rotation.array)

        norm_strain = Normalize(vmin=strain_range_percent[0], vmax=strain_range_percent[1])
        euu_rgb = cm_strain(norm_strain(euu_pct))[:, :, :3]
        evv_rgb = cm_strain(norm_strain(evv_pct))[:, :, :3]
        euv_rgb = cm_strain(norm_strain(euv_pct))[:, :, :3]

        title_fs = 16
        ax[0].imshow(euu_rgb * self.mask[:, :, np.newaxis],)
        ax[1].imshow(evv_rgb * self.mask[:, :, np.newaxis],)
        ax[2].imshow(euv_rgb * self.mask[:, :, np.newaxis],)

        ax[0].set_title(r"$\epsilon_{uu}$ $\updownarrow$", fontsize=title_fs)
        ax[1].set_title(r"$\epsilon_{vv}$ $\leftrightarrow$", fontsize=title_fs)
        ax[2].set_title(r"$\epsilon_{uv}$ $\nearrow\!\!\!\!\!\!\!\!\!\:\swarrow$", fontsize=title_fs)

        if plot_rotation:
            norm_rot = Normalize(vmin=rotation_range_degrees[0], vmax=rotation_range_degrees[1])
            rot_rgb = cm_rot(norm_rot(rot_deg))[:, :, :3]
            ax[3].imshow(rot_rgb * self.mask[:, :, np.newaxis],)
            ax[3].set_title(r"Rotation $\circlearrowleft$", fontsize=title_fs)

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            a.set_facecolor("black")
            a.set_aspect("equal")

        if plot_scalebar:
            scalebar_kwargs = {}
            for key, value in kwargs.items():
                if key.startswith('scalebar_'):
                    scalebar_key = key[len('scalebar_'):]
                    scalebar_kwargs[scalebar_key] = value
            
            scalebar_defaults = {
                'sampling': self.ds_sampling,
                'units': self.ds_units,
                'length': None,
                'width_px': 1,
                'pad_px': 0.5,
                'color': 'black',
                'loc': 'lower left',
                'fontsize': 12,
                'bold': True,
            }
            scalebar_defaults.update(scalebar_kwargs)
            scalebar_config = ScalebarConfig(**scalebar_defaults)
            add_scalebar_to_ax(
                ax[0],
                array_size=int(self.ds_shape[0]),
                sampling=scalebar_config.sampling,
                length_units=scalebar_config.length,
                units=scalebar_config.units,
                width_px=scalebar_config.width_px,
                pad_px=scalebar_config.pad_px,
                color=scalebar_config.color,
                loc=scalebar_config.loc,
                fontsize=scalebar_config.fontsize,
                bold=scalebar_config.bold,
            )

        if is_horizontal:
            fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.16, wspace=0.05)
            if plot_rotation:
                pos3 = ax[3].get_position()
                ax[3].set_position([pos3.x0 + 0.03, pos3.y0, pos3.width, pos3.height])
            
            cb_orientation = "horizontal"
            cb_size = 0.02
            cb_pad = 0.02
            
            b0 = ax[0].get_position()
            b2 = ax[2].get_position()
            strain_cb_pos = [b0.x0, b0.y0 - cb_pad - cb_size, b2.x1 - b0.x0, cb_size]
            
            if plot_rotation:
                b3 = ax[3].get_position()
                rot_cb_pos = [b3.x0, b0.y0 - cb_pad - cb_size, b3.x1 - b3.x0, cb_size]
                last_pos = b3
            else:
                rot_cb_pos = None
                last_pos = b2
                
        else:
            fig.subplots_adjust(left=0.04, right=0.80, top=0.98, bottom=0.04, hspace=0.15)
            
            cb_orientation = "vertical"
            cb_size = 0.02
            cb_pad = 0.02
            
            b0 = ax[0].get_position()
            b2 = ax[2].get_position()
            strain_cb_pos = [b0.x1 + cb_pad, b2.y0, cb_size, b0.y1 - b2.y0]
            
            if plot_rotation:
                b3 = ax[3].get_position()
                rot_cb_pos = [b0.x1 + cb_pad, b3.y0, cb_size, b3.y1 - b3.y0]
                last_pos = b3
            else:
                rot_cb_pos = None
                last_pos = b2

        cax1 = fig.add_axes(strain_cb_pos)
        sm_strain = ScalarMappable(norm=norm_strain, cmap=cm_strain)
        cbar1 = fig.colorbar(sm_strain, cax=cax1, orientation=cb_orientation)
        cbar1.set_label("Strain (%)", fontsize=title_fs)
        cbar1.ax.tick_params(labelsize=12)

        if plot_rotation and rot_cb_pos is not None:
            cax2 = fig.add_axes(rot_cb_pos)
            sm_rot = ScalarMappable(norm=norm_rot, cmap=cm_rot)
            cbar2 = fig.colorbar(sm_rot, cax=cax2, orientation=cb_orientation)
            cbar2.set_label("Rotation (deg)", fontsize=title_fs)
            cbar2.ax.tick_params(labelsize=12)
        
        if plot_gvecs:
            if self.u_ref is None or self.v_ref is None:
                print("Warning: u_ref and v_ref not found. Call fit_strain() first.")
                return fig, ax
            
            if is_horizontal:
                ref_width = last_pos.width * 0.8
                ref_left = last_pos.x1 - 0.035
                ref_ax = fig.add_axes([ref_left, last_pos.y0, ref_width, last_pos.height])
            else: 
                ref_height = last_pos.height * 0.5
                ref_bottom = last_pos.y0 - ref_height - 0.05
                ref_width = last_pos.width * 0.8
                ref_left = last_pos.x0 + (last_pos.width - ref_width) / 2  # Center it
                ref_ax = fig.add_axes([ref_left, ref_bottom, ref_width, ref_height])
            
            ref_ax.set_xlim(-1.5, 1.5)
            ref_ax.set_ylim(-1.5, 1.5)
            ref_ax.set_aspect('equal')
            ref_ax.axis('off')
            u_norm = self.u_ref / np.linalg.norm(self.u_ref)
            v_norm = self.v_ref / np.linalg.norm(self.v_ref)
            
            u_row, u_col = u_norm
            v_row, v_col = v_norm
            arrow_props_ref = dict(arrowstyle='->', lw=3, mutation_scale=25)
            
            u_arrow = FancyArrowPatch(
                (0, 0), (u_col, -u_row),
                color='darkred', **arrow_props_ref
            )
            ref_ax.add_patch(u_arrow)
            
            v_arrow = FancyArrowPatch(
                (0, 0), (v_col, -v_row),
                color='darkblue', **arrow_props_ref
            )
            ref_ax.add_patch(v_arrow)
            ref_ax.text(u_col * 1.3, -u_row * 1.3, r'$\mathbf{g}_{1}$',
                    fontsize=14, fontweight='bold', color='darkred',
                    ha='center', va='center')
            
            ref_ax.text(v_col * 1.3, -v_row * 1.3, r'$\mathbf{g}_{2}$',
                    fontsize=14, fontweight='bold', color='darkblue',
                    ha='center', va='center')
        

        return fig, ax

