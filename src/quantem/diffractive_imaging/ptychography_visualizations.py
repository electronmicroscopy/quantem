import warnings
from typing import Any, Literal, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal.windows import tukey

from quantem.core import config
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.ptycho_utils import AffineTransform
from quantem.diffractive_imaging.ptychography_base import PtychographyBase, Snapshot


class PtychographyVisualizations(PtychographyBase):
    def _scalebar_real(self) -> dict[str, Any]:
        return {"sampling": self.sampling[1], "units": "Å"}

    def _scalebar_recip(self) -> dict[str, Any]:
        return {"sampling": self.reciprocal_sampling[1], "units": r"$\mathrm{A^{-1}}$"}

    def show_obj(
        self,
        obj: np.ndarray | None = None,
        snapshot_iter: int | None = None,
        slice_index: int | None = None,
        cbar: bool = False,
        **kwargs,
    ):
        """
        Show the projected object.

        Parameters
        ----------
        obj: np.ndarray | None, optional
            The object to show. If None, the summed object from the last iteration is shown.
        snapshot_iter: int | None, optional
            The iteration number to show. If None, the last iteration is shown.
        slice_index: int | None, optional
            The slice index to show. If None, the object is shown as a 2D image.
        cbar: bool, optional
            Whether to show a colorbar, by default False
        **kwargs: dict, optional
            Additional arguments passed to show_2d
        """
        obj_iter = "Final"
        if obj is None:
            if snapshot_iter is not None:
                snp = self.get_snapshot_by_iter(snapshot_iter, closest=True, cropped=True)
                obj_np = snp["obj"]
                obj_iter = snp["iteration"]
            else:
                obj_np = self.obj_cropped
        else:
            obj_np = self._to_numpy(obj)
            if obj_np.ndim == 2:
                obj_np = obj_np[None, ...]

        ph_cmap = kwargs.pop("cmap", config.get("viz.phase_cmap"))

        t = kwargs.pop("title", "")
        if len(t) > 0 and not t.endswith(" "):
            t += " "
        if obj_iter != "Final":
            t += f"Iter {obj_iter} "

        if slice_index is not None:
            obj_np = obj_np[slice_index][None]
            t += f"Slice {slice_index} "
        elif obj_np.shape[0] > 1:
            t += "Projected "

        ims = []
        titles = []
        cmaps = []
        # obj_np dtype depends on obj_type:
        #   potential  -> real values to plot directly
        #   pure_phase -> real phase to plot directly (no np.angle wrap)
        #   complex    -> complex; extract amp & phase via np.abs / np.angle
        if self.obj_type == "potential":
            ims.append(np.abs(obj_np).sum(0))
            titles.append(t + "Potential")
            cmaps.append(ph_cmap)
        elif self.obj_type == "pure_phase":
            ims.append(obj_np.sum(0))
            titles.append(t + "Pure Phase")
            cmaps.append(ph_cmap)
        else:
            ims.extend([np.angle(obj_np).sum(0), np.abs(obj_np).sum(0)])
            titles.extend([t + "Phase", t + "Amplitude"])
            cmaps.extend([ph_cmap, "gray"])

        scalebar = [self._scalebar_real()] + [None] * (len(ims) - 1)

        show_2d(
            ims,
            title=titles,
            cmap=cmaps,
            cbar=cbar,
            scalebar=scalebar,
            **kwargs,
        )

    def show_obj_fft(
        self,
        obj: np.ndarray | None = None,
        tukey_alpha: float = 0.5,
        pad: int = 0,
        show_obj: bool = False,
        snapshot_iter: int | None = None,
        return_fft: bool = False,
        **kwargs,
    ):
        """
        Show the Fourier transform of the object.
        Parameters
        ----------
        obj: np.ndarray | None, optional
            The object to show. If None, the object from the last iteration is shown.
        tukey_alpha: float, optional
            The alpha parameter for the tukey window, by default 0.5
        pad: int, optional
            The padding to add to the object, by default 0
        show_obj: bool, optional
            Whether to show the object, by default False
        snapshot_iter: int | None, optional
            The iteration number of the snapshot to show. If None, the last iteration is shown.
        return_fft: bool, optional
            Whether to return the Fourier transform, by default False
        **kwargs: dict, optional
            Additional arguments passed to show_2d

        Returns
        -------
        np.ndarray | None
            The Fourier transform of the object if return_fft is True, otherwise None
        """
        obj_iter = "Final"
        if obj is None:
            if snapshot_iter is not None:
                snp = self.get_snapshot_by_iter(snapshot_iter, closest=True, cropped=True)
                obj_np = snp["obj"]
                obj_iter = snp["iteration"]
            else:
                obj_np = self.obj_cropped
        else:
            obj_np = self._to_numpy(obj)
            if obj_np.ndim == 2:
                obj_np = obj_np[None, ...]

        window_2d = (
            tukey(obj_np.shape[-2], tukey_alpha)[:, None]
            * tukey(obj_np.shape[-1], tukey_alpha)[None, :]
        )
        # Build the complex transmission function and apply the spatial window:
        #   potential  -> real values, plotted in real space
        #   pure_phase -> real phase; transmission = exp(1j*phase)
        #   complex    -> already complex transmission
        if self.obj_type == "potential":
            windowed_obj = obj_np.sum(0) * window_2d
        elif self.obj_type == "pure_phase":
            windowed_obj = np.exp(1j * obj_np.sum(0)) * window_2d
        else:
            windowed_obj = np.abs(obj_np).sum(0) * np.exp(1j * np.angle(obj_np).sum(0)) * window_2d
        obj_pad = np.pad(windowed_obj, pad, mode="constant", constant_values=0)

        obj_fft = np.fft.fftshift(np.fft.fft2(obj_pad))

        fft_sampling = 1 / (self.sampling[1] * obj_pad.shape[1])
        fft_scalebar = {"sampling": fft_sampling, "units": r"$\mathrm{A^{-1}}$"}

        t = kwargs.pop("title", "")
        if len(t) > 0 and not t.endswith(" "):
            t += " "
        if obj_iter != "Final":
            t += f"Iter {obj_iter} "

        if show_obj:
            obj_scalebar = self._scalebar_real()
            if self.obj_type == "potential":
                obj_show = obj_pad
            else:  # complex or pure phase just show the phase
                obj_show = np.angle(obj_pad)
            fig, ax = show_2d(
                [
                    obj_show,
                    np.abs(obj_fft),
                ],
                title=[t + "Object", t + "Fourier Transform"],
                scalebar=[obj_scalebar, fft_scalebar],
                **kwargs,
            )
            ax[1].set_aspect(obj_pad.shape[-1] / obj_pad.shape[-2])
        else:
            fig, ax = show_2d(
                np.abs(obj_fft),
                scalebar=fft_scalebar,
                title=t + "Fourier Transform",
                **kwargs,
            )
            ax.set_aspect(obj_pad.shape[-1] / obj_pad.shape[-2])
        if return_fft:
            return obj_fft
        else:
            return

    def show_probe(
        self,
        probe: np.ndarray | None = None,
        snapshot_iter: int | None = None,
        sum_probes: bool = False,
        **kwargs,
    ):
        """
        Show the probe, each probe mode is shown separately.
        Parameters
        ----------
        probe: np.ndarray | None, optional
            The probe to show. If None, the probe from the last iteration is shown.
        snapshot_iter: int | None, optional
            The index of the snapshot to show. If None, the last iteration is shown.
        sum_probes: bool, optional
            Whether to sum the probes, by default False
        **kwargs: dict, optional
            Additional arguments passed to show_2d

        Returns
        -------
        None
            The probe is shown in a new figure
        """
        probe_iter = "Final"
        if probe is None:
            if snapshot_iter is not None:
                snp = self.get_snapshot_by_iter(snapshot_iter, closest=True, cropped=True)
                probe = snp["probe"]
                probe_iter = snp["iteration"]
            else:
                probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        t = kwargs.pop("title", "")
        if len(t) > 0 and not t.endswith(" "):
            t += " "
        if sum_probes and self.num_probes > 1:
            t += "Summed "
        if probe_iter != "Final":
            t += f"Iter {probe_iter} "

        scalebar = [self._scalebar_real()]
        if sum_probes:
            probes = [np.fft.fftshift(probe.sum(0))]
        else:
            probes = [np.fft.fftshift(probe[i]) for i in range(len(probe))]
            scalebar += [None] * (len(probes) - 1)

        if len(probes) > 1:
            prb_intensities = self.get_probe_intensities(probe)
            titles = [
                t + f"Probe {i + 1}/{len(prb_intensities)}: {prb_intensity * 100:.1f}%"
                for i, prb_intensity in enumerate(prb_intensities)
            ]
        else:
            titles = [t + "Probe"]

        show_2d(probes, title=titles, scalebar=scalebar, **kwargs)

    def show_probe_top_bottom(
        self,
        probe: np.ndarray | None = None,
        snapshot_iter: int | None = None,
        probe_index: int | None = 0,
        quantity: Literal["intensity", "amplitude", "phase"] = "intensity",
        fftshift: bool = True,
        return_arrays: bool = False,
        **kwargs,
    ):
        """
        Show the real-space probe at the top and bottom object surfaces.

        The bottom surface probe is computed by propagating the reconstructed
        probe through the multislice free-space propagators only. The reconstructed
        object is not applied.

        Parameters
        ----------
        probe : np.ndarray | None, optional
            Probe array to show. If None, the current reconstructed probe is used.
        snapshot_iter : int | None, optional
            Snapshot iteration to show. If None, the final/current probe is shown.
        probe_index : int | None, optional
            Probe mode index to show. If None, mixed-state modes are summed
            incoherently as intensity.
        quantity : {"intensity", "amplitude", "phase"}, optional
            Quantity to display for a single probe mode, by default "intensity".
        fftshift : bool, optional
            Whether to center the real-space probe for display.
        return_arrays : bool, optional
            If True, return the displayed top and bottom arrays in addition to
            the figure and axes.
        **kwargs
            Additional arguments passed to show_2d.

        Returns
        -------
        tuple
            ``(fig, axs)`` by default, or ``(fig, axs, arrays)`` if
            return_arrays is True.
        """
        if probe is None:
            if snapshot_iter is not None:
                snp = self.get_snapshot_by_iter(snapshot_iter, closest=True, cropped=True)
                probe = snp["probe"]
            else:
                probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        self.compute_propagator_arrays()

        top = probe.copy()
        bottom = top.copy()
        for prop in self._to_numpy(self.propagators):
            bottom = np.fft.ifft2(np.fft.fft2(bottom) * prop)

        if probe_index is None:
            if quantity != "intensity":
                raise ValueError("probe_index=None is only supported for quantity='intensity'")
            top_img = np.sum(np.abs(top) ** 2, axis=0)
            bottom_img = np.sum(np.abs(bottom) ** 2, axis=0)
            label = "Summed Probe Intensity"
            cmap = kwargs.pop("cmap", "magma")
        else:
            if probe_index < 0 or probe_index >= top.shape[0]:
                raise ValueError(
                    f"probe_index must be between 0 and {top.shape[0] - 1}, got {probe_index}"
                )
            top_probe = top[probe_index]
            bottom_probe = bottom[probe_index]

            if quantity == "intensity":
                top_img = np.abs(top_probe) ** 2
                bottom_img = np.abs(bottom_probe) ** 2
                cmap = kwargs.pop("cmap", "magma")
            elif quantity == "amplitude":
                top_img = np.abs(top_probe)
                bottom_img = np.abs(bottom_probe)
                cmap = kwargs.pop("cmap", "gray")
            elif quantity == "phase":
                top_img = np.angle(top_probe)
                bottom_img = np.angle(bottom_probe)
                cmap = kwargs.pop("cmap", config.get("viz.phase_cmap"))
            else:
                raise ValueError(
                    f"quantity must be 'intensity', 'amplitude', or 'phase', got {quantity}"
                )
            label = f"Probe {probe_index + 1} {quantity.capitalize()}"

        if fftshift:
            top_img = np.fft.fftshift(top_img)
            bottom_img = np.fft.fftshift(bottom_img)

        scalebar = [self._scalebar_real(), None]
        titles = [f"Top Surface {label}", f"Bottom Surface {label}"]

        fig, axs = show_2d(
            [top_img, bottom_img],
            title=titles,
            cmap=cmap,
            scalebar=scalebar,
            **kwargs,
        )
        if return_arrays:
            return fig, axs, {"top": top_img, "bottom": bottom_img}
        return fig, axs

    def show_fourier_probe(self, probe: np.ndarray | None = None, **kwargs):
        """
        Show the Fourier transform of the probe.
        Parameters
        ----------
        probe: np.ndarray | None, optional
            The probe to show. If None, the probe from the last iteration is shown.
        **kwargs: dict, optional
            Additional arguments passed to show_2d

        Returns
        -------
        None
            The Fourier transform of the probe is shown in a new figure
        """
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        probes = [np.fft.fftshift(np.fft.fft2(probe[i])) for i in range(len(probe))]
        scalebar = [self._scalebar_recip()] + [None] * (len(probes) - 1)
        if len(probes) > 1:
            titles = self.get_probe_intensities(probe)
            titles = [
                f"Fourier Probe {i + 1}/{len(titles)}: {t * 100:.1f}%"
                for i, t in enumerate(titles)
            ]
        else:
            titles = "Fourier Probe"
        show_2d(probes, title=titles, scalebar=scalebar, **kwargs)

    def show_obj_and_probe(
        self,
        cbar: bool = False,
        snapshot_iter: int | None = None,
        sum_probes: bool = False,
        figax: tuple | None = None,
        **kwargs,
    ):
        """
        Shows the projected object and the probe in one figure. Calls show_obj and show_probe.
        Parameters
        ----------
        cbar: bool, optional
            Whether to show a colorbar, by default False
        snapshot_iter: int | None, optional
            The iteration number of the snapshot to show. If None, the last iteration is shown.
        **kwargs: dict, optional
            Additional arguments passed to show_obj and show_probe

        Returns
        -------
        None
            The object and probe are shown in a new figure
        """
        axsize = kwargs.pop("axsize", (5, 5))
        if sum_probes:
            num_probes = 1
        else:
            num_probes = self.num_probes
        ncols = 2 + num_probes if self.obj_type == "complex" else 1 + num_probes
        if figax is None:
            fig, axs = plt.subplots(
                ncols=ncols, figsize=kwargs.pop("figsize", (axsize[0] * ncols, axsize[1]))
            )
        else:
            fig, axs = figax
        self.show_obj(
            figax=(fig, axs[:-num_probes]), cbar=cbar, snapshot_iter=snapshot_iter, **kwargs
        )
        self.show_probe(
            figax=(fig, axs[-num_probes:]),
            cbar=False,
            snapshot_iter=snapshot_iter,
            sum_probes=sum_probes,
            **kwargs,
        )
        if figax is None:
            plt.tight_layout()
            plt.show()

    def show_obj_slices(
        self,
        obj: np.ndarray | None = None,
        cbar: bool = False,
        interval_type: Literal["quantile", "manual", "minmax", "abs"] = "quantile",
        interval_scaling: Literal["each", "all"] = "each",
        max_width: int = 4,
        return_fig: bool = False,
        **kwargs,
    ):
        """
        Show the object slices.
        Parameters
        ----------
        obj: np.ndarray | None, optional
            The object to show. If None, the object from the last iteration is shown.
        cbar: bool, optional
            Whether to show a colorbar, by default False
        interval_type: Literal["quantile", "manual", "minmax", "abs"], optional
            The interval type to use for the colorbar, by default "quantile"
        interval_scaling: Literal["each", "all"], optional
            The interval scaling to use for the colorbar, by default "each"
        max_width: int, optional
            The maximum width of the object slices, by default 4
        return_fig: bool, optional
            If True, return ``(fig, axs)`` for saving or further customization.
        **kwargs: dict, optional
            Additional arguments passed to show_2d

        Returns
        -------
        tuple | None
            ``(fig, axs)`` if return_fig is True, otherwise None.
        """
        if obj is None:
            obj = self.obj_cropped
        else:
            obj = self._to_numpy(obj)
            if obj.ndim == 2:
                obj = obj[None, ...]

        # slice_thicknesses has length num_slices - 1; entry i is the gap to the next slice
        thicknesses = self.slice_thicknesses
        t_parts = []
        for i in range(len(obj)):
            t_part = f"{i + 1}/{len(obj)}"
            if i < len(thicknesses):
                t_part += f" | {thicknesses[i]:.1f} Å"
            t_parts.append(t_part)

        if self.obj_type == "potential":
            objs_flat = [np.abs(obj[i]) for i in range(len(obj))]
            titles_flat = [f"Potential {t_parts[i]}" for i in range(len(obj))]
        elif self.obj_type == "pure_phase":
            objs_flat = [obj[i] for i in range(len(obj))]
            titles_flat = [f"Pure Phase {t_parts[i]}" for i in range(len(obj))]
        else:
            objs_flat = [np.angle(obj[i]) for i in range(len(obj))]
            titles_flat = [f"Phase {t_parts[i]}" for i in range(len(obj))]

        # Nest lists with max length max_width
        objs = [objs_flat[i : i + max_width] for i in range(0, len(objs_flat), max_width)]
        titles = [titles_flat[i : i + max_width] for i in range(0, len(titles_flat), max_width)]

        scalebars: list = [[None for _ in row] for row in objs]
        scalebars[0][0] = self._scalebar_real()

        if interval_type == "quantile":
            norm = {"interval_type": "quantile"}
            # TODO -- make this work with interval_scaling
            if interval_scaling == "all":
                warnings.warn(
                    "interval_scaling='all' is not yet supported for quantile normalization"
                )
        elif interval_type in ["manual", "minmax", "abs"]:
            norm: dict[str, Any] = {"interval_type": "manual"}
            if interval_scaling == "all":
                norm["vmin"] = np.min(objs_flat)
                norm["vmax"] = np.max(objs_flat)
            else:
                norm["vmin"] = kwargs.pop("vmin", None)
                norm["vmax"] = kwargs.pop("vmax", None)
        else:
            raise ValueError(f"Unknown interval type: {interval_type}")

        fig, axs = show_2d(
            objs,
            title=titles,
            cmap=kwargs.pop("cmap", config.get("viz.phase_cmap")),
            norm=norm,
            cbar=cbar,
            scalebar=scalebars,
            **kwargs,
        )
        if return_fig:
            return fig, axs
        return None

    def plot_losses(self, figax: tuple | None = None, plot_lrs: bool = True):
        """
        Plot the losses and learning rates.
        Parameters
        ----------
        figax: tuple | None, optional
            The figure and axes to plot on. If None, a new figure is created.
        plot_lrs: bool, optional
            Whether to plot the learning rates, by default True

        Returns
        -------
        None
        """

        if figax is None:
            _fig, ax = plt.subplots()
        else:
            _fig, ax = figax

        lw = 2
        lines = []
        iters = np.arange(len(self.iter_losses))
        # colors = plt.cm.Set1.colors  # type:ignore
        colors = config.get("viz.colors.set")  # [1:]
        if len(self.val_iter_losses) > 0:
            lines.extend(ax.semilogy(iters, self.iter_losses, c="k", label="train loss", lw=lw))
            lines.extend(
                ax.semilogy(
                    np.arange(len(self.val_iter_losses)),
                    self.val_iter_losses,
                    c=colors[6],
                    label="val loss",
                    lw=lw,
                )
            )
        else:
            lines.extend(ax.semilogy(iters, self.iter_losses, c="k", label="loss", lw=lw))

        ax.set_ylabel("Loss", color="k")
        ax.tick_params(axis="y", which="both", colors="k")
        ax.spines["left"].set_color("k")
        ax.set_xlabel("Iterations")

        # check if all lrs are constant and if so, don't plot lr
        if all(np.all(lr == lr[0]) for lr in self.iter_lrs.values() if len(lr) > 0):
            plot_lrs = False

        if plot_lrs and len(self.iter_lrs) > 0:
            nx = ax.twinx()
            ax.set_zorder(2)  # putting the twinx behind
            nx.set_zorder(1)
            ax.patch.set_visible(False)
            nx.spines["left"].set_visible(False)
            color_idx = 0

            # Sort optimizers: object first, then probe, then the rest
            sorted_items = sorted(
                self.iter_lrs.items(),
                key=lambda x: (0 if x[0] == "object" else 1 if x[0] == "probe" else 2, x[0]),
            )
            for lr_type, lr_values in sorted_items:
                if len(lr_values) > 0:
                    # Create iterations array that matches lr_values length
                    lr_iters = np.arange(len(lr_values))
                    linestyles = ["-", "--", ":", "-."]
                    linestyle = linestyles[color_idx % len(linestyles)]
                    if lr_type == "probe":
                        zorder = 3
                    elif lr_type == "object":
                        zorder = 2
                    else:
                        zorder = 1
                    lines.extend(
                        nx.semilogy(
                            lr_iters,
                            lr_values,
                            c=colors[color_idx % len(colors)],
                            label=f"{lr_type} LR",
                            linestyle=linestyle,
                            zorder=zorder,
                            lw=lw,
                        )
                    )
                    color_idx += 1

            nx.set_ylabel("LRs", c=colors[0])
            nx.spines["right"].set_color(colors[0])
            nx.tick_params(axis="y", which="both", colors=colors[0])

        else:
            # No learning rates to plot, add to title
            # set title to each lr type
            title = ""
            for lr_type, lr_values in self.iter_lrs.items():
                if len(lr_values) > 0:
                    title += f"{lr_type} LR: {lr_values[0]:.1e} | "
            ax.set_title(title[:-3], fontsize=10)

        labs = [lin.get_label() for lin in lines]
        if len(labs) > 1:
            ax.legend(lines, labs, loc="upper right")

        ax.set_xbound(-2, np.max(iters if np.any(iters) else [1]) + 2)
        if figax is None:
            plt.tight_layout()
            plt.show()

    def visualize(self, cbar: bool = True, return_fig: bool = False):
        """
        Plot losses and show object and probe.
        Parameters
        ----------
        cbar: bool, optional
            Whether to show a colorbar, by default True
        return_fig: bool, optional
            If True, return ``(fig, axs)`` instead of calling ``plt.show()``.

        Returns
        -------
        tuple | None
            ``(fig, axs)`` if return_fig is True, otherwise None.
        """
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        self.plot_losses(figax=(fig, ax_top))

        n_bot = 3 if self.obj_type == "complex" else 2
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_bot, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)])
        self.show_obj_and_probe(figax=(fig, axs_bot), cbar=cbar, sum_probes=True)
        plt.suptitle(
            f"Final loss: {self.iter_losses[-1]:.3e} | Iters: {len(self.iter_losses)}",
            fontsize=14,
            y=0.95,
        )
        if return_fig:
            return fig, (ax_top, axs_bot)
        plt.show()
        return None

    def show_iters(
        self,
        show_probe: bool = True,
        show_object: bool = True,
        iters: list[int] | slice | None = None,
        every_nth: int | None = None,
        max_n: int | None = None,
        cbar: bool = False,
        norm: Literal["quantile", "manual", "minmax", "abs"] = "quantile",
        interval_scaling: Literal["each", "all"] = "each",
        max_width: int = 4,
        cropped: bool = True,
        closest: bool = True,
        **kwargs,
    ):
        """
        Display object and/or probe reconstructions from stored iteration snapshots.

        Parameters
        ----------
        show_probe : bool, optional
            Whether to show probe reconstructions, by default True
        show_object : bool, optional
            Whether to show object reconstructions, by default True
        iters : list[int] | slice | None, optional
            Specific iter iterations to display. If None, shows all available iters.
            Takes precedence over every_nth.
        every_nth : int | None, optional
            Show every nth iter instead of all. Only used if iters is None
        max_n : int | None, optional
            Maximum number of iterations to display
        cbar : bool, optional
            Whether to show colorbars, by default False
        norm : str, optional
            Normalization method for object display, by default "quantile". Ignored if
            show_probe and show_object are both True.
        interval_scaling : str, optional
            How to scale intervals: "each" for per-image scaling, "all" for global scaling across
            all iterations, by default "each". Ignored if show_probe and show_object are both
            True.
        max_width : int, optional
            Maximum number of images per row, by default 4
        cropped : bool, optional
            Whether to show cropped objects (default True) or full objects
        closest : bool, optional
            Whether to fall back to the closest stored snapshot when an exact iteration is
            not available, by default True
        **kwargs
            Additional arguments passed to show_2d
        """
        if not self.snapshots:
            print(
                "No iteration snapshots available. Use store_snapshots=True during reconstruction."
            )
            return

        if not show_object and not show_probe:
            print("Must show at least one of object or probe")
            return

        all_iterations = [snapshot["iteration"] for snapshot in self.snapshots]

        if iters is not None:
            if isinstance(iters, slice):
                selected_iterations = all_iterations[iters]
            else:
                selected_iterations = iters
        elif every_nth is not None:
            selected_iterations = all_iterations[::every_nth]
        else:
            selected_iterations = all_iterations

        if max_n is not None:
            selected_iterations = selected_iterations[:max_n]  # truncate to max_n

        if len(selected_iterations) == 0:
            print("No valid iterations selected for display")
            return

        selected_snapshots = [
            self.get_snapshot_by_iter(i, closest=closest, cropped=cropped)
            for i in selected_iterations
        ]

        if show_object and show_probe:
            if norm != "quantile" or interval_scaling != "each":
                warnings.warn(
                    "norm and interval_scaling are ignored when both the object and probe "
                    "are shown; pass show_probe=False to use them.",
                    stacklevel=2,
                )
            self._show_object_and_probe_iters(selected_snapshots, cbar, max_width, **kwargs)
        elif show_object:
            self._show_object_iters_only(
                selected_snapshots, norm, interval_scaling, cbar, max_width, **kwargs
            )
        elif show_probe:
            self._show_probe_iters_only(selected_snapshots, cbar, max_width, **kwargs)

    def _show_object_iters_only(
        self,
        snapshots: list[Snapshot],
        norm: Literal["quantile", "manual", "minmax", "abs"],
        interval_scaling: Literal["each", "all"],
        cbar: bool,
        max_width: int,
        **kwargs,
    ):
        """Display only object reconstructions from iteration snapshots."""
        ph_cmap = kwargs.pop("cmap", config.get("viz.phase_cmap"))

        all_images = []
        all_titles = []
        all_cmaps = []

        for snapshot in snapshots:
            obj = snapshot["obj"]
            iteration = snapshot["iteration"]
            title_prefix = f"Iter {iteration} "

            if self.obj_type == "potential":
                all_images.append(np.abs(obj).sum(0))
                all_titles.append(title_prefix + "Potential")
                all_cmaps.append(ph_cmap)
            elif self.obj_type == "pure_phase":
                all_images.append(obj.sum(0))
                all_titles.append(title_prefix + "Phase")
                all_cmaps.append(ph_cmap)
            else:  # complex
                all_images.extend([np.angle(obj).sum(0), np.abs(obj).sum(0)])
                all_titles.extend([title_prefix + "Phase", title_prefix + "Amplitude"])
                all_cmaps.extend([ph_cmap, "gray"])

        if norm == "quantile":
            norm_dict = {"interval_type": "quantile"}
            # TODO: implement global quantile scaling for interval_scaling="all"
        elif norm in ["manual", "minmax", "abs"]:
            norm_dict: dict[str, Any] = {"interval_type": "manual"}
            # Calculate global vmin/vmax if interval_scaling="all" and objects are shown
            if interval_scaling == "all":
                if len(all_images) > 0:
                    all_values_flat = np.concatenate([arr.ravel() for arr in all_images])
                    norm_dict["vmin"] = float(np.min(all_values_flat))
                    norm_dict["vmax"] = float(np.max(all_values_flat))
            else:
                norm_dict["vmin"] = kwargs.pop("vmin", None)
                norm_dict["vmax"] = kwargs.pop("vmax", None)
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        images_grid = [all_images[i : i + max_width] for i in range(0, len(all_images), max_width)]
        titles_grid = [all_titles[i : i + max_width] for i in range(0, len(all_titles), max_width)]
        cmaps_grid = [all_cmaps[i : i + max_width] for i in range(0, len(all_cmaps), max_width)]

        scalebars: list = [[None for _ in row] for row in images_grid]
        if scalebars:
            scalebars[0][0] = self._scalebar_real()

        show_2d(
            images_grid,
            title=titles_grid,
            cmap=cmaps_grid,
            norm=norm_dict,
            cbar=cbar,
            scalebar=scalebars,
            **kwargs,
        )

    def _show_probe_iters_only(
        self, snapshots: list[Snapshot], cbar: bool, max_width: int, **kwargs
    ):
        """Display only probe reconstructions from iteration snapshots."""
        all_probes = []
        all_titles = []

        for snapshot in snapshots:
            probe = snapshot["probe"]
            iteration = snapshot["iteration"]

            if probe.ndim == 3 and probe.shape[0] > 1:
                probe_display = np.fft.fftshift(probe.sum(0))
                title = f"Iter {iteration} Coherently summed Probe"
            else:
                probe_display = np.fft.fftshift(probe[0] if probe.ndim == 3 else probe)
                title = f"Iter {iteration} Probe"

            all_probes.append(probe_display)
            all_titles.append(title)

        probes_grid = [all_probes[i : i + max_width] for i in range(0, len(all_probes), max_width)]
        titles_grid = [all_titles[i : i + max_width] for i in range(0, len(all_titles), max_width)]

        # Set up scalebars
        scalebars: list = [[None for _ in row] for row in probes_grid]
        if scalebars:
            scalebars[0][0] = self._scalebar_real()

        show_2d(
            probes_grid,
            title=titles_grid,
            scalebar=scalebars,
            cbar=cbar,
            **kwargs,
        )

    def _show_object_and_probe_iters(
        self,
        snapshots: list[Snapshot],
        cbar: bool,
        max_width: int,
        **kwargs,
    ):
        """Display both object and probe reconstructions from iteration snapshots."""
        ph_cmap = kwargs.pop("cmap", config.get("viz.phase_cmap"))

        all_images = []
        all_titles = []
        all_cmaps = []

        for i in range(0, len(snapshots), max_width // 2):
            row_images = []
            row_titles = []
            row_cmaps = []

            for j in range(max_width // 2):
                if i + j >= len(snapshots):
                    continue
                snapshot = snapshots[i + j]
                obj = snapshot["obj"]
                probe = snapshot["probe"]
                iteration = snapshot["iteration"]

                if self.obj_type == "potential":
                    row_images.append(np.abs(obj).sum(0))
                    row_titles.append(f"Iter {iteration} Potential")
                    row_cmaps.append(ph_cmap)
                elif self.obj_type == "pure_phase":
                    row_images.append(obj.sum(0))
                    row_titles.append(f"Iter {iteration} Phase")
                    row_cmaps.append(ph_cmap)
                else:  # complex
                    row_images.extend([np.angle(obj).sum(0), np.abs(obj).sum(0)])
                    row_titles.extend(
                        [
                            f"Iter {iteration} Phase",
                            f"Iter {iteration} Amplitude",
                        ]
                    )
                    row_cmaps.extend([ph_cmap, "gray"])

                # Process probe
                if probe.ndim == 3 and probe.shape[0] > 1:
                    probe_display = np.fft.fftshift(probe.sum(0))
                    probe_title = f"Iter {iteration} Summed Probe"
                else:
                    probe_display = np.fft.fftshift(probe[0] if probe.ndim == 3 else probe)
                    probe_title = f"Iter {iteration} Probe"

                row_images.append(probe_display)
                row_titles.append(probe_title)
                row_cmaps.append(None)

            all_images.append(row_images)
            all_titles.append(row_titles)
            all_cmaps.append(row_cmaps)

        scalebars: list = [[None for _ in row] for row in all_images]
        if scalebars:
            scalebars[0][0] = self._scalebar_real()
            scalebars[0][1] = self._scalebar_real()

        show_2d(
            all_images,
            title=all_titles,
            cmap=all_cmaps,
            cbar=cbar,
            scalebar=scalebars,
            **kwargs,
        )

    def show_scan_positions(
        self,
        plot_radii: bool = True,
        num_probes: int | None = None,
        axsize: tuple[int, int] = (10, 10),
        edgecolors: str = "red",
        linewidths: float = 0.5,
        **kwargs,
    ):
        r"""Show scan positions, probe radii, and overlap metadata

        Visualize probe coverage to verify sufficient overlap for
        ptychographic reconstruction. The probe radius is estimated as
        :math:`r = 0.61 \lambda / \alpha + |\Delta f| \cdot \alpha`
        where :math:`\lambda` is the electron wavelength, :math:`\alpha` is
        the convergence semi-angle, and :math:`\Delta f` is the defocus.
        Overlap is :math:`(d - s)/d = 1 - s/d` where :math:`d = 2r` is
        the probe diameter and :math:`s` is the scan step size.

        Parameters
        ----------
        plot_radii : bool, optional
            Whether to plot the probe radius. Default is True
        num_probes : int | None, optional
            Number of probe positions to display. Default is 3 rows
            of scan positions
        axsize : tuple[int, int], optional
            Size of the figure axes. Default is (10, 10)
        edgecolors : str, optional
            Edge color for the probe circles. Default is "red"
        linewidths : float, optional
            Line width of the probe circles. Default is 0.5
        **kwargs
            Additional keyword arguments passed to ``matplotlib.axes.Axes.scatter``

        Examples
        --------
        >>> ptycho.show_scan_positions()
        Scan grid: 192 x 192 positions
        FOV: 133.46 x 133.46 Å
        Probe diameter (rough estimate): 3.46 Å
        Step size: 0.70 Å
        Probe overlap (rough estimate): 79.8%

        >>> ptycho.show_scan_positions(num_probes=10, edgecolors="b", linewidths=0.5)
        """
        # for each scan position, sum the intensity of self.probe at that position
        scan_positions = self.dset.scan_positions_px.cpu().detach().numpy()
        probe_params = self.probe_model.probe_params
        probe_radius_px = None
        conv_angle = probe_params.get("semiangle_cutoff")
        defocus = probe_params.get("defocus", 0)
        energy = probe_params.get("energy")
        scan_gpts = self.dset.gpts
        scan_sampling = self.dset.scan_sampling
        fov = scan_sampling * (np.array(scan_gpts) - 1)
        print(f"Scan grid: {scan_gpts[0]} x {scan_gpts[1]} positions")
        print(f"Object FOV: {fov[0]:.2f} x {fov[1]:.2f} Å")

        if conv_angle is not None and energy is not None:
            from quantem.core.utils.utils import electron_wavelength_angstrom

            wavelength = electron_wavelength_angstrom(energy)
            conv_angle_rad = conv_angle * 1e-3
            # For defocused probe: radius ≈ |defocus| * convergence_angle + diffraction_limit
            diffraction_limit_A = 0.61 * wavelength / conv_angle_rad
            defocus_blur_A = abs(defocus) * conv_angle_rad
            probe_radius_A = diffraction_limit_A + defocus_blur_A
            probe_radius_px = probe_radius_A / self.sampling[0]
            probe_diameter_A = 2 * probe_radius_A
            probe_step_size_A = scan_sampling[0]
            overlap = max(0.0, 1 - probe_step_size_A / probe_diameter_A)
            print(f"Probe diameter (rough estimate): {probe_diameter_A:.2f} Å")
            print(f"Step size: {probe_step_size_A:.2f} Å")
            print(f"Probe overlap (rough estimate): {overlap * 100:.1f}%")

        _fig, ax = show_2d(self._get_probe_overlap(), title="probe overlap", axsize=axsize)
        if probe_radius_px is not None and plot_radii:
            # plot a circle with the probe radius for each probe position
            if num_probes is None:
                num_probes = 3 * scan_gpts[1]  # 3 rows worth
            ax.scatter(
                scan_positions[:num_probes, 1],
                scan_positions[:num_probes, 0],
                s=probe_radius_px**2,
                facecolors="none",
                edgecolors=edgecolors,
                linewidths=linewidths,
                **kwargs,
            )
        plt.show()

    def show_updated_scan_positions(
        self,
        scan_positions_px: np.ndarray | torch.Tensor | None = None,
        initial_scan_positions_px: np.ndarray | torch.Tensor | None = None,
        scale_arrows: float = 1.0,
        plot_arrow_freq: int | None = None,
        plot_cropped_rotated_fov: bool = True,
        cbar: bool = True,
        verbose: bool = True,
        return_fig: bool = False,
        **kwargs,
    ):
        r"""Show changes to scan positions during ptychographic reconstruction.

        The default plot compares the current learned scan positions against the
        initial scan positions from preprocessing. Stored positions are in object
        pixels; this visualization converts them to Å for the axis labels and
        colorbar.

        Parameters
        ----------
        scan_positions_px : np.ndarray | None, optional
            Updated scan positions in object pixels. If None, uses
            ``self.dset.scan_positions_px``.
        initial_scan_positions_px : np.ndarray | None, optional
            Reference scan positions in object pixels. If None, uses
            ``self.dset.initial_scan_positions_px``.
        scale_arrows : float, optional
            Scaling factor applied to displacement vectors before plotting.
            Default is 1.
        plot_arrow_freq : int | None, optional
            If provided, plot every Nth row and column of a raster scan grid.
        plot_cropped_rotated_fov : bool, optional
            If True, plot positions in the same cropped/rotated FOV convention as
            ``show_obj``. Default is True.
        cbar : bool, optional
            Whether to show a colorbar for displacement magnitudes. Default is True.
        verbose : bool, optional
            Whether to print summary displacement statistics. Default is True.
        return_fig : bool, optional
            If True, return ``(fig, ax)`` instead of calling ``plt.show()``.
        **kwargs
            Additional keyword arguments passed to ``matplotlib.axes.Axes.quiver``.
            ``figsize``, ``figax``, ``cmap``, and ``title`` are consumed by this
            method.
        """

        if scan_positions_px is None:
            scan_positions_px = self.dset.scan_positions_px
        if initial_scan_positions_px is None:
            initial_scan_positions_px = self.dset.initial_scan_positions_px

        scan_positions_px = self._to_numpy(scan_positions_px).astype(float, copy=False)
        initial_scan_positions_px = self._to_numpy(initial_scan_positions_px).astype(
            float, copy=False
        )

        if scan_positions_px.ndim == 3:
            scan_positions_px = cast(np.ndarray, scan_positions_px.mean(axis=0))
        if initial_scan_positions_px.ndim == 3:
            initial_scan_positions_px = cast(np.ndarray, initial_scan_positions_px.mean(axis=0))

        if scan_positions_px.ndim != 2 or scan_positions_px.shape[-1] != 2:
            raise ValueError(
                "scan_positions_px must have shape (num_positions, 2), "
                f"got {scan_positions_px.shape}"
            )
        if initial_scan_positions_px.ndim != 2 or initial_scan_positions_px.shape[-1] != 2:
            raise ValueError(
                "initial_scan_positions_px must have shape (num_positions, 2), "
                f"got {initial_scan_positions_px.shape}"
            )
        if scan_positions_px.shape != initial_scan_positions_px.shape:
            raise ValueError(
                "scan_positions_px and initial_scan_positions_px must have the same shape, "
                f"got {scan_positions_px.shape} and {initial_scan_positions_px.shape}"
            )

        sampling = np.asarray(self.sampling)
        scan_positions = scan_positions_px * sampling
        initial_scan_positions = initial_scan_positions_px * sampling

        if plot_cropped_rotated_fov:
            angle = (
                self.dset.com_rotation_rad
                if self.dset.com_transpose
                else -self.dset.com_rotation_rad
            )
            tf = AffineTransform(angle=angle)
            origin = initial_scan_positions.mean(axis=0)
            initial_scan_positions = tf(initial_scan_positions, origin=origin)
            scan_positions = tf(scan_positions, origin=origin)

            obj_shape = self.obj_cropped.shape[-2:]
            center_shift = initial_scan_positions.mean(axis=0) - (
                np.array(obj_shape) / 2 * sampling
            )
            initial_scan_positions -= center_shift
            scan_positions -= center_shift
        else:
            obj_shape = self.obj_shape_full[-2:]

        if plot_arrow_freq is not None:
            freq = int(plot_arrow_freq)
            if freq <= 0:
                raise ValueError(f"plot_arrow_freq must be positive, got {plot_arrow_freq}")

            rshape = tuple(self.dset.gpts) + (2,)
            if np.prod(self.dset.gpts) != initial_scan_positions.shape[0]:
                raise ValueError(
                    "plot_arrow_freq only supports full raster scan grids with "
                    f"{np.prod(self.dset.gpts)} positions, got {initial_scan_positions.shape[0]}"
                )

            initial_scan_positions = initial_scan_positions.reshape(rshape)[
                ::freq, ::freq
            ].reshape(-1, 2)
            scan_positions = scan_positions.reshape(rshape)[::freq, ::freq].reshape(-1, 2)

        deltas = scan_positions - initial_scan_positions
        norms = np.linalg.norm(deltas, axis=1)

        if verbose:
            print(
                "Updated scan position shifts: "
                f"mean={norms.mean():.4g} Å, "
                f"rms={np.sqrt(np.mean(norms**2)):.4g} Å, "
                f"max={norms.max():.4g} Å"
            )

        extent = [
            0,
            sampling[1] * obj_shape[1],
            sampling[0] * obj_shape[0],
            0,
        ]

        figsize = kwargs.pop("figsize", (4, 4))
        figax = kwargs.pop("figax", None)
        cmap = kwargs.pop("cmap", "Reds")
        title = kwargs.pop("title", "Updated probe positions")

        if figax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figax

        quiver_kwargs = {
            "angles": "xy",
            "scale_units": "xy",
            "scale": 1,
            "cmap": cmap,
        }
        quiver_kwargs.update(kwargs)

        im = ax.quiver(
            initial_scan_positions[:, 1],
            initial_scan_positions[:, 0],
            deltas[:, 1] * scale_arrows,
            deltas[:, 0] * scale_arrows,
            norms,
            **quiver_kwargs,
        )

        if cbar:
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
            fig.add_axes(ax_cb)
            cb = fig.colorbar(im, cax=ax_cb)
            cb.set_label("Δ [Å]", rotation=0, ha="left", va="bottom")
            cb.ax.yaxis.set_label_coords(0.5, 1.01)

        ax.set_ylabel("x [Å]")
        ax.set_xlabel("y [Å]")
        ax.set_xlim((extent[0], extent[1]))
        ax.set_ylim((extent[2], extent[3]))
        ax.set_aspect("equal")
        ax.set_title(title)

        if return_fig:
            return fig, ax

        plt.show()
        return None

    def show_fourier_probe_and_amplitudes(
        self,
        probe: np.ndarray | None = None,
        amplitudes: np.ndarray | None = None,
        fft_shift: bool = False,
        **kwargs,
    ):
        """show the fourier probe and amplitudes, useful for debugging
        Parameters
        ----------
        probe: np.ndarray | None, optional
            The probe to show. If None, the probe from the last iteration is shown.
        amplitudes: np.ndarray | None, optional
            The amplitudes to show. If None, the amplitudes from the last iteration are shown.
        fft_shift: bool, optional
            Whether to fftshift the probe and amplitudes, by default False
        **kwargs: dict, optional
            Additional arguments passed to show_2d
        """
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        probe_plot = np.abs(np.fft.fft2(probe[0]))

        if amplitudes is None:
            amplitudes = self._to_numpy(self.dset.centered_amplitudes.sum(0))
        else:
            amplitudes = self._to_numpy(amplitudes.sum(0))

        scalebar = [self._scalebar_recip()]

        if fft_shift:
            probe_plot = np.fft.fftshift(probe_plot)
        else:
            amplitudes = np.fft.fftshift(amplitudes)

        figsize = kwargs.pop("figsize", (10, 5))
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        show_2d(probe_plot, title="fourier probe", scalebar=scalebar, figax=(fig, ax[0]), **kwargs)
        show_2d(amplitudes, title="amplitudes", figax=(fig, ax[1]), **kwargs)
