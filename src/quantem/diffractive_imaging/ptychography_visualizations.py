from typing import Any, Literal, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey

from quantem.core import config
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.ptycho_utils import center_crop_arr
from quantem.diffractive_imaging.ptychography_base import PtychographyBase


class PtychographyVisualizations(PtychographyBase):
    def show_obj(
        self,
        obj: np.ndarray | None = None,
        cbar: bool = False,
        # norm: Literal["quantile", "manual", "minmax", "abs"] = "quantile",
        **kwargs,
    ):
        if obj is None:
            obj = self.obj_cropped
        else:
            obj = self._to_numpy(obj)
            if obj.ndim == 2:
                obj = obj[None, ...]

        # if norm == "quantile":
        #     norm_dict = {"interval_type": "quantile"}
        # elif norm in ["manual", "minmax", "abs"]:
        #     norm_dict = {"interval_type": "manual"}
        # else:
        #     raise ValueError(f"Unknown norm type: {norm}")

        ph_cmap = kwargs.pop("cmap", config.get("viz.phase_cmap"))
        if obj.shape[0] > 1:
            t = "Summed "
        else:
            t = ""

        ims = []
        titles = []
        cmaps = []
        if self.obj_type == "potential":
            ims.append(np.abs(obj).sum(0))
            titles.append(t + "Potential")
            cmaps.append(ph_cmap)
        elif self.obj_type == "pure_phase":
            ims.append(np.angle(obj).sum(0))
            titles.append(t + "Pure Phase")
            cmaps.append(ph_cmap)
        else:
            ims.extend([np.angle(obj).sum(0), np.abs(obj).sum(0)])
            titles.extend([t + "Phase", t + "Amplitude"])
            cmaps.extend([ph_cmap, "gray"])

        scalebar = [{"sampling": self.sampling[0], "units": "Å"}] + [None] * (len(ims) - 1)

        show_2d(
            ims,
            title=titles,
            cmap=cmaps,
            # norm=norm_dict,
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
        snapshot_idx: int | None = None,
        return_fft: bool = False,
        **kwargs,
    ):
        if obj is None:
            if snapshot_idx is not None:
                obj_np = self.epoch_snapshots[snapshot_idx]["obj"]
                obj_np = self._crop_rotate_obj_fov(obj_np).sum(0)  # type:ignore # FIXME
                if self.obj_type == "pure_phase":
                    obj_np = np.exp(1j * np.angle(obj_np))
            else:
                obj_np = self.obj_cropped.sum(0)
        else:
            obj_np = self._to_numpy(obj)
            if obj_np.ndim == 3:
                obj_np = obj.sum(0)
            else:
                if obj_np.ndim != 2:
                    raise ValueError(f"obj must be 2D, got {obj_np.ndim}D")

        window_2d = (
            tukey(obj_np.shape[0], tukey_alpha)[:, None]
            * tukey(obj_np.shape[1], tukey_alpha)[None, :]
        )
        if self.obj_type == "potential":
            windowed_obj = obj_np * window_2d
        else:
            windowed_obj = np.abs(obj_np) * window_2d * np.exp(1j * np.angle(obj_np) * window_2d)
        obj_pad = np.pad(windowed_obj, pad, mode="constant", constant_values=0)

        obj_fft = np.fft.fftshift(np.fft.fft2(obj_pad))

        fft_sampling = 1 / (self.sampling[0] * obj_pad.shape[0])
        fft_scalebar = {"sampling": fft_sampling, "units": r"$\mathrm{A^{-1}}$"}

        if show_obj:
            obj_scalebar = {"sampling": self.sampling[0], "units": "Å"}
            if self.obj_type == "potential" or self.obj_type == "complex":
                obj_show = obj_pad
            else:  # self.obj_type == "pure_phase":
                obj_show = np.angle(obj_pad)
            stitle = kwargs.pop("title", "")
            if snapshot_idx is not None:
                stitle += f" epoch {self.epoch_snapshots[snapshot_idx]['iteration']}"
            if len(stitle) > 0:
                stitle = stitle + " "
            show_2d(
                [
                    obj_show,
                    np.abs(obj_fft),
                ],
                title=[stitle + "Object", stitle + "Fourier Transform"],
                scalebar=[obj_scalebar, fft_scalebar],
                **kwargs,
            )
        else:
            show_2d(np.abs(obj_fft), scalebar=fft_scalebar, **kwargs)
        if return_fft:
            return obj_fft
        else:
            return

    def show_probe(self, probe: np.ndarray | None = None):
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        probes = [np.fft.fftshift(probe[i]) for i in range(len(probe))]
        scalebar = [{"sampling": self.sampling[0], "units": "Å"}] + [None] * (len(probes) - 1)
        if len(probes) > 1:
            titles = self.get_probe_intensities(probe)
            titles = [f"Probe {i + 1}/{len(titles)}: {t * 100:.1f}%" for i, t in enumerate(titles)]
        else:
            titles = "Probe"
        show_2d(probes, title=titles, scalebar=scalebar)

    def show_fourier_probe(self, probe: np.ndarray | None = None):
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        probes = [np.fft.fftshift(np.fft.fft2(probe[i])) for i in range(len(probe))]
        scalebar = [{"sampling": self.reciprocal_sampling[0], "units": r"$\mathrm{A^{-1}}$"}] + [
            None
        ] * (len(probes) - 1)
        if len(probes) > 1:
            titles = self.get_probe_intensities(probe)
            titles = [
                f"Fourier Probe {i + 1}/{len(titles)}: {t * 100:.1f}%"
                for i, t in enumerate(titles)
            ]
        else:
            titles = "Fourier Probe"
        show_2d(probes, title=titles, scalebar=scalebar)

    def show_fourier_probe_and_amplitudes(
        self,
        probe: np.ndarray | None = None,
        amplitudes: np.ndarray | None = None,
        fft_shift: bool = False,
        **kwargs,
    ):
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

        scalebar = [{"sampling": self.reciprocal_sampling[0], "units": r"$\mathrm{A^{-1}}$"}]

        if fft_shift:
            probe_plot = np.fft.fftshift(probe_plot)
        else:
            amplitudes = np.fft.fftshift(amplitudes)

        figsize = kwargs.pop("figsize", (10, 5))
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        show_2d(probe_plot, title="fourier probe", scalebar=scalebar, figax=(fig, ax[0]), **kwargs)
        show_2d(amplitudes, title="amplitudes", figax=(fig, ax[1]), **kwargs)

    def show_obj_and_probe(self, cbar: bool = False, figax=None):
        """shows the summed object and summed probe"""
        ims = []
        titles = []
        cmaps = []
        if self.obj_type == "potential":
            ims.append(np.abs(self.obj_cropped).sum(0))
            titles.append("Potential")
            cmaps.append(config.get("viz.phase_cmap"))
        elif self.obj_type == "pure_phase":
            ims.append(np.angle(self.obj_cropped).sum(0))
            titles.append("Pure Phase")
            cmaps.append(config.get("viz.phase_cmap"))
        else:
            ims.append(np.angle(self.obj_cropped).sum(0))
            ims.append(np.abs(self.obj_cropped).sum(0))
            titles.extend(["Phase", "Amplitude"])
            cmaps.extend([config.get("viz.phase_cmap"), config.get("viz.cmap")])

        ims.append(np.fft.fftshift(self.probe.sum(0)))
        titles.append("Probe")
        cmaps.append(None)
        scalebar = [{"sampling": self.sampling[0], "units": "Å"}] + [None] * (len(ims) - 1)
        cbars = [True] * (len(ims) - 1) + [False] if cbar else False
        show_2d(
            ims,
            title=titles,
            cmap=cmaps,
            scalebar=scalebar,
            cbar=cbars,
            figax=figax,
            tight_layout=True if figax is None else False,
        )

    def show_obj_slices(
        self,
        obj: np.ndarray | None = None,
        cbar: bool = False,
        interval_type: Literal["quantile", "manual"] = "quantile",
        interval_scaling: Literal["each", "all"] = "each",
        max_width: int = 4,
        **kwargs,
    ):
        if obj is None:
            obj = self.obj_cropped
        else:
            obj = self._to_numpy(obj)
            if obj.ndim == 2:
                obj = obj[None, ...]

        if self.obj_type == "potential":
            objs_flat = [np.abs(obj[i]) for i in range(len(obj))]
            titles_flat = [f"Potential {i + 1}/{len(obj)}" for i in range(len(obj))]
        elif self.obj_type == "pure_phase":
            objs_flat = [np.angle(obj[i]) for i in range(len(obj))]
            titles_flat = [f"Pure Phase {i + 1}/{len(obj)}" for i in range(len(obj))]
        else:
            objs_flat = [np.angle(obj[i]) for i in range(len(obj))]
            titles_flat = [f"Phase {i + 1}/{len(obj)}" for i in range(len(obj))]

        # Nest lists with max length max_width
        objs = [objs_flat[i : i + max_width] for i in range(0, len(objs_flat), max_width)]
        titles = [titles_flat[i : i + max_width] for i in range(0, len(titles_flat), max_width)]

        scalebars: list = [[None for _ in row] for row in objs]
        scalebars[0][0] = {"sampling": self.sampling[0], "units": "Å"}

        if interval_type == "quantile":
            norm = {"interval_type": "quantile"}
            # TODO -- make this work with interval_scaling
        elif interval_type in ["manual", "minmax", "abs"]:
            norm: dict[str, Any] = {"interval_type": "manual"}
            if interval_scaling == "all":
                norm["vmin"] = np.min(objs_flat)
                norm["vmax"] = np.max(objs_flat)
            else:
                norm["vmin"] = kwargs.get("vmin")
                norm["vmax"] = kwargs.get("vmax")
        else:
            raise ValueError(f"Unknown interval type: {interval_type}")

        show_2d(
            objs,
            title=titles,
            cmap=config.get("viz.phase_cmap"),
            norm=norm,
            cbar=cbar,
            scalebar=scalebars,
        )

    def plot_losses(self, figax: tuple | None = None, plot_lrs: bool = True):
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        lw = 2
        lines = []
        epochs = np.arange(len(self.epoch_losses))
        # colors = plt.cm.Set1.colors  # type:ignore
        colors = config.get("viz.colors.set")  # [1:]
        lines.extend(ax.semilogy(epochs, self.epoch_losses, c="k", label="loss", lw=lw))
        ax.set_ylabel("Loss", color="k")
        ax.tick_params(axis="y", which="both", colors="k")
        ax.spines["left"].set_color("k")
        ax.set_xlabel("Epochs")

        # check if all lrs are constant and if so, don't plot lr
        if all(np.all(lr == self.epoch_lrs["object"][0]) for lr in self.epoch_lrs.values()):
            plot_lrs = False

        if plot_lrs and len(self.epoch_lrs) > 0:
            nx = ax.twinx()
            nx.spines["left"].set_visible(False)
            color_idx = 0

            # Sort optimizers: object first, then probe, then the rest
            sorted_items = sorted(
                self.epoch_lrs.items(),
                key=lambda x: (0 if x[0] == "object" else 1 if x[0] == "probe" else 2, x[0]),
            )
            for lr_type, lr_values in sorted_items:
                if len(lr_values) > 0:
                    # Create epochs array that matches lr_values length
                    lr_epochs = np.arange(len(lr_values))
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
                            lr_epochs,
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

            labs = [lin.get_label() for lin in lines]
            nx.legend(lines, labs, loc="upper right")
        else:
            # No learning rates to plot, add to title
            # set title to each lr type
            title = ""
            for lr_type, lr_values in self.epoch_lrs.items():
                title += f"{lr_type} LR: {lr_values[0]:.1e} | "
            ax.set_title(title[:-3], fontsize=10)

        ax.set_xbound(-2, np.max(epochs if np.any(epochs) else [1]) + 2)
        if figax is None:
            plt.tight_layout()
            plt.show()

    def visualize(self, cbar: bool = True):
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        self.plot_losses(figax=(fig, ax_top))

        n_bot = 3 if self.obj_type == "complex" else 2
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_bot, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)])
        self.show_obj_and_probe(figax=(fig, axs_bot), cbar=cbar)
        plt.suptitle(
            f"Final loss: {self.epoch_losses[-1]:.3e} | Epochs: {len(self.epoch_losses)}",
            fontsize=14,
            y=0.95,
        )
        plt.show()

    def show_epochs(
        self,
        show_probe: bool = True,
        show_object: bool = True,
        epochs: list[int] | slice | None = None,
        every_nth: int | None = None,
        max_n: int | None = None,
        cbar: bool = False,
        norm: Literal["quantile", "manual", "minmax", "abs"] = "quantile",
        interval_scaling: Literal["each", "all"] = "each",
        max_width: int = 4,
        cropped: bool = True,
        **kwargs,
    ):
        """
        Display object and/or probe reconstructions from stored epoch snapshots.

        Parameters
        ----------
        show_probe : bool, optional
            Whether to show probe reconstructions, by default True
        show_object : bool, optional
            Whether to show object reconstructions, by default True
        epochs : list[int] | slice | None, optional
            Specific epoch iterations to display. If None, shows all available epochs
        every_nth : int | None, optional
            Show every nth epoch instead of all. Overrides epochs parameter
        max_epochs : int | None, optional
            Maximum number of epochs to display
        cbar : bool, optional
            Whether to show colorbars, by default False
        norm : str, optional
            Normalization method for object display, by default "quantile"
        interval_scaling : str, optional
            How to scale intervals: "each" for per-image scaling, "all" for global scaling across all epochs, by default "each"
        max_width : int, optional
            Maximum number of images per row, by default 4
        cropped : bool, optional
            Whether to show cropped objects (default True) or full objects
        **kwargs
            Additional arguments passed to show_2d
        """
        if not self.epoch_snapshots:
            print("No epoch snapshots available. Use store_iterations=True during reconstruction.")
            return

        if not show_object and not show_probe:
            print("Must show at least one of object or probe")
            return

        all_iterations = [snapshot["iteration"] for snapshot in self.epoch_snapshots]

        if every_nth is not None:
            selected_indices = list(range(0, len(self.epoch_snapshots), every_nth))
        elif epochs is not None:
            if isinstance(epochs, slice):
                selected_indices = list(range(*epochs.indices(len(self.epoch_snapshots))))
            else:
                selected_indices = []
                for epoch in epochs:
                    try:
                        idx = all_iterations.index(epoch)
                        selected_indices.append(idx)
                    except ValueError:
                        print(f"Warning: Epoch {epoch} not found in snapshots")
        else:
            selected_indices = list(range(len(self.epoch_snapshots)))

        if max_n is not None:
            selected_indices = selected_indices[:max_n]  # TEST

        if not selected_indices:
            print("No valid epochs selected for display")
            return

        selected_snapshots = [self.epoch_snapshots[i] for i in selected_indices]

        if norm == "quantile":
            norm_dict = {"interval_type": "quantile"}
            # TODO: implement global quantile scaling for interval_scaling="all"
        elif norm in ["manual", "minmax", "abs"]:
            norm_dict: dict[str, Any] = {"interval_type": "manual"}

            # Calculate global vmin/vmax if interval_scaling="all" and objects are shown
            if interval_scaling == "all" and show_object:
                all_object_values = []
                for snapshot in selected_snapshots:
                    obj = cast(np.ndarray, snapshot["obj"])

                    if cropped:
                        obj = self._crop_epoch_obj(obj)

                    if obj.ndim == 3 and obj.shape[0] > 1:
                        obj_display = obj.sum(0)
                    else:
                        obj_display = obj[0] if obj.ndim == 3 else obj

                    if self.obj_type == "potential":
                        all_object_values.append(np.abs(obj_display))
                    elif self.obj_type == "pure_phase":
                        all_object_values.append(np.angle(obj_display))
                    else:  # complex
                        all_object_values.extend([np.angle(obj_display), np.abs(obj_display)])

                if all_object_values:
                    all_values_flat = np.concatenate([arr.ravel() for arr in all_object_values])
                    norm_dict["vmin"] = float(np.min(all_values_flat))
                    norm_dict["vmax"] = float(np.max(all_values_flat))
            else:
                norm_dict["vmin"] = kwargs.get("vmin")
                norm_dict["vmax"] = kwargs.get("vmax")
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        if show_object and show_probe:
            self._show_object_and_probe_epochs(
                selected_snapshots, norm_dict, cbar, max_width, cropped, **kwargs
            )
        elif show_object:
            self._show_object_epochs_only(
                selected_snapshots, norm_dict, cbar, max_width, cropped, **kwargs
            )
        elif show_probe:
            self._show_probe_epochs_only(selected_snapshots, cbar, max_width, **kwargs)

    def _crop_epoch_obj(self, obj: np.ndarray) -> np.ndarray:
        """Apply the same cropping logic as obj_cropped property to epoch snapshot objects."""
        cropped = self._crop_rotate_obj_fov(obj)
        if self.obj_type == "pure_phase":
            cropped = np.exp(1j * np.angle(cropped))
        cropped = center_crop_arr(cropped, tuple(self.obj_shape_crop))
        return cropped

    def _show_object_epochs_only(self, snapshots, norm_dict, cbar, max_width, cropped, **kwargs):
        """Display only object reconstructions from epoch snapshots."""
        ph_cmap = config.get("viz.phase_cmap")

        all_images = []
        all_titles = []
        all_cmaps = []

        for snapshot in snapshots:
            obj = snapshot["obj"]
            iteration = snapshot["iteration"]

            if cropped:
                obj = self._crop_epoch_obj(obj)

            if obj.ndim == 3 and obj.shape[0] > 1:
                obj_display = obj.sum(0)
                title_prefix = f"Epoch {iteration} Summed "
            else:
                obj_display = obj[0] if obj.ndim == 3 else obj
                title_prefix = f"Epoch {iteration} "

            if self.obj_type == "potential":
                all_images.append(np.abs(obj_display))
                all_titles.append(title_prefix + "Potential")
                all_cmaps.append(ph_cmap)
            elif self.obj_type == "pure_phase":
                all_images.append(np.angle(obj_display))
                all_titles.append(title_prefix + "Phase")
                all_cmaps.append(ph_cmap)
            else:  # complex
                all_images.extend([np.angle(obj_display), np.abs(obj_display)])
                all_titles.extend([title_prefix + "Phase", title_prefix + "Amplitude"])
                all_cmaps.extend([ph_cmap, "gray"])

        images_grid = [all_images[i : i + max_width] for i in range(0, len(all_images), max_width)]
        titles_grid = [all_titles[i : i + max_width] for i in range(0, len(all_titles), max_width)]
        cmaps_grid = [all_cmaps[i : i + max_width] for i in range(0, len(all_cmaps), max_width)]

        scalebars: list = [[None for _ in row] for row in images_grid]
        if scalebars:
            scalebars[0][0] = {"sampling": self.sampling[0], "units": "Å"}

        show_2d(
            images_grid,
            title=titles_grid,
            cmap=cmaps_grid,
            norm=norm_dict,
            cbar=cbar,
            scalebar=scalebars,
            **kwargs,
        )

    def _show_probe_epochs_only(self, snapshots, cbar, max_width, **kwargs):
        """Display only probe reconstructions from epoch snapshots."""
        all_probes = []
        all_titles = []

        for snapshot in snapshots:
            probe = snapshot["probe"]
            iteration = snapshot["iteration"]

            if probe.ndim == 3 and probe.shape[0] > 1:
                probe_display = np.fft.fftshift(probe.sum(0))
                title = f"Epoch {iteration} Summed Probe"
            else:
                probe_display = np.fft.fftshift(probe[0] if probe.ndim == 3 else probe)
                title = f"Epoch {iteration} Probe"

            all_probes.append(probe_display)
            all_titles.append(title)

        probes_grid = [all_probes[i : i + max_width] for i in range(0, len(all_probes), max_width)]
        titles_grid = [all_titles[i : i + max_width] for i in range(0, len(all_titles), max_width)]

        # Set up scalebars
        scalebars: list = [[None for _ in row] for row in probes_grid]
        if scalebars:
            scalebars[0][0] = {
                "sampling": self.reciprocal_sampling[0],
                "units": r"$\mathrm{A^{-1}}$",
            }

        show_2d(
            probes_grid,
            title=titles_grid,
            scalebar=scalebars,
            cbar=cbar,
            **kwargs,
        )

    def _show_object_and_probe_epochs(
        self, snapshots, norm_dict, cbar, max_width, cropped, **kwargs
    ):
        """Display both object and probe reconstructions from epoch snapshots."""
        ph_cmap = config.get("viz.phase_cmap")

        all_images = []
        all_titles = []
        all_cmaps = []
        all_scalebars = []

        for snapshot in snapshots:
            obj = snapshot["obj"]
            probe = snapshot["probe"]
            iteration = snapshot["iteration"]

            # Apply cropping if requested
            if cropped:
                obj = self._crop_epoch_obj(obj)

            row_images = []
            row_titles = []
            row_cmaps = []
            row_scalebars = []

            # Process object
            if obj.ndim == 3 and obj.shape[0] > 1:
                obj_display = obj.sum(0)
                obj_prefix = "Summed "
            else:
                obj_display = obj[0] if obj.ndim == 3 else obj
                obj_prefix = ""

            if self.obj_type == "potential":
                row_images.append(np.abs(obj_display))
                row_titles.append(f"Epoch {iteration} {obj_prefix}Potential")
                row_cmaps.append(ph_cmap)
                row_scalebars.append({"sampling": self.sampling[0], "units": "Å"})
            elif self.obj_type == "pure_phase":
                row_images.append(np.angle(obj_display))
                row_titles.append(f"Epoch {iteration} {obj_prefix}Phase")
                row_cmaps.append(ph_cmap)
                row_scalebars.append({"sampling": self.sampling[0], "units": "Å"})
            else:  # complex
                row_images.extend([np.angle(obj_display), np.abs(obj_display)])
                row_titles.extend(
                    [
                        f"Epoch {iteration} {obj_prefix}Phase",
                        f"Epoch {iteration} {obj_prefix}Amplitude",
                    ]
                )
                row_cmaps.extend([ph_cmap, "gray"])
                row_scalebars.extend([{"sampling": self.sampling[0], "units": "Å"}, None])

            # Process probe
            if probe.ndim == 3 and probe.shape[0] > 1:
                probe_display = np.fft.fftshift(probe.sum(0))
                probe_title = f"Epoch {iteration} Summed Probe"
            else:
                probe_display = np.fft.fftshift(probe[0] if probe.ndim == 3 else probe)
                probe_title = f"Epoch {iteration} Probe"

            row_images.append(probe_display)
            row_titles.append(probe_title)
            row_cmaps.append(None)
            row_scalebars.append(
                {"sampling": self.reciprocal_sampling[0], "units": r"$\mathrm{A^{-1}}$"}
            )

            all_images.append(row_images)
            all_titles.append(row_titles)
            all_cmaps.append(row_cmaps)
            all_scalebars.append(row_scalebars)

        show_2d(
            all_images,
            title=all_titles,
            cmap=all_cmaps,
            norm=norm_dict,
            cbar=cbar,
            scalebar=all_scalebars,
            **kwargs,
        )

    def show_scan_positions(self, plot_radii: bool = True):
        # for each scan position, sum the intensity of self.probe at that position
        scan_positions = self.dset.scan_positions_px.cpu().detach().numpy()

        probe_params = self.probe_model.probe_params
        probe_radius_px = None

        conv_angle = probe_params.get("semiangle_cutoff")
        defocus = probe_params.get("defocus", 0)
        energy = probe_params.get("energy")

        if conv_angle is not None and energy is not None:
            from quantem.core.utils.utils import electron_wavelength_angstrom

            wavelength = electron_wavelength_angstrom(energy)
            conv_angle_rad = conv_angle * 1e-3

            # For defocused probe: radius ≈ |defocus| * convergence_angle + diffraction_limit
            diffraction_limit_angstrom = 0.61 * wavelength / conv_angle_rad
            defocus_blur_angstrom = abs(defocus) * conv_angle_rad
            probe_radius_angstrom = diffraction_limit_angstrom + defocus_blur_angstrom
            probe_radius_px = probe_radius_angstrom / self.sampling[0]

        _fig, ax = show_2d(self._get_probe_overlap(), title="probe overlap")
        if probe_radius_px is not None and plot_radii:
            # plot a circle with the probe radius for each probe position
            ax.scatter(
                scan_positions[:, 1],
                scan_positions[:, 0],
                s=probe_radius_px**2,
                edgecolors="red",
                c="none",
                linestyle="--",
            )
        plt.show()
