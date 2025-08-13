from typing import Any, Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from quantem.core import config
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.ptychography_base import PtychographyBase


# TODO make dataclass
class PtychographyVisualizations(PtychographyBase):
    def show_obj(
        self,
        obj: np.ndarray | None = None,
        cbar: bool = False,
        interval_type: Literal["quantile", "manual"] = "quantile",
        **kwargs,
    ):
        if obj is None:
            obj = self.obj_cropped
        else:
            obj = self._to_numpy(obj)
            if obj.ndim == 2:
                obj = obj[None, ...]

        if interval_type == "quantile":
            norm = {"interval_type": "quantile"}
        elif interval_type in ["manual", "minmax", "abs"]:
            norm = {"interval_type": "manual"}
        else:
            raise ValueError(f"Unknown interval type: {interval_type}")

        ph_cmap = config.get("viz.phase_cmap")
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
            norm=norm,
            cbar=cbar,
            scalebar=scalebar,
            **kwargs,
        )

    def show_probe(self, probe: np.ndarray | None = None):
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        probes = [np.fft.fftshift(probe[i]) for i in range(len(probe))]
        scalebar = {"sampling": self.reciprocal_sampling[0], "units": r"$\mathrm{A^{-1}}$"}
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
        if len(probes) > 1:
            titles = self.get_probe_intensities(probe)
            titles = [
                f"Fourier Probe {i + 1}/{len(titles)}: {t * 100:.1f}%"
                for i, t in enumerate(titles)
            ]
        else:
            titles = "Fourier Probe"
        show_2d(probes, title=titles)

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

        lines = []
        epochs = np.arange(len(self.epoch_losses))
        # colors = plt.cm.tab20.colors # type:ignore  # TODO make a better dual categorical cmap
        # colors = ["blue", "royalblue", "darkorange", "orange", "green", "limegreen", "red", "coral"]
        # colors = ["royalblue", "darkorange", "limegreen", "coral"]
        colors = plt.cm.Set1.colors  # type:ignore
        lines.extend(ax.semilogy(epochs, self.epoch_losses, c="k", label="loss"))
        ax.set_ylabel("Loss", color="k")
        ax.tick_params(axis="y", which="both", colors="k")
        ax.spines["left"].set_color("k")
        # for label in ax.get_yticklabels():
        #     label.set_color(colors[0])
        ax.set_xlabel("Epochs")
        nx = ax.twinx()
        nx.spines["left"].set_visible(False)

        if plot_lrs:
            obj_recon_types = np.array([mode.split("-")[0] for mode in self.epoch_recon_types])
            probe_recon_types = np.array([mode.split("-")[1] for mode in self.epoch_recon_types])
            obj_modes = np.unique(obj_recon_types)
            probe_modes = np.unique(probe_recon_types)

            for i, (omode, pmode) in enumerate(zip(obj_modes, probe_modes)):
                if "GD" in omode or "GD" in pmode:
                    GD_inds = np.where(obj_recon_types == omode)[0]
                    lines.append(
                        ax.axvspan(
                            min(epochs[GD_inds]) - 0.1,
                            max(epochs[GD_inds]) + 0.1,
                            color=colors[1],
                            alpha=0.3,
                            label="GD",
                        )
                    )
                else:
                    obj_inds = np.where(obj_recon_types == omode)[0]
                    lines.extend(
                        nx.semilogy(
                            epochs[obj_inds],
                            self.epoch_lrs["object"][obj_inds],
                            c=colors[i],
                            label=omode + " LR",
                        )
                    )
                    probe_inds = np.where(probe_recon_types == pmode)[0]
                    if "probe" in self.epoch_lrs.keys():
                        lines.extend(
                            nx.semilogy(
                                epochs[probe_inds],
                                self.epoch_lrs["probe"][probe_inds],
                                c=colors[i],
                                label=pmode + " LR",
                                linestyle="--",
                            )
                        )
                nx.set_ylabel("LR", c=colors[0])
                nx.spines["right"].set_color(colors[0])
                nx.tick_params(axis="y", which="both", colors=colors[0])

        labs = [lin.get_label() for lin in lines]
        nx.legend(lines, labs, loc="upper center")
        nx.set_ylabel("LRs")
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
            y=0.94,
        )
        plt.show()

    def show_object_epochs(self):
        # image grid
        pass
