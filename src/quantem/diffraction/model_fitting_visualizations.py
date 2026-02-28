from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from quantem.core.visualization import show_2d

if TYPE_CHECKING:
    from quantem.diffraction.model_fitting import ModelDiffraction


class ModelDiffractionVisualizations:
    def plot_losses(
        self, figax: tuple[Any, Any] | None = None, plot_lrs: bool = True
    ) -> tuple[Any, Any]:
        md = cast("ModelDiffraction", self)

        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        mean_hist = md.fit_history.get("mean")
        losses = np.asarray([] if mean_hist is None else mean_hist.losses, dtype=np.float64)
        if losses.size == 0:
            ax.text(
                0.5,
                0.5,
                "No fit history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            if figax is None:
                plt.tight_layout()
                plt.show()
            return fig, ax

        iters = np.arange(losses.size)
        lines: list[Any] = []
        lines.extend(ax.semilogy(iters, losses, c="k", lw=2, label="loss"))
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss", color="k")
        ax.tick_params(axis="y", which="both", colors="k")
        ax.spines["left"].set_color("k")
        ax.set_xbound(-2, max(1, int(iters.max())) + 2)

        lrs = np.asarray([] if mean_hist is None else mean_hist.lrs, dtype=np.float64)
        if plot_lrs and lrs.size > 0:
            if lrs.size == losses.size and not np.allclose(lrs, lrs[0]):
                ax_lr = ax.twinx()
                ax.set_zorder(2)
                ax_lr.set_zorder(1)
                ax.patch.set_visible(False)
                ax_lr.spines["left"].set_visible(False)
                lines.extend(
                    ax_lr.semilogy(
                        np.arange(lrs.size), lrs, c="tab:blue", lw=2, ls="--", label="LR"
                    )
                )
                ax_lr.set_ylabel("LR", color="tab:blue")
                ax_lr.tick_params(axis="y", which="both", colors="tab:blue")
                ax_lr.spines["right"].set_color("tab:blue")
            else:
                ax.set_title(f"LR: {float(lrs[-1]):.2e}", fontsize=10)

        labels = [line.get_label() for line in lines]
        if len(labels) > 1:
            ax.legend(lines, labels, loc="upper right")

        if figax is None:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def visualize(
        self,
        *,
        power: float = 0.25,
        cbar: bool = False,
        axsize: tuple[int, int] = (6, 6),
    ) -> tuple[Any, Any]:
        md = cast("ModelDiffraction", self)

        if md.image_ref is None:
            md.preprocess()
        if md.image_ref is None or md.model is None or md.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")

        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        md.plot_losses(figax=(fig, ax_top), plot_lrs=True)

        ref = np.asarray(md.image_ref, dtype=np.float32)
        pred = md.render_current
        refp = ref if power == 1.0 else np.maximum(ref, 0.0) ** float(power)
        predp = pred if power == 1.0 else np.maximum(pred, 0.0) ** float(power)
        vmin = float(min(refp.min(), predp.min()))
        vmax = float(max(refp.max(), predp.max()))

        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.15)
        axs = np.array(
            [fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])], dtype=object
        )
        show_2d(
            [refp, predp],
            figax=(fig, axs),
            title=["image_ref", "model"],
            cmap="gray",
            cbar=bool(cbar),
            returnfig=False,
            axsize=axsize,
            vmin=vmin,
            vmax=vmax,
        )

        mean_hist = md.fit_history.get("mean")
        if mean_hist is not None and len(mean_hist.losses) > 0:
            fig.suptitle(
                f"Final loss: {mean_hist.losses[-1]:.3e} | Iters: {len(mean_hist.losses)}",
                fontsize=13,
                y=0.98,
            )
        plt.show()
        return fig, axs

    def plot_mean_model(
        self,
        *,
        power: float = 0.25,
        returnfig: bool = False,
        axsize: tuple[int, int] = (6, 6),
        **_: Any,
    ) -> tuple[Any, Any] | None:
        md = cast("ModelDiffraction", self)
        if md.image_ref is None:
            md.preprocess()
        if md.image_ref is None or md.model is None or md.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")

        ref = np.asarray(md.image_ref, dtype=np.float32)
        pred = md.render_current

        refp = ref if power == 1.0 else np.maximum(ref, 0.0) ** float(power)
        predp = pred if power == 1.0 else np.maximum(pred, 0.0) ** float(power)
        vmin = float(min(refp.min(), predp.min()))
        vmax = float(max(refp.max(), predp.max()))

        fig, ax = show_2d(
            [refp, predp],
            title=["image_ref", "model"],
            cmap="gray",
            cbar=False,
            returnfig=True,
            axsize=axsize,
            vmin=vmin,
            vmax=vmax,
        )
        if returnfig:
            return fig, ax
        return None
