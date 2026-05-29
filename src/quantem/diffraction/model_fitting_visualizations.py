from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from quantem.core import config
from quantem.core.visualization import show_2d

if TYPE_CHECKING:
    from quantem.diffraction.model_fitting import ModelDiffraction


class ModelDiffractionVisualizations:
    def _plot_overlays(
        self,
        ax: Any,
        origin_rc: np.ndarray,
        disk_centers_rc: np.ndarray,
        *,
        overlay_origin: bool = True,
        overlay_disks: bool = True,
        origin_marker_kwargs: dict[str, Any] | None = None,
        disk_marker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Plot origin and disk-center overlays on an axis.

        Parameters
        ----------
        ax : Any
            Matplotlib axis receiving overlays.
        origin_rc : np.ndarray
            Origin coordinate as ``(row, col)``.
        disk_centers_rc : np.ndarray
            Disk centers as ``(N, 2)`` in ``(row, col)`` order.
        overlay_origin : bool, optional
            If ``True``, plot origin marker.
        overlay_disks : bool, optional
            If ``True``, plot disk-center markers.
        origin_marker_kwargs : dict[str, Any] | None, optional
            Matplotlib kwargs merged onto origin marker defaults.
        disk_marker_kwargs : dict[str, Any] | None, optional
            Matplotlib kwargs merged onto disk marker defaults.

        Returns
        -------
        None
        """
        colors = config.get("viz.colors.set")
        if overlay_origin and origin_rc.shape == (2,):
            kw_origin = {
                "marker": "+",
                "color": colors[0],
                "markersize": 10,
                "markeredgewidth": 3,
                "linestyle": "None",
            }
            if origin_marker_kwargs is not None:
                kw_origin.update(origin_marker_kwargs)
            ax.plot(float(origin_rc[1]), float(origin_rc[0]), **kw_origin)

        if overlay_disks and disk_centers_rc.ndim == 2 and disk_centers_rc.shape[0] > 0:
            kw_disks = {
                "marker": "x",
                "color": colors[1],
                "markersize": 5,
                "markeredgewidth": 2.0,
                "linestyle": "None",
            }
            if disk_marker_kwargs is not None:
                kw_disks.update(disk_marker_kwargs)
            ax.plot(disk_centers_rc[:, 1], disk_centers_rc[:, 0], **kw_disks)

    def plot_losses(
        self, 
        figax: tuple[Any, Any] | None = None, 
        plot_lrs: bool = True, 
        plot_individual: bool = False,
        individual_row: int = 0,
        individual_col: int = 0,
    ) -> tuple[Any, Any]:
        md = cast("ModelDiffraction", self)
        colors = config.get("viz.colors.set")
        loss_color = "k"
        lr_color = colors[8]

        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax
        if plot_individual:
            mean_hist = md.fit_history.get(f"individual_{individual_row}_{individual_col}")
            losses = np.asarray([] if mean_hist is None else mean_hist.losses, dtype=np.float64)
        else:
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
        lines.extend(ax.semilogy(iters, losses, c=loss_color, lw=2, label="loss"))
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss", color=loss_color)
        ax.tick_params(axis="y", which="both", colors=loss_color)
        ax.spines["left"].set_color(loss_color)
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
                    ax_lr.semilogy(np.arange(lrs.size), lrs, c=lr_color, lw=2, ls="--", label="LR")
                )
                ax_lr.set_ylabel("LR", color=lr_color)
                ax_lr.tick_params(axis="y", which="both", colors=lr_color)
                ax_lr.spines["right"].set_color(lr_color)
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
        individual_loss: bool = False,
        pattern_row: int = 0,
        pattern_col: int = 0,
        power: float = 1,
        cbar: bool = False,
        axsize: tuple[int, int] = (6, 6),
        overlay: bool = True,
        overlay_origin: bool = True,
        overlay_disks: bool = True,
        overlay_on: Literal["model", "both"] = "model",
        origin_marker_kwargs: dict[str, Any] | None = None,
        disk_marker_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """
        Visualize fit losses with reference/model image panels.

        Parameters
        ----------
        power : float, optional
            Power-law display scaling.
        cbar : bool, optional
            If ``True``, draw colorbars.
        axsize : tuple[int, int], optional
            Axis size passed through to ``show_2d``.
        overlay : bool, optional
            If ``True``, draw coordinate overlays.
        overlay_origin : bool, optional
            If ``True``, include origin marker in overlays.
        overlay_disks : bool, optional
            If ``True``, include disk-center markers in overlays.
        overlay_on : {"model", "both"}, optional
            Which image panel(s) receive overlays.
        origin_marker_kwargs : dict[str, Any] | None, optional
            Marker kwargs override for origin marker.
        disk_marker_kwargs : dict[str, Any] | None, optional
            Marker kwargs override for disk-center markers.

        Returns
        -------
        tuple[Any, Any]
            ``(fig, axs)`` for further editing.

        Raises
        ------
        RuntimeError
            If model/context are not defined.
        ValueError
            If ``overlay_on`` is invalid.
        """
        md = cast("ModelDiffraction", self)

        if md.image_ref is None:
            md.preprocess()
        if md.image_ref is None or md.model is None or md.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        if md.dataset.shape[0] <= pattern_row or md.dataset.shape[1] <= pattern_col:
            raise ValueError("individual row or column outside bounds of dataset")

        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        md.plot_losses(figax=(fig, ax_top), plot_lrs=True, plot_individual=individual_loss,individual_row=pattern_row,individual_col=pattern_col)

        if individual_loss:
            pred = md.render_individual_pattern(row=pattern_row, col=pattern_col)
            ref = np.asarray(md.dataset.array[pattern_row, pattern_col], dtype=np.float32)
        else:
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
            cmap=config.get("viz.cmap"),
            cbar=bool(cbar),
            returnfig=False,
            axsize=axsize,
            vmin=vmin,
            vmax=vmax,
        )

        if overlay:
            if overlay_on not in ("model", "both"):
                raise ValueError("overlay_on must be 'model' or 'both'.")
            origin_rc, disk_centers_rc = md.get_overlay_coordinates()
            axes = [axs[1]] if overlay_on == "model" else [axs[0], axs[1]]
            for ax in axes:
                self._plot_overlays(
                    ax,
                    origin_rc,
                    disk_centers_rc,
                    overlay_origin=overlay_origin,
                    overlay_disks=overlay_disks,
                    origin_marker_kwargs=origin_marker_kwargs,
                    disk_marker_kwargs=disk_marker_kwargs,
                )

        mean_hist = md.fit_history.get("mean")
        if individual_loss:
            mean_hist = md.fit_history.get(f"individual_{pattern_row}_{pattern_row}")
        if mean_hist is not None and len(mean_hist.losses) > 0:
            fig.suptitle(
                f"Final loss: {mean_hist.losses[-1]:.3e} | Iters: {len(mean_hist.losses)}",
                fontsize=13,
                y=0.98,
            )
        plt.show()
        return fig, axs

    def plot_model(
        self,
        *,
        plot_mean_model: bool = False,
        plot_individual_model: bool = False,
        pattern_row: int = 0,
        pattern_col: int= 0,
        power: float = 0.25,
        returnfig: bool = False,
        axsize: tuple[int, int] = (6, 6),
        overlay: bool = True,
        overlay_origin: bool = True,
        overlay_disks: bool = True,
        overlay_on: Literal["model", "both"] = "model",
        origin_marker_kwargs: dict[str, Any] | None = None,
        disk_marker_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[Any, Any] | None:
        """
        Plot reference and indiviual diffraction images.

        Parameters
        ----------
        power : float, optional
            Power-law display scaling.
        returnfig : bool, optional
            If ``True``, return ``(fig, ax)``.
        axsize : tuple[int, int], optional
            Axis size passed through to ``show_2d``.
        overlay : bool, optional
            If ``True``, draw coordinate overlays.
        overlay_origin : bool, optional
            If ``True``, include origin marker in overlays.
        overlay_disks : bool, optional
            If ``True``, include disk-center markers in overlays.
        overlay_on : {"model", "both"}, optional
            Which image panel(s) receive overlays.
        origin_marker_kwargs : dict[str, Any] | None, optional
            Marker kwargs override for origin marker.
        disk_marker_kwargs : dict[str, Any] | None, optional
            Marker kwargs override for disk-center markers.
        **_ : Any
            Ignored extra kwargs for backward compatibility.

        Returns
        -------
        tuple[Any, Any] | None
            Figure/axes tuple when ``returnfig=True``; otherwise ``None``.

        Raises
        ------
        RuntimeError
            If model/context are not defined.
        ValueError
            If ``overlay_on`` is invalid.
        """
        md = cast("ModelDiffraction", self)
        if md.image_ref is None:
            md.preprocess()
        if md.image_ref is None or md.model is None or md.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        if plot_mean_model and plot_individual_model:
            raise RuntimeError("can only plot mean or plot individual, not both")
        if plot_individual_model:
            pred = md.render_individual_pattern(pattern_row, pattern_col)
            ref = np.asarray(md.dataset.array[pattern_row, pattern_col], dtype=np.float32)
        elif plot_mean_model:
            ref = np.asarray(md.image_ref, dtype=np.float32)
            pred = md.render_mean_refined
        else:
            ref = np.asarray(md.image_ref, dtype=np.float32)
            pred = md.render_current

        refp = ref if power == 1.0 else np.maximum(ref, 0.0) ** float(power)
        predp = pred if power == 1.0 else np.maximum(pred, 0.0) ** float(power)
        vmin = float(min(refp.min(), predp.min()))
        vmax = float(max(refp.max(), predp.max()))

        t1 = kwargs.pop("title", "")
        fig, ax = show_2d(
            [refp, predp],
            title=[t1 + " image_ref", t1 + " model"],
            cmap=config.get("viz.cmap"),
            cbar=True,
            returnfig=True,
            axsize=axsize,
            vmin=vmin,
            vmax=vmax,
            **kwargs, 
        )
        if overlay:
            if overlay_on not in ("model", "both"):
                raise ValueError("overlay_on must be 'model' or 'both'.")
            origin_rc, disk_centers_rc = md.get_overlay_coordinates()
            axs_arr = np.asarray(ax, dtype=object).reshape(-1)
            axes = [axs_arr[1]] if overlay_on == "model" else [axs_arr[0], axs_arr[1]]
            for a in axes:
                self._plot_overlays(
                    a,
                    origin_rc,
                    disk_centers_rc,
                    overlay_origin=overlay_origin,
                    overlay_disks=overlay_disks,
                    origin_marker_kwargs=origin_marker_kwargs,
                    disk_marker_kwargs=disk_marker_kwargs,
                )
        if returnfig:
            return fig, ax
        return None

    def visualize_components(
        self,
        components: str | list[str],
        *,
        power: float = 0.25,
        cbar: bool = False,
        axsize: tuple[int, int] = (6, 6),
        returnfig: bool = False,
        overlay: bool = True,
        overlay_origin: bool = True,
        overlay_disks: bool = True,
        origin_marker_kwargs: dict[str, Any] | None = None,
        disk_marker_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any] | None:
        """
        Render and display a summed component image.

        Parameters
        ----------
        components : str | list[str]
            Component name or list of component names. Multiple names are
            composited by summing component renders.
        power : float, optional
            Power-law display scaling.
        cbar : bool, optional
            If ``True``, draw colorbars.
        axsize : tuple[int, int], optional
            Axis size passed through to ``show_2d``.
        returnfig : bool, optional
            If ``True``, return ``(fig, ax)``.
        overlay : bool, optional
            If ``True``, draw coordinate overlays on all component panels.
        overlay_origin : bool, optional
            If ``True``, include origin marker in overlays.
        overlay_disks : bool, optional
            If ``True``, include disk-center markers in overlays.
        origin_marker_kwargs : dict[str, Any] | None, optional
            Marker kwargs override for origin marker.
        disk_marker_kwargs : dict[str, Any] | None, optional
            Marker kwargs override for disk-center markers.

        Returns
        -------
        tuple[Any, Any] | None
            Figure/axes tuple when ``returnfig=True``; otherwise ``None``.
        """
        md = cast("ModelDiffraction", self)
        names = [components] if isinstance(components, str) else list(components)
        if len(names) == 0:
            raise ValueError("components must contain at least one component name.")

        rendered = [
            np.asarray(md.get_rendered_component(name), dtype=np.float32) for name in names
        ]
        summed = np.sum(np.stack(rendered, axis=0), axis=0)
        summed_scaled = summed if power == 1.0 else np.maximum(summed, 0.0) ** float(power)
        title = names[0] if len(names) == 1 else " + ".join(names)

        fig, ax = show_2d(
            summed_scaled,
            title=title,
            cmap=config.get("viz.cmap"),
            cbar=bool(cbar),
            returnfig=True,
            axsize=axsize,
        )

        if overlay:
            origin_rc, disk_centers_rc = md.get_overlay_coordinates()
            self._plot_overlays(
                ax,
                origin_rc,
                disk_centers_rc,
                overlay_origin=overlay_origin,
                overlay_disks=overlay_disks,
                origin_marker_kwargs=origin_marker_kwargs,
                disk_marker_kwargs=disk_marker_kwargs,
            )

        if returnfig:
            return fig, ax
        return None
