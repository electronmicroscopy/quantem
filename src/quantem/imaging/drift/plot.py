
import matplotlib.pyplot as plt
import numpy as np

from quantem.core.visualization import show_2d
from quantem.imaging.drift import diagnostics


def plot_transformed_images(self, show_knots: bool = True, **kwargs):
    self._ensure_warped_images()
    fig, ax = show_2d(
        list(self.images_warped.array),
        **kwargs,
    )
    if show_knots:
        for a0 in range(self.shape[0]):
            x = self.knots[a0][0]
            y = self.knots[a0][1]
            ax[a0].plot(
                y,
                x,
                color="r",
            )


def plot_convergence(
    self,
    figsize=(8, 3),
    **kwargs,
):
    """
    Plot the convergence of the drift correction.
    """
    sub = np.abs(self.error_track[:, 0] - 2) < 0.1
    error = self.error_track[:, 1]
    it = np.arange(error.shape[0])

    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    color = (1, 0, 0)  # red

    # Plot Affine
    if np.any(~sub):
        ax[0].plot(
            it[~sub],
            100 * error[~sub],
            marker="o",
            color=color,
            linestyle="-",
            label="Affine",
            **kwargs,
        )
        ax[0].set_xlabel("Affine Iterations")
        ax[0].set_ylabel("Mean Error [%]")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    else:
        ax[0].axis("off")

    # Plot Non-Rigid
    if np.any(sub):
        first_true = np.argmax(sub)
        if first_true > 0:
            sub[first_true - 1] = True

        ax[1].plot(
            it[sub],
            100 * error[sub],
            marker="o",
            color=color,
            linestyle="-",
            label="Non-Rigid",
            **kwargs,
        )
        ax[1].set_xlabel("Non-Rigid Iterations")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    else:
        ax[1].axis("off")

    plt.tight_layout()

    return self


def _ensure_warped_images(self):
    """Lazily populate images_warped from current knots if marked stale."""
    if getattr(self, "_images_warped_stale", False):
        self._warp_and_translate_torch(
            self._max_image_shift_cached, upsample_factor=8,
            solve_translation=False)
        self._images_warped_stale = False


def plot_merged_images(self, show_knots: bool = True, **kwargs):
    """
    Plot the current transformed images, with knot overlays.
    """
    self._ensure_warped_images()
    fig, ax = show_2d(
        self.images_warped.array.mean(0),
        **kwargs,
    )
    if show_knots:
        for a0 in range(self.shape[0]):
            x = self.knots[a0][0]
            y = self.knots[a0][1]
            ax.plot(
                y,
                x,
            )


def _comparison_pair(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compare scan 0 with scan 1, or with the mean of all remaining scans."""
    if stack.shape[0] == 2:
        return stack[0], stack[1]
    return np.mean(stack[1:], axis=0), stack[0]


def _registration_overlay(
    reference: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build percentile-normalized RGB agreement and difference images."""
    values = np.concatenate((reference[mask], moving[mask]))
    if values.size:
        low, high = np.percentile(values, (1.0, 99.0))
    else:
        low, high = 0.0, 1.0
    scale = max(float(high - low), np.finfo(np.float32).eps)
    first = np.clip((reference - low) / scale, 0.0, 1.0)
    second = np.clip((moving - low) / scale, 0.0, 1.0)
    overlay = np.stack((first, second, np.zeros_like(first)), axis=-1)
    difference = np.abs(first - second)
    overlay[~mask] = 0.0
    difference[~mask] = 0.0
    return overlay, difference


def _plot_registration_diagnostics(
    correction,
    *,
    stages: tuple[str, ...] | list[str] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot stage-by-stage registration on one fixed measured footprint."""
    selected, stacks, common_mask, _ = diagnostics._registration_data(
        correction,
        stages,
    )
    if figsize is None:
        figsize = (11.0, 3.4 * len(selected))
    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=figsize,
        squeeze=False,
    )
    for row_index, stage in enumerate(selected):
        reference, moving = _comparison_pair(stacks[stage])
        metrics = diagnostics._pair_metrics(reference, moving, common_mask)
        overlay, difference = _registration_overlay(reference, moving, common_mask)
        axes[row_index, 0].imshow(overlay)
        axes[row_index, 0].set_title(
            f"{stage}: RGB registration\ncommon NCC {float(metrics['common_ncc']):.4f}"
        )
        axes[row_index, 1].imshow(difference, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_index, 1].set_title(
            "Percentile-normalized |difference|\n"
            f"native MAD {float(metrics['mean_absolute_difference']):.4g}; "
            f"RMS {float(metrics['root_mean_square_difference']):.4g}"
        )
        axes[row_index, 2].imshow(common_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].set_title(
            f"Fixed common coverage\n{float(common_mask.mean()):.1%} of canvas"
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Drift registration diagnostics",
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout()
    return figure, axes


def _plot_displacement_diagnostics(
    correction,
    *,
    stage: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot scan-line-origin displacement without changing fitted knots."""
    if stage is None:
        stage = diagnostics._select_stages(correction, None)[-1]
    else:
        stage = diagnostics._select_stages(correction, (stage,))[0]
    fields = diagnostics._displacement_fields(correction, stage)
    rows = diagnostics._displacement_rows(correction, (stage,))
    if figsize is None:
        figsize = (11.0, 3.6 * len(fields))
    figure, axes = plt.subplots(
        len(fields),
        2,
        figsize=figsize,
        squeeze=False,
    )
    for image_index, (field, metrics) in enumerate(zip(fields, rows, strict=True)):
        scan_lines = np.arange(field.shape[1])
        row_displacement = np.mean(field[0], axis=1)
        column_displacement = np.mean(field[1], axis=1)
        axes[image_index, 0].plot(
            scan_lines,
            row_displacement,
            label="row displacement",
        )
        axes[image_index, 0].plot(
            scan_lines,
            column_displacement,
            label="column displacement",
        )
        axes[image_index, 0].axhline(0.0, color="0.3", linewidth=0.7)
        axes[image_index, 0].set_xlabel("Scan line")
        axes[image_index, 0].set_ylabel("Displacement (px)")
        axes[image_index, 0].set_title(
            f"Image {image_index}: scan-line origins\n"
            f"endpoint displacement {float(metrics['endpoint_displacement_px']):.3g} px"
        )
        axes[image_index, 0].legend(frameon=False)
        axes[image_index, 0].grid(alpha=0.25)

        magnitude = np.linalg.norm(field, axis=0)
        if magnitude.shape[1] == 1:
            axes[image_index, 1].plot(
                scan_lines,
                magnitude[:, 0],
                color="#0072B2",
            )
            axes[image_index, 1].set_xlabel("Scan line")
            axes[image_index, 1].set_ylabel("Displacement magnitude (px)")
            axes[image_index, 1].grid(alpha=0.25)
        else:
            image = axes[image_index, 1].imshow(
                magnitude,
                cmap="viridis",
                aspect="auto",
                origin="upper",
            )
            axes[image_index, 1].set_xlabel("Knot along fast scan")
            axes[image_index, 1].set_ylabel("Scan line")
            figure.colorbar(
                image,
                ax=axes[image_index, 1],
                label="Displacement (px)",
            )
        axes[image_index, 1].set_title(
            f"Image {image_index}: displacement magnitude\n"
            f"maximum {float(metrics['max_displacement_px']):.3g} px"
        )
    figure.suptitle(
        f"Drift displacement diagnostics: {stage}",
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout()
    return figure, axes
