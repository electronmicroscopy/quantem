
import matplotlib.pyplot as plt
import numpy as np

from quantem.core.visualization import show_2d


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
