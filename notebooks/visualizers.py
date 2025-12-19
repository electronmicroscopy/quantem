from matplotlib import pyplot as plt
import numpy as np


def plot_image_diffs(im1, im2, subtitles=None, extent=None, suptitle=''):
    diff = im2 - im1
    
    max_val = np.nanmax([im1, im2])
    min_val = np.nanmin([im1, im2])
    diff_extent = np.nanmax(np.abs(diff))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
    fig.suptitle(suptitle)

    if subtitles is not None:
        for i, title in enumerate(subtitles):
            axes[i].set_title(title)
    
    if extent is None:
        for ax in axes:
            ax.set_axis_off()
        extent = [-1,1,-1,1] # Dummy values that will not be presented

    im = axes[0].imshow(im1, vmin=min_val, vmax=max_val, extent=extent)
    im = axes[1].imshow(im2, vmin=min_val, vmax=max_val, extent=extent)
    fig.colorbar(im, ax=axes[:2], fraction=0.046, pad=0.04)

    im = axes[2].imshow(diff, vmin=-diff_extent, vmax=diff_extent, cmap='seismic', extent=extent)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)


def plot_dose_dependent_strip(dose_levels_to_recon, crop=16):
    # plotting the diff images in a vertical strip
    # in each, plotting the base image as a small insert
    plt.figure(figsize=(3, len(dose_levels_to_recon.keys())*3))

    for i, (dose_level, (off_img, on_img)) in enumerate(dose_levels_to_recon.items()):
        on_img = on_img[crop:-crop, crop:-crop]
        off_img = off_img[crop:-crop, crop:-crop]
        diff = on_img - off_img
        diff_extent = np.nanmax(np.abs(diff))

        ax = plt.subplot(len(dose_levels_to_recon.keys()), 1, i+1)
        ax.set_title(f'Dose level: {dose_level}')
        ax.set_axis_off()

        im = ax.imshow(diff, vmin=-diff_extent, vmax=diff_extent, cmap='seismic')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # insert the low dose image as an inset
        inset_ax = ax.inset_axes([0, 0, 0.33, 0.33])
        inset_ax.imshow(on_img)
        inset_ax.set_axis_off()

    plt.show()