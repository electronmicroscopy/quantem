from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.signal.windows import tukey

from quantem.core.datastructures.dataset3d import Dataset3d


class TimeSeries(Dataset3d):
    """
    TimeSeries is for aligning sequential 2D time series (in situ) data. Includes option to padding and edge blending
    for the alignment process.

    This class supports Dataset3d input arrays (time, x, y).
    """

    def pad_and_blend(
        self,
        pad_width: int | tuple[int, int] | None,
        pad_val: str | int | None = 0,
        edge_blend: int | tuple[int, int] | None = (8,8),
        modify_in_place: bool = False,
    ) -> NDArray:
        """
        Generate a padded and/or edge blended image stack.

        Parameters
        ----------
        pad_width: int | tuple[int, int] | None
            Amount of padding in x and y to apply to an image stack. Defaults to None.
        pad_val: str | None = 0
            Value for the padding background. If str with `mean`, `median`, `max`, or `min`, the padding value 
            will be calculated. Defaults to 0.
        edge_blend: int | tuple[int, int] | None = (8,8)
            Amount of edge blending in x and y to apply to an image stack. Defaults in edge blending 8 in x and y.
        modify_in_place: bool = False
            If True, modifies the dataset directly. Defaults to False, returns a new dataset.
        
        Returns
        -------
        stack_pad: NDArray
            The padded image stack data (time, x, y).

        Raises
        ------
        ValueError
            If edge blending value is outside of allowable range.
            
        """
        # Padding value dictionary 
        agg_func = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min
        }

        if pad_val in agg_func:
            pad_val = agg_func[pad_val](self.array)
        
        # Padding values
        if pad_width is None:
            pad_width = np.zeros(2,dtype = 'int')
        else:
            pad_width = np.array(pad_width)
            if pad_width.size == 1:
                pad_width = np.array(
                    [pad_width,pad_width],
                )

        # Apply padding
        padded = self.pad(
            pad_width = ((0,0),(0,2 * pad_width[0]),(0,2 * pad_width[1])),
            modify_in_place = modify_in_place
        )

        # Initialize empty array
        stack_pad = pad_val * np.ones(padded.shape)

        # Edge blend values
        if edge_blend is not None:
            edge_blend = np.array(edge_blend)
            
            if edge_blend.size == 1:
                edge_blend = np.array(
                    [edge_blend,edge_blend],
                )
            
            # Create tukey window function for blending
            wx = tukey(self.shape[1],alpha=2*edge_blend[0]/self.shape[1])[None,:,None]
            wy = tukey(self.shape[2],alpha=2*edge_blend[1]/self.shape[2])[None,None,:]
            window = wx * wy

            # Combine tukey window and images
            stack_pad[
                :, 
                pad_width[0]:stack_pad.shape[1]-pad_width[0], 
                pad_width[1]:stack_pad.shape[2]-pad_width[1]] = \
                self.array * window + pad_val * (1 - window)
        else:
            # Add image to stack
            stack_pad[
                :, 
                pad_width[0]:stack_pad.shape[1]-pad_width[0], 
                pad_width[1]:stack_pad.shape[2]-pad_width[1]] = \
                self.array 

        # Ensure edge blend values are within range
        if edge_blend[0] < 0 or edge_blend[0] > stack_pad.shape[1] / 2:
            if edge_blend[1] < 0 or edge_blend[1] > stack_pad.shape[2] / 2:
                raise ValueError("edge_blend is outside of allowable range.")
        
        return stack_pad
        
    def align_image(
        im_ref: NDArray | Any,
        im: NDArray | Any,
        return_aligned_image = True,
        window_size: int | None = 7,            
    ) -> NDArray: 
        """
        Use 2D DFT cross correlation alignment on an image to a reference image.

        Parameters
        ----------
        im_ref: NDArray | Any
            First image to use as reference.
        im: NDArray | Any
            Second image to shift with respect to the reference.
        return_aligned_image = True
            If True, returns the shifted second image.
        window_size: int | None = 7
            Size of the window around the subpixel position. If None, defaults to 7.

        Returns
        -------
        im_shift: NDArray
            shifted image with respect to the shifted coordinates
        shift: NDArray
            shifted coordinates (x,y) between two images
        """
        # Take 2D DFT of an image (im) and the image before it as reference (im_ref)
        G_ref = np.fft.fft2(im_ref) 
        G_a0 = np.fft.fft2(im) 
        
        im_corr = np.real(np.fft.ifft2(G_ref * np.conj(G_a0)))

        # Index position of the maximum  
        nrows, ncols = im_corr.shape
        peak_r, peak_c = np.unravel_index(np.argmax(im_corr), im_corr.shape)
        
        # Constants
        half = window_size // 2

        # Build periodic window
        # Padding of the extra pixels
        r_idx = np.mod(np.arange(peak_r - half, peak_r + half + 1), nrows)
        c_idx = np.mod(np.arange(peak_c - half, peak_c + half + 1), ncols)
        window = im_corr[np.ix_(r_idx, c_idx)]

        # Center of mass within window
        com_r, com_c = ndimage.center_of_mass(window)

        # Adjust back to full image coordinates (with periodicity)
        shift_row = (((peak_r - half + com_r) + nrows/2) % nrows) - nrows/2
        shift_col = (((peak_c - half + com_c) + ncols/2) % ncols) - ncols/2
        shift = np.array((shift_row,shift_col))

        if return_aligned_image:
            kx_shift = -2*np.pi*np.fft.fftfreq(im.shape[0])
            ky_shift = -2*np.pi*np.fft.fftfreq(im.shape[1])
            im_shift = np.real(np.fft.ifft2(
                G_a0 * np.exp(1j * (kx_shift[:,None] * shift[0] + ky_shift[None,:] * shift[1]))
            ))
            return im_shift, shift
        else:
            return shift
        

    def align_stack(
        stack_pad: NDArray | Any, 
        window_size: int | None = 7,
        running_average_frames: float | int = 20.0,
        correlation_power = 1.0,
    ) -> NDArray:
        """
        Use 2D DFT cross correlation alignment on an entire image stack.

        Parameters
        ----------
        stack_pad: NDArray | Any
            Unaligned image stack data (time, x, y).
        window_size: int| None = 7
            Size of the window around the subpixel position. If None, defaults to 7.
        running_average_frames: float | int = 20.0
            Maximum number of images for the running average applied to the reference. If None, default is 20.0.
        correlation_power: int | None = 1.0
            Option for either cross correlation (1.0) or phase correlation (0.0). If None, defaults to 1.0 (cross correlation)
            
        Returns
        -------
        stack_aligned: NDArray
            Aligned image stack data (time, x, y).
        xy_shift: NDArray
            Shifted coordinates (x, y) for all images.
        """
        # Initializing aligned stack and shifts
        nframes, nx, ny = stack_pad.shape
        stack_aligned = np.zeros_like(stack_pad)
        stack_aligned[0] = stack_pad[0]
        xy_shift = np.zeros((nframes, 2))

        # Angular frequencies
        kx_shift = -2 * np.pi * np.fft.fftfreq(nx)
        ky_shift = -2 * np.pi * np.fft.fftfreq(ny)

        # Precomputing FFTs for all frames
        G_all = np.fft.fft2(stack_pad, axes=(1, 2))
        G_ref = G_all[0].copy()

        # Constants
        half = window_size // 2
        nrows, ncols = nx, ny

        for a0 in range(1, nframes):
            G_a0 = G_all[a0]
            if correlation_power < 1.0:
                m = G_ref * np.conj(G_a0)
                im_corr = np.real(np.fft.ifft2(
                    (np.abs(m)**correlation_power) * np.exp(1j*np.angle(m))
                ))
            else:
                im_corr = np.real(np.fft.ifft2(G_ref * np.conj(G_a0)))

            # Get subpixel position
            peak_r, peak_c = np.unravel_index(np.argmax(im_corr), im_corr.shape)
            
            # Padding of the extra pixels
            r_idx = np.mod(np.arange(peak_r - half, peak_r + half + 1), nrows)
            c_idx = np.mod(np.arange(peak_c - half, peak_c + half + 1), ncols)
            window = im_corr[np.ix_(r_idx, c_idx)]

            # Center of mass within window
            com_r, com_c = ndimage.center_of_mass(window)

            # Adjust back to full image coordinates (with periodicity)
            shift_row = (((peak_r - half + com_r) + nrows / 2) % nrows) - nrows / 2
            shift_col = (((peak_c - half + com_c) + ncols / 2) % ncols) - ncols / 2
            shift = np.array((shift_row, shift_col))
            xy_shift[a0] = shift

            # Apply shift
            phase = np.exp(1j * (kx_shift[:, None] * shift[0] + ky_shift[None, :] * shift[1]))
            im_shift = np.real(np.fft.ifft2(G_a0 * phase))
            stack_aligned[a0] = im_shift

            # Updating reference using running average
            weight = max(1 / (a0 + 1), 1 / running_average_frames)
            G_ref = G_ref * (1 - weight) + G_a0 * phase * weight

        return stack_aligned, xy_shift