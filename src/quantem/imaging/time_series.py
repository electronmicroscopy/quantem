from collections.abc import Sequence
from typing import List, Optional, Union

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm
from scipy import ndimage
from IPython.display import display
import plotly.graph_objects as go

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d


class TimeSeries(Dataset3d):

    def pad_images(
        self,
        pad_width: Union[int, tuple[int, int], None],
        pad_val: [str , None] = None,
        edge_blend: Union[int, tuple[int, int], None] = (8,8),
        modify_in_place: bool = False,
    ):

        # padding value dictionary 
        agg_func = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min
        }

        if pad_val in agg_func:
            pad_val = agg_func[pad_val](self.array)
        elif pad_val is None:
            pad_val = 0  
        
        # padding values
        if pad_width is None:
            pad_width = np.zeros(2,dtype = 'int')
        else:
            pad_width = np.array(pad_width)
            if pad_width.size == 1:
                pad_width = np.array(
                    [pad_width,pad_width],
                )

        # apply padding
        padded = self.pad(
            pad_width = ((0,0),(0,2 * pad_width[0]),(0,2 * pad_width[1])),
            modify_in_place = modify_in_place
        )

        # initialize empty array
        stack_pad = pad_val * np.ones(padded.shape)

        # edge blend values
        if edge_blend is not None:
            edge_blend = np.array(edge_blend)
            
            if edge_blend.size == 1:
                edge_blend = np.array(
                    [edge_blend,edge_blend],
                )
            
            # create tukey window function for blending
            wx = tukey(self.shape[1],alpha=2*edge_blend[0]/self.shape[1])[None,:,None]
            wy = tukey(self.shape[2],alpha=2*edge_blend[1]/self.shape[2])[None,None,:]
            window = wx * wy

            # combine tukey window and images
            stack_pad[
                :, 
                pad_width[0]:stack_pad.shape[1]-pad_width[0], 
                pad_width[1]:stack_pad.shape[2]-pad_width[1]] = \
                self.array * window + pad_val * (1 - window)
        else:
            # add image to stack
            stack_pad[
                :, 
                pad_width[0]:stack_pad.shape[1]-pad_width[0], 
                pad_width[1]:stack_pad.shape[2]-pad_width[1]] = \
                self.array 
        
        print(stack_pad.shape)

        # ensure edge blend values are within range
        if edge_blend[0] < 0 or edge_blend[0] > stack_pad.shape[1] / 2:
            if edge_blend[1] < 0 or edge_blend[1] > stack_pad.shape[2] / 2:
                raise ValueError("edge_blend is outside of allowable range.")
        
        return stack_pad
        
    def align_images(
        im_ref,
        im,
        return_aligned_image=True,
        window_size = 7,            
    ):
        """
        Using 2D cross correlation on an entire image stack.

        Parameters
        ----------
        stack_pad
            The unaligned image stack data
        window_size
            The size of the window around the subpixel position. If None, default is 7

        Returns
        -------
        shift
            shifted coordinates (x,y)
        im_shift
            shifted image with respect to the shifted coordinates
        """
        
        # take 2D DFT of an image (im) and the image before it as reference (im_ref)
        G_ref = np.fft.fft2(im_ref) 
        G = np.fft.fft2(im) 
        
        im_corr = np.real(np.fft.ifft2(G_ref * np.conj(G)))

        # index position of the maximum 
        nrows, ncols = im_corr.shape
        peak_r, peak_c = np.unravel_index(np.argmax(im_corr), im_corr.shape)
        half = window_size // 2

        # TODO replace COM with DFT upsampling (already implemented in quantem)

        # Build periodic window

        # padding of the extra pixels
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
                G * np.exp(1j * (kx_shift[:,None] * shift[0] + ky_shift[None,:] * shift[1]))
            ))
            return shift,im_shift

        else:
            return shift
        

    def align_stack(
        stack_pad, 
        window_size = 7,
        running_average_frames = 20.0,
    ):
        """
        Using 2D cross correlation on an entire image stack.

        Parameters
        ----------
        stack_pad
            The unaligned image stack data.
        window_size
            The size of the window around the subpixel position. If None, default is 7.
        running_average_frames = 20.0
            ... If None, default is 20.0.

        """

        # initialize aligned stack
        stack_aligned = np.zeros_like(stack_pad)
        
        # indexing the aligned stack the same as the padded stack
        stack_aligned[0] = stack_pad[0]
        
        # initialize shifted 
        dxy = np.zeros((stack_pad.shape[0],2))

        # angular frequencies
        kx_shift = -2*np.pi*np.fft.fftfreq(stack_pad.shape[1])
        ky_shift = -2*np.pi*np.fft.fftfreq(stack_pad.shape[2])
        G_ref = np.fft.fft2(stack_pad[0])
        
        for a0 in range(1,stack_pad.shape[0]):
            G = np.fft.fft2(stack_pad[a0])
            im_corr = np.real(np.fft.ifft2(G_ref * np.conj(G)))

            # Get subpixel shift
                                 
            # Get subpixel position
            nrows, ncols = im_corr.shape
            peak_r, peak_c = np.unravel_index(np.argmax(im_corr), im_corr.shape)
            half = window_size // 2

            # Build periodic window
            r_idx = np.mod(np.arange(peak_r - half, peak_r + half + 1), nrows)
            c_idx = np.mod(np.arange(peak_c - half, peak_c + half + 1), ncols)
            window = im_corr[np.ix_(r_idx, c_idx)]

            # Center of mass within window
            com_r, com_c = ndimage.center_of_mass(window)

            # Adjust back to full image coordinates (with periodicity)
            shift_row = (((peak_r - half + com_r) + nrows/2) % nrows) - nrows/2
            shift_col = (((peak_c - half + com_c) + ncols/2) % ncols) - ncols/2
            shift = np.array((shift_row,shift_col))

            dxy[a0] = shift

            # Aligned image
            G_shift = G * np.exp(1j * (kx_shift[:,None] * shift[0] + ky_shift[None,:] * shift[1]))
            im_shift = np.real(np.fft.ifft2(G_shift))

            stack_aligned[a0] = im_shift
            weight = np.maximum(1/(a0+1),1/running_average_frames)
            G_ref = G_ref * (1-weight) + G_shift * weight

        return stack_aligned, dxy
    
def play_stack_plotly(stack, fps=30, vmin=None, vmax=None, cmap='inferno'):
    """
    Play a stack of images as a movie using plotly, with play/pause and frame slider.

    Parameters:
    -----------
    stack : np.ndarray
        Image stack with shape (num_frames, height, width).
    fps : int, optional
        Frames per second for playback. Default is 30.
    vmin : float, optional
        Minimum intensity for display. Defaults to stack min.
    vmax : float, optional
        Maximum intensity for display. Defaults to stack max.
    cmap : str, optional
        Colormap for display. Default is 'gray'.
    """
    if vmin is None:
        vmin = float(np.min(stack))
    if vmax is None:
        vmax = float(np.max(stack))

    frames = [
        go.Frame(
            data=[go.Heatmap(z=frame, colorscale=cmap, zmin=vmin, zmax=vmax, showscale=False)],
            name=str(k)
        ) for k, frame in enumerate(stack)
    ]

    fig = go.Figure(
        data=[go.Heatmap(z=stack[0], colorscale=cmap, zmin=vmin, zmax=vmax, showscale=False)],
        frames=frames
    )

    fig.update_layout(
        width=600, height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 1000/fps}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ],
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {"label": str(k), "method": "animate", "args": [[str(k)], {"frame": {"duration": 0}, "mode": "immediate"}]}
                for k in range(len(stack))
            ],
        }]
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    display(fig)

    return fig














