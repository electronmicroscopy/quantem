from collections.abc import Sequence
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d


class TimeSeries(Dataset3d):

    def pad_images(
        self,
        pad_width: Union[int, tuple[int, int], None],
        pad_val = None,
        edge_blend: Union[int, tuple[int, int], None] = (8,8),
        modify_in_place: bool = False,
    ):
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

        return stack_pad

# cap edge blending to 0-half-width (stack_pad.shape)







        # print(self.shape, window.shape)


        # stack_pad = pad_val * np.ones(s_pad.shape)

        # print(stack_pad.shape)

        # # blend edge
        # wx = tukey(s_pad.shape[1],alpha=2*edge_blend[0]/s_pad.shape[1])[None,:,None]
        # wy = tukey(s_pad.shape[2],alpha=2*edge_blend[1]/s_pad.shape[2])[None,None,:]
        # window = wx * wy
   
        # print(wx.shape, wy.shape)
        # print(window)

        



        # shift to center


#eventually center the image.... 






