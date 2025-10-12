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
        pad_width: Union[int, tuple[int, int]],
        pad_val = None,
        edge_blend: Union[int, tuple[int, int], None] = None,
        # modify_in_place: bool = True,
    ):
        # padding values
        pad_x = 2*pad_width[0]
        if len(pad_width) == 1:
            pad_y = pad_x
        else:
            pad_y = 2*pad_width[1]

        # apply padding
        padded = self.pad(
            pad_width = ((0,0),(0,pad_x),(0,pad_y)),
            # modify_in_place = modify_in_place
        )

        #blend edge
        wx = tukey(self.shape[1],alpha=2*edge_blend[0]/self.shape[1])[None,:,None]
        wy = tukey(self.shape[2],alpha=2*edge_blend[1]/self.shape[2])[None,None,:]
        window = wx * wy

        stack_pad = pad_val * np.ones(padded.shape)
        # print(pad_val)

        
        # print(stack_pad[:, pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]])
        print(stack_pad.shape, self.shape, window.shape)
        
        r = stack_pad[:, pad_width[0]:-1*pad_width[0], pad_width[1]:-1*pad_width[1]]
        s = self.array * window

        print(r.shape, s.shape)

        stack_pad[:, pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]] = \
            self.array * window + pad_val * (1 - window)


        return stack_pad









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






