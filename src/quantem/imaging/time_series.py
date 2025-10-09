from collections.abc import Sequence
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
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
        edge_blend: Union[int, tuple[int, int], None] = None,
        modify_in_place: bool = True,
    ):
        # padding values
        pad_x = pad_width[0]
        if len(pad_width) == 1:
            pad_y = pad_x
        else:
            pad_y = pad_width[1]

        # apply padding
        self.pad(
            pad_width = ((0,0),(0,pad_x),(0,pad_y)),
            modify_in_place = modify_in_place,
        )

        # blend edge



        # shift to center

#ref pad_value in drift.py


    def align_images(
        self,
    ):
        pass


#eventually center the image.... 








