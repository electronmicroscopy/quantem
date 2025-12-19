from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d
from scipy.signal.windows import tukey

from quantem.core.datastructures.dataset import Dataset
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize

# add autoserialize from quantem

# from quantem.core.utils.imaging_utils import ImagingUtils

class TimeSeries(AutoSerialize):
    """
    TimeSeries is for aligning sequential 2D time series (in situ) data.

    This class supports Dataset3d input arrays (time, x, y).
    """

    _token = object()

    def __init__(
        self,
        data,
        pad_shape,
        pad_val,
        blend_shape,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use TimeSeries.from_array() or .from_data() to instantiate this class."
            )

        self._data = data
        self._orig_shape = self.data.shape[:]
        self._pad_shape = pad_shape
        self._pad_val = pad_val
        self._blend_shape = blend_shape


    @classmethod
    def from_array(
        cls, 
        data: NDArray, 
        pad_shape,
        pad_val,
        blend_shape,
    ):
        if not isinstance(data, np.ndarray):
            raise TypeError(f'Data should be numpy array, got type {type(data)}')
        
        arr = np.asarray(data)

        if arr.ndim != 3:
            raise ValueError('Data must be a 3D array with shape 3 (time, rows, columns).') # probably wrong
        
        # creating a Dataset3d from array (calls dataset3d.py)
        ds = Dataset3d.from_array(
            array = arr,
            name = 'TimeSeries.stack',
            origin = np.zeros(3),
            sampling = np.ones(3),
            units = ['index','pixels','pixels'],
        )

        # creating a series class with the newly created Dataset3d (calls _init_)
        series = cls(
            data=ds,
            pad_shape = pad_shape,
            pad_val = pad_val,
            blend_shape = blend_shape,
            _token=cls._token,
        )

        series.pad_shape = pad_shape
        series.pad_val = pad_val
        series.blend_shape = blend_shape

        # apply preprocessing
        series.preprocess()
        
        return series
    
    # @classmethod
    # def from_file():


    @property
    def data(self) -> Dataset3d:
        return self._data

    @property
    def orig_shape(self) -> tuple[int, int, int]:
        return self._orig_shape
    
    @property
    def array(self) -> NDArray:
        return self._data.array
    
    @property
    def shape(self) -> tuple[int, int, int]:
        return self._data.array.shape
    
    @property
    def n_frames(self) -> int:
        return int(self.shape[0])
    
    @property
    def filtered(self):
        return self._filtered
    
    @property
    def pad_shape(self) -> int | tuple[int, int]:
        return self._pad_shape
    
    @property
    def align_coords(self):
        return self._align_coords
    
    @property
    def align_im(self):
        return self._align_im

    @pad_shape.setter
    def pad_shape(self, value):
        if isinstance(value, (int, np.integer)):
            pad_r = pad_c = int(value)

        else:
            if not isinstance(value, tuple) and len(value) == 2:
                raise TypeError('pad_shape should be an int value or 2-value int sequence.')
            
            if len(value) != 2:
                raise TypeError('pad_shape int sequence should have length 2.')
            
            pad_r = int(value[0])
            pad_c = int(value[1])
            
            
        if pad_r < 0 or pad_c < 0:
            raise TypeError('pad_shape should have non-negative int values.')
        
        self._pad_shape = (pad_r, pad_c)

    @property
    def pad_val(self) -> str | int | float:
        return self._pad_val

    @pad_val.setter
    def pad_val(self, value: str | int | float ):

        if isinstance(value, (float, int)):
            value_flt = float(value)

        elif isinstance(value, str):
            pad_mode = value.lower()
            
            modes = {
                'mean': np.mean,
                'median': np.median,
                'max': np.max,
                'min': np.min
            }

            if pad_mode in modes:
                value_flt = float(modes[pad_mode](self.data.array))

            elif str(value) not in modes:
                try:
                    value_flt = float(value)
                
                except ValueError:
                    raise ValueError('pad_val should be a float, int, or str (supported modes include mean, median, max, min).')
        
        else:
            raise TypeError('pad_val should be a float, int, or str (supported modes include mean, median, max, min).')


        if value_flt < 0:
            raise ValueError('pad_val should be a non-negative value.')
            
        self._pad_val = value_flt
            
    @property
    def blend_shape(self) -> tuple[int,int]:
        return self._blend_shape

    @blend_shape.setter
    def blend_shape(self, value):
        if isinstance(value, (int, float)):
            blend_r = blend_c = int(value)

        elif isinstance(value, tuple) and len(value) == 2:
            blend_r, blend_c = map(int, value)

        else: 
            raise TypeError('blend_shape should be an int value or 2-value int sequence.')

        if blend_r < 0 or blend_c < 0:
            raise TypeError('pad_shape should have non-negative int values.')
        
        self._blend_shape = (blend_r, blend_c)

    
    def preprocess(
        self, 
        modify_in_place: bool = True,
        **kwargs: Any,
        ):
        # call from imaging_utils.pad and imaging_utils.blend_shape
        
        print(self.pad_shape)

        pad_width = ((0,0),(0,2 * self.pad_shape[0]),(0,2 * self.pad_shape[1]))

        padded = np.pad(self.array, pad_width=pad_width, **kwargs)

        stack_pad = self.pad_val * np.ones(padded.shape)

        # Edge blend values
        if self.blend_shape is not None:
            # Create tukey window function for blending
            wx = tukey(self.shape[1],alpha=2*self.blend_shape[0]/self.shape[1])[None,:,None]
            wy = tukey(self.shape[2],alpha=2*self.blend_shape[1]/self.shape[2])[None,None,:]
            window = wx * wy

            # Combine tukey window and images
            stack_pad[
                :, 
                self.pad_shape[0]:stack_pad.shape[1]-self.pad_shape[0], 
                self.pad_shape[1]:stack_pad.shape[2]-self.pad_shape[1]] = \
                self.array * window + self.pad_val * (1 - window)
        else:
            # Add image to stack
            stack_pad[
                :, 
                self.pad_shape[0]:stack_pad.shape[1]-self.pad_shape[0], 
                self.pad_shape[1]:stack_pad.shape[2]-self.pad_shape[1]] = \
                self.array         

        if modify_in_place is True:
            self._data.array = stack_pad


        return stack_pad

    def edge_filtering(
            self,
            sigma_edge,
            sf_val,
            modify_in_place: bool = True,
            ):
        """
        """

        edge_stack = np.zeros_like(self.array)
        edge_stack[0] = self.array[0]
        
        sf_val = np.array(sf_val)
        if sf_val.size == 1:
            sf_val = np.array(
                [sf_val, sf_val],
                )

        r = np.arange(
            -np.ceil(2.0*sigma_edge),
            np.ceil(2.0*sigma_edge+1),
        )

        k = np.exp(
            (r[:,None]**2) / (-2*sigma_edge**2)
        )

        sf = np.array([
            [-sf_val[0],0,sf_val[1]],
        ])

        for i in range(len(self.array)):

            im_x = convolve2d(self.array[i], sf, mode='same', boundary='symm')
            im_x = convolve2d(im_x, k, mode='same', boundary='symm')
            im_x = convolve2d(im_x, k.T, mode='same', boundary='symm')

            im_y = convolve2d(self.array[i], sf.T, mode='same', boundary='symm')
            im_y = convolve2d(im_y, k, mode='same', boundary='symm')
            im_y = convolve2d(im_y, k.T, mode='same', boundary='symm')

            im_edge = np.sqrt(im_x**2 + im_y**2)

            edge_stack[i] = im_edge

        if modify_in_place is True:
            self._filtered = edge_stack

        return im_edge


    def align_stack(
        self,
        running_average_frames: float | int = 20.0,
        correlation_power = 1.0,
        edge_prefilter = False,
        # edge_width = 1.0,
        # upsample_factor: int = 1,
    ) -> NDArray:
        """
        """
        
        if edge_prefilter is True:
            stack_pad = self.filtered
        else:
            stack_pad = self.array
        


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

        #filtering image > FFT both images iteratively (within for loop)

        for a0 in range(1, nframes):
            G_a0 = G_all[a0]
            m = G_ref * np.conj(G_a0)
            if correlation_power < 1.0:
                im_corr = np.real(np.fft.ifft2(
                    (np.abs(m)**correlation_power) * np.exp(1j*np.angle(m))
                ))
            else:
                im_corr = np.real(np.fft.ifft2(m))

            # Coarse peak
            peak = np.unravel_index(np.argmax(im_corr), im_corr.shape)
            x0, y0 = peak

            # Parabolic refinement
            x_inds = np.mod(x0 + np.arange(-1, 2), m.shape[0]).astype(int)
            y_inds = np.mod(y0 + np.arange(-1, 2), m.shape[1]).astype(int)

            vx = im_corr[x_inds, y0]
            vy = im_corr[x0, y_inds]

            def parabolic_peak(v):
                return (v[2] - v[0]) / (4 * v[1] - 2 * v[2] - 2 * v[0])

            dx = parabolic_peak(vx)
            dy = parabolic_peak(vy)

            # print(dx,dy)

            x0 = (x0 + dx) % m.shape[0]
            y0 = (y0 + dy) % m.shape[1]

            # print('x0,y0', x0, y0)

            shifts = (x0, y0)
            shifts = (shifts + 0.5 * np.array(m.shape)) % m.shape - 0.5 * np.array(m.shape)

            xy_shift[a0] = shifts

            # Apply shift
            phase = np.exp(1j * (kx_shift[:, None] * shifts[0] + ky_shift[None, :] * shifts[1]))
            im_shift = np.real(np.fft.ifft2(G_a0 * phase))
            stack_aligned[a0] = im_shift

            # Updating reference using running average
            weight = max(1 / (a0 + 1), 1 / running_average_frames)
            G_ref = G_ref * (1 - weight) + G_a0 * phase * weight

        # say time series with you edge pad it... if you know all shifts, then you can center after knowing all the shifts from edge belending of actual image
        # separate finding dxy shifts from applying them
        #all sigma edge for alignment and for centering (separately)



        self._align_coords = xy_shift
        self._align_im = stack_aligned

    
    
    #TODO use coordinates from edge filtered stack to real stack
    
    
    
    
    
    
    
    
    
    
    
    
    # def align_stack(
    #     self,
    #     window_size: int | None = 7,
    #     running_average_frames: float | int = 20.0,
    #     correlation_power = 1.0,
    #     edge_prefilter = False,
    #     edge_width = 1.0,
    #     # upsample_factor: int = 1,
    # ) -> NDArray:
    #     """
    #     Use 2D DFT cross correlation alignment on an entire image stack.

    #     Parameters
    #     ----------
    #     stack_pad: NDArray | Any
    #         Unaligned image stack data (time, x, y).
    #     window_size: int| None = 7
    #         Size of the window around the subpixel position. If None, defaults to 7.
    #     running_average_frames: float | int = 20.0
    #         Maximum number of images for the running average applied to the reference. If None, default is 20.0.
    #     correlation_power: int | None = 1.0
    #         Option for either cross correlation (1.0) or phase correlation (0.0). If None, defaults to 1.0 (cross correlation)
            
    #     Returns
    #     -------
    #     stack_aligned: NDArray
    #         Aligned image stack data (time, x, y).
    #     xy_shift: NDArray
    #         Shifted coordinates (x, y) for all images.
    #     """
    #     # Initializing aligned stack and shifts
    #     nframes, nx, ny = stack_pad.shape
    #     stack_aligned = np.zeros_like(self.array)
    #     stack_aligned[0] = self.array[0]
    #     xy_shift = np.zeros((nframes, 2))

    #     # Angular frequencies
    #     kx_shift = -2 * np.pi * np.fft.fftfreq(nx)
    #     ky_shift = -2 * np.pi * np.fft.fftfreq(ny)

    #     # Precomputing FFTs for all frames
    #     G_all = np.fft.fft2(stack_pad, axes=(1, 2))
    #     G_ref = G_all[0].copy()

    #     # Constants
    #     half = window_size // 2
    #     nrows, ncols = nx, ny

    #     for a0 in range(1, nframes):
    #         G_a0 = G_all[a0]
    #         m = G_ref * np.conj(G_a0)
    #         if correlation_power < 1.0:
    #             im_corr = np.real(np.fft.ifft2(
    #                 (np.abs(m)**correlation_power) * np.exp(1j*np.angle(m))
    #             ))
    #         else:
    #             im_corr = np.real(np.fft.ifft2(m))

    #         # Coarse peak
    #         peak = np.unravel_index(np.argmax(im_corr), im_corr.shape)
    #         x0, y0 = peak

    #         # Parabolic refinement
    #         x_inds = np.mod(x0 + np.arange(-1, 2), m.shape[0]).astype(int)
    #         y_inds = np.mod(y0 + np.arange(-1, 2), m.shape[1]).astype(int)

    #         vx = im_corr[x_inds, y0]
    #         vy = im_corr[x0, y_inds]

    #         def parabolic_peak(v):
    #             return (v[2] - v[0]) / (4 * v[1] - 2 * v[2] - 2 * v[0])

    #         dx = parabolic_peak(vx)
    #         dy = parabolic_peak(vy)

    #         # print(dx,dy)

    #         x0 = (x0 + dx) % m.shape[0]
    #         y0 = (y0 + dy) % m.shape[1]

    #         # print('x0,y0', x0, y0)

    #         shifts = (x0, y0)
    #         shifts = (shifts + 0.5 * np.array(m.shape)) % m.shape - 0.5 * np.array(m.shape)

    #         xy_shift[a0] = shifts

    #         # Apply shift
    #         phase = np.exp(1j * (kx_shift[:, None] * shifts[0] + ky_shift[None, :] * shifts[1]))
    #         im_shift = np.real(np.fft.ifft2(G_a0 * phase))
    #         stack_aligned[a0] = im_shift

    #         # Updating reference using running average
    #         weight = max(1 / (a0 + 1), 1 / running_average_frames)
    #         G_ref = G_ref * (1 - weight) + G_a0 * phase * weight

    #     return stack_aligned, xy_shift



    # def align(
    #     self,
    #     window_size,
    #     running_average_frames,
    #     correlation_power,
    #     ):
    #     # call from imaging_utils.edge_filtering and imaging_utils.cross_correlation_shift
    #     raise NotImplementedError





        # if dataset is already loaded in as an array

        # validate all pad arguments

        # series = cls(...)

    # @classmethod
    # def from_file(cls, file_path, file_type, *pad_args):
    #     # if loading dataset directly from a file (.h5, .png stack, video stack, etc.)
    #     # make Dataset3D.from_file(), similar to Dataset2D.from_file > add to quantem.core.io.file_readers for stacks, videos, etc.

    #     # validate all pad arguments

    #     # series = cls(...)



