from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import tukey

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.imaging_utils import edge_filter


class TimeSeries(AutoSerialize):
    """
    TimeSeries is for aligning sequential 2D time series (in situ) data.

    This class supports Dataset3d input arrays (time, rows, columns).
    """

    _token = object()

    def __init__(
        self,
        data: NDArray,
        pad_shape: int | tuple[int,int],
        pad_val: str | int | float,
        blend_shape: int | tuple[int,int],
        _token: object | None = None,
    ):
        """
        Initialize a preprocessed (padded and edge_blended) 3D dataset (time, rows, columns).

        Parameters
        ----------
        data: NDArray
            The initial image stack 3D dataset (time, rows, columns) 
        pad_shape: int | tuple[int,int]
            Amount of padding for each row and column to apply to an image stack.
        pad_val: str | int | float
            Value for the padding background. If str is `mean`, `median`, `max`, 
            or `min`, the padding value will be calculated.
        blend_shape: int | tuple[int,int]
            Amount of edge blending in each row and column to apply to an image stack.
        _token: object | None = None
            Token to prevent direct instantiation, by default None.
        """
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
        pad_shape: int | tuple[int,int],
        pad_val: str | int | float,
        blend_shape: int | tuple[int,int],
    ):
        """
        Creates a time series Dataset3D from a 3D array.

        Parameters
        ----------
        data: NDArray
            The initial image stack 3D dataset (time, rows, columns) 
        pad_shape: int | tuple[int,int]
            Amount of padding for each row and column to apply to an image stack.
        pad_val: str | int | float
            Value for the padding background. If str is `mean`, `median`, `max`, 
            or `min`, the padding value will be calculated.
        blend_shape: int | tuple[int,int]
            Amount of edge blending in each row and column to apply to an image stack.

        Returns
        ----------
        series: cls
        A new instance of the class containing the preprocessed
        time-series dataset.
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(f'Data should be numpy array, got type {type(data)}')
        
        arr = np.asarray(data)

        if arr.ndim != 3:
            raise ValueError('Data must be a 3D array with shape 3 (time, rows, columns).') # probably wrong
        
        ds = Dataset3d.from_array(
            array = arr,
            name = 'TimeSeries.stack',
            origin = np.zeros(3),
            sampling = np.ones(3),
            units = ['index','pixels','pixels'],
        )

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

        series.preprocess()
        
        return series
    
    #TODO @ classmethod ... def from_file():

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
    def pad_shape(self) -> int | tuple[int, int]:
        return self._pad_shape
    
    @property
    def align_coords(self) -> NDArray:
        return self._align_coords
    
    @property
    def align_im(self) -> NDArray:
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
                    raise ValueError('pad_val should be a float, int, or str (supported ' \
                    'modes include mean, median, max, min).')
        
        else:
            raise TypeError('pad_val should be a float, int, or str (supported modes ' \
            'include mean, median, max, min).')

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
        """
        Applies padding and edge blending to an image stack.

        Parameters
        ----------
        modify_in_place: bool = True
            If True, modifies the dataset directly. If False, returns a new dataset.
        
        Returns
        ----------

        """

        pad_width = ((0,0),(0,2 * self.pad_shape[0]),(0,2 * self.pad_shape[1]))

        padded = np.pad(self.array, pad_width=pad_width, **kwargs)

        stack_pad = self.pad_val * np.ones(padded.shape)

        # Edge blend values
        if self.blend_shape is not None:
            # Create tukey window function for blending
            wr = tukey(self.shape[1],alpha=2*self.blend_shape[0]/self.shape[1])[None,:,None]
            wc = tukey(self.shape[2],alpha=2*self.blend_shape[1]/self.shape[2])[None,None,:]
            window = wr * wc

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

        return self.array


    def align_stack(
        self,
        running_average_frames: float | int = 20.0,
        correlation_power: int | None = 1.0,
        sigma_edge: int = 0.7,
        sf_val: int | tuple[int,int] = 0.5,
    ) -> NDArray:
        """
        Calculates all alignment coordinate shifts for an edge-filtered image 
        stack and applies the shifts to the unfiltered stack.

        running_average_frames: float | int = 20.0,
            Maximum number of images for the running average applied to the 
            reference. If None, default to 20.0.
        correlation_power: int | None = 1.0
            Option for either cross correlation (1.0) or phase correlation (0.0). 
            If None, defaults to 1.0 (cross correlation).
        sigma_edge: int = 0.7
            Standard deviation (sigma) of the 1D Gaussian kernel.
        sf_val: int | Tuple[int,int] = 0.5
            Scale factor(s) for the symmetric/asymmetric finite-difference gradient kernel.
        """
        
        stack_aligned = np.zeros_like(self.array)
        stack_aligned[0] = self.array[0]

        nframes, nx, ny = self.shape 
        kr_shift = -2 * np.pi * np.fft.fftfreq(nx)
        kc_shift = -2 * np.pi * np.fft.fftfreq(ny)
        rc_shift = np.zeros((nframes, 2))

        for a0 in range(0, nframes - 1):
            if a0 == 0:
                filtered_im_0 = edge_filter(
                    im = self.array[0],
                    sigma_edge = sigma_edge, 
                    sf_val = sf_val,    
                )

                G_filter_ref = np.fft.fft2(filtered_im_0)
            
            filtered_im = edge_filter(
                im = self.array[a0 + 1],
                sigma_edge = sigma_edge, 
                sf_val = sf_val,    
            )

            G_filter_a0 = np.fft.fft2(filtered_im)

            m = G_filter_ref * np.conj(G_filter_a0)
            if correlation_power < 1.0:
                im_corr = np.real(np.fft.ifft2(
                    (np.abs(m)**correlation_power) * np.exp(1j*np.angle(m))
                ))
            else:
                im_corr = np.real(np.fft.ifft2(m))
            
            # Coarse peak
            peak = np.unravel_index(np.argmax(im_corr), im_corr.shape)
            r0, c0 = peak

            # Parabolic refinement
            r_inds = np.mod(r0 + np.arange(-1, 2), m.shape[0]).astype(int)
            c_inds = np.mod(c0 + np.arange(-1, 2), m.shape[1]).astype(int)

            vr = im_corr[r_inds, c0]
            vc = im_corr[r0, c_inds]

            def parabolic_peak(v):
                return (v[2] - v[0]) / (4 * v[1] - 2 * v[2] - 2 * v[0])

            dr = parabolic_peak(vr)
            dc = parabolic_peak(vc)

            r0 = (r0 + dr) % m.shape[0]
            c0 = (c0 + dc) % m.shape[1]

            shifts = (r0, c0)
            shifts = (shifts + 0.5 * np.array(m.shape)) % m.shape - 0.5 * np.array(m.shape)
            rc_shift[a0 + 1] = shifts

            phase = np.exp(1j * (kr_shift[:, None] * shifts[0] + kc_shift[None, :] * shifts[1]))

            # Apply shift to original image
            G_a0 = np.fft.fft2(self.array[a0 + 1])

            im_shift = np.real(np.fft.ifft2(G_a0 * phase))
            stack_aligned[a0 + 1] = im_shift

            # Updating reference using running average
            weight = max(1 / (a0 + 1), 1 / running_average_frames)
            G_filter_ref = G_filter_ref * (1 - weight) + G_filter_a0 * phase * weight
            
        self._align_coords = rc_shift
        self._align_im = stack_aligned
