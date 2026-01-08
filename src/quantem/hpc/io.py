"""
GPU-accelerated HDF5 loading and binning for 4D-STEM diffraction data.

Provides high-performance bitshuffle+LZ4 decompression directly to GPU memory.

Examples
--------
>>> from quantem.hpc import load, bin
>>> data = load('/path/to/file.h5')
>>> binned = bin(data, factor=2)
"""

from __future__ import annotations

import cupy as cp
import h5py
import hdf5plugin  # noqa: F401 - registers bitshuffle filter
import numpy as np
from numba import njit, prange

__all__ = ["GPUDecompressor", "load", "clear_memory", "bin"]

# CUDA LZ4 decompression kernel (adapted from NVIDIA nvcomp, BSD-3-Clause)
_CUDA_LZ4_SOURCE = r'''
/*
 * LZ4 decompression kernel extracted from NVIDIA nvcomp
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 * BSD-3-Clause License
 */
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned long size_t;

using offset_type = uint16_t;
using word_type = uint32_t;
using position_type = uint32_t;
using double_word_type = uint64_t;
using item_type = uint32_t;

constexpr const int DECOMP_THREADS_PER_CHUNK = 32;
constexpr const int DECOMP_CHUNKS_PER_BLOCK = 2;
constexpr const position_type DECOMP_INPUT_BUFFER_SIZE
    = DECOMP_THREADS_PER_CHUNK * sizeof(double_word_type);
constexpr const position_type DECOMP_BUFFER_PREFETCH_DIST
    = DECOMP_INPUT_BUFFER_SIZE / 2;

inline __device__ void syncCTA() {
    if (DECOMP_THREADS_PER_CHUNK > 32) __syncthreads();
    else __syncwarp();
}

inline __device__ int warpBallot(int vote) {
    return __ballot_sync(0xffffffff, vote);
}

inline __device__ offset_type readWord(const uint8_t* const address) {
    offset_type word = 0;
    for (size_t i = 0; i < sizeof(offset_type); ++i)
        word |= address[i] << (8 * i);
    return word;
}

struct token_type {
    position_type num_literals;
    position_type num_matches;

    __device__ bool hasNumLiteralsOverflow() const { return num_literals >= 15; }
    __device__ bool hasNumMatchesOverflow() const { return num_matches >= 19; }

    __device__ position_type numLiteralsOverflow() const {
        return hasNumLiteralsOverflow() ? num_literals - 15 : 0;
    }
    __device__ uint8_t numLiteralsForHeader() const {
        return hasNumLiteralsOverflow() ? 15 : num_literals;
    }
    __device__ position_type numMatchesOverflow() const {
        return hasNumMatchesOverflow() ? num_matches - 19 : 0;
    }
    __device__ uint8_t numMatchesForHeader() const {
        return hasNumMatchesOverflow() ? 15 : num_matches - 4;
    }
    __device__ position_type lengthOfLiteralEncoding() const {
        if (hasNumLiteralsOverflow()) {
            const position_type num = numLiteralsOverflow();
            return (num / 0xff) + 1;
        }
        return 0;
    }
    __device__ position_type lengthOfMatchEncoding() const {
        if (hasNumMatchesOverflow()) {
            const position_type num = numMatchesOverflow();
            return (num / 0xff) + 1;
        }
        return 0;
    }
};

class BufferControl {
public:
    __device__ BufferControl(uint8_t* const buffer, const uint8_t* const compData,
                             const position_type length)
        : m_offset(0), m_length(length), m_buffer(buffer), m_compData(compData) {}

    inline __device__ position_type readLSIC(position_type& idx) const {
        position_type num = 0;
        uint8_t next = 0xff;
        while (next == 0xff && idx < end()) {
            next = rawAt(idx)[0];
            ++idx;
            num += next;
        }
        while (next == 0xff) {
            next = m_compData[idx];
            ++idx;
            num += next;
        }
        return num;
    }

    inline __device__ const uint8_t* raw() const { return m_buffer; }
    inline __device__ const uint8_t* rawAt(const position_type i) const {
        return raw() + (i - begin());
    }

    inline __device__ uint8_t operator[](const position_type i) const {
        if (i >= m_offset && i - m_offset < DECOMP_INPUT_BUFFER_SIZE)
            return m_buffer[i - m_offset];
        return m_compData[i];
    }

    inline __device__ void setAndAlignOffset(const position_type offset) {
        const uint8_t* const alignedPtr = reinterpret_cast<const uint8_t*>(
            (reinterpret_cast<size_t>(m_compData + offset) / sizeof(double_word_type))
            * sizeof(double_word_type));
        m_offset = alignedPtr - m_compData;
    }

    inline __device__ void loadAt(const position_type offset) {
        setAndAlignOffset(offset);
        if (m_offset + DECOMP_INPUT_BUFFER_SIZE <= m_length) {
            const double_word_type* const word_data
                = reinterpret_cast<const double_word_type*>(m_compData + m_offset);
            double_word_type* const word_buffer
                = reinterpret_cast<double_word_type*>(m_buffer);
            word_buffer[threadIdx.x] = word_data[threadIdx.x];
        } else {
            #pragma unroll
            for (int i = threadIdx.x; i < DECOMP_INPUT_BUFFER_SIZE;
                 i += DECOMP_THREADS_PER_CHUNK) {
                if (m_offset + i < m_length)
                    m_buffer[i] = m_compData[m_offset + i];
            }
        }
        syncCTA();
    }

    inline __device__ position_type begin() const { return m_offset; }
    inline __device__ position_type end() const { return m_offset + DECOMP_INPUT_BUFFER_SIZE; }

private:
    int64_t m_offset;
    const position_type m_length;
    uint8_t* const m_buffer;
    const uint8_t* const m_compData;
};

inline __device__ void coopCopyNoOverlap(uint8_t* const dest, const uint8_t* const source,
                                         const position_type length) {
    for (position_type i = threadIdx.x; i < length; i += blockDim.x)
        dest[i] = source[i];
}

inline __device__ void coopCopyRepeat(uint8_t* const dest, const uint8_t* const source,
                                      const position_type dist, const position_type length) {
    for (position_type i = threadIdx.x; i < length; i += blockDim.x)
        dest[i] = source[i % dist];
}

inline __device__ void coopCopyOverlap(uint8_t* const dest, const uint8_t* const source,
                                       const position_type dist, const position_type length) {
    if (dist < length) coopCopyRepeat(dest, source, dist, length);
    else coopCopyNoOverlap(dest, source, length);
}

inline __device__ token_type decodePair(const uint8_t num) {
    return token_type{static_cast<uint8_t>((num & 0xf0) >> 4),
                      static_cast<uint8_t>(num & 0x0f)};
}

inline __device__ void decompressStream(uint8_t* buffer, uint8_t* decompData,
                                        const uint8_t* compData, const position_type comp_end) {
    BufferControl ctrl(buffer, compData, comp_end);
    ctrl.loadAt(0);
    position_type decomp_idx = 0;
    position_type comp_idx = 0;

    while (comp_idx < comp_end) {
        if (comp_idx + DECOMP_BUFFER_PREFETCH_DIST > ctrl.end())
            ctrl.loadAt(comp_idx);

        token_type tok = decodePair(*ctrl.rawAt(comp_idx));
        ++comp_idx;

        position_type num_literals = tok.num_literals;
        if (tok.num_literals == 15)
            num_literals += ctrl.readLSIC(comp_idx);
        const position_type literalStart = comp_idx;

        if (num_literals + comp_idx > ctrl.end())
            coopCopyNoOverlap(decompData + decomp_idx, compData + comp_idx, num_literals);
        else
            coopCopyNoOverlap(decompData + decomp_idx, ctrl.rawAt(comp_idx), num_literals);

        comp_idx += num_literals;
        decomp_idx += num_literals;

        if (comp_idx < comp_end) {
            offset_type offset;
            if (comp_idx + sizeof(offset_type) > ctrl.end())
                offset = readWord(compData + comp_idx);
            else
                offset = readWord(ctrl.rawAt(comp_idx));

            comp_idx += sizeof(offset_type);

            position_type match = 4 + tok.num_matches;
            if (tok.num_matches == 15)
                match += ctrl.readLSIC(comp_idx);

            if (offset <= num_literals
                && (ctrl.begin() <= literalStart && ctrl.end() >= literalStart + num_literals)) {
                coopCopyOverlap(decompData + decomp_idx,
                                ctrl.rawAt(literalStart + (num_literals - offset)), offset, match);
                syncCTA();
            } else {
                syncCTA();
                coopCopyOverlap(decompData + decomp_idx,
                                decompData + decomp_idx - offset, offset, match);
            }
            decomp_idx += match;
        }
    }
}

inline __device__ uint32_t read32be_batch(const uint8_t* address) {
    return ((uint32_t)(255 & address[0]) << 24 | (uint32_t)(255 & address[1]) << 16 |
            (uint32_t)(255 & address[2]) << 8  | (uint32_t)(255 & address[3]));
}

extern "C" __global__ void h5lz4dc_batched(
    const uint8_t* const compressed, const uint32_t* const chunk_offsets,
    const uint32_t* const block_starts, const uint32_t* const block_counts,
    const uint32_t* const block_offsets, const uint32_t blocksize,
    const uint32_t frame_bytes, uint8_t* const decompressed
) {
    const int frame_id = blockIdx.z;
    const int block_in_frame = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t chunk_offset = chunk_offsets[frame_id];
    const uint32_t block_offset = block_offsets[frame_id];
    const uint32_t num_blocks = block_counts[frame_id];
    __shared__ uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * DECOMP_CHUNKS_PER_BLOCK];

    if (block_in_frame < num_blocks) {
        const uint32_t block_start = block_starts[block_offset + block_in_frame];
        const uint8_t* input = compressed + chunk_offset + block_start + 4;
        const uint32_t comp_size = read32be_batch(compressed + chunk_offset + block_start);
        uint8_t* output = decompressed + frame_id * frame_bytes + block_in_frame * blocksize;
        decompressStream(buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE, output, input, comp_size);
    }
}

extern "C" __global__ void shuf_8192_32_batched(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out, const uint32_t frame_u32s
) {
    const int frame_id = blockIdx.z;
    const uint32_t* frame_in = in + frame_id * frame_u32s;
    uint32_t* frame_out = out + frame_id * frame_u32s;
    __shared__ uint32_t smem[32][33];

    smem[threadIdx.y][threadIdx.x] = frame_in[threadIdx.x + threadIdx.y * 64 +
                                               blockIdx.x * 2048 + blockIdx.y * 32];
    __syncthreads();

    uint32_t v = smem[threadIdx.x][threadIdx.y];
    #pragma unroll 32
    for (int i = 0; i < 32; i++)
        smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU, v & (1U << i));
    __syncthreads();

    frame_out[threadIdx.x + threadIdx.y * 32 + blockIdx.y * 1024 + blockIdx.x * 2048] =
        smem[threadIdx.x][threadIdx.y];
}
'''

# Compile CUDA kernels at import time for fast first load()
_cuda_module = cp.RawModule(code=_CUDA_LZ4_SOURCE, options=("-std=c++11", "-w"))
_h5lz4dc_kernel = _cuda_module.get_function("h5lz4dc_batched")
_bitshuffle_kernel = _cuda_module.get_function("shuf_8192_32_batched")

class GPUDecompressor:
    """GPU-accelerated decompressor for bitshuffle+LZ4 HDF5 datasets.

    Uses pinned memory and CUDA kernels for maximum throughput.
    CUDA kernels are compiled at module import time.
    """

    def __init__(
        self,
        max_compressed_bytes: int = 1024 * 1024 * 1024,
        max_frames: int = 100000,
        max_frame_bytes: int = 192 * 192 * 4,
        n_blocks_per_frame: int = 18,
    ):
        """Initialize the decompressor with pre-allocated buffers.

        Parameters
        ----------
        max_compressed_bytes : int, optional
            Maximum size of compressed data, by default 1GB.
        max_frames : int, optional
            Maximum number of frames to support, by default 100000.
        max_frame_bytes : int, optional
            Maximum bytes per frame, by default 147456 (192x192 uint32).
        n_blocks_per_frame : int, optional
            LZ4 blocks per frame, by default 18 for 192x192 uint32.
        """
        self.max_compressed_bytes = max_compressed_bytes
        self.max_frames = max_frames
        self.max_frame_bytes = max_frame_bytes
        self.n_blocks_per_frame = n_blocks_per_frame
        self._h5lz4dc = _h5lz4dc_kernel
        self._shuf = _bitshuffle_kernel
        # Pinned memory for fast CPU->GPU transfers
        self._pinned_mem = cp.cuda.alloc_pinned_memory(max_compressed_bytes)
        self._pinned_buffer = np.frombuffer(
            self._pinned_mem, dtype=np.uint8, count=max_compressed_bytes
        )
        # Pre-allocated metadata arrays
        self._chunk_sizes = np.zeros(max_frames, dtype=np.uint32)
        self._chunk_offsets = np.zeros(max_frames, dtype=np.uint32)
        self._block_counts = np.zeros(max_frames, dtype=np.uint32)
        self._block_starts_flat = np.zeros(max_frames * n_blocks_per_frame, dtype=np.uint32)
        self._block_offsets = np.zeros(max_frames + 1, dtype=np.uint32)
        # Pre-allocate all GPU buffers for fast first load()
        self._concat_gpu = cp.empty(max_compressed_bytes, dtype=cp.uint8)
        total_output_bytes = max_frames * max_frame_bytes
        self._lz4_output = cp.empty(total_output_bytes, dtype=cp.uint8)
        self._shuffled_output = cp.empty(total_output_bytes, dtype=cp.uint8)

    def load(
        self,
        filepath: str,
        dataset_path: str = "entry/data/data",
    ) -> cp.ndarray:
        """Load and decompress a bitshuffle+LZ4 HDF5 dataset to GPU.

        Parameters
        ----------
        filepath : str
            Path to the HDF5 file.
        dataset_path : str, optional
            Path to the dataset within the HDF5 file, by default "entry/data/data".

        Returns
        -------
        cp.ndarray
            CuPy array on GPU with shape (n_frames, height, width).
        """
        with h5py.File(filepath, "r") as f:
            ds = f[dataset_path]
            n_frames = ds.shape[0]
            frame_shape = ds.shape[1:]
            dtype = ds.dtype
            frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)

            # Reallocate GPU buffers only if dataset exceeds pre-allocated size
            total_needed = n_frames * frame_bytes
            if total_needed > len(self._lz4_output):
                self._lz4_output = cp.empty(total_needed, dtype=cp.uint8)
                self._shuffled_output = cp.empty(total_needed, dtype=cp.uint8)
            # Read chunks into pinned memory
            offset = 0
            for i in range(n_frames):
                _, raw = ds.id.read_direct_chunk((i, 0, 0))
                chunk_len = len(raw)
                self._chunk_offsets[i] = offset
                self._chunk_sizes[i] = chunk_len
                self._pinned_buffer[offset : offset + chunk_len] = np.frombuffer(
                    raw, dtype=np.uint8
                )
                offset += chunk_len
            total_compressed = offset
        # Parse headers
        _parse_headers(
            self._pinned_buffer,
            self._chunk_sizes,
            self._chunk_offsets,
            self._block_starts_flat,
            self._block_counts,
            n_frames,
            self.n_blocks_per_frame,
        )
        # Compute block offsets
        self._block_offsets[1 : n_frames + 1] = np.cumsum(self._block_counts[:n_frames])
        total_blocks = int(self._block_offsets[n_frames])
        # Transfer to GPU
        self._concat_gpu[:total_compressed].set(self._pinned_buffer[:total_compressed])
        chunk_offsets_gpu = cp.asarray(self._chunk_offsets[:n_frames])
        block_starts_gpu = cp.asarray(self._block_starts_flat[:total_blocks])
        block_counts_gpu = cp.asarray(self._block_counts[:n_frames])
        block_offsets_gpu = cp.asarray(self._block_offsets[: n_frames + 1])
        # LZ4 decompress
        block_size = 8192
        max_blocks = int(self._block_counts[:n_frames].max())
        max_batch = 10000
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            self._h5lz4dc(
                ((max_blocks + 1) // 2, 1, batch_n),
                (32, 2, 1),
                (
                    self._concat_gpu,
                    chunk_offsets_gpu[start:],
                    block_starts_gpu,
                    block_counts_gpu[start:],
                    block_offsets_gpu[start:],
                    np.uint32(block_size),
                    np.uint32(frame_bytes),
                    self._lz4_output[byte_offset:],
                ),
            )
        # Bitshuffle
        n_8kb = frame_bytes // 8192
        frame_u32s = frame_bytes // 4
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            self._shuf(
                (n_8kb, 2, batch_n),
                (32, 32, 1),
                (
                    self._lz4_output[byte_offset:].view(cp.uint32),
                    self._shuffled_output[byte_offset:].view(cp.uint32),
                    np.uint32(frame_u32s),
                ),
            )
        cp.cuda.Device().synchronize()
        total_bytes = n_frames * frame_bytes
        return self._shuffled_output[:total_bytes].view(dtype).reshape(
            (n_frames,) + frame_shape
        )

@njit(cache=True, parallel=True)
def _parse_headers(
    pinned_buffer,
    chunk_sizes,
    chunk_offsets,
    block_starts_out,
    block_counts_out,
    n_frames,
    n_blocks_per_frame,
):
    """Parse bitshuffle+LZ4 chunk headers in parallel."""
    for i in prange(n_frames):
        offset = chunk_offsets[i]
        chunk = pinned_buffer[offset : offset + chunk_sizes[i]]

        # Parse header (first 12 bytes)
        uncomp_size = (
            int(chunk[0]) << 56
            | int(chunk[1]) << 48
            | int(chunk[2]) << 40
            | int(chunk[3]) << 32
            | int(chunk[4]) << 24
            | int(chunk[5]) << 16
            | int(chunk[6]) << 8
            | int(chunk[7])
        )
        block_size = (
            int(chunk[8]) << 24
            | int(chunk[9]) << 16
            | int(chunk[10]) << 8
            | int(chunk[11])
        )
        n_blocks = (uncomp_size + block_size - 1) // block_size
        block_counts_out[i] = n_blocks
        pos = 12
        base_idx = i * n_blocks_per_frame
        for b in range(n_blocks):
            block_starts_out[base_idx + b] = pos
            comp_size = (
                int(chunk[pos]) << 24
                | int(chunk[pos + 1]) << 16
                | int(chunk[pos + 2]) << 8
                | int(chunk[pos + 3])
            )
            pos += 4 + comp_size

# Pre-allocate decompressor at import for fast first load()
_default_decompressor = GPUDecompressor()

def load(filepath: str, dataset_path: str = "entry/data/data") -> cp.ndarray:
    """Load a bitshuffle+LZ4 compressed HDF5 dataset directly to GPU.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    dataset_path : str, optional
        Path to the dataset within the HDF5 file, by default "entry/data/data".

    Returns
    -------
    cp.ndarray
        CuPy array on GPU with shape (n_frames, height, width).
    """
    return _default_decompressor.load(filepath, dataset_path)

def bin(
    data: cp.ndarray,
    factor: int = 2,
    dtype: type | np.dtype | None = None,
    reduction: str = "sum",
) -> cp.ndarray:
    """Apply spatial binning to a stack of 2D images on GPU.

    Parameters
    ----------
    data : cp.ndarray
        CuPy array with shape (n_frames, height, width) or (height, width).
    factor : int, optional
        Binning factor (2 for 2x2, 4 for 4x4, etc.), by default 2.
    dtype : type or np.dtype, optional
        Output dtype. If None, uses uint32 for int input (sum), float32 for mean.
    reduction : str, optional
        Reduction method - 'sum' (default) or 'mean'.

    Returns
    -------
    cp.ndarray
        Binned CuPy array with reduced spatial dimensions.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got '{reduction}'")
    if factor == 1:
        return data
    is_2d = data.ndim == 2
    if is_2d:
        data = data[None, :, :]
    if data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    n, h, w = data.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Dimensions ({h}, {w}) not divisible by factor {factor}")
    reshaped = data.reshape(n, h // factor, factor, w // factor, factor)
    if dtype is None:
        dtype = cp.float32 if reduction == "mean" else (
            cp.uint32 if cp.issubdtype(data.dtype, cp.integer) else cp.float32
        )
    if reduction == "mean":
        result = reshaped.mean(axis=(2, 4), dtype=dtype)
    else:
        result = reshaped.sum(axis=(2, 4), dtype=dtype)
    return result[0] if is_2d else result

def clear_memory() -> None:
    """Release GPU memory pools."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass