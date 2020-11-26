# Copyright (c) 2020, NVIDIA CORPORATION.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cuseries.ED.stream_dists_fft import fft_sdist, fft_mdist, fft_zdist
from cuseries.ED.stream_dists_kernels import sdist_kernel, mdist_kernel, zdist_kernel
from cuseries.ED.stream_dists_helpers import mnorm, znorm

import cupy as cp
from numba import cuda

__all__ = ["sdist", "mdist", "zdist"]

def sdist(Q, S, mode="fft"):
    """
    Rolling Euclidean Distance 

    Arguments:
    -------
      Q: cupy.core.core.ndarray or numba.cuda.DeviceNDArray or cudf.Series or numpy.ndarray
        the input query of length m to be aligned
      S: cupy.core.core.ndarray or numba.cuda.DeviceNDArray or cudf.Series or numpy.ndarray
        the input stream of length n>=m to be scanned
      mode: str
        either "naive" or "fft"
    Returns
    -------
    cupy.core.core.ndarray
        the computed distance array of length n-m+1

    """
        
    if not isinstance(Q, cp.core.core.ndarray):
        Q = cp.asarray(Q)
        
    if not isinstance(S, cp.core.core.ndarray):
        S = cp.asarray(S)
    
    assert(Q.dtype == S.dtype)
    assert((len(Q.shape) == len(S.shape) == 1 and Q.shape[0] <= S.shape[0])) 

    if mode == "fft":
        Z = fft_sdist(Q, S)
    else:
        stream = cuda.stream()
        Z = cp.empty(len(S)-len(Q)+1, dtype=Q.dtype)
        sdist_kernel[80*32, 64, stream](Q, S, Z)
        stream.synchronize()
    
    return Z

def mdist(Q, S, mode="fft"):
    
    """
    Rolling mean-adjusted Euclidean Distance

    Arguments:
    -------
      Q: cupy.core.core.ndarray or numba.cuda.DeviceNDArray or cudf.Series or numpy.ndarray
        the input query of length m to be aligned
      S: cupy.core.core.ndarray or numba.cuda.DeviceNDArray or cudf.Series or numpy.ndarray
        the input stream of length n>=m to be scanned
      mode: str
        either "naive" or "fft"
    Returns
    -------
    cupy.core.core.ndarray
        the computed distance array of length n-m+1

    """
    
    if not isinstance(Q, cp.core.core.ndarray):
        Q = cp.asarray(Q)
        
    if not isinstance(S, cp.core.core.ndarray):
        S = cp.asarray(S)
    
    assert(Q.dtype == S.dtype)
    assert((len(Q.shape) == len(S.shape) == 1 and Q.shape[0] <= S.shape[0]))
    
    if mode == "fft":
        Z = fft_mdist(Q, S)
    else:
        stream = cuda.stream()
        Z = cp.empty(len(S)-len(Q)+1, dtype=Q.dtype)
        mdist_kernel[80*32, 64, stream](mnorm(Q), S, Z)
        stream.synchronize()
    
    return Z

def zdist(Q, S, mode="fft", epsilon=1e-6):
    """
    Rolling mean- and amplitude-adjusted Euclidean Distance 

    Arguments:
    -------
      Q: cupy.core.core.ndarray or numba.cuda.DeviceNDArray or cudf.Series or numpy.ndarray
        the input query of length m to be aligned
      S: cupy.core.core.ndarray or numba.cuda.DeviceNDArray or cudf.Series or numpy.ndarray
        the input stream of length n>=m to be scanned
      epsilon: float
        non-negative number for regularizing zero stdev
      mode: str
        either "naive" or "fft"
    Returns
    -------
    cupy.core.core.ndarray
        the computed distance array of length n-m+1

    """
    
    if not isinstance(Q, cp.core.core.ndarray):
        Q = cp.asarray(Q)
        
    if not isinstance(S, cp.core.core.ndarray):
        S = cp.asarray(S)
    
    assert(epsilon > 0)
    assert(Q.dtype == S.dtype)
    assert((len(Q.shape) == len(S.shape) == 1 and Q.shape[0] <= S.shape[0]))
    assert(cp.std(Q, ddof=0) > 0)
        
    if mode == "fft":
        Z = fft_zdist(Q, S, epsilon)
    else:
        stream = cuda.stream()    
        Z = cp.empty(len(S)-len(Q)+1, dtype=Q.dtype)
        zdist_kernel[80*32, 64, stream](znorm(Q, epsilon), S, Z, epsilon)
        stream.synchronize()
    
    return Z
