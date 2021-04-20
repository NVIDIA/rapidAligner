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

from numba import cuda, float64
from math import sqrt

###############################################################################
# plain rolling Euclidean distance
###############################################################################

@cuda.jit
def sdist_kernel(Q, S, out):
    """Euclidean Distance naive kernel: nothing cached"""
    
    warpDim = cuda.blockDim.x  // 32
    warpIdx = cuda.threadIdx.x // 32
    laneIdx = cuda.threadIdx.x  % 32
    
    lower  = cuda.blockIdx.x*warpDim+warpIdx
    stride = cuda.gridDim.x*warpDim
    
    for position in range(lower, S.shape[0]-Q.shape[0]+1, stride):
    
        accum = float64(0)
        for index in range(laneIdx, Q.shape[0], 32):
            value  = Q[index]-S[position+index]
            accum += value*value
        
        for delta in [16, 8, 4, 2, 1]:
            value  = cuda.shfl_down_sync(0xFFFFFFFF, accum, delta)
            accum += value
    
        if laneIdx == 0:
            out[position] = accum
        
###############################################################################
# mean-adjusted rolling Euclidean distance
###############################################################################

@cuda.jit(max_registers=63)
def mdist_kernel(Q, S, out):
    """mean-adjusted Euclidean Distance naive kernel: nothing cached"""
    
    warpDim = cuda.blockDim.x  // 32
    warpIdx = cuda.threadIdx.x // 32
    laneIdx = cuda.threadIdx.x  % 32
    
    lower  = cuda.blockIdx.x*warpDim+warpIdx
    stride = cuda.gridDim.x*warpDim
    
    for position in range(lower, S.shape[0]-Q.shape[0]+1, stride):
    
        accum = float64(0)
        for index in range(laneIdx, Q.shape[0], 32):
            accum += S[position+index]
            
        for delta in [16, 8, 4, 2, 1]:
            accum += cuda.shfl_xor_sync(0xFFFFFFFF, accum, delta)
        
        mean = accum/Q.shape[0]
        accum = float64(0)
        for index in range(laneIdx, Q.shape[0], 32):
            value  = Q[index]-S[position+index]+mean
            accum += value*value
        
        for delta in [16, 8, 4, 2, 1]:
            value  = cuda.shfl_down_sync(0xFFFFFFFF, accum, delta)
            accum += value
    
        if laneIdx == 0:
            out[position] = accum
            
###############################################################################
# mean- and amplitude-adjusted rolling Euclidean distance
###############################################################################

@cuda.jit(max_registers=63)
def zdist_kernel(Q, S, out, epsilon):
    """z-normalized Euclidean Distance naive kernel: nothing cached"""
    
    warpDim = cuda.blockDim.x  // 32
    warpIdx = cuda.threadIdx.x // 32
    laneIdx = cuda.threadIdx.x  % 32
    
    lower  = cuda.blockIdx.x*warpDim+warpIdx
    stride = cuda.gridDim.x*warpDim
    
    for position in range(lower, S.shape[0]-Q.shape[0]+1, stride):
    
        accum1 = float64(0)
        accum2 = float64(0)
        for index in range(laneIdx, Q.shape[0], 32):
            value   = S[position+index]
            accum1 += value
            accum2 += value*value
                
        for delta in [16, 8, 4, 2, 1]:
            accum1 += cuda.shfl_xor_sync(0xFFFFFFFF, accum1, delta)
            accum2 += cuda.shfl_xor_sync(0xFFFFFFFF, accum2, delta)
        
        mean  = accum1/Q.shape[0]
        sigma = accum2/Q.shape[0]-mean*mean
        sigma = sqrt(sigma) if sigma > 0.0 else epsilon
        
        accum = float64(0)
        for index in range(laneIdx, Q.shape[0], 32):
            value  = Q[index]-(S[position+index]-mean)/sigma
            accum += value*value
        
        for delta in [16, 8, 4, 2, 1]:
            accum += cuda.shfl_down_sync(0xFFFFFFFF, accum, delta)
            
        if laneIdx == 0:
            out[position] = accum
        