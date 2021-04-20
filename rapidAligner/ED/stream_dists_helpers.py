
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

import cupy as cp

###############################################################################
# helpers to avoid redundant code 
###############################################################################

def cumsum(x, Kahan=0):
    """
    Wrapper for exclusive prefix sum computation with an optional
    refinement step using a approach similar to Kahan summation.
    This function is not exposed to the user.

    Arguments:
    -------
      x: cupy.core.core.ndarray
        the input array of length n to be scanned with operation +
      Kahan: int
        non-negative number of Kahan summation adjustment rounds
    Returns
    -------
    cupy.core.core.ndarray
        the computed exclusive prefix scan of length n+1

    """

    assert(isinstance(Kahan, int) and Kahan >= 0)

    # allocate an empty array with leading 0
    y = cp.empty(len(x)+1, dtype=x.dtype)
    y[0] = 0

    # compute the inclusive prefix sum starting at entry 1
    cp.cumsum(x, out=y[1:])
    
    # basically exploit that (d/dt int f(t) dt) - f(t) = r = 0 forall f(t)
    # in case delta is non-vanishing due to numeric inaccuracies, we add
    # the prefix scan of r to the final result (inaccuracies might add up)
    if Kahan:
        r = x-cp.diff(y)
        if(cp.max(cp.abs(r))):
            y += cumsum(r, Kahan-1)
    return y

def mnorm(x):
    """
    Mean-adjustment of a given time series. Afterwards the time series
    has vanishing mean, i.e. sum_i x[i] = 0
    
    Arguments:
    -------
      x: cupy.core.core.ndarray
        the input array of length n to be normalized
    Returns
    -------
    cupy.core.core.ndarray
        the mean-adjusted array of length n
    
    
    """
    
    return x-cp.mean(x)

def znorm(x, epsilon):
    """
    Mean- and amplitude-adjustment of a given time series. Afterwards the time series
    has vanishing mean, i.e. sum_i x[i] = 0 and unit standard devitation i.e.
    sum_i x[i]*x[i] = n where n is the length of the sequence x
    
    Arguments:
    -------
      x: cupy.core.core.ndarray
        the input array of length n to be normalized
    Returns
    -------
    cupy.core.core.ndarray
        the mean-adjusted array of length n
    
    
    """    
    
    return (x-cp.mean(x))/max(cp.std(x, ddof=0), epsilon)

        