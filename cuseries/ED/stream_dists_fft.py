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

from cuseries.ED.stream_dists_helpers import cumsum
from cuseries.ED.stream_dists_helpers import mnorm, znorm
import cupy as cp
import math

def fft_sdist(Q, S, alignment=10000, Kahan=0):
    """
    Rolling Euclidean Distance using FFT to run in loglinear time
    
    Equation exploiting cross-correlation (Fourier) theorem:
    
    d[k] = sum_i (Q[i] - S[i+k])**2
         = sum_i (Q[i]**2 - 2*Q[i]*S[i+k] + S[i+k]**2)
         = sum_i Q[i]**2 - 2*correlation[k] + sum_i S[i+k]**2
         = sum_i Q[i]**2 - 2*correlation[k] + Y[k] 

    Arguments:
    -------
      Q: cupy.core.core.ndarray
        the input query of length m to be aligned
      S: cupy.core.core.ndarray
        the input stream of length n>=m to be scanned
      Kahan: int
        non-negative number of Kahan summation adjustment rounds
    Returns
    -------
    cupy.core.core.ndarray
        the computed distance array of length n-m+1

    """
    
    assert(Q.dtype == S.dtype)
    
    m = len(Q)
    n = (len(S)+alignment-1)//alignment*alignment
    iS = cp.zeros(n, dtype=S.dtype)
    iS[:len(S)] = S
        
    Y = cumsum(iS**2, Kahan)    
    Y = Y[+m:]-Y[:-m]        
    E = cp.zeros(n, dtype=Q.dtype)
    E[:m] = Q
    R = cp.fft.irfft(cp.fft.rfft(E).conj()*cp.fft.rfft(iS), n=n)
    
    return (cp.sum(cp.square(Q))-2*R[:-m+1]+Y)[:len(S)-m+1]

def fft_mdist(Q, S, alignment=10000, Kahan=0):
    """
    Rolling mean-adjusted Euclidean Distance using FFT to run in loglinear time
    
    Equation exploiting cross-correlation (Fourier) theorem:
    
    d[k] = sum_i (f(Q[i]) - f(S[i+k]))**2
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*f(S[i+k]) + f(S[i+k])**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*(S[i+k]-mu[k]) + (S[i+k]-mu[k])**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + 2*f(Q[i])*mu[k] + (S[i+k]-mu[k])**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + (S[i+k]-mu[k])**2)
         since sum_i f(Q[i]) = 0 by definition
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + S[i+k]**2 - 2*S[i+k]*mu[k] + mu[k]**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + S[i+k]**2 - 2*|Q|*mu[k]*mu[k] + mu[k]**2)
         = sum_i f(Q[i])**2 - 2*correlation(k)  + Y[k] - 2*X[k]**2/|Q| + X[k]**2/|Q| 
         = sum_i f(Q[i])**2 - 2*correlation(k)  + Y[k] -   X[k]**2/|Q| 
         = sum_i f(Q[i])**2 - 2*correlation(k)  + |Q|*variance[k] 

    Arguments:
    -------
      Q: cupy.core.core.ndarray
        the input query of length m to be aligned
      S: cupy.core.core.ndarray
        the input stream of length n>=m to be scanned
      Kahan: int
        non-negative number of Kahan summation adjustment rounds
    Returns
    -------
    cupy.core.core.ndarray
        the computed distance array of length n-m+1

    """
    
    m, Q = len(Q), mnorm(Q)
    n = (len(S)+alignment-1)//alignment*alignment
    iS = cp.zeros(n).astype(S.dtype)
    iS[:len(S)] = S
    
    X, Y = cumsum(iS, Kahan), cumsum(iS**2, Kahan)
    X = X[+m:]-X[:-m]
    Y = Y[+m:]-Y[:-m]
    Z = Y-X*X/m
    E = cp.zeros(n, dtype=Q.dtype)
    E[:m] = Q
    R = cp.fft.irfft(cp.fft.rfft(E).conj()*cp.fft.rfft(iS), n=n)
        
    return (cp.sum(cp.square(Q))-2*R[:-m+1]+Z)[:len(S)-m+1]

def fft_zdist(Q, S, epsilon, alignment=10000, Kahan=0):    
    """
    Rolling mean- and amplitude-adjusted Euclidean Distance using FFT to run in loglinear time
    
    Equation exploiting cross-correlation (Fourier) theorem:
    
    d[k] = sum_i (f(Q[i]) - f(S[i+k]))**2
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*f(S[i+k]) + f(S[i+k])**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*(S[i+k]-mu[k]) + (S[i+k]-mu[k])**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + 2*f(Q[i])*mu[k] + (S[i+k]-mu[k])**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + (S[i+k]-mu[k])**2)
         since sum_i f(Q[i]) = 0 by definition
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + S[i+k]**2 - 2*S[i+k]*mu[k] + mu[k]**2)
         = sum_i (f(Q[i])**2 - 2*f(Q[i])*S[i+k] + S[i+k]**2 - 2*|Q|*mu[k]*mu[k] + mu[k]**2)
         = sum_i f(Q[i])**2 - 2*correlation(k)  + Y[k] - 2*X[k]**2/|Q| + X[k]**2/|Q| 
         = sum_i f(Q[i])**2 - 2*correlation(k)  + Y[k] -   X[k]**2/|Q| 
         = sum_i f(Q[i])**2 - 2*correlation(k)  + |Q|*variance[k] 

    Arguments:
    -------
      Q: cupy.core.core.ndarray
        the input query of length m to be aligned
      S: cupy.core.core.ndarray
        the input stream of length n>=m to be scanned
      epsilon: float
        non-negative number for regularizing zero stdev
      Kahan: int
        non-negative number of Kahan summation adjustment rounds
    Returns
    -------
    cupy.core.core.ndarray
        the computed distance array of length n-m+1

    """
    
    assert(epsilon > 0)
       
    m, Q = len(Q), znorm(Q, epsilon)
    n = (len(S)+alignment-1)//alignment*alignment
    iS = cp.zeros(n, dtype=S.dtype)
    iS[:len(S)] = S
    delta = n-len(S)
    
    X, Y = cumsum(iS, Kahan), cumsum(iS**2, Kahan)
    X = X[+m:]-X[:-m]
    Y = Y[+m:]-Y[:-m]
    Z = cp.sqrt(cp.maximum(Y/m-cp.square(X/m), 0))
    E = cp.zeros(n, dtype=Q.dtype)
    E[:m] = Q
    R = cp.fft.irfft(cp.fft.rfft(E).conj()*cp.fft.rfft(iS), n=n)
    F = cp.where(Z > 0 , 2*(m-R[:-m+1]/Z), m*cp.ones_like(Z))
    
    return F[:len(S)-m+1]

