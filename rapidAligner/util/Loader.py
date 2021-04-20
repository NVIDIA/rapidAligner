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

__all__ = ['FakeSeriesGenerator', 'ECGLoader']

import os
import urllib
import zipfile
import cupy as cp
from scipy.io import loadmat

class ECGLoader:
    
    def __init__(self, root='./data/ECG', url=None):
        
        self.root = root
        
        assert url != None, \
        "provide the URL to 22h of ECG data stated on the bottom of https://www.cs.ucr.edu/~eamonn/UCRsuite.html"
            
        filename = os.path.join(root, 'ECG_one_day.zip')
        
        if not os.path.isdir(root):
            os.makedirs(root)
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename)
                    
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(root)
    
    @property
    def subject(self, alpha=400.0, beta=50.0):
        return alpha*loadmat(os.path.join(self.root, 'ECG_one_day','ECG.mat'))['ECG'].flatten()+beta
    
    @property
    def query(self):
        return loadmat(os.path.join(self.root, 'ECG_one_day','ECG_query.mat'))['ecg_query'].flatten()
    
    @property
    def data(self):
        return self.query, self.subject
        

class FakeSeriesGenerator:
    
    def __init__(self, query_length=3600, subject_length=2**20, seed=None, beta=1.0):
        
        self.query_length = query_length
        self.subject_length = subject_length
        self.beta = beta
        
        assert isinstance(query_length, int) and query_length > 0
        assert isinstance(subject_length, int) and subject_length > 0
        assert query_length <= subject_length
        assert isinstance(beta, float) and beta >= 0
        
        if isinstance(seed, int):
            cp.random.seed(seed)
            
        noise   = cp.random.uniform(-1, +1, self.subject_length+self.query_length)
        kernel  = cp.exp(-self.beta*cp.linspace(0, 1, self.subject_length+self.query_length))
        kernel /= cp.sqrt(cp.sum(kernel**2))
        
        self.signal = cp.fft.irfft(cp.fft.rfft(noise)*cp.fft.rfft(kernel), n=self.subject_length+self.query_length)
        
    @property
    def subject(self):
        return self.signal[self.query_length:].get()
    
    @property
    def query(self, alpha=2.0, beta=1.20):
        
        signal = self.signal[:self.query_length]
        mean = cp.mean(signal)
        
        return (alpha*(signal-mean)+beta*mean).get()
    
    @property
    def data(self):
        return self.query, self.subject