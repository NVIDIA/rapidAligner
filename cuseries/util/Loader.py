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

import os
import urllib
import zipfile
import cupy as cp
from scipy.io import loadmat

class ECGLoader:
    
    def __init__(self, root='./data/ECG'):
        
        self.root = root
        
        url = "https://www.cs.ucr.edu/~eamonn/ECG_one_day.zip"
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
        
        