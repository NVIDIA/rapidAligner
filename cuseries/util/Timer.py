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

from numba import cuda

class cudaTimer:

    def __init__(self, label='', gpu=0):
    
        
        self.label = label
        self.gpu = gpu
        self.start = cuda.event()
        self.end   = cuda.event()
        cuda.select_device(self.gpu)
        self.start.record(),
   
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
    
        cuda.select_device(self.gpu)
        suffix = 'ms ('+self.label+')' if self.label else 'ms'
        self.end.record()
        self.end.synchronize()
        time = cuda.event_elapsed_time(self.start, self.end)
        print('elapsed time:', int(time), suffix)

