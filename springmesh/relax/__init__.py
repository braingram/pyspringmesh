#!/usr/bin/env python


from . import dynamic
from . import parallel
from . import standard

__all__ = ['dynamic', 'parallel', 'standard']


has_cuda = False
try:
    import pycuda
    has_cuda = True
except ImportError:
    has_cuda = False

if has_cuda:
    try:
        from . import cuda
        __all__.append('cuda')
    except pycuda.driver.RuntimeError:
       print("Error enabling cuda relax method")
