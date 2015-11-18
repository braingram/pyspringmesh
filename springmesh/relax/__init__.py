#!/usr/bin/env python


from . import dynamic
from . import parallel
from . import standard

__all__ = ['dynamic', 'parallel', 'standard']


try:
   from . import cuda
   __all__.append('cuda')
except ImportError:
   print("Error importing cuda relax method")
