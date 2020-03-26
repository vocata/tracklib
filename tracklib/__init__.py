'''
tracking library which contains a varity of algorithms:

'filter': tracking filter
'model': dynamic motion model
'math': numerical algorithm
'utils': misclineous tools
'''
from __future__ import division, absolute_import, print_function


__version__ = '1.0'
__all__ = ['filter', 'math', 'model', 'utils']

from . import filter
from . import model
from . import math
from . import utils
from .utils import *
