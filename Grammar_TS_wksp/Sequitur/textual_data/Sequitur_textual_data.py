'''
Created on 18 avr. 2014

@author: francois
'''

import numpy as np

au = np.array(list('ABC'), 'U1')

print au.dtype

au_as_int = au.view(np.uint32)

print au_as_int

print ''.join(au_as_int.view('U1'))