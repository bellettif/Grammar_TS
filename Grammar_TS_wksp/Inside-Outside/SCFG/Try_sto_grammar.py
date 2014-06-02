'''
Created on 1 juin 2014

@author: francois
'''

from SCFG_c import *

import numpy as np
from matplotlib import pyplot as plt

from sto_grammar import SCFG

N = 7

terms = ['a', 'b', 'c']

M = len(terms)

A = np.zeros((N, N, N), dtype = np.double)
B = np.zeros((N, M), dtype = np.double)

A[0, 1, 2] = 1.0
A[0, 3, 4] = 1.0
A[0, 5, 6] = 1.0
A[0, 1, 1] = 1.0
A[0, 3, 3] = 1.0
A[0, 5, 5] = 1.0

A[2, 0, 1] = 1.0

A[4, 0, 3] = 1.0

A[6, 0, 5] = 1.0

B[1, 0] = 1.0
B[3, 1] = 1.0
B[5, 2] = 1.0

for i in xrange(N):
    total = np.sum(A[i]) + np.sum(B[i])
    A[i] /= total
    B[i] /= total
    
first_try = SCFG()
first_try.init_from_A_B(A, B, terms)

rules = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [],
              []),
         1 : ([],
              [],
              ['a'],
              [1.0]),
         2 : ([[0, 1]],
              [1.0],
              [],
              []),
         3 : ([],
              [],
              ['b'],
              [1.0]),
         4 : ([[0, 3]],
              [1.0],
              [],
              []),
         5 : ([],
              [],
              ['c'],
              [1.0]),
         6 : ([[0, 5]],
              [1.0],
              [],
              [])}

second_try = SCFG()
second_try.init_from_rule_dict(rules)

print 'First try A:'
print first_try.A
print ''

print 'Second try A:'
print second_try.A
print ''

print 'First try B'
print first_try.B
print ''

print 'Second try B'
print second_try.B
print ''


