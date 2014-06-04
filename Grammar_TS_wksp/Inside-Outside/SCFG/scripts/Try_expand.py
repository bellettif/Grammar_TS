'''
Created on 1 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from SCFG.sto_grammar import SCFG

import time

from Grammar_examples import palindrom_grammar
from Grammar_examples import palindrom_grammar_2
from Grammar_examples import palindrom_grammar_3

selected_grammar = palindrom_grammar_2

N = selected_grammar.N
M = selected_grammar.M

selected_grammar.A = np.ones((N, N, N))

for i in xrange(N):
    total = np.sum(selected_grammar.A[i]) + np.sum(selected_grammar.B[i])
    selected_grammar.A[i] /= total
    selected_grammar.B[i] /= total

for i in xrange(N):
    print np.sum(selected_grammar.A[i]) + np.sum(selected_grammar.B[i])

selected_grammar.expand(2)

for i in xrange(N + 1):
    print np.sum(selected_grammar.A[i]) + np.sum(selected_grammar.B[i])

print ''

print selected_grammar.A

selected_grammar.merge(1, 2)