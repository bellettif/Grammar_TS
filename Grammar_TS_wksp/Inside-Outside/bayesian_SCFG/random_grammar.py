'''
Created on 16 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from SCFG.sto_grammar import SCFG

alpha_D = 0.01
alpha_T = 0.01

n_D = 16
n_T = 4
N = n_D + n_T

terminal_chars = ['a', 'b', 'c', 'd']
M = len(terminal_chars)

A = np.zeros((N, N, N))
B = np.zeros((N, M))

for i in xrange(n_D):
    beta = np.random.dirichlet(alpha_D * np.ones(N))
    beta_beta = np.outer(beta, beta)
    weights = np.random.dirichlet(alpha_D * np.ones(N ** 2))
    A[i, :, :] = np.reshape(weights, (N, N))
    
A[np.where(A <= 1e-6)] = 0


print A
"""
print A

for i in xrange(n_D, N):
    B[i,i - n_D] = 1.0
    print B[i, :]
    print ''
    
random_grammar = SCFG()
random_grammar.init_from_A_B(A, B, terminal_chars)

sentences = random_grammar.produce_sentences(100)

for sentence in sentences:
    print sentence
"""