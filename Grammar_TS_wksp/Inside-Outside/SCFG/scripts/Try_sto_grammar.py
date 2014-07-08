'''
Created on 1 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from SCFG.sto_grammar import SCFG

import time

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

n_samples = 20
n_iterations = 100

sentences = first_try.produce_sentences(n_samples)

first_try.plot_stats(100000, max_represented = 100)

for sentence in sentences:
    print sentence
    
N = first_try.N
M = first_try.M

A_proposal = np.random.uniform(0.0, 1.0, (N, N, N))
B_proposal = np.random.uniform(0.0, 1.0, (N, M))

A_proposal = np.maximum(A_proposal, 0.1 * np.ones((N, N, N)))
B_proposal = np.maximum(B_proposal, 0.1 * np.ones((N, M)))
    
A_proposal = np.asanyarray(A_proposal, dtype = np.double)
B_proposal = np.asanyarray(B_proposal, dtype = np.double)
    
estim_A, estim_B, likelihoods = first_try.estimate_A_B(sentences,
                                                       n_iterations, 
                                                       'explicit',
                                                       A_proposal,
                                                       B_proposal)

print "Done"

print estim_A

exact_likelihood = first_try.estimate_likelihoods(sentences)
exact_likelihoods = np.zeros((n_iterations + 1, len(sentences)))

for i in xrange(n_iterations + 1):
    exact_likelihoods[i] = exact_likelihood

print likelihoods.shape
print exact_likelihoods.shape

first_try.plot_grammar_matrices('Examples','Estimated', estim_A, estim_B)
first_try.plot_grammar_matrices('Examples','Exact')
first_try.compare_grammar_matrices_3('Examples', 'Random_init', 
                                   estim_A, estim_B,
                                   A_proposal, B_proposal)


print likelihoods

plt.plot(np.log(likelihoods))
plt.plot(np.log(exact_likelihoods), linestyle = "--")
plt.show()