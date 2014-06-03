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

selected_grammar = palindrom_grammar

N = selected_grammar.N
M = selected_grammar.M

n_samples = 100
n_iterations = 50

sentences = selected_grammar.produce_sentences(n_samples)
    
exact_likelihood = selected_grammar.estimate_likelihoods(sentences)
exact_likelihoods = np.zeros((n_iterations + 1, len(sentences)))
for i in xrange(n_iterations + 1):
    exact_likelihoods[i] = exact_likelihood
    
A_proposal = np.random.uniform(0.0, 1.0, (N, N, N))
B_proposal = np.random.uniform(0.0, 1.0, (N, M))
A_proposal = np.maximum(A_proposal, 0.1 * np.ones((N, N, N)))
B_proposal = np.maximum(B_proposal, 0.1 * np.ones((N, M)))

estim_A, estim_B, likelihoods = selected_grammar.estimate_A_B(sentences,
                                                              n_iterations, 
                                                              'explicit',
                                                              A_proposal,
                                                              B_proposal)

selected_grammar.merge(2, 4)
N = selected_grammar.N - 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'explicit',
                                                                  A_proposal,
                                                                  B_proposal)

selected_grammar.merge(3, 5)
N = selected_grammar.N - 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'explicit',
                                                                  A_proposal,
                                                                  B_proposal)


plt.subplot(311)
plt.plot(np.log(likelihoods))
plt.plot(np.log(exact_likelihoods), linestyle = "--")
plt.subplot(312)
plt.plot(np.log(new_likelihoods))
plt.plot(np.log(exact_likelihoods), linestyle = "--")
plt.subplot(313)
plt.plot(np.log(new_likelihoods))
plt.plot(np.log(exact_likelihoods), linestyle = "--")
plt.show()