'''
Created on 1 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy

from SCFG.sto_grammar import SCFG

from Grammar_examples import palindrom_grammar

selected_grammar = palindrom_grammar

n_samples = 100
sentences = selected_grammar.produce_sentences(n_samples)

n_iterations = 50

mean = 0
sigma = 0.1
noise_source = np.random.normal

fold_grammar = copy.deepcopy(selected_grammar)
fold_grammar.merge(1, 3)
fold_grammar.merge(2, 4)

temp_exact_likelihoods = selected_grammar.estimate_likelihoods(sentences)
exact_likelihoods = np.zeros((n_iterations + 1, len(sentences)), dtype = np.double)

for i in xrange(n_iterations + 1):
    exact_likelihoods[i,:] = temp_exact_likelihoods

estim_A, estim_B, likelihoods = selected_grammar.estimate_A_B(sentences,
                                                              n_iterations,
                                                              'perturbated',
                                                              noise_source_A = noise_source, 
                                                              param_1_A = mean, 
                                                              param_2_A = sigma, 
                                                              epsilon_A = 0.01, 
                                                              noise_source_B = noise_source,
                                                              param_1_B = mean,
                                                              param_2_B = sigma,
                                                              epsilon_B = 0.01)

fold_estim_A, fold_estim_B, fold_likelihoods = fold_grammar.estimate_A_B(sentences,
                                                                         n_iterations,
                                                                         'perturbated',
                                                                         noise_source_A = noise_source, 
                                                                         param_1_A = mean, 
                                                                         param_2_A = sigma, 
                                                                         epsilon_A = 0.01, 
                                                                         noise_source_B = noise_source,
                                                                         param_1_B = mean,
                                                                         param_2_B = sigma,
                                                                         epsilon_B = 0.01)

plt.subplot(211)
plt.plot(np.log(likelihoods))
plt.plot(np.log(exact_likelihoods), linestyle = '--')
plt.ylabel("Log lk")
plt.title("Estimation without folding")
plt.subplot(212)
plt.plot(np.log(fold_likelihoods))
plt.plot(np.log(exact_likelihoods), linestyle = '--')
plt.ylabel("Log lk")
plt.title("Estimation with 2 foldings")
plt.savefig("Effect_of_folding.png", dpi = 300)
plt.close()

def normalize_slices(A, B):
    assert(A.ndim == 3)
    assert(B.ndim == 2)
    assert(A.shape[0] == A.shape[1] == A.shape[2] == B.shape[0])
    for i in xrange(A.shape[0]):
        total = np.sum(A[i,:,:]) + np.sum(B[i,:])
        A[i,:,:] /= total
        B[i,:] /= total
    return A, B

A = selected_grammar.A
B = selected_grammar.B
terms = selected_grammar.term_chars

N = A.shape[0]
M = B.shape[1]

small_sigma = 0.1
medium_sigma = 0.5
big_sigma = 1.0

n_grammars = 100

zero_sigma_likelihoods = []
for i in xrange(n_grammars):
    zero_sigma_likelihoods.append(selected_grammar.estimate_likelihoods(sentences))

small_sigma_likelihoods = []
for i in xrange(n_grammars):
    temp_grammar = SCFG()
    new_A = A + np.random.normal(0.0, small_sigma, (N, N, N))
    new_A = np.maximum(new_A, 0.01 * np.ones((N, N, N)))
    new_B = np.copy(B)
    
    temp_grammar.init_from_A_B(new_A,
                               new_B,
                               terms)
    small_sigma_likelihoods.append(temp_grammar.estimate_likelihoods(sentences))
   
medium_sigma_likelihoods = []
for i in xrange(n_grammars):
    temp_grammar = SCFG()
    new_A = A + np.random.normal(0.0, medium_sigma, (N, N, N))
    new_A = np.maximum(new_A, 0.01 * np.ones((N, N, N)))
    new_B = np.copy(B)
    
    temp_grammar.init_from_A_B(new_A,
                               new_B,
                               terms)
    medium_sigma_likelihoods.append(temp_grammar.estimate_likelihoods(sentences))
   
big_sigma_likelihoods = []
for i in xrange(n_grammars):
    temp_grammar = SCFG()
    new_A = A + np.random.normal(0.0, big_sigma, (N, N, N))
    new_A = np.maximum(new_A, 0.01 * np.ones((N, N, N)))
    new_B = np.copy(B)
    
    temp_grammar.init_from_A_B(new_A,
                               new_B,
                               terms)
    big_sigma_likelihoods.append(temp_grammar.estimate_likelihoods(sentences))
    

plt.subplot(221)
plt.plot(np.log(zero_sigma_likelihoods))
plt.title('Exact likelihoods')
plt.ylabel("Log lk")
plt.subplot(222)
plt.plot(np.log(small_sigma_likelihoods))
plt.title('Perturbation 0.1 sigma')
plt.ylabel("Log lk")
plt.subplot(223)
plt.plot(np.log(medium_sigma_likelihoods))
plt.title('Perturbation 0.5 sigma')
plt.ylabel("Log lk")
plt.subplot(224)
plt.plot(np.log(big_sigma_likelihoods))
plt.title('Perturbation 1.0 sigma')
plt.ylabel("Log lk")
plt.savefig("Stability_of_lk.png", dpi = 300)
plt.close()




