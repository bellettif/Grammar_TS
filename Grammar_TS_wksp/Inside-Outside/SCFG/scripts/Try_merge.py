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

mean_A = 0.0
sigma_A = 0.05
epsilon_A = 0.01

mean_B = 0.0
sigma_B = 0.05
epsilon_B = 0.01

sentences = selected_grammar.produce_sentences(n_samples)
    
exact_likelihood = selected_grammar.estimate_likelihoods(sentences)
    
A_proposal = np.random.uniform(0.0, 1.0, (N, N, N))
B_proposal = np.random.uniform(0.0, 1.0, (N, M))
A_proposal = np.maximum(A_proposal, 0.1 * np.ones((N, N, N)))
B_proposal = np.maximum(B_proposal, 0.1 * np.ones((N, M)))

all_likelihoods = []

estim_A, estim_B, likelihoods = selected_grammar.estimate_A_B(sentences,
                                                              n_iterations, 
                                                              'explicit',
                                                              A_proposal,
                                                              B_proposal)
all_likelihoods.extend(np.copy(likelihoods))

selected_grammar.expand(3)
selected_grammar.blur_A_B('keep_zeros_B',
                          noise_source_A = np.random.normal,
                          param_1_A = mean_A,
                          param_2_A = sigma_A,
                          epsilon_A = epsilon_A,
                          noise_source_B = np.random.normal,
                          param_1_B = mean_B,
                          param_2_B = sigma_B,
                          epsilon_B = epsilon_B)
N = selected_grammar.N + 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'exact')
all_likelihoods.extend(np.copy(new_likelihoods))

selected_grammar.merge(2, 4)
selected_grammar.blur_A_B('keep_zeros_B',
                          noise_source_A = np.random.normal,
                          param_1_A = mean_A,
                          param_2_A = sigma_A,
                          epsilon_A = epsilon_A,
                          noise_source_B = np.random.normal,
                          param_1_B = mean_B,
                          param_2_B = sigma_B,
                          epsilon_B = epsilon_B)
N = selected_grammar.N - 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'exact')
all_likelihoods.extend(np.copy(new_likelihoods))

selected_grammar.merge(3, 5)
selected_grammar.blur_A_B('keep_zeros_B',
                          noise_source_A = np.random.normal,
                          param_1_A = mean_A,
                          param_2_A = sigma_A,
                          epsilon_A = epsilon_A,
                          noise_source_B = np.random.normal,
                          param_1_B = mean_B,
                          param_2_B = sigma_B,
                          epsilon_B = epsilon_B)
N = selected_grammar.N - 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'exact')
all_likelihoods.extend(np.copy(new_likelihoods))

selected_grammar.expand(4)
selected_grammar.blur_A_B('keep_zeros_B',
                          noise_source_A = np.random.normal,
                          param_1_A = mean_A,
                          param_2_A = sigma_A,
                          epsilon_A = epsilon_A,
                          noise_source_B = np.random.normal,
                          param_1_B = mean_B,
                          param_2_B = sigma_B,
                          epsilon_B = epsilon_B)
N = selected_grammar.N + 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'exact')
all_likelihoods.extend(np.copy(new_likelihoods))

selected_grammar.merge(1, 3)
selected_grammar.blur_A_B('keep_zeros_B',
                          noise_source_A = np.random.normal,
                          param_1_A = mean_A,
                          param_2_A = sigma_A,
                          epsilon_A = epsilon_A,
                          noise_source_B = np.random.normal,
                          param_1_B = mean_B,
                          param_2_B = sigma_B,
                          epsilon_B = epsilon_B)
N = selected_grammar.N - 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'exact')
all_likelihoods.extend(np.copy(new_likelihoods))

selected_grammar.expand(1)
selected_grammar.blur_A_B('keep_zeros_B',
                          noise_source_A = np.random.normal,
                          param_1_A = mean_A,
                          param_2_A = sigma_A,
                          epsilon_A = epsilon_A,
                          noise_source_B = np.random.normal,
                          param_1_B = mean_B,
                          param_2_B = sigma_B,
                          epsilon_B = epsilon_B)
N = selected_grammar.N + 1
A_proposal = estim_A
B_proposal = estim_B
estim_A, estim_B, new_likelihoods = selected_grammar.estimate_A_B(sentences,
                                                                  n_iterations, 
                                                                  'exact',
                                                                  A_proposal,
                                                                  B_proposal)
all_likelihoods.extend(np.copy(new_likelihoods))




all_likelihoods = np.asanyarray(all_likelihoods)

print all_likelihoods.shape
print all_likelihoods

exact_likelihoods = np.zeros((all_likelihoods.shape[0], len(sentences)))
for i in xrange(all_likelihoods.shape[0]):
    exact_likelihoods[i,:] = exact_likelihood

plt.plot(np.log(all_likelihoods))
plt.plot(np.log(exact_likelihoods), linestyle = "--")
plt.show()