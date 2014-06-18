'''
Created on 16 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
from Grammar_examples.Grammar_examples import word_grammar
from SCFG.grammar_distance import Grammar_distance
from SCFG.internal_grammar_distance import compute_distance_matrix,\
    compute_internal_distance_matrix_MC

from SCFG.sto_grammar import SCFG

sentences = word_grammar.produce_sentences(100)

N = word_grammar.N + word_grammar.M
M = len(word_grammar.term_chars)

print word_grammar.term_chars

first_distances = []
last_distances = []
all_last_likelihoods = []
all_first_likelihoods = []

n_trials = 10

original_text = '\r'.join([' '.join(x) for x in sentences])

print original_text + '\n'

folder_path = '/Users/francois/Grammar_TS/Grammar_TS_wksp/Inside-Outside/bayesian_SCFG/Results_1'
folder_name = 'Original'
word_grammar.plot_grammar_matrices(folder_path, 
                                   folder_name)

first_n_iters = 35
second_n_iters = 15

n_samples = 1e4

for i in xrange(n_trials):
    print 'Doing %d' % i
    #
    new_A, new_B, likelihoods = word_grammar.estimate_A_B(samples = sentences, 
                                                          n_iterations = first_n_iters, 
                                                          init_option = 'random',
                                                          A_proposal = np.zeros((N, N, N)),
                                                          B_proposal = np.zeros((N, M)),
                                                          term_chars = word_grammar.term_chars,
                                                          noise_source_A = 0,
                                                          param_1_A = 0.1,
                                                          param_2_A = 0.1,
                                                          epsilon_A = 0,
                                                          noise_source_B = 0,
                                                          param_1_B = 0.1,
                                                          param_2_B = 0,
                                                          epsilon_B = 0)
    word_grammar_estim = SCFG()
    word_grammar_estim.init_from_A_B(new_A, new_B, word_grammar.term_chars)
    gd = Grammar_distance(word_grammar, word_grammar_estim)
    first_distances.append(gd.compute_distance(n_samples))
    all_first_likelihoods.append(np.log(np.copy(likelihoods[-1])))
    folder_name = "Estim_%d_%d" % (i, first_n_iters)
    word_grammar_estim.plot_grammar_matrices(folder_path, 
                                             folder_name)
    distance_matrix = compute_distance_matrix(word_grammar, word_grammar_estim, n_samples)
    plt.imshow(distance_matrix,
               interpolation = 'None')
    plt.savefig(folder_path + '/' + folder_name + '/distance_matrix.png')
    plt.close()
    estimated_text = '\r'.join([' '.join(x) for x in word_grammar_estim.produce_sentences(20)])
    print 'Likelihoods: '
    print likelihoods[-1]
    print estimated_text
    print ''
    #
    new_A, new_B, likelihoods = word_grammar.estimate_A_B(samples = sentences, 
                                                          n_iterations = 15, 
                                                          init_option = 'explicit',
                                                          A_proposal = new_A,
                                                          B_proposal = new_B,
                                                          term_chars = word_grammar.term_chars)
    word_grammar_estim = SCFG()
    word_grammar_estim.init_from_A_B(new_A, new_B, word_grammar.term_chars)
    gd = Grammar_distance(word_grammar, word_grammar_estim)
    last_distances.append(gd.compute_distance(n_samples))
    all_last_likelihoods.append(np.log(np.copy(likelihoods[-1])))
    folder_name = "Estim_%d_%d" % (i, second_n_iters)
    word_grammar_estim.plot_grammar_matrices(folder_path, 
                                             folder_name)
    distance_matrix = compute_distance_matrix(word_grammar, word_grammar_estim, n_samples)
    plt.imshow(distance_matrix,
               interpolation = 'None')
    plt.savefig(folder_path + '/' + folder_name + '/distance_matrix.png')
    plt.close()
    estimated_text = '\r'.join([' '.join(x) for x in word_grammar_estim.produce_sentences(20)])
    print 'Likelihoods: '
    print likelihoods[-1]
    print estimated_text
    print ''
    #
    print '\n'
    print 'Done'


plt.subplot(211)
for i in xrange(len(first_distances)):
    plt.plot(first_distances[i] * np.ones(len(all_first_likelihoods[i])), 
             all_first_likelihoods[i],
             linestyle = 'None',
             marker = 'v',
             alpha = 0.1)
plt.subplot(212)
for i in xrange(len(last_distances)):
    plt.plot(last_distances[i] * np.ones(len(all_last_likelihoods[i])), 
             all_last_likelihoods[i],
             linestyle = 'None',
             marker = '^',
             alpha = 0.1)
plt.show()

"""
plt.plot(likelihoods)
plt.show()
"""