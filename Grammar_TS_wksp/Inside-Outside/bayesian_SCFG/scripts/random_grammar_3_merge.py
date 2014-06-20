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
import copy

from SCFG.sto_grammar import SCFG

def plot_lks(left_grammar, 
             right_grammar, 
             sample_sentences,
             file_name):
    sentence_set = list(set([' '.join(sentence) for sentence in sample_sentences]))
    sentence_set = [sentence.split(' ') for sentence in sentence_set]
    left_lks = np.log(left_grammar.estimate_likelihoods(sentence_set))
    right_lks = np.log(right_grammar.estimate_likelihoods(sentence_set))
    plt.scatter(left_lks, right_lks, alpha = 0.1, marker = 'o')
    plt.savefig(file_name, dpi = 300)
    plt.close()


model_gram = word_grammar

n_sentences = 200

sentences = model_gram.produce_sentences(n_sentences)

N = model_gram.N + model_gram.M
M = len(model_gram.term_chars)

print model_gram.term_chars

first_distances = []
last_distances = []
all_last_likelihoods = []
all_first_likelihoods = []

n_trials = 10
n_samples = 1e3

original_text = '\r'.join([' '.join(x) for x in sentences])

print original_text + '\n'

folder_path = '/Users/francois/Grammar_TS/Grammar_TS_wksp/Inside-Outside/bayesian_SCFG/Results_3'
folder_name = 'Model'
model_gram.plot_grammar_matrices(folder_path,
                                 folder_name)

plot_lks(model_gram, 
         model_gram, 
         sentences, 
         'Model_model_model_samples_lk.png')

distance_matrix_model = model_gram.compute_internal_distance_matrix(n_samples)

plt.matshow(distance_matrix_model)
plt.savefig('Distance_matrix_model.png')
plt.close()

new_A, new_B, likelihoods = word_grammar.estimate_A_B(samples = sentences, 
                                                      n_iterations = 50, 
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

plt.plot(np.log(likelihoods))
plt.savefig('Likelihoods.png')
plt.close()

estim_gram = SCFG()
estim_gram.init_from_A_B(new_A, new_B,
                         model_gram.term_chars)

distance_matrix_estim = estim_gram.compute_internal_distance_matrix(n_samples)

plt.matshow(distance_matrix_estim)
plt.savefig('Distance_matrix_estim.png')
plt.close()

distance_matrix_model_estim = compute_distance_matrix(model_gram,
                                                      estim_gram,
                                                      n_samples)
plt.matshow(distance_matrix_model_estim)
plt.savefig('Distance_matrix_model_estim.png')
plt.close()

estim_sentences = estim_gram.produce_sentences(n_sentences)

print '\nBefore fold:'
for sentence in estim_sentences:
    print ' '.join(sentence)

plot_lks(model_gram,
         estim_gram,
         sentences,
         'Model_estim_model-samples_lk.png')

plot_lks(model_gram,
         estim_gram,
         estim_sentences,
         'Model_estim_estim-samples_lk.png')

estim_gram.plot_grammar_matrices(folder_path,
                                 'Estim')

print estim_gram.ranked_internal_distances


#
#    Folding
#


estim_gram_folded = copy.deepcopy(estim_gram)
estim_gram_folded.merge_on_closest()

distance_matrix_estim_folded = estim_gram_folded.compute_internal_distance_matrix(n_samples)

plt.matshow(distance_matrix_estim_folded)
plt.savefig('Distance_matrix_estim-folded.png')
plt.close()

distance_matrix_model_estim_folded = compute_distance_matrix(model_gram,
                                                              estim_gram_folded,
                                                              n_samples)
plt.matshow(distance_matrix_model_estim_folded)
plt.savefig('Distance_matrix_model_estim-folded.png')
plt.close()

distance_matrix_estim_estim_folded = compute_distance_matrix(estim_gram,
                                                              estim_gram_folded,
                                                              n_samples)
plt.matshow(distance_matrix_estim_estim_folded)
plt.savefig('Distance_matrix_estim_estim-folded.png')
plt.close()

estim_folded_sentences = estim_gram_folded.produce_sentences(n_sentences)

print '\nAfter fold:'
for sentence in estim_folded_sentences:
    print ' '.join(sentence)

plot_lks(model_gram,
         estim_gram_folded,
         sentences,
         'Model_estim-folded_model-samples_lk.png')

plot_lks(model_gram,
         estim_gram_folded,
         estim_sentences,
         'Model_estim-folded_estim-samples_lk.png')

plot_lks(model_gram,
         estim_gram_folded,
         estim_folded_sentences,
         'Model_estim-folded_folded-samples_lk.png')

plot_lks(estim_gram,
         estim_gram_folded,
         sentences,
         'Estim_estim-folded_model-samples_lk.png')

plot_lks(estim_gram,
         estim_gram_folded,
         estim_sentences,
         'Estim_estim-folded_estim-samples_lk.png')

plot_lks(estim_gram,
         estim_gram_folded,
         estim_folded_sentences,
         'Estim_estim-folded_folded-samples_lk.png')

estim_gram_folded.plot_grammar_matrices(folder_path,
                                        'Estim_folded')