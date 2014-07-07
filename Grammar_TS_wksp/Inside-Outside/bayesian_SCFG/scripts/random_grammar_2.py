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

def plot_lks(left_grammar, 
             right_grammar, 
             sample_sentences,
             file_name):
    sentence_set = list(set([' '.join(sentence) for sentence in sample_sentences]))
    sentence_set = [sentence.split(' ') for sentence in sentence_set]
    left_lks = np.log(left_grammar.estimate_likelihoods(sentence_set))
    right_lks = np.log(right_grammar.estimate_likelihoods(sentence_set))
    plt.scatter(left_lks, right_lks, alpha = 0.1, marker = 'o')
    """
    print np.all(left_lks == right_lks)
    if(np.all(left_lks == right_lks)):
        left_lk_count = {}
        for i in xrange(len(sentence_set)):
            sentence = sentence_set[i]
            left_lk = left_lks[i]
            right_lk = right_lks[i]
            if left_lk not in left_lk_count:
                left_lk_count[left_lk] = 0
            plt.text(right_lk, 
                     left_lk - left_lk_count[left_lk] * 0.1,
                     ' '.join(sentence),
                     fontsize = 6)
            left_lk_count[left_lk] += 1
    else:
        for i, sentence in enumerate(sentence_set):
            plt.text(right_lks[i], 
                     left_lks[i],
                     ' '.join(sentence),
                     fontsize = 6)            
    fig = plt.gcf()
    fig.set_size_inches((20, 20))
    """
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

folder_path = '/Users/francois/Grammar_TS/Grammar_TS_wksp/Inside-Outside/bayesian_SCFG/Results_7'
folder_name = 'Model'
model_gram.plot_grammar_matrices(folder_path,
                                 folder_name)

plot_lks(model_gram, 
         model_gram, 
         sentences, 
         'Model_model_model_samples_lk.png')

distance_matrix_model = compute_distance_matrix(model_gram, 
                                                model_gram,
                                                n_samples)

plt.matshow(distance_matrix_model)
plt.savefig('Distance_matrix_model.png')

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

plt.plot(likelihoods)
plt.savefig('Likelihoods.png')
plt.close()

estim_gram = SCFG()
estim_gram.init_from_A_B(new_A, new_B,
                         model_gram.term_chars)

distance_matrix_estim = compute_distance_matrix(estim_gram,
                                                estim_gram,
                                                n_samples)

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

for sentence in estim_sentences:
    print ' '.join(sentence)

plot_lks(estim_gram, 
         estim_gram, 
         sentences, 
         'Estim_estim_model_samples_lk.png')

plot_lks(model_gram,
         estim_gram,
         sentences,
         'Model_estim_model_samples_lk.png')

plot_lks(model_gram,
         estim_gram,
         estim_sentences,
         'Model_estim_estim_samples_lk.png')

estim_gram.plot_grammar_matrices(folder_path,
                                 'Estim')




