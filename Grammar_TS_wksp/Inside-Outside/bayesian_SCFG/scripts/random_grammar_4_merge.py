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
import cPickle as pickle

from SCFG.sto_grammar import SCFG, Looping_der_except

def plot_lks(left_grammar,
             right_grammar, 
             sample_sentences,
             title,
             xlabel,
             ylabel,
             file_name):
    sentence_set = list(set([' '.join(sentence) for sentence in sample_sentences]))
    sentence_set = [sentence.split(' ') for sentence in sentence_set]
    left_lks = np.log(left_grammar.estimate_likelihoods(sentence_set))
    right_lks = np.log(right_grammar.estimate_likelihoods(sentence_set))
    plt.scatter(left_lks, right_lks, alpha = 0.1, marker = 'o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_name, dpi = 300)
    plt.close()


model_gram = word_grammar

model_gram.draw_grammar('Model_grammar.png')

n_sentences = 100

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

n_reductions = model_gram.M + 4

original_text = '\r'.join([' '.join(x) for x in sentences])
print original_text + '\n'


folder_path = '/Users/francois/Grammar_TS/Grammar_TS_wksp/Inside-Outside/bayesian_SCFG/Results_7'
folder_name = 'Model'
model_gram.plot_grammar_matrices(folder_path,
                                 folder_name)

plot_lks(model_gram, 
         model_gram,
         sentences, 
         'Model grammar samples log lk',
         'Model grammar lk',
         'Model grammar lk',
         'Model_model_model_samples_lk.png')

distance_matrix_model = model_gram.compute_internal_distance_matrix(n_samples)
plt.matshow(distance_matrix_model)
plt.title('Distance matrix estimated estimated')
plt.xlabel('Model rules')
plt.xlabel('Model rules')
plt.savefig('Distance_matrix_model.png')

iteration = 0
 
while iteration < n_trials:
    try:
        print 'Doing iteration %d' % iteration
        if iteration == 0:
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
        else:
            print estim_gram.A.shape
            print estim_gram.B.shape
            N = estim_gram.A.shape[0]
            M = estim_gram.B.shape[1]
            estim_gram.A = np.maximum(estim_gram.A, 1e-6 * np.ones((N, N, N)))
            estim_gram.B = np.maximum(estim_gram.B, 1e-6 * np.ones((N, M)))
            new_A, new_B, likelihoods = word_grammar.estimate_A_B(samples = sentences, 
                                                                  n_iterations = 50, 
                                                                  init_option = 'explicit',
                                                                  A_proposal = estim_gram.A,
                                                                  B_proposal = estim_gram.B,
                                                                  term_chars = word_grammar.term_chars)
        plt.plot(np.log(likelihoods))
        plt.title('Log likelihoods iteration %d' % iteration)
        plt.ylabel('Sample log lk')
        plt.xlabel('EM step')
        plt.savefig('Likelihoods_%d.png' % iteration)
        plt.close()
        #
        estim_gram = SCFG()
        estim_gram.init_from_A_B(new_A, new_B,
                                 model_gram.term_chars)
        
        distance_matrix_estim = estim_gram.compute_internal_distance_matrix(n_samples)
        plt.matshow(distance_matrix_estim)
        plt.title('Distance matrix estimated, iter %d' % iteration)
        plt.xlabel('Estimated rules')
        plt.xlabel('Estimated rules')
        plt.savefig('Distance_matrix_estim_%d.png' % iteration)
        plt.close()
        #
        distance_matrix_model_estim = compute_distance_matrix(model_gram,
                                                              estim_gram,
                                                              n_samples)
        plt.matshow(distance_matrix_model_estim)
        plt.title('Distance matrix model estimated, iter %d' % iteration)
        plt.xlabel('Model rules')
        plt.ylabel('Estimated rules')
        plt.savefig('Distance_matrix_model_estim_%d.png' % iteration)
        plt.close()
        #
        estim_sentences = estim_gram.produce_sentences(n_sentences)
        #
        print '\nEstimation %d:' % iteration
        for sentence in estim_sentences:
            print ' '.join(sentence)
        print ''
        #
        plot_lks(model_gram,
                 estim_gram,
                 sentences,
                 'Model grammar samples lk, iter %d' % iteration,
                 'Model grammar log lk',
                 'Estim grammar log lk',
                 'Model_estim_model-samples_lk_%d.png' % iteration)
        #
        plot_lks(model_gram,
                 estim_gram,
                 estim_sentences,
                 'Estimated grammar samples lk, iter %d' % iteration,
                 'Model grammar log lk',
                 'Estim grammar log lk',
                 'Model_estim_estim-samples_lk_%d.png' % iteration)
        #
        estim_gram.plot_grammar_matrices(folder_path,
                                         'Estim_%d' % iteration)
        estim_gram.draw_grammar('Estimated_grammar_%d.png' % iteration)
        #
        #    Folding
        #
        estim_gram.merge_on_closest()
        estim_gram.draw_grammar('Estimated_grammar_folded_%d.png' % iteration)
        #
        distance_matrix_estim_folded = estim_gram.compute_internal_distance_matrix(n_samples)
        
        plt.matshow(distance_matrix_estim_folded)
        plt.title('Distance matrix estimated folded, iter %d' % iteration)
        plt.xlabel('Estimated rules folded')
        plt.xlabel('Estimated rules folded')
        plt.savefig('Distance_matrix_estim-folded_%d.png' % iteration)
        plt.close()
        #
        distance_matrix_model_estim_folded = compute_distance_matrix(model_gram,
                                                                      estim_gram,
                                                                      n_samples)
        plt.matshow(distance_matrix_model_estim_folded)
        plt.title('Distance matrix model estimated folded, iter %d' % iteration)
        plt.xlabel('Model rules')
        plt.xlabel('Estimated rules folded')
        plt.savefig('Distance_matrix_model_estim-folded_%d.png' % iteration)
        plt.close()
        saved_estim_gram = copy.deepcopy(estim_gram)
        iteration += 1
        print '\n'
    except Looping_der_except as e:
        print '\n\n\n'
        print '\t'
        print "'\t------------ ERROR ------------"
        print '\t'
        print e
        print '\n\n\n'
        estim_gram = saved_estim_gram
    
pickle.dump(saved_estim_gram, open('Final_grammar.pi', 'wb'))

saved_estim_gram.draw_grammar('Final_grammar_draw.png')

