'''
Created on 18 juil. 2014

@author: francois
'''

import numpy as np
import cPickle as pickle
import os

from SCFG.sto_grammar import SCFG, normalize_slices
from grammar_examples.grammar_examples import produce_word_grammar
from SCFG.grammar_distance import compute_distance_matrix

from matplotlib import pyplot as plt

probas = [0.5, 0.5, 0.5, 0.5]
probas = np.asarray(probas)
probas /= np.sum(probas)


target_grammar, rule_nick_names = produce_word_grammar(probas[0],
                                           probas[1],
                                           probas[2],
                                           probas[3])

model_grammar = SCFG()
model_grammar.init_from_A_B(target_grammar.A, 
                            target_grammar.B,
                            target_grammar.term_chars)

model_grammar.write_signature_to_file(10000,
                                      50,
                                      'model_sign.png',
                                      'Model signature')

n_in_signature = 50
n_samples = 1000
n_trials = 100
n_iterations = 20
max_n_steps = 3
n_iteration_post_trimming = 10
threshold = 0.01
N_range = np.arange(8, 9)
n_productions = 100

n_samples_MC = 1000

samples = model_grammar.produce_sentences(n_samples)

all_symbols = []
for sample in samples:
    all_symbols.extend(sample)
sample_term_symbols = list(set(all_symbols))

M = len(sample_term_symbols)

print "Sample term symbols:"
print sample_term_symbols

result_folder = 'results_word_grammar_8_symbols_elimination/'

os.mkdir(result_folder)

model_name = 'Word grammar'

model_grammar.draw_grammar(result_folder + model_name + ' graph.png')
model_grammar.write_sentences_to_file(n_productions,
                                      result_folder + model_name + ' prod.txt')

model_lks = np.log(model_grammar.estimate_likelihoods(samples))

estimation_results = {}
best_lks = {}
all_lks = []

for N in N_range:
    estimation_results[N] = {}
    max_lk_N = 0
    arg_max_lk_N = 0
    first = True
    for i_trial in range(n_trials):
        print "Doing trial %d with %d symbols" % (i_trial, N)
        A_init = np.random.uniform(0.01, 1.0, (N, N, N))
        B_init = np.random.uniform(0.01, 1.0, (N, M))
        normalize_slices(A_init, B_init)
        exp_grammar = SCFG()
        exp_grammar.init_from_A_B(A_init,
                                  B_init,
                                  sample_term_symbols)
        est_lk = []
        for step in xrange(max_n_steps - 1):
            est_A, est_B, temp = exp_grammar.estimate_A_B(samples,
                                                            n_iterations,
                                                            init_option = 'exact',
                                                            trim_near_zeros= True,
                                                            trim_threshold= threshold,
                                                            n_iterations_post= n_iteration_post_trimming)
            temp = np.log(temp)
            for line in temp:
                est_lk.append(line)
            exp_grammar.A = est_A
            exp_grammar.B = est_B
            exp_grammar.compute_internal_distance_matrix(n_samples = n_samples_MC,
                                                         JS = False)
            if not exp_grammar.is_diagonal():
                print '\t\tGrammar is not diagonal'
                exp_grammar.merge_on_closest()
                exp_grammar.add_virgin_rule()
            else:
                print '\t\tGrammar is diagonal'
                break
        print step
        est_A, est_B, temp = exp_grammar.estimate_A_B(samples,
                                                        n_iterations,
                                                        init_option = 'exact',
                                                        trim_near_zeros= True,
                                                        trim_threshold= threshold,
                                                        n_iterations_post= n_iteration_post_trimming)
        temp = np.log(temp)
        for line in temp:
            est_lk.append(line)
        est_lk = np.vstack(est_lk)
        print est_lk.shape
        estimation_results[N][i_trial] = (est_A, est_B, est_lk)
        avg_lk = np.mean(est_lk[-1,:])
        all_lks.append((N, i_trial, avg_lk))
        if first:
            max_lk_N = avg_lk
            arg_max_lk_N = 0
        if avg_lk > max_lk_N:
            max_lk_N = avg_lk
            arg_max_lk_N = i_trial
    best_lks[N] = (arg_max_lk_N, max_lk_N)
    print ''

all_lks = filter(lambda x : not np.isnan(x[-1]), all_lks)  
all_lks.sort(key = (lambda x : -x[-1]))

pickle.dump(all_lks, open(result_folder + 'all_lks.pi', 'wb'))
pickle.dump(best_lks, open(result_folder + 'best_lks.pi', 'wb'))
pickle.dump(estimation_results, open(result_folder + 'estimated_result.pi', 'wb'))

all_sample_lks = []
for rank, (N, i_trial, avg_lks) in enumerate(all_lks[:20]):
    est_grammar = SCFG()
    A, B, lks = estimation_results[N][i_trial]
    est_grammar.init_from_A_B(A, B, sample_term_symbols)
    est_grammar.draw_grammar(result_folder + 
                             ('grammar_%d_%d_%d_lk_%.2f_graph.png' 
                              % (rank, N, i_trial, avg_lks)))
    #
    sample_lks = np.log(est_grammar.estimate_likelihoods(samples))
    all_sample_lks.append(sample_lks)
    plt.scatter(model_lks, sample_lks)
    plt.title('%s versus est. grammar (%d symbols, trial %d)' 
              % (model_name, N, i_trial))
    plt.xlabel('Model log lk')
    plt.ylabel('Estimated grammar log lk')
    plt.savefig(result_folder + ('grammar_%d_%d_%d_lk_%.2f_lks.png' 
                              % (rank, N, i_trial, avg_lks)))
    plt.close()
    #
    plt.plot(lks)
    plt.title('%s est. grammar (%d symbols, trial %d) EM' 
              % (model_name, N, i_trial))
    plt.xlabel('Sample log lk')
    plt.ylabel('EM iteration')
    plt.savefig(result_folder + ('grammar_%d_%d_%d_lk_%.2f_EM.png' 
                              % (rank, N, i_trial, avg_lks)))
    plt.close()
    #
    dist_matrix_KL = compute_distance_matrix(model_grammar,
                                             est_grammar,
                                             n_samples, JS = False)
    plt.imshow(dist_matrix_KL, cmap = 'PuOr',
               interpolation = 'None')
    plt.title('KL dist matrix %s vs %s' 
              % ('est. grammar %d symbol, %d trial' % (N, i_trial),
                 'model grammar'))
    plt.colorbar()
    plt.savefig(result_folder + ('grammar_%d_%d_%d_lk_%.2f_cross_matrix.png' 
                              % (rank, N, i_trial, avg_lks)),
                dpi = 600)
    plt.close()
    #
    dist_matrix_KL = compute_distance_matrix(model_grammar,
                                             model_grammar,
                                             n_samples, JS = False)
    plt.imshow(dist_matrix_KL, cmap = 'PuOr',
               interpolation = 'None')
    plt.title('KL internal dist matrix %s' 
              % ('est. grammar %d symbol, %d trial' % (N, i_trial)))
    plt.colorbar()
    plt.savefig(result_folder + ('grammar_%d_%d_%d_lk_%.2f_inter_matrix.png' 
                              % (rank, N, i_trial, avg_lks)),
                dpi = 600)
    plt.close()
    #
    est_grammar.write_sentences_to_file(n_productions,
                                        result_folder + ('grammar_%d_%d_%d_lk_%.2f_prod.txt' 
                                        % (rank, N, i_trial, avg_lks)))
    est_grammar.write_signature_to_file(n_productions, n_in_signature,
                                        result_folder + ('grammar_%d_%d_%d_lk_%.2f_sign.png' 
                                        % (rank, N, i_trial, avg_lks)),
                                        '%s est. grammar (%d symbols, trial %d) signature' 
              % (model_name, N, i_trial))
    
all_sample_lks = np.asanyarray(all_sample_lks)
for sample_lk in all_sample_lks:
    plt.plot(model_lks, sample_lk,
             marker = 'x', linestyle = 'None')
plt.title('%s versus est. grammars (%d symbols)' 
          % (model_name, N))
plt.ylabel('Estimated grammar log lk')
plt.savefig(result_folder + ('all_grammars_%d.png' 
                          % N),
            dpi = 600)
plt.close()
    