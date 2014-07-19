'''
Created on 19 juil. 2014

@author: francois
'''

import numpy as np
import cPickle as pickle
from matplotlib import pyplot as plt
import multiprocessing as multi
import os
import copy

from SCFG.sto_grammar import SCFG, normalize_slices

N_PROCESSES = 4

import load_data

data = load_data.no_rep_achu_file_contents
file_names = load_data.achu_file_names

result_folder = 'achu'
source_name = 'Achu'

for key, value in data.iteritems():
    data[key] = value.split(' ')[:40]


def do_test_with_N_symbols(N, folder_name):
    #
    n_trials = 20
    n_iterations = 50
    n_iteration_post_trimming = 20
    threshold = 0.01
    n_productions = 100
    #
    samples = copy.deepcopy([data[key] for key in file_names])
    #
    all_symbols = []
    for sample in samples:
        all_symbols.extend(sample)
    all_symbols = list(set(all_symbols))
    #
    sample_term_symbols = all_symbols
    M = len(sample_term_symbols)
    #
    estimation_results = {}
    all_lks = []
    #
    os.mkdir(folder_name)
    #
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
        est_A, est_B, est_lk = exp_grammar.estimate_A_B(samples,
                                                        n_iterations,
                                                        init_option = 'exact',
                                                        trim_near_zeros= True,
                                                        trim_threshold= threshold,
                                                        n_iterations_post= n_iteration_post_trimming)
        est_lk = np.log(est_lk)
        estimation_results[i_trial] = (est_A, est_B, est_lk)
        avg_lk = np.mean(est_lk[:,-1])
        all_lks.append((N, i_trial, avg_lk))
        if first:
            max_lk_N = avg_lk
            arg_max_lk_N = 0
        if avg_lk > max_lk_N:
            max_lk_N = avg_lk
            arg_max_lk_N = i_trial
        best_lks = (arg_max_lk_N, max_lk_N)
        print ''
    all_lks.sort(key = (lambda x : -x[-1]))
    pickle.dump(all_lks, open(folder_name + 'all_lks.pi', 'wb'))
    pickle.dump(best_lks, open(folder_name + 'best_lks.pi', 'wb'))
    pickle.dump(estimation_results, open(folder_name + 'estimated_result.pi', 'wb'))
    all_sample_lks = []
    for rank, (N, i_trial, avg_lks) in enumerate(all_lks[:20]):
        est_grammar = SCFG()
        A, B, lks = estimation_results[i_trial]
        est_grammar.init_from_A_B(A, B, sample_term_symbols)
        est_grammar.draw_grammar(folder_name + 
                                 ('grammar_%d_%d_%d_lk_%.2f_graph.png' 
                                  % (rank, N, i_trial, avg_lks)))
        #
        sample_lks = np.log(est_grammar.estimate_likelihoods(samples))
        all_sample_lks.append(sample_lks)
        plt.scatter(range(len(sample_lks)), sample_lks)
        plt.title('%s versus est. grammar (%d symbols, trial %d)' 
                  % (source_name, N, i_trial))
        plt.xticks(range(len(sample_lks)), file_names,
                   rotation = 'vertical', fontsize = 8)
        plt.ylabel('Estimated grammar log lk')
        plt.savefig(folder_name + ('grammar_%d_%d_%d_lk_%.2f_lks.png' 
                                  % (rank, N, i_trial, avg_lks)))
        plt.close()
        #
        plt.plot(lks)
        plt.title('%s est. grammar (%d symbols, trial %d) EM' 
                  % (source_name, N, i_trial))
        plt.xlabel('Sample log lk')
        plt.ylabel('EM iteration')
        plt.savefig(folder_name + ('grammar_%d_%d_%d_lk_%.2f_EM.png' 
                                  % (rank, N, i_trial, avg_lks)))
        plt.close()
        #
        est_grammar.write_sentences_to_file(n_productions,
                                            folder_name + ('grammar_%d_%d_%d_lk_%.2f_prod.txt' 
                                            % (rank, N, i_trial, avg_lks)))
        est_grammar.write_signature_to_file(n_productions, 20,
                                            folder_name + ('grammar_%d_%d_%d_lk_%.2f_sign.png' 
                                            % (rank, N, i_trial, avg_lks)),
                                            '%s est. grammar (%d symbols, trial %d) signature' 
                  % (folder_name, N, i_trial))
    all_sample_lks = np.asanyarray(all_sample_lks)
    for sample_lk in all_sample_lks:
        plt.plot(range(len(sample_lks)), sample_lk,
                 marker = 'x', linestyle = 'None')
    filtered_lks = np.ravel(all_sample_lks)
    filtered_lks = filter(lambda x : np.isfinite(x) and not np.isnan(x), filtered_lks)
    avg = np.mean(filtered_lks)
    plt.title('%s, lk est. grammars (%d symbols, avg = %.2f)' 
              % (source_name, N, avg))
    plt.xticks(range(len(sample_lks)), file_names,
               rotation = 'vertical', fontsize = 6)
    plt.ylabel('Estimated grammar log lk')
    plt.savefig(folder_name + ('all_grammars_%d_%.2f.png' 
                              % (N, avg)),
                dpi = 600)
    plt.close()
        
for N in xrange(6, 14):
    folder_name = result_folder + '_' + str(N) + '/'
    do_test_with_N_symbols(N, folder_name)
    
    