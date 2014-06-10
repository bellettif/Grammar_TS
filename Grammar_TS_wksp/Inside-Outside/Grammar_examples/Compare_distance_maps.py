'''
Created on 4 juin 2014

@author: francois
'''

from matplotlib import pyplot as plt

import numpy as np
import time

import cPickle as pickle


from SCFG.grammar_distance import Grammar_distance

from Grammar_examples import palindrom_grammar_1, \
                             palindrom_grammar_2, \
                             palindrom_grammar_3, \
                             repetition_grammar_1, \
                             repetition_grammar_2, \
                             embedding_grammar_central_1, \
                             embedding_grammar_central_2, \
                             embedding_grammar_left_right_1, \
                             embedding_grammar_left_right_2, \
                             name_grammar_1, \
                             name_grammar_2, \
                             action_grammar_1, \
                             action_grammar_2

all_grammars = {'palindrom_grammar_1' : palindrom_grammar_1,
               'palindrom_grammar_2' : palindrom_grammar_2,
               'palindrom_grammar_3' : palindrom_grammar_3,
               'repetition_grammar_1' : repetition_grammar_1,
               'repetition_grammar_2' : repetition_grammar_2,
               'embedding_grammar_central_1' : embedding_grammar_central_1,
               'embedding_grammar_central_2' : embedding_grammar_central_2,
               'embedding_grammar_left_right_1' : embedding_grammar_left_right_1,
               'embedding_grammar_left_right_2' : embedding_grammar_left_right_2,
               'name_grammar_1' : name_grammar_1,
               'name_grammar_2' : name_grammar_2,
               'action_grammar_1' : action_grammar_1,
               'action_grammar_2' : action_grammar_2}


n_samples = 1e6

for epsilon in np.linspace(0, 0.5, 10, True):
    n_grammars = len(all_grammars.keys())
    #dists_lk = np.zeros((n_grammars, n_grammars))
    dists_MC = np.zeros((n_grammars, n_grammars))
    tot_time_lk = 0
    tot_time_MC = 0
    for i, key_1 in enumerate(all_grammars.keys()):
        for j, key_2 in enumerate(all_grammars.keys()):
            print 'Computing lk distance between %s and %s' % (key_1, key_2)
            dst = Grammar_distance(all_grammars[key_1], all_grammars[key_2])
            begin = time.clock()
            #dists_lk[i][j] = dst.compute_distance(n_samples)
            tot_time_lk +=  time.clock() - begin
            print 'Computing MC distance between %s and %s' % (key_1, key_2)
            begin = time.clock()
            dists_MC[i][j] = dst.compute_distance_MC(n_samples,
                                                     max_length = 0,
                                                     epsilon = epsilon)
            tot_time_MC += time.clock() - begin
            print ''
    print 'Total time for lk: %f' % tot_time_lk
    print 'Total time for MC: %f' % tot_time_MC
    """          
    plt.matshow(dists_lk)
    plt.title('Distance lk')
    plt.xticks(range(1,n_grammars+1), all_grammars.keys(), rotation = 'vertical', fontsize = 3)
    plt.yticks(range(1,n_grammars+1), all_grammars.keys(), fontsize = 3)
    plt.savefig('Distance_comparison/Dist_matrix_comparison_lk', dpi = 600)
    """
    plt.matshow(dists_MC)
    plt.title('Distance MC')
    plt.xticks(range(1,n_grammars+1), all_grammars.keys(), rotation = 'vertical', fontsize = 3)
    plt.yticks(range(1,n_grammars+1), all_grammars.keys(), fontsize = 3)
    plt.savefig('Distance_comparison/Dist_matrix_comparison_MC_%f.png' % epsilon, dpi = 600)
    





