'''
Created on 4 juin 2014

@author: francois
'''

from matplotlib import pyplot as plt

import numpy as np

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
max_represented = 100

for name, grammar in all_grammars.iteritems():
    grammar.plot_stats(n_samples, 
                       max_represented = 40,
                       filename = 'stats_' + name + '.png')
    
n_grammars = len(all_grammars.keys())
dist_dict = {}
dists = np.zeros((n_grammars, n_grammars))
for i, key_1 in enumerate(all_grammars.keys()):
    dist_dict[key_1] = {}
    for j, key_2 in enumerate(all_grammars.keys()):
        print 'Computing distance between %s and %s' % (key_1, key_2)
        dst = Grammar_distance(all_grammars[key_1], all_grammars[key_2])
        dist_dict[key_1][key_2] = dst.compute_distance(n_samples)
        print 'Distance between %s and %s = %f' % (key_1, key_2, dist_dict[key_1][key_2])
        dists[i][j] = dist_dict[key_1][key_2]
        
plt.matshow(dists)
plt.xticks(range(1,n_grammars+1), all_grammars.keys(), rotation = 'vertical', fontsize = 3)
plt.yticks(range(1,n_grammars+1), all_grammars.keys(), fontsize = 3)
plt.savefig('Dist_matrix', dpi = 600)

pickle.dump(dist_dict, open('Dist_dict.pi', 'wb'))
