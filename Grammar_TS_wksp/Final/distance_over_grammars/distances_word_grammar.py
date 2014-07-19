'''
Created on 18 juil. 2014

@author: francois
'''

import numpy as np
import cPickle as pickle

from SCFG.sto_grammar import SCFG, normalize_slices
from grammar_examples.grammar_examples import produce_word_grammar
from SCFG.grammar_distance import compute_distance
from plot_convention.colors import b_blue, b_orange

from matplotlib import pyplot as plt

n_grammars = 100

n_samples = 1e4
probas = [0.5, 0.5, 0.5, 0.5]
probas = np.asarray(probas)
probas /= np.sum(probas)
print probas
original_grammar, rule_nick_names = produce_word_grammar(probas[0],
                                           probas[1],
                                           probas[2],
                                           probas[3])
KL_distances = np.zeros(n_grammars)
JS_distances = np.zeros(n_grammars)
euclidian_distances = np.zeros(n_grammars)
proba_strings = []

for i in xrange(n_grammars):
    new_probas = np.random.uniform(0.01, 1, 4)
    proba_strings.append(('%.2f %.2f %.2f %.2f' % (new_probas[0],
                                                     new_probas[1],
                                                     new_probas[2],
                                                     new_probas[3])))
    euclidian_distances[i] = np.sqrt(np.sum((new_probas - probas)**2))
    new_grammar, rule_nick_names = produce_word_grammar(new_probas[0],
                                           new_probas[1],
                                           new_probas[2],
                                           new_probas[3])
    KL_distances[i] = compute_distance(original_grammar,
                                       new_grammar,
                                       n_samples,
                                       JS = False)
    JS_distances[i] = compute_distance(original_grammar,
                                       new_grammar,
                                       n_samples,
                                       JS = True)
    print '%d done' % i

plt.scatter(euclidian_distances, KL_distances, color = b_orange)
plt.scatter(euclidian_distances, JS_distances, color = b_blue)
plt.xlabel('Euclidian distance between source probas')
plt.xticks(euclidian_distances, proba_strings, 
           rotation = 'vertical',
           fontsize = 4)
plt.ylabel('Sym divergence')
plt.title('Sym divergences of word grammars')
plt.legend(('KL', 'JS'))
plt.savefig('Sym_word_grammar.png', dpi = 600)