'''
Created on 20 mai 2014

@author: francois
'''

import SCFG_c

import numpy as np
from matplotlib import pyplot as plt

from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG

rule_1 = Sto_rule(1,
                  [0.2, 0.3, 0.1, 0.4],
                  [[1, 1], [1, 2], [2, 1], [2, 2]],
                  [2.4, 0.1, 0.1],
                  ['Bernadette', 'Colin', 'Michel'])

rule_2 = Sto_rule(2,
                  [0.2, 0.3, 0.4, 0.1],
                  [[2, 2], [2, 1], [1, 2], [1, 1]],
                  [0.1, 2.0],
                  ['Pierre', 'Mathieu'])

rule_3 = Sto_rule(3,
                  [0.6, 0.5, 0.4, 0.8],
                  [[1, 1], [1, 2], [2, 1], [2, 2]],
                  [],
                  [])

main_grammar = SCFG([rule_1, rule_2, rule_3], 3)

sentences = main_grammar.producte_sentences(100)

print sentences

main_grammar.print_parameters()

A = np.copy(main_grammar.A)
B = np.copy(main_grammar.B)

A += np.random.normal(0.0, 0.01, (A.shape[0], A.shape[1], A.shape[2]))
B += np.random.normal(0.0, 0.01, (B.shape[0], B.shape[1]))

for i in xrange(A.shape[0]):
    total = np.sum(A[i, :, :]) + np.sum(B[i, :])
    A[i, :, :] /= total
    B[i, :] /= total

log_lks = []
other_log_lks = []
for sentence in sentences:
    E, F, log_lk = main_grammar.compute_inside_outside(sentence,
                                                       main_grammar.A,
                                                       main_grammar.B)
    E, F, other_log_lk = main_grammar.compute_inside_outside(sentence,
                                                             A,
                                                             B)
    """
    print sentence
    print '\n'
    print E
    print '\n'
    print F
    print '\n'
    print '------------------------------------'
    """
    log_lks.append(log_lk)
    other_log_lks.append(other_log_lk)

#print log_lks

plt.plot(log_lks, linestyle = "None", marker = "o", color = 'b')
plt.plot(other_log_lks, linestyle = "None", marker = "x", color = 'r')
plt.show()

main_grammar.print_parameters()





