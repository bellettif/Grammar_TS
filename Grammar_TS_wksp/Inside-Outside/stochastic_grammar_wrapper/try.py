'''
Created on 20 mai 2014

@author: francois
'''

import SCFG_c

import numpy as np

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
                  [1.0, 2.0],
                  ['Pierre', 'Mathieu'])

rule_3 = Sto_rule(3,
                  [0.6, 0.5, 0.4, 0.8],
                  [[1, 1], [1, 2], [2, 1], [2, 2]],
                  [],
                  [])

#rule_1.print_state()
#rule_2.print_state()
#rule_3.print_state()

main_grammar = SCFG([rule_1, rule_2, rule_3], 3)

sentences = SCFG_c.compute_derivations(rule_3,
                                      main_grammar,
                                      1000)

print sentences

A, B = SCFG_c.compute_A_and_B(main_grammar)

E, F = SCFG_c.compute_inside_outside(main_grammar,
                                     sentences[5],
                                     A, B)

A_estim, B_estim = SCFG_c.estimate_model(main_grammar,
                                         sentences,
                                         A, B,
                                         100)

print A
print '\n'
print B
print '\n'
print A_estim
print '\n'
print B_estim





