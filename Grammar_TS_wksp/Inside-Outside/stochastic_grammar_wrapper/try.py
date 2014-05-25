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

E, F = main_grammar.compute_inside_outside(sentences[5],
                                           main_grammar.A,
                                           main_grammar.B)

main_grammar.print_parameters()





