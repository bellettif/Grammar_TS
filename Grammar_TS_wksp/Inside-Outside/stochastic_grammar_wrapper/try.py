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

rule_1.print_state()
rule_2.print_state()
rule_3.print_state()

SCFG_c.create_c_grammar([rule_1, rule_2, rule_3],
                        3)

main_grammar = SCFG([rule_1, rule_2, rule_3], 3)
