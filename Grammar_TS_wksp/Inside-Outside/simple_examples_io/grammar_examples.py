'''
Created on 25 mai 2014

@author: francois
'''

from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG

import copy

#    First simple grammar
rule_S = Sto_rule(0,
                  [0.3, 0.3, 0.2, 0.2],
                  [[1, 3], [2, 4], [1, 1], [2, 2]],
                  [],
                  [])
#
rule_A = Sto_rule(1,
                  [],
                  [],
                  [1.0],
                  ['a'])
#
rule_B = Sto_rule(2,
                  [],
                  [],
                  [1.0],
                  ['b'])
#
rule_C = Sto_rule(3,
                  [1.0],
                  [[0, 1]],
                  [],
                  [])
#
rule_D = Sto_rule(4,
                  [1.0],
                  [[0, 2]],
                  [],
                  [])
#
paper_grammar = SCFG(copy.deepcopy([rule_S, rule_A, rule_B, rule_C, rule_D]),
                     0)


#
#    The palindrom grammar from the paper
#
rule_S = Sto_rule(0,
                  [3.0, 3.0, 3.0, 1.0, 1.0, 1.0],
                  [[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
                  [],
                  [])
#
rule_A = Sto_rule(1,
                  [],
                  [],
                  [1.0],
                  ['a'])
#
rule_B = Sto_rule(2,
                  [1.0],
                  [[0, 1]],
                  [],
                  [])
#
rule_C = Sto_rule(3,
                  [],
                  [],
                  [1.0],
                  ['b'])
#
rule_D = Sto_rule(4,
                  [1.0],
                  [[0, 3]],
                  [],
                  [])
#
rule_E = Sto_rule(5,
                  [],
                  [],
                  [1.0],
                  ['c'])
#
rule_F = Sto_rule(6,
                  [1.0],
                  [[0, 5]],
                  [],
                  [])
#
palindrom_grammar = SCFG(copy.deepcopy([rule_S, 
                          rule_A, rule_B, rule_C,
                          rule_D, rule_E, rule_F]),
                         0)