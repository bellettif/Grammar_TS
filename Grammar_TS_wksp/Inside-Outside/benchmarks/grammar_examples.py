'''
Created on 25 mai 2014

@author: francois
'''

from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG
from learning_rate_analyst import Learning_rate_analyst

#A first very basic grammar
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
grammar_1 = SCFG([rule_1, rule_2, rule_3], 3)

#The palindrom grammar from the paper
rule_S = Sto_rule(0,
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
                  [],
                  [])
rule_A = Sto_rule(1,
                  [],
                  [],
                  [1.0],
                  ['a'])
rule_B = Sto_rule(2,
                  [1.0],
                  [[0, 1]],
                  [],
                  [])
rule_C = Sto_rule(3,
                  [],
                  [],
                  [1.0],
                  ['b'])
rule_D = Sto_rule(4,
                  [1.0],
                  [[0, 3]],
                  [],
                  [])
rule_E = Sto_rule(5,
                  [],
                  [],
                  [1.0],
                  ['c'])
rule_F = Sto_rule(6,
                  [1.0],
                  [[0, 5]],
                  [],
                  [])
palindrom_grammar = SCFG([rule_S, 
                          rule_A, rule_B, rule_C,
                          rule_D, rule_E, rule_F],
                         0)

#An example of simple action grammar
rule_wildcard = Sto_rule(5,
                         [],
                         [],
                         [0.25, 0.25, 0.25, 0.25],
                         ["Try_begin", "Succeed_begin", "Try_end", "Succeed_end"])
rule_try_begin = Sto_rule(1,
                          [0.1, 0.1],
                          [[1, 5], [5, 1]],
                          [1.0],
                          ["Try_begin"])
rule_succeed_begin = Sto_rule(2,
                              [0.1, 0.1],
                              [[2, 5], [5, 2]],
                              [1.0],
                              ["Succeed_begin"])
rule_try_end = Sto_rule(3,
                        [0.1, 0.1],
                        [[3, 5], [5, 3]],
                        [1.0],
                        ["Try_end"])
rule_succeed_end = Sto_rule(4,
                            [0.1, 0.1],
                            [[4, 5], [5, 4]],
                            [1.0],
                            ["Succeed_end"])
rule_begin = Sto_rule(6,
                     [0.5, 0.5, 0.05, 0.05],
                     [[1, 2], [1, 6], [5, 6], [6, 5]],
                     [],
                     [])
rule_end = Sto_rule(7,
                    [0.5, 0.5, 0.05, 0.05],
                    [[3, 4], [3, 7], [5, 7], [7, 5]],
                    [],
                    [])
rule_root = Sto_rule(0,
                     [1.0],
                     [[6, 7]],
                     [],
                     [])
action_grammar = SCFG([rule_wildcard, 
                       rule_try_begin, rule_succeed_begin,
                       rule_try_end, rule_succeed_end,
                       rule_begin, rule_end,
                       rule_root], 0)



