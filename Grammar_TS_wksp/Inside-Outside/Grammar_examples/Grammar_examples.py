'''
Created on 2 juin 2014

@author: francois
'''

import matplotlib.pyplot as plt

from SCFG.sto_grammar import SCFG

palindrom_1_rules = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [],
              []),
         1 : ([],
              [],
              ['a'],
              [1.0]),
         2 : ([[0, 1]],
              [1.0],
              [],
              []),
         3 : ([],
              [],
              ['b'],
              [1.0]),
         4 : ([[0, 3]],
              [1.0],
              [],
              []),
         5 : ([],
              [],
              ['c'],
              [1.0]),
         6 : ([[0, 5]],
              [1.0],
              [],
              [])}
palindrom_grammar_1 = SCFG()
palindrom_grammar_1.init_from_rule_dict(palindrom_1_rules)
    
palindrom_rules_2 = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
                      [1.0, 1.0, 1.0, 1.2, 1.2, 1.2],
                      [],
                      []),
                 1 : ([],
                      [],
                      ['a'],
                      [1.0]),
                 2 : ([[0, 1]],
                      [1.0],
                      [],
                      []),
                 3 : ([],
                      [],
                      ['b'],
                      [1.0]),
                 4 : ([[0, 3]],
                      [1.0],
                      [],
                      []),
                 5 : ([],
                      [],
                      ['c'],
                      [1.0]),
                 6 : ([[0, 5]],
                      [1.0],
                      [],
                      [])}
palindrom_grammar_2 = SCFG()
palindrom_grammar_2.init_from_rule_dict(palindrom_rules_2)

palindrom_rules_3 = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
                      [1.2, 0.8, 1.2, 1.2, 1.2, 1.2],
                      [],
                      []),
                 1 : ([],
                      [],
                      ['a'],
                      [1.0]),
                 2 : ([[0, 1]],
                      [1.0],
                      [],
                      []),
                 3 : ([],
                      [],
                      ['b'],
                      [1.0]),
                 4 : ([[0, 3]],
                      [1.0],
                      [],
                      []),
                 5 : ([],
                      [],
                      ['c'],
                      [1.0]),
                 6 : ([[0, 5]],
                      [1.0],
                      [],
                      [])}
palindrom_grammar_3 = SCFG()
palindrom_grammar_3.init_from_rule_dict(palindrom_rules_3)

repetition_1_rules = {0 : ([[0, 1], [2, 3]],
                       [0.6, 0.2],
                       [],
                       []),
                   1 : ([[2, 3]],
                        [1.0],
                        [],
                        []),
                   2 : ([],
                        [],
                        ['a'],
                        [1.0]),
                   3 : ([],
                        [],
                        ['b'],
                        [1.0])}
repetition_grammar_1 = SCFG()
repetition_grammar_1.init_from_rule_dict(repetition_1_rules)

repetition_2_rules = {0 : ([[0, 1], [2, 3]],
                       [0.8, 0.2],
                       [],
                       []),
                   1 : ([[2, 3]],
                        [1.0],
                        [],
                        []),
                   2 : ([],
                        [],
                        ['a'],
                        [1.0]),
                   3 : ([],
                        [],
                        ['b'],
                        [1.0])}
repetition_grammar_2 = SCFG()
repetition_grammar_2.init_from_rule_dict(repetition_2_rules)


embedding_rules_central_1 = {0 : ([[1, 2]],
                               [1.0],
                               [],
                               []),
                          1 : ([],
                               [],
                               ['a'],
                               [1.0]),
                          2 : ([[3, 1]],
                               [1.0],
                               [],
                               []),
                          3 : ([[3, 4], [4, 4]],
                               [0.6, 0.2],
                               ['b'],
                               [0.2]),
                          4 : ([],
                               [],
                               ['b'],
                               [1.0])}
embedding_grammar_central_1 = SCFG()
embedding_grammar_central_1.init_from_rule_dict(embedding_rules_central_1)

embedding_rules_central_2 = {0 : ([[1, 2]],
                               [1.0],
                               [],
                               []),
                          1 : ([],
                               [],
                               ['a'],
                               [1.0]),
                          2 : ([[3, 1]],
                               [1.0],
                               [],
                               []),
                          3 : ([[3, 4], [4, 4]],
                               [0.8, 0.2],
                               ['b'],
                               [0.2]),
                          4 : ([],
                               [],
                               ['b'],
                               [1.0])}
embedding_grammar_central_2 = SCFG()
embedding_grammar_central_2.init_from_rule_dict(embedding_rules_central_2)

embedding_grammar_left_right_rules_1 = {0 : ([[1, 3]],
                                          [1.0],
                                          [],
                                          []),
                                      1 : ([[1, 2], [2, 2]],
                                           [0.6, 0.2],
                                           ['a'],
                                           [0.2]),
                                      2 : ([],
                                           [],
                                           ['a'],
                                           [1.0]),
                                      3 : ([[4, 1]],
                                           [1.0],
                                           [],
                                           []),
                                      4 : ([],
                                           [],
                                           ['b'],
                                           [1.0])}
embedding_grammar_left_right_1 = SCFG()
embedding_grammar_left_right_1.init_from_rule_dict(embedding_grammar_left_right_rules_1)

embedding_grammar_left_right_rules_2 = {0 : ([[1, 3]],
                                          [1.0],
                                          [],
                                          []),
                                      1 : ([[1, 2], [2, 2]],
                                           [0.8, 0.2],
                                           ['a'],
                                           [0.2]),
                                      2 : ([],
                                           [],
                                           ['a'],
                                           [1.0]),
                                      3 : ([[4, 1]],
                                           [1.0],
                                           [],
                                           []),
                                      4 : ([],
                                           [],
                                           ['b'],
                                           [1.0])}
embedding_grammar_left_right_2 = SCFG()
embedding_grammar_left_right_2.init_from_rule_dict(embedding_grammar_left_right_rules_2)



class CSGrammar:
    
    def __init__(self,
                N, M = 1):
        self.N = N
        self.M = M
        
    def produce_sentences(self, n_samples):
        return [self.produce_sentence() for i in xrange(n_samples)]
    
    def produce_sentence(self):
        return (['a'] * self.N) + (['b'] * self.M) + (['a'] * self.N)
        

CSExample = CSGrammar(3, 2)

name_grammar_1_rules = {0 : ([[1, 1], [1, 2], [2, 1], [2, 2], [3, 0]],
                          [0.8, 0.5, 0.2, 0.8, 0.2],
                          [],
                          []),
                      1 : ([[1, 1], [1, 2], [2, 1], [2, 2]],
                           [0.2, 0.3, 0.1, 0.4],
                           ['a', 'b', 'c'],
                           [2.3, 2.5, 1.4]),
                      2 : ([[2, 2], [2, 1], [1, 2], [1, 1]],
                           [0.2, 0.3, 0.4, 2.1],
                           ['a', 'd'],
                           [1.3, 3.2]),
                      3 : ([[0, 1], [0, 2]],
                           [0.3, 0.4],
                           [],
                           [])}
name_grammar_1 = SCFG()
name_grammar_1.init_from_rule_dict(name_grammar_1_rules)

name_grammar_2_rules = {0 : ([[1, 1], [1, 2], [2, 1], [2, 2], [3, 0]],
                          [0.6, 0.3, 0.1, 0.6, 0.9],
                          [],
                          []),
                      1 : ([[1, 1], [1, 2], [2, 1], [2, 2]],
                           [0.5, 0.6, 0.4, 0.1],
                           ['a', 'b', 'c'],
                           [1.3, 2.9, 2.4]),
                      2 : ([[2, 2], [2, 1], [1, 2], [1, 1]],
                           [0.2, 0.3, 0.4, 2.1],
                           ['a', 'd'],
                           [2.3, 1.2]),
                      3 : ([[0, 1], [0, 2]],
                           [0.9, 0.3],
                           [],
                           [])}
name_grammar_2 = SCFG()
name_grammar_2.init_from_rule_dict(name_grammar_2_rules)

action_grammar_1_rules = {0 : ([[6, 7]],
                             [1.0],
                             [],
                             []),
                        1 : ([[1, 5], [5, 1]],
                             [0.1, 0.1],
                             ["a"],
                             [1.0]),
                        2 : ([[2, 5], [5, 2]],
                             [0.1, 0.1],
                             ["b"],
                             [1.0]),
                        3 : ([[3, 5], [5, 3]],
                             [0.1, 0.1],
                             ["c"],
                             [1.0]),
                        4 : ([[4, 5], [5, 4]],
                             [0.1, 0.1],
                             ["d"],
                             [1.0]),
                        5 : ([],
                             [],
                             ["a", "b", "c", "d"],
                             [0.25, 0.25, 0.25, 0.25]),
                        6 : ([[1, 2], [1, 6], [5, 6], [6, 5]],
                             [0.5, 0.5, 0.05, 0.05],
                             [],
                             []),
                        7 : ([[3, 4], [3, 7], [5, 7], [7, 5]],
                             [0.5, 0.5, 0.05, 0.05],
                             [],
                             [])}
action_grammar_1 = SCFG()
action_grammar_1.init_from_rule_dict(action_grammar_1_rules)

action_grammar_2_rules = {0 : ([[6, 7]],
                             [1.0],
                             [],
                             []),
                        1 : ([[1, 5], [5, 1]],
                             [0.1, 0.1],
                             ["Try_begin"],
                             [1.0]),
                        2 : ([[2, 5], [5, 2]],
                             [0.3, 0.3],
                             ["Succeed_begin"],
                             [1.0]),
                        3 : ([[3, 5], [5, 3]],
                             [0.1, 0.1],
                             ["Try_end"],
                             [1.0]),
                        4 : ([[4, 5], [5, 4]],
                             [0.3, 0.3],
                             ["Succeed_end"],
                             [1.0]),
                        5 : ([],
                             [],
                             ["Try_begin", "Succeed_begin", "Try_end", "Succeed_end"],
                             [0.25, 0.25, 0.25, 0.25]),
                        6 : ([[1, 2], [1, 6], [5, 6], [6, 5]],
                             [0.4, 0.4, 0.1, 0.1],
                             [],
                             []),
                        7 : ([[3, 4], [3, 7], [5, 7], [7, 5]],
                             [0.4, 0.5, 0.1, 0.1],
                             [],
                             [])}
action_grammar_2 = SCFG()
action_grammar_2.init_from_rule_dict(action_grammar_2_rules)


