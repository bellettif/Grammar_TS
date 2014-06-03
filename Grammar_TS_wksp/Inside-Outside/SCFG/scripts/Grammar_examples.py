'''
Created on 2 juin 2014

@author: francois
'''

from SCFG.sto_grammar import SCFG

palindrom_rules = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
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
palindrom_grammar = SCFG()
palindrom_grammar.init_from_rule_dict(palindrom_rules)

palindrom_rules_2 = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]],
                      [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
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
                      [1.0, 1.0, 1.0, 1.5, 1.5, 1.5],
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

repetition_rules = {0 : ([[0, 1], [2, 3]],
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
repetition_grammar = SCFG()
repetition_grammar.init_from_rule_dict(repetition_rules)

embedding_rules_central = {0 : ([[1, 2]],
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
embedding_grammar_central = SCFG()
embedding_grammar_central.init_from_rule_dict(embedding_rules_central)

embedding_grammar_left_right_rules = {0 : ([[1, 3]],
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
embedding_grammar_left_right = SCFG()
embedding_grammar_left_right.init_from_rule_dict(embedding_grammar_left_right_rules)

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


