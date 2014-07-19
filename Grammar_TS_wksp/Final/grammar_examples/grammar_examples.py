'''
Created on 2 juin 2014

@author: francois
'''

import matplotlib.pyplot as plt

from SCFG.sto_grammar import SCFG

def produce_simple_grammar(p_S_AC, p_S_BD, p_S_AA, p_S_BB):
    assert(p_S_AC + p_S_BD + p_S_AA + p_S_BB == 1.0)
    simple_grammar_rules = {0: ( # rule S
                                [[1, 3], [2, 4], [1, 1], [2, 2]], # derivation pairs
                                [p_S_AC, p_S_BD, p_S_AA, p_S_BB], # derivation pairs' probas
                                [], # terminal symbols
                                []  # terminal symbols' weights
                                ),
                            1 : ( # rule A
                                 [],
                                 [],
                                 ['a'],
                                 [1.0]
                                 ),
                            2 : ( # rule B
                                 [],
                                 [],
                                 ['b'],
                                 [1.0]
                                 ),
                            3 : ( # rule C
                                 [[0, 1]],
                                 [1.0],
                                 [],
                                 []
                                 ),
                            4 : ( # rule D
                                  [[0, 2]],
                                  [1.0],
                                  [],
                                  []
                                  )}
    rule_nick_names = {0: 'S', 1 : 'A', 2 : 'B', 3 : 'C', 4 : 'D'}
    simple_grammar = SCFG()
    simple_grammar.init_from_rule_dict(simple_grammar_rules)
    return simple_grammar, rule_nick_names

def produce_palindrom_grammar(p_S_AB, 
                              p_S_CD, 
                              p_S_EF,
                              p_S_AA,
                              p_S_CC,
                              p_S_EE):
    palindrom_rules = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]], # S
                  [p_S_AB, p_S_CD, p_S_EF, p_S_AA, p_S_CC, p_S_EE],
                  [],
                  []),
             1 : ([], # A
                  [],
                  ['a'],
                  [1.0]),
             2 : ([[0, 1]], # embed A
                  [1.0],
                  [],
                  []),
             3 : ([], # B
                  [],
                  ['b'],
                  [1.0]),
             4 : ([[0, 3]], # embed B
                  [1.0],
                  [],
                  []),
             5 : ([], # C
                  [],
                  ['c'],
                  [1.0]),
             6 : ([[0, 5]], # embed C
                  [1.0],
                  [],
                  [])}
    rule_nick_names = {0: 'S',
                       1: 'A',
                       2: 'embed A',
                       3 : 'B',
                       4 : 'embed B',
                       5 : 'C',
                       6 : 'embed C'}
    palindrom_grammar = SCFG()
    palindrom_grammar.init_from_rule_dict(palindrom_rules)
    return palindrom_grammar, rule_nick_names

def produce_word_grammar(p_NP_DTNN,
                         p_NP_DTJJNP,
                         p_JJNN_JJNN,
                         p_JJNN_JJJJNN):
    word_grammar_rules = {0 : ([[1, 2]], # root
                             [1.0],
                             [],
                             []),
                        1 : ([[3, 4], [3, 5]], # NP
                             [p_NP_DTNN, p_NP_DTJJNP],
                             [],
                             []),
                        2 : ([[7, 1]], # VP
                             [1.0],
                             [],
                             []),
                        3 : ([], # DT
                             [],
                             ['the', 'a'],
                             [0.5, 0.5]),
                        4 : ([], # NN
                             [],
                             ['mouse', 'cat', 'dog'],
                             [0.3, 0.3, 0.3]),
                        5 : ([[6, 4], [6, 5]], # JJNN 
                              [p_JJNN_JJNN, p_JJNN_JJJJNN],
                              [],
                              []),
                        6 : ([], # JJ
                             [],
                             ['big', 'black'],
                             [0.5, 0.5]),
                        7 : ([], # VB
                             [],
                             ['chased', 'ate'],
                             [0.5, 0.5])}
    word_grammar = SCFG()
    word_grammar.init_from_rule_dict(word_grammar_rules)
    rule_nick_names = {0 : 'root',
                       1 : 'NP',
                       2 : 'VP',
                       3 : 'DT',
                       4 : 'NN',
                       5 : 'JJNN',
                       6 : 'JJ',
                       7 : 'VB'}
    return word_grammar, rule_nick_names

palindrom_rules = {0 : ([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3], [5, 5]], # S
                  [0.3, 0.3, 0.3, 0.2, 0.2, 0.2],
                  [],
                  []),
             1 : ([], # A
                  [],
                  ['a'],
                  [1.0]),
             2 : ([[0, 1]], # embed A
                  [1.0],
                  [],
                  []),
             3 : ([], # B
                  [],
                  ['b'],
                  [1.0]),
             4 : ([[0, 3]], # embed B
                  [1.0],
                  [],
                  []),
             5 : ([], # C
                  [],
                  ['c'],
                  [1.0]),
             6 : ([[0, 5]], # embed C
                  [1.0],
                  [],
                  [])}
palindrom_rule_nick_names = {0: 'S',
                   1: 'A',
                   2: 'embed A',
                   3 : 'B',
                   4 : 'embed B',
                   5 : 'C',
                   6 : 'embed C'}
palindrom_grammar = SCFG()
palindrom_grammar.init_from_rule_dict(palindrom_rules)

redundant_palindrom_rules = \
            {0 : ([[1, 2], [1, 7], [3, 4], [3, 8], [5, 6], [5, 9], [1, 1], [3, 3], [5, 5]], # S
                  [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2],
                  [],
                  []),
             1 : ([], # A
                  [],
                  ['a'],
                  [1.0]),
             2 : ([[0, 1]], # embed A
                  [1.0],
                  [],
                  []),
             3 : ([], # B
                  [],
                  ['b'],
                  [1.0]),
             4 : ([[0, 3]], # embed B
                  [1.0],
                  [],
                  []),
             5 : ([], # C
                  [],
                  ['c'],
                  [1.0]),
             6 : ([[0, 5]], # embed C
                  [1.0],
                  [],
                  []),
             7 : ([[0, 1]], # embed A bis
                  [1.0],
                  [],
                  []),
             8 : ([[0, 3]], # embed B bis
                  [1.0],
                  [],
                  []),
             9 : ([[0, 5]], # embed C bis
                  [1.0],
                  [],
                  [])}
redundant_palindrom_rule_nick_names = {0: 'S',
                   1: 'A',
                   2: 'embed A',
                   3 : 'B',
                   4 : 'embed B',
                   5 : 'C',
                   6 : 'embed C',
                   7 : 'embed A bis',
                   8 : 'embed B bis',
                   9 : 'embed C bis'}
redundant_palindrom_grammar = SCFG()
redundant_palindrom_grammar.init_from_rule_dict(redundant_palindrom_rules)

word_grammar_rules = {0 : ([[1, 2]], # root
                             [1.0],
                             [],
                             []),
                        1 : ([[3, 4], [3, 5]], # NP
                             [0.5, 0.5],
                             [],
                             []),
                        2 : ([[7, 1]], # VP
                             [1.0],
                             [],
                             []),
                        3 : ([], # DT
                             [],
                             ['the', 'a'],
                             [0.5, 0.5]),
                        4 : ([], # NN
                             [],
                             ['mouse', 'cat', 'dog'],
                             [0.3, 0.3, 0.3]),
                        5 : ([[6, 4], [6, 5]], # JJNN 
                              [0.5, 0.5],
                              [],
                              []),
                        6 : ([], # JJ
                             [],
                             ['big', 'black'],
                             [0.5, 0.5]),
                        7 : ([], # VB
                             [],
                             ['chased', 'ate'],
                             [0.5, 0.5])}
word_grammar = SCFG()
word_grammar.init_from_rule_dict(word_grammar_rules)
word_grammar_rule_nick_names = {0 : 'root',
                               1 : 'NP',
                               2 : 'VP',
                               3 : 'DT',
                               4 : 'NN',
                               5 : 'JJNN',
                               6 : 'JJ',
                               7 : 'VB'}

redundant_word_grammar_rules = {0 : ([[1, 2], [1, 9], [8, 2], [8, 9]], # root
                             [0.25, 0.25, 0.25, 0.25],
                             [],
                             []),
                        1 : ([[3, 4], [3, 5]], # NP
                             [0.5, 0.5],
                             [],
                             []),
                        2 : ([[7, 1]], # VP
                             [1.0],
                             [],
                             []),
                        3 : ([], # DT
                             [],
                             ['the', 'a'],
                             [0.5, 0.5]),
                        4 : ([], # NN
                             [],
                             ['mouse', 'cat', 'dog'],
                             [0.3, 0.3, 0.3]),
                        5 : ([[6, 4], [6, 5]], # JJNN 
                              [0.5, 0.5],
                              [],
                              []),
                        6 : ([], # JJ
                             [],
                             ['big', 'black'],
                             [0.5, 0.5]),
                        7 : ([], # VB
                             [],
                             ['chased', 'ate'],
                             [0.5, 0.5]),
                        8 : ([[3, 4], [3, 5]], # NP bis
                             [0.5, 0.5],
                             [],
                             []),
                        9 : ([[7, 1]], # VP bis
                             [1.0],
                             [],
                             [])}
redundant_word_grammar = SCFG()
redundant_word_grammar.init_from_rule_dict(redundant_word_grammar_rules)
redundant_word_grammar_rule_nick_names = {0 : 'root',
                   1 : 'NP',
                   2 : 'VP',
                   3 : 'DT',
                   4 : 'NN',
                   5 : 'JJNN',
                   6 : 'JJ',
                   7 : 'VB',
                   8 : 'NP bis',
                   9 : 'VP bis'}