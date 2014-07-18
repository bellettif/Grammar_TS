'''
Created on 16 juil. 2014

@author: francois
'''

import numpy as np

from grammar_examples.grammar_examples import produce_simple_grammar, produce_palindrom_grammar
from study_grammar import study_grammar

grammar_name = 'Simple grammar'
grammar_ex, rule_nick_names = produce_simple_grammar(0.3, 
                                                     0.3, 
                                                     0.2,
                                                     0.2)

study_grammar(grammar_name,
              grammar_ex,
              rule_nick_names)

grammar_name = 'Palindrom grammar'
probas = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2]
probas /= np.sum(probas)
grammar_ex, rule_nick_names = produce_palindrom_grammar(probas[0], 
                                                        probas[1], 
                                                        probas[2], 
                                                        probas[3],
                                                        probas[4],
                                                        probas[5])
study_grammar(grammar_name,
              grammar_ex,
              rule_nick_names)