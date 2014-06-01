'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np

class Grammar_distance():
    
    def __init__(self,
                 first_grammar,
                 second_grammar):
        self.left_grammar = first_grammar
        self.right_grammar = second_grammar
    
    def compute_distance(self,
                         n_samples):
        left_samples = self.left_grammar.produce_sentences(n_samples)
        right_samples = self.right_grammar.produce_sentences(n_samples)       
        left_right_probas = self.left_grammar.compute_probas(right_samples)
        left_left_probas = self.left_grammar.compute_probas(left_samples)
        right_left_probas = self.right_grammar.compute_probas(left_samples)
        right_right_probas = self.right_grammar.compute_probas(right_samples)
        #
        left_result = 0
        for i, left_right_proba in enumerate(left_right_probas):
            if left_right_proba == 0:
                continue
            left_left_proba = left_left_probas[i]
            left_result += np.log(left_right_proba / left_left_proba) * left_right_proba
        left_result /= float(n_samples)
        #
        right_result = 0
        for i, right_left_proba in enumerate(right_left_probas):
            if right_left_proba == 0:
                continue
            right_right_proba = right_right_probas[i]
            right_result += np.log(right_left_proba / right_right_proba) * right_left_proba
        right_result /= float(n_samples)
        return left_result + right_result
    
    
from benchmarks.grammar_examples import grammar_1
from benchmarks.grammar_examples_2 import grammar_2

grammar_distance = Grammar_distance(grammar_1, grammar_2)
print grammar_distance.compute_distance(100000)

grammar_distance = Grammar_distance(grammar_1, grammar_1)
print grammar_distance.compute_distance(100000)

grammar_distance = Grammar_distance(grammar_2, grammar_1)
print grammar_distance.compute_distance(100000)

grammar_distance = Grammar_distance(grammar_2, grammar_2)
print grammar_distance.compute_distance(100000)
