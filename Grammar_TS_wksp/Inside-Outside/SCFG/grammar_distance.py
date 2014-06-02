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
                         n_samples,
                         max_length = 0):
        left_samples = self.left_grammar.produce_sentences(n_samples, max_length)
        right_samples = self.right_grammar.produce_sentences(n_samples, max_length)       
        left_right_probas = self.left_grammar.estimate_likelihoods(right_samples, max_length)
        left_left_probas = self.left_grammar.estimate_likelihoods(left_samples, max_length)
        right_left_probas = self.right_grammar.estimate_likelihoods(left_samples, max_length)
        right_right_probas = self.right_grammar.estimate_likelihoods(right_samples, max_length)
        print np.sum(left_left_probas)
        print np.sum(right_right_probas)
        print np.sum(left_right_probas)
        print np.sum(right_left_probas)
        #
        selection = np.where(left_left_probas != 0)
        left_result = np.sum(np.log(left_left_probas[selection] / right_left_probas[selection]) * left_left_probas[selection])
        left_result /= float(n_samples)
        #
        selection = np.where(right_right_probas != 0)
        right_result = np.sum(np.log(right_right_probas[selection] / left_right_probas[selection]) * right_right_probas[selection])
        right_result /= float(n_samples)
        return left_result + right_result
