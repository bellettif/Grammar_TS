'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np

from SCFG.sto_grammar import compute_KL_signature

class Grammar_distance():
    
    def __init__(self,
                 first_grammar,
                 second_grammar):
        self.left_grammar = first_grammar
        self.right_grammar = second_grammar
        self.left_signature = 0
        self.right_signature = 0
    
    def compute_distance(self,
                         n_samples,
                         max_length = 0):
        if sorted(self.left_grammar.term_chars) != sorted(self.right_grammar.term_chars):
            return np.inf
        left_samples = self.left_grammar.produce_sentences(n_samples, max_length)
        right_samples = self.right_grammar.produce_sentences(n_samples, max_length)
        print len(left_samples)
        print len(right_samples)
        left_right_probas = self.left_grammar.estimate_likelihoods(right_samples)
        left_left_probas = self.left_grammar.estimate_likelihoods(left_samples)
        right_left_probas = self.right_grammar.estimate_likelihoods(left_samples)
        right_right_probas = self.right_grammar.estimate_likelihoods(right_samples)
        #
        selection = np.where(left_left_probas != 0)
        mid_probas = 0.5 * (left_left_probas + right_left_probas)
        left_result = np.sum(np.log2(left_left_probas[selection] / mid_probas[selection]) * left_left_probas[selection])
        left_result /= float(n_samples)
        #
        selection = np.where(right_right_probas != 0)
        mid_probas = 0.5 * (right_right_probas + left_right_probas)
        right_result = np.sum(np.log2(right_right_probas[selection] / mid_probas[selection]) * right_right_probas[selection])
        right_result /= float(n_samples)
        return np.sqrt(left_result + right_result)

    def compute_distance_MC(self,
                            n_samples,
                            max_length = 0,
                            epsilon = 0):
        if sorted(self.left_grammar.term_chars) != sorted(self.right_grammar.term_chars):
            return np.inf
        if self.left_signature == 0:
            self.left_signature = self.left_grammar.compute_signature(n_samples,
                                                             max_length = max_length)
        if self.right_signature == 0:
            self.right_signature = self.right_grammar.compute_signature(n_samples,
                                                               max_length = max_length)
        merged_signature = self.left_signature.items() + self.right_signature.items()
        if max_length == 0 and epsilon != 0:
            merged_signature.sort(key = (lambda x : -x[1]))
            total_counts = float(sum([x[1] for x in merged_signature]))
            current_total = 0
            max_length = 0
            for sentence, count in merged_signature:
                current_total += count / total_counts
                if current_total >= 1.0 - epsilon:
                    max_length = len(sentence.split(' '))
                    break
        left_signature = dict(filter(lambda x : len(x[0].split(' ')) <= max_length, 
                                     self.left_signature.items()))
        right_signature = dict(filter(lambda x : len(x[0].split(' ')) <= max_length,
                                      self.right_signature.items()))
        return compute_KL_signature(left_signature,
                                    right_signature)
        
        