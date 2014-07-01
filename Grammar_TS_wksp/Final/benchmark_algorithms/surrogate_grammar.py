'''
Created on 1 juil. 2014

@author: francois
'''

import string
import copy
import numpy as np

class Surrogate_grammar:

    def __init__(self, 
                 terminal_symbols,
                 n_layers,
                 wildcard_symbol = 'w'):
        self.grammar = {}
        self.wildcard_symbol = wildcard_symbol
        self.hashcodes = {}
        #
        #    Create grammar
        #        Non terminal symbols first
        #
        rule_index = 0
        layer_index = 0
        last_created = []
        to_create = [rule_index]
        for layer_index in range(1, n_layers + 1):
            print layer_index
            for i, next_lhs in enumerate(to_create):
                left_index = 2 ** (layer_index) + 2 * i
                right_index = 2 ** (layer_index) + 2 * i + 1
                self.grammar['r%d_' % next_lhs] = ('r%d_' % (left_index),
                                              'r%d_' % (right_index))
                last_created.append(left_index)
                last_created.append(right_index)
            to_create = copy.deepcopy(last_created)
            last_created = []
        #
        #    Linking lowest level layer rules to
        #        pairs of terminal symbols
        #    
        all_pairs = [(x, y) for x in terminal_symbols 
                     for y in terminal_symbols]
        np.random.shuffle(all_pairs)
        all_pairs[:len(to_create)]
        for i in xrange(len(to_create)):
            self.grammar['r%d_' % to_create[i]] = all_pairs[i]
        #
        #    Creating list of all rules
        #
        self.all_rules = self.grammar.keys()
        #
        def compute_hashcode(key):
            if key in self.hashcodes:
                return self.hashcodes[key]
            else:
                if key in self.grammar:
                    left, right = self.grammar[key]
                    self.hashcodes[key] = '>' + compute_hashcode(left) + '-' \
                                            + compute_hashcode(right) + '<'
                    return self.hashcodes[key]
                else:
                    return key
        for rule in self.all_rules:
            compute_hashcode(rule)
        
    def unfold_noise_less(self, root_sequence):
        to_unfold = filter(lambda x : x != self.wildcard_symbol,
                           root_sequence)
        next_to_unfold = []
        finished = False
        while not finished:
            finished = True
            for symbol in to_unfold:
                if symbol in self.grammar:
                    next_to_unfold.extend(self.grammar[symbol])
                    finished = False
                else:
                    next_to_unfold.append(symbol)
            to_unfold = copy.deepcopy(next_to_unfold)
            next_to_unfold = []
        return to_unfold
    
    def unfold_noisy(self, 
                     root_sequence, 
                     wildcard_distribution):
        to_unfold = copy.deepcopy(root_sequence)
        next_to_unfold = []
        finished = False
        while not finished:
            finished = True
            for symbol in to_unfold:
                if symbol in self.grammar:
                    next_to_unfold.extend(self.grammar[symbol])
                    finished = False
                else:
                    next_to_unfold.append(symbol)
            to_unfold = copy.deepcopy(next_to_unfold)
            next_to_unfold = []
        wildcard_distrib_items = wildcard_distribution.items()
        wildcard_productions = [x[0] for x in wildcard_distrib_items]
        wildcard_weights = [x[1] for x in wildcard_distrib_items]
        for i, symbol in enumerate(to_unfold):
            if symbol == self.wildcard_symbol:
                to_unfold[i] = np.random.choice(wildcard_productions, p = wildcard_weights)
        return to_unfold
    
    def produce_sentence(self,
                         n_roots,
                         n_wildcards):
        root_sequence = list(np.random.permutation(self.all_rules)[:n_roots]) + n_wildcards * ['w']
        root_sequence = list(np.random.permutation(root_sequence))
        noiseless_sequence = self.unfold_noise_less(root_sequence)
        proba_dist_symbols = list(set(noiseless_sequence))
        proba_dist_wildcard = {}
        #
        #    Computing noise less distribution so as to
        #        get wildcard distribution
        #
        for symbol in proba_dist_symbols:
            proba_dist_wildcard[symbol] = float(len(filter(lambda x : x == symbol, 
                                                           noiseless_sequence))) \
                                            / float(len(noiseless_sequence))
        #
        #    Producing noisy sequence
        #
        noisy_sequence = self.unfold_noisy(root_sequence, proba_dist_wildcard)
        return noisy_sequence
        """
        #
        #    Check that the wildcard distribution constraint is met
        #
        proba_dist_wildcard_ex_post = {}
        for symbol in proba_dist_symbols:
            proba_dist_wildcard_ex_post[symbol] = float(len(filter(lambda x : x == symbol, 
                                                                   noisy_sequence))) \
                                                    / float(len(noisy_sequence))
        """
        
    def produce_sentences(self,
                          n_roots,
                          n_wildcards,
                          n_sentences):
        sentences = [' '.join(self.produce_sentence(n_roots, n_wildcards))
                     for i in xrange(n_sentences)]
        return sentences