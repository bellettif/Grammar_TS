'''
Created on 5 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from Rule import *

class Grammar:
    
    def __init__(self, rule_dict):
        self.rule_dict = rule_dict
        for rule in self.rule_dict.values():
            rule.compute_barcode(self.rule_dict)
        all_symbols = rule_dict.keys()
        for rule in rule_dict.values():
            all_symbols.extend(rule.rhs)
        all_symbols = list(set(all_symbols))
        self.terminals = sorted(filter(lambda x : x not in rule_dict,
                                all_symbols))
        self.non_terminals = sorted(filter(lambda x : x in rule_dict,
                                    all_symbols))
        self.all_symbols = sorted(list(set.union(
                              set(self.terminals),
                              set(self.non_terminals))))
        
    def rand_seq(self, n_symbols, freqs):
        freqs = np.asarray(freqs)
        freqs /= np.sum(freqs)
        reduced_form = [np.random.choice(self.all_symbols, p = freqs)
                        for i in xrange(n_symbols)]
        freqs = zip(self.all_symbols, freqs)
        expanded_form = []
        for x in reduced_form:
            if x not in self.rule_dict:
                expanded_form.append(x)
            else:
                expanded_form.extend(self.rule_dict[x].barcode)
        return freqs, reduced_form, expanded_form
        
    def rand_seq_non_term(self, n_symbols, freqs):
        freqs = np.asarray(freqs)
        freqs /= np.sum(freqs)
        reduced_form = [np.random.choice(self.non_terminals, p = freqs)
                        for i in xrange(n_symbols)]
        freqs = zip(self.non_terminals, freqs)
        expanded_form = []
        for x in reduced_form:
            if x not in self.rule_dict:
                expanded_form.append(x)
            else:
                expanded_form.extend(self.rule_dict[x].barcode)
                
        return freqs, reduced_form, expanded_form
        