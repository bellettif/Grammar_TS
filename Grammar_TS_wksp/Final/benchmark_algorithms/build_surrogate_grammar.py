'''
Created on 1 juil. 2014

@author: francois
'''

import string
import copy
import numpy as np

from Surrogate_grammar import Surrogate_grammar

all_symbols = string.ascii_lowercase[:7]
n_layers = 5

s_g = Surrogate_grammar(terminal_symbols = all_symbols,
                        n_layers = n_layers)

n_roots = 32
n_wildcards = 128
n_sentences = 16

sentences = s_g.produce_sentences(n_roots, 
                                  n_wildcards,
                                  n_sentences)

for sentence in sentences:
    print sentence