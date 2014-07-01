'''
Created on 1 juil. 2014

@author: francois
'''

import string
import copy
import numpy as np
import cPickle as pickle

from surrogate_grammar import Surrogate_grammar

all_symbols = string.ascii_lowercase[:7]
n_layers = 5

s_g = Surrogate_grammar(terminal_symbols = all_symbols,
                        n_layers = n_layers)

pickle.dump(s_g, open('surrogate_grammar.pi', 'wb'))

