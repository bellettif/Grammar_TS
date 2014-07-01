'''
Created on 1 juil. 2014

@author: francois
'''

import string
import copy
import numpy as np

all_symbols = string.ascii_lowercase[:7]

n_layers = 5

grammar = {}
rule_index = 0
layer_index = 0
last_created = []

to_create = [rule_index]

for layer_index in range(1, n_layers + 1):
    print layer_index
    for i, next_lhs in enumerate(to_create):
        left_index = 2 ** (layer_index) + 2 * i
        right_index = 2 ** (layer_index) + 2 * i + 1
        grammar['r%d_' % next_lhs] = ('r%d_' % (left_index),
                                      'r%d_' % (right_index))
        last_created.append(left_index)
        last_created.append(right_index)
    to_create = copy.deepcopy(last_created)
    last_created = []
    
all_pairs = [(x, y) for x in all_symbols for y in all_symbols]

np.random.shuffle(all_pairs)

all_pairs[:len(to_create)]

for i in xrange(len(to_create)):
    grammar['r%d_' % to_create[i]] = all_pairs[i]
    
all_rules = grammar.keys()

wildcard_symbol = 'w'

def unfold_noise_less(root_sequence):
    to_unfold = filter(lambda x : x != wildcard_symbol,
                       root_sequence)
    next_to_unfold = []
    finished = False
    while not finished:
        finished = True
        for symbol in to_unfold:
            if symbol in grammar:
                next_to_unfold.extend(grammar[symbol])
                finished = False
            else:
                next_to_unfold.append(symbol)
        to_unfold = copy.deepcopy(next_to_unfold)
        next_to_unfold = []
    return to_unfold

def unfold_noisy(root_sequence, wildcard_distribution):
    to_unfold = copy.deepcopy(root_sequence)
    next_to_unfold = []
    finished = False
    while not finished:
        finished = True
        for symbol in to_unfold:
            if symbol in grammar:
                next_to_unfold.extend(grammar[symbol])
                finished = False
            else:
                next_to_unfold.append(symbol)
        to_unfold = copy.deepcopy(next_to_unfold)
        next_to_unfold = []
    wildcard_distrib_items = wildcard_distribution.items()
    wildcard_productions = [x[0] for x in wildcard_distrib_items]
    wildcard_weights = [x[1] for x in wildcard_distrib_items]
    for i, symbol in enumerate(to_unfold):
        if symbol == wildcard_symbol:
            to_unfold[i] = np.random.choice(wildcard_productions, p = wildcard_weights)
    return to_unfold

n_roots = 32
n_wildcards = 128
n_sequences = 18

root_sequence = list(np.random.permutation(all_rules)[:n_roots]) + n_wildcards * ['w']
root_sequence = list(np.random.permutation(root_sequence))

noiseless_sequence = unfold_noise_less(root_sequence)

proba_dist_symbols = list(set(noiseless_sequence))

proba_dist_wildcard = {}
for symbol in proba_dist_symbols:
    proba_dist_wildcard[symbol] = float(len(filter(lambda x : x == symbol, 
                                                   noiseless_sequence))) \
                                    / float(len(noiseless_sequence))

noisy_sequence = unfold_noisy(root_sequence, proba_dist_wildcard)
    
proba_dist_wildcard_ex_post = {}
for symbol in proba_dist_symbols:
    proba_dist_wildcard_ex_post[symbol] = float(len(filter(lambda x : x == symbol, 
                                                           noisy_sequence))) \
                                            / float(len(noisy_sequence))
                                            
print proba_dist_wildcard

print '\n'

print proba_dist_wildcard_ex_post