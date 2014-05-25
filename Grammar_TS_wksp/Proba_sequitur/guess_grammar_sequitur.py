'''
Created on 22 mai 2014

@author: francois
'''

import cPickle as pickle
import re
import load_data
import string
import numpy as np

from Proba_sequitur import init_barelk, pair_counts_multi
from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG
from stochastic_grammar_wrapper import SCFG_c

_epsilon = 0.01

best_symbols = pickle.load(open('best_symbols_achu.pi', 'rb'))
last_parses = pickle.load(open('last_parses_achu.pi', 'rb'))

print best_symbols

sequences = load_data.achu_file_contents.values()

all_symbols = []
for sequence in sequences:
    all_symbols.extend(sequence)
all_symbols = set(all_symbols)
all_symbols = filter(lambda x : ' ' != x, all_symbols)

atom_probas = init_barelk(sequences)
pair_counts = pair_counts_multi(sequences,
                                all_symbols)
total_pair_counts = float(sum(pair_counts.values()))

preterminals = [string.upper(x) for x in all_symbols]

preterminal_rules = {}
current_i = 1
for preterminal in preterminals:
    weights = [1.0 if x == string.lower(preterminal)
               else _epsilon for x in all_symbols]
    preterminal_rules[current_i] = Sto_rule(current_i,
                                           [], 
                                           [], 
                                           weights,
                                           all_symbols)
    current_i += 1

for current_rule in preterminal_rules.values():
    current_rule.print_state()

Rule_1 = Sto_rule(7,
                  [1.0], 
                  [[1, 6]], 
                  [],
                  [])

proba_b = atom_probas['b']
proba_e = atom_probas['e']
Rule_2 = Sto_rule(8,
                  [proba_b, proba_e],
                  [[2, 1], [5, 1]],
                  [],
                  [])

proba_c = atom_probas['c']
Rule_3 = Sto_rule(9,
                  [proba_b, proba_c],
                  [[4, 2], [4, 5]],
                  [],
                  [])

proba_2 = pair_counts['b-a'] + pair_counts['e-a']
proba_3 = pair_counts['d-b'] + pair_counts['d-e']
Rule_4 = Sto_rule(10,
                  [proba_2, proba_3],
                  [[7, 8], [7, 9]],
                  [],
                  [])

Rule_5 = Sto_rule(11,
                  [1.0],
                  [[9, 7]],
                  [],
                  [])

all_rules = {}
for current_rule in preterminal_rules.values():
    all_rules[current_rule.rule_name] = current_rule
    
print all_rules
"""
for current_rule in [Rule_1, Rule_2, Rule_3, Rule_4, Rule_5]:
    all_rules[current_rule.rule_name] = current_rule
"""

Root_rule = Sto_rule(0,
                     np.ones(len(all_rules)),
                     [[x, x] for x in all_rules.keys()],
                     [],
                     [])

all_rules[0] = Root_rule

main_grammar = SCFG(all_rules.values(),
                    0)

split_sequences = [seq.split(' ')[:2] for seq in sequences]

print split_sequences

A, B = SCFG_c.compute_A_and_B(main_grammar)

print A
print ''
print B
print ''

A_estim, B_estim = SCFG_c.estimate_model(main_grammar,
                                         split_sequences,
                                         A, B,
                                         10)


print A_estim
print ''
print B_estim





