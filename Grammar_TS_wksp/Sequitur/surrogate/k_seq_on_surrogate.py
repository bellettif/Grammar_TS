'''
Created on 5 mai 2014

@author: francois
'''

import numpy as np
import string

from Rule import *
from Grammar import *
from k_compression.k_sequitur import k_seq_compress
from k_seq_grammar_analysis import *

grammar = {'-2' : Rule('-2', ['-1', '3']),
           '-1' : Rule('-1', ['4', '5']),
           '-5' : Rule('-5', ['1', '3']),
           '-3' : Rule('-3', ['-1', '-5']),
           '-6' : gen_framing_rule('-3', '6', '7', 3, 2),
           '-4' : gen_power_rule('-4', '8', '9', 3)}

my_grammar = Grammar(grammar)
for rule in my_grammar.rule_dict.values():
    print rule.barcode

non_terminal_weights = np.random.power(5, len(my_grammar.non_terminals))
terminal_weights = np.random.power(5, len(my_grammar.terminals))

all_weights = list(non_terminal_weights)
all_weights.extend(list(terminal_weights))

freqs, reduced_form, expended_form =  my_grammar.rand_seq(1000, all_weights)

print freqs
print reduced_form
print ''.join(expended_form)

expended_form = filter(lambda x : (x != '>') and (x != '<'), expended_form)

expended_form = map(int, expended_form)

expended_form = np.asarray(expended_form, dtype = np.int32)

print expended_form

inferred_grammar = k_seq_compress(expended_form, 20, 20)

print inferred_grammar

plot_rules(inferred_grammar, 'Inferred', 'Example.png')
