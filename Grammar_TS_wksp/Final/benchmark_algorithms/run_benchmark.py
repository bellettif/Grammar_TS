'''
Created on 1 juil. 2014

@author: francois
'''

import string

from benchmark import run_complete_benchmark
from surrogate_grammar import Surrogate_grammar

from symbol_atomic_density import compute_symbol_density, \
                                    plot_frequencies

n_roots = 128
n_wildcards = 32
n_sentences = 18

#
#    Generate sequences of symbols
#
all_symbols = list(string.ascii_lowercase[:7]) + 5* ['a']
n_layers = 5
s_g = Surrogate_grammar(terminal_symbols = all_symbols,
                        n_layers = n_layers)
"""
#
#    Compute individual symbol frequencies
#
symbol_density = compute_symbol_density(input_sentences)
plot_frequencies(symbol_density, 'Density_5_reps.png')
"""
#
#    Run benchmark
#
k_set = range(2, 12)
run_complete_benchmark(k_set, 
                       n_rounds = 4,
                       filename = 'Benchmark_5_reps.png',
                       surrogate_grammar = s_g,
                       n_trials = 10,
                       n_roots = n_roots,
                       n_wildcards = n_wildcards,
                       n_sentences = n_sentences)

#
#    Generate sequences of symbols
#
all_symbols = list(string.ascii_lowercase[:7]) + 3* ['a']
n_layers = 5
s_g = Surrogate_grammar(terminal_symbols = all_symbols,
                        n_layers = n_layers)
"""
#
#    Compute individual symbol frequencies
#
symbol_density = compute_symbol_density(input_sentences)
plot_frequencies(symbol_density, 'Density_3_reps.png')
"""
#
#    Run benchmark
#
k_set = range(2, 12)
run_complete_benchmark(k_set, 
                       n_rounds = 4,
                       filename = 'Benchmark_3_reps.png',
                       surrogate_grammar = s_g,
                       n_trials = 10,
                       n_roots = n_roots,
                       n_wildcards = n_wildcards,
                       n_sentences = n_sentences)

all_symbols = list(string.ascii_lowercase[:7])
n_layers = 5
s_g = Surrogate_grammar(terminal_symbols = all_symbols,
                        n_layers = n_layers)
"""
#
#    Compute individual symbol frequencies
#
symbol_density = compute_symbol_density(input_sentences)
plot_frequencies(symbol_density, 'Density_0_reps.png')
"""
#
#    Run benchmark
#
k_set = range(2, 12)
run_complete_benchmark(k_set, 
                       n_rounds = 4,
                       filename = 'Benchmark_0_reps.png',
                       surrogate_grammar = s_g,
                       n_trials = 10,
                       n_roots = n_roots,
                       n_wildcards = n_wildcards,
                       n_sentences = n_sentences)
