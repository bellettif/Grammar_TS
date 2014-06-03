'''
Created on 2 juin 2014

@author: francois
'''

from matplotlib import pyplot as plt

from SCFG.grammar_distance import Grammar_distance
from SCFG.sto_grammar import SCFG

from Grammar_examples import palindrom_grammar
from Grammar_examples import palindrom_grammar_2
from Grammar_examples import palindrom_grammar_3

n_samples = 100000

palindrom_grammar.plot_stats(n_samples, max_represented = 100)
palindrom_grammar_2.plot_stats(n_samples, max_represented = 100)
palindrom_grammar_3.plot_stats(n_samples, max_represented = 100)


gd_1_1 = Grammar_distance(palindrom_grammar, palindrom_grammar)
print '1 1'
print gd_1_1.compute_distance(n_samples)
print '1 1'
print gd_1_1.compute_distance(n_samples)

gd_1_2 = Grammar_distance(palindrom_grammar, palindrom_grammar_2)
print '1 2'
print gd_1_2.compute_distance(n_samples)
print '1 2'
print gd_1_2.compute_distance(n_samples)

gd_2_2 = Grammar_distance(palindrom_grammar_2, palindrom_grammar_2)
print '2 2'
print gd_2_2.compute_distance(n_samples)
print '2 2'
print gd_2_2.compute_distance(n_samples)

gd_2_1 = Grammar_distance(palindrom_grammar_2, palindrom_grammar)
print '2 1'
print gd_2_1.compute_distance(n_samples)
print '2 1'
print gd_2_1.compute_distance(n_samples)

gd_3_1 = Grammar_distance(palindrom_grammar_3, palindrom_grammar)
print '3 1'
print gd_3_1.compute_distance(n_samples)
print '3 1'
print gd_3_1.compute_distance(n_samples)

gd_3_2 = Grammar_distance(palindrom_grammar_3, palindrom_grammar_2)
print '3 2'
print gd_3_2.compute_distance(n_samples)
print '3 2'
print gd_3_2.compute_distance(n_samples)