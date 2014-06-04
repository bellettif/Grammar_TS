'''
Created on 2 juin 2014

@author: francois
'''

from matplotlib import pyplot as plt

from SCFG.grammar_distance import Grammar_distance
from SCFG.sto_grammar import SCFG, compute_KL_signature

from Grammar_examples import palindrom_grammar
from Grammar_examples import palindrom_grammar_2
from Grammar_examples import palindrom_grammar_3

import time

n_samples = 100000
epsilon = 0.05

palindrom_grammar_signature = palindrom_grammar.compute_signature(n_samples, epsilon)
palindrom_grammar_2_signature = palindrom_grammar_2.compute_signature(n_samples, epsilon)
palindrom_grammar_3_signature = palindrom_grammar_3.compute_signature(n_samples, epsilon)

gd_1_1 = Grammar_distance(palindrom_grammar, palindrom_grammar)
print '1 1'
begin = time.clock()
print gd_1_1.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '1 1'
begin = time.clock()
print gd_1_1.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '1 1 signature'
begin = time.clock()
print compute_KL_signature(palindrom_grammar.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
begin = time.clock()
print compute_KL_signature(palindrom_grammar.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
print ''

gd_1_2 = Grammar_distance(palindrom_grammar, palindrom_grammar_2)
print '1 2'
begin = time.clock()
print gd_1_2.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '1 2'
begin = time.clock()
print gd_1_2.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '1 2 signature'
begin = time.clock()
print compute_KL_signature(palindrom_grammar.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar_2.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
begin = time.clock()
print compute_KL_signature(palindrom_grammar.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar_2.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
print ''

gd_2_2 = Grammar_distance(palindrom_grammar_2, palindrom_grammar_2)
print '2 2'
begin = time.clock()
print gd_2_2.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '2 2'
begin = time.clock()
print gd_2_2.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '2 2 signature'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_2.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar_2.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_2.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar_2.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
print ''

gd_2_1 = Grammar_distance(palindrom_grammar_2, palindrom_grammar)
print '2 1'
begin = time.clock()
print gd_2_1.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '2 1'
begin = time.clock()
print gd_2_1.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '2 1 signature'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_2.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_2.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
print ''

gd_3_1 = Grammar_distance(palindrom_grammar_3, palindrom_grammar)
print '3 1'
begin = time.clock()
print gd_3_1.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '3 1'
begin = time.clock()
print gd_3_1.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '3 1 signature'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_3.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_3.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
print ''

gd_3_2 = Grammar_distance(palindrom_grammar_3, palindrom_grammar_2)
print '3 2'
begin = time.clock()
print gd_3_2.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '3 2'
begin = time.clock()
print gd_3_2.compute_distance(n_samples)
print 'Took ' + str(time.clock() - begin) + ' seconds'
print '3 2 signature'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_3.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar_2.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
begin = time.clock()
print compute_KL_signature(palindrom_grammar_3.compute_signature(n_samples, epsilon, 8),
                           palindrom_grammar_2.compute_signature(n_samples, epsilon, 8))

print 'Took ' + str(time.clock() - begin) + ' seconds'
print ''