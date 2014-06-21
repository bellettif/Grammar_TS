'''
Created on 20 juin 2014

@author: francois
'''

import copy
import numpy as np
import string

from k_compression.k_sequitur import k_Sequitur

#
#    Linguistic example
#
example_sequence = ['the', 'little', 'cat', 'chases', 'the', 'mouse']
example_sequence.extend(['the', 'little', 'cat', 'catches', 'the', 'mouse'])
example_sequence.extend(['the', 'big', 'cat', 'chases', 'the', 'little', 'cat'])
example_sequence.extend(['the', 'little', 'cat', 'runs', 'away', 'from', 'the', 'big', 'cat'])
#
print '--------------------------------------'
print '\tLinguistic example'
print ' '.join(example_sequence)
#
sequi = k_Sequitur(example_sequence)
sequi.run()
sequi.draw_graph('Linguistic.png')
#
print ' '.join(sequi.compressed_sequence)
for lhs, array in sequi.grammar.items():
    print '%s (%d) -> (%s) - (hashcode: %s)' % (lhs, 
                                                sequi.ref_counts[lhs], '-'.join(array),
                                                sequi.rule_to_hashcode[lhs])
print '\n'
#
#    Repetition example
#   
repetition_example = []
for i in xrange(10):
    if np.random.choice([True, False]):
        repetition_example.extend(['a', 'b'])
    else:
        repetition_example.extend(['c', 'd'])
print '--------------------------------------'
print '\tNot noisy repetition example'
print ' '.join(repetition_example)
sequi = k_Sequitur(repetition_example)
sequi.run()
sequi.draw_graph('Repetition_not_noisy.png')
#
print ' '.join(sequi.compressed_sequence)
for lhs, array in sequi.grammar.items():
    print '%s (%d) -> (%s) - (hashcode: %s)' % (lhs, 
                                                sequi.ref_counts[lhs], '-'.join(array),
                                                sequi.rule_to_hashcode[lhs])
print '\n'
#
#    Repetition example with noise
#
repetition_example = []
for i in xrange(10):
    decision = np.random.choice(['ab', 'cd', 'noise'], 1, p = [0.4, 0.4, 0.2])
    if decision == 'ab':
        repetition_example.extend(['a', 'b'])
        continue
    if decision == 'cd':
        repetition_example.extend(['c', 'd'])
        continue
    repetition_example.append(string.ascii_lowercase[np.random.randint(4, 26)])   
print '--------------------------------------'
print '\tNoisy repetition example'
print ' '.join(repetition_example)
sequi = k_Sequitur(repetition_example)
sequi.run()
sequi.draw_graph('Repetition_noisy.png')
#
print ' '.join(sequi.compressed_sequence)
for lhs, array in sequi.grammar.items():
    print '%s (%d) -> (%s) - (hashcode: %s)' % (lhs, 
                                                sequi.ref_counts[lhs], '-'.join(array),
                                                sequi.rule_to_hashcode[lhs])
print '\n'
#
#    Palindrom example
#
palindrom_example = ['a', 'b', 'c', 'd']
palindrom_example_rev = copy.deepcopy(palindrom_example)
palindrom_example_rev.reverse()
palindrom_example = palindrom_example + palindrom_example_rev
palindrom_example = 3 * palindrom_example
#
print '--------------------------------------'
print '\tPalindrom example'
print ' '.join(palindrom_example)
#
sequi = k_Sequitur(palindrom_example)
sequi.run()
sequi.draw_graph('Palindrom.png')
#
print ' '.join(sequi.compressed_sequence)
for lhs, array in sequi.grammar.items():
    print '%s (%d) -> (%s) - (hashcode: %s)' % (lhs, 
                                                sequi.ref_counts[lhs], '-'.join(array),
                                                sequi.rule_to_hashcode[lhs])
print '\n'