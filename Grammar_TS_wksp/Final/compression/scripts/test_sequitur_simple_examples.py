'''
Created on 20 juin 2014

@author: francois
'''

import copy
import numpy as np

from compression.sequitur import Sequitur

#
#    Linguistic example
#
example_sequence = ['the', 'little', 'cat', 'chases', 'the', 'mouse']
example_sequence.extend(['the', 'little', 'cat', 'catches', 'the', 'mouse'])
example_sequence.extend(['the', 'big', 'cat', 'chases', 'the', 'little', 'cat'])
example_sequence.extend(['the', 'little', 'cat', 'runs', 'away', 'from', 'the', 'big', 'cat'])
#
print example_sequence
#
sequi = Sequitur(example_sequence)
sequi.run()
#
print sequi.compressed_sequence
for lhs, array in sequi.grammar.items():
    print '%s (%d) -> (%s) - (hashcode: %s)' % (lhs, 
                                                sequi.ref_counts[lhs], '-'.join(array),
                                                sequi.rule_to_hashcode[lhs])
print '\n'
#
#    Repetition example with noise
#   
repetition_example = []
for i in xrange(100):
    if np.random.uniform(0, 1.0) > 0.1:
        repetition_example.extend(['a', 'b'])
    else:
        if np.random.uniform(0, 1.0) > 0.5:
            repetition_example.append('a')
        else:
            repetition_example.append('b')
print repetition_example
sequi = Sequitur(repetition_example)
sequi.run()
#
print sequi.compressed_sequence
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
print palindrom_example
#
sequi = Sequitur(palindrom_example)
sequi.run()
#
print sequi.compressed_sequence
for lhs, array in sequi.grammar.items():
    print '%s (%d) -> (%s) - (hashcode: %s)' % (lhs, 
                                                sequi.ref_counts[lhs], '-'.join(array),
                                                sequi.rule_to_hashcode[lhs])
print '\n'