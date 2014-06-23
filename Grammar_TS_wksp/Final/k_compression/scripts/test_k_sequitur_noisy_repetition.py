'''
Created on 20 juin 2014

@author: francois
'''

import copy
import numpy as np
import string

from k_compression.k_sequitur import k_Sequitur

#
#    Repetition example with noise
#
repetition_example = []
for i in xrange(40):
    decision = np.random.choice(['ab', 'cd', 'noise'], 1, p = [0.4, 0.4, 0.2])
    if decision == 'ab':
        repetition_example.extend(['a', 'b'])
        continue
    if decision == 'cd':
        repetition_example.extend(['c', 'd'])
        continue
    repetition_example.append(string.ascii_lowercase[np.random.randint(0, 4)])   
print '--------------------------------------'
print '\tNoisy repetition example'
print ' '.join(repetition_example)
for k in range(2, 13):
    sequi = k_Sequitur(repetition_example, k)
    sequi.run()
    sequi.draw_graph('Repetition_noisy_%d.png' % k)
    #
    print 'k = %d' % k
    print ' '.join(sequi.compressed_sequence)
    for lhs, array in sequi.grammar.items():
        print '%s (%d) -> (%s) - (hashcode: %s)' % (lhs, 
                                                    sequi.ref_counts[lhs], '-'.join(array),
                                                    sequi.rule_to_hashcode[lhs])
    print '\n'