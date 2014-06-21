'''
Created on 14 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

import load_data
from load_data import int_to_char, \
                        char_to_int

from plot_convention import colors

from compression.sequitur import Sequitur

achu_seq_files = load_data.achu_file_contents
    
example_file = achu_seq_files['achuSeq_1']
example_file = example_file.split(' ')

sequi = Sequitur(example_file)
sequi.run()

print 'Done'

example_sequence = ['the', 'little', 'cat', 'chases', 'the', 'mouse']
example_sequence.extend(['the', 'little', 'cat', 'catches', 'the', 'mouse'])
example_sequence.extend(['the', 'big', 'cat', 'chases', 'the', 'little', 'cat'])
example_sequence.extend(['the', 'little', 'cat', 'runs', 'away', 'from', 'the', 'big', 'cat'])

print example_sequence

sequi = Sequitur(example_sequence)
sequi.run()

print sequi.compressed_sequence
for lhs, array in sequi.grammar.items():
    print '%s (%d) -> (%s)' % (lhs, sequi.ref_counts[lhs], '-'.join(array))
    
sequi.draw_graph('test.png')