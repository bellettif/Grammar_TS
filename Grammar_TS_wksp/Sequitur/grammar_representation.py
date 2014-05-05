'''
Created on 18 avr. 2014

@author: francois
'''

import time
import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib

from compression.sequitur import run as sequitur_algo
import information_th.entropy_stats as inf_th
from loader import load_sequence as load_sequence
from grammar_graph import grammar_to_graph

data_folder = 'compression/Sequitur_lk/data/'

data_files = os.listdir(data_folder)
data_files = filter(lambda x : '.csv' in x, data_files)

sequences = {}

for data_file in data_files:
    sequences[data_file] = load_sequence(data_folder + data_file)
    
compression_results = {}
for file_name, sequence in sequences.iteritems():
    #if(file_name != 'oldoSeq_2.csv'): continue
    print file_name
    print sequence[-40:]
    print 'Sequence length %d' % len(sequence)
    start_running = time.clock()
    grammar = sequitur_algo(sequence)
    print time.clock() - start_running
    print 'Grammar length %d' % len(grammar.keys())
    print 'Compressed length %d' % len(grammar[0][0])
    grammar[0][0] = grammar[0][0][-20:]
    grammar_to_graph('Plots/' + file_name.split('.')[0] + 'grammar.png', grammar)
    print '\n'
    

    