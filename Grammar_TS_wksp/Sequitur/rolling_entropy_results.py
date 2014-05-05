'''
Created on 18 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib

from compression.sequitur import run as sequitur_algo
import information_th.entropy_stats as inf_th
from loader import load_sequence as load_sequence

data_folder = 'compression/Sequitur_lk/data/'

data_files = os.listdir(data_folder)
data_files = filter(lambda x : '.csv' in x, data_files)

sequences = {}

for data_file in data_files:
    sequences[data_file] = load_sequence(data_folder + data_file)
    
for file_name, sequence in sequences.iteritems():
    grammar = sequitur_algo(sequence)
    compressed = grammar[0][0]
    k = 10
    rolling_meas = inf_th.compute_rolling_entropy(sequence, k)
    rolling_compressed = inf_th.compute_rolling_entropy(compressed, k)
    plt.subplot(211)
    plt.plot(rolling_meas)
    plt.title(file_name)
    plt.xlabel('Index in uncompressed sequence')
    plt.ylabel('Entropy')
    plt.subplot(212)
    plt.plot(rolling_compressed)
    plt.xlabel('Index in compressed sequene')
    plt.ylabel('Entropy')
    plt.savefig('Plots/rolling_entropy_%s.png' % file_name, dpi = 300)
    plt.close()
    