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
    
compression_results = {}
for file_name, sequence in sequences.iteritems():
    grammar = sequitur_algo(sequence)
    compressed = grammar[0][0]
    compression_results[file_name] = {'grammar_len' : len(grammar.keys()) - 1,
                                      'len_before' : len(sequence),
                                      'len_after' : len(compressed),
                                      'entropy_before' : inf_th.compute_entropy(sequence),
                                      'entropy_after' : inf_th.compute_entropy(compressed)}
    
n = len(compression_results.keys())
file_names = []
grammar_lengths = []
entropy_bef = []
entropy_aft = []
len_bef = []
len_aft = []

for file_name, result in compression_results.iteritems():
    file_names.append(file_name)
    grammar_lengths.append(result['grammar_len'])
    entropy_bef.append(result['entropy_before'])
    entropy_aft.append(result['entropy_after'])
    len_bef.append(result['len_before'])
    len_aft.append(result['len_after'])
    
for i, file_name in enumerate(file_names):
    plt.text(len_bef[i], len_aft[i], file_name, size = 'xx-small')
plt.scatter(len_bef, len_aft, c = range(n), cmap = plt.cm.jet, s = 5 * np.asarray(grammar_lengths), alpha = 0.5)
plt.title("Length reduction (blob size prop. to number of rules)")
plt.xlabel('Length before compression')
plt.ylabel('Length after compression')
plt.savefig('Plots/Length_results_seq.png', dpi = 300)
plt.close()

for i, file_name in enumerate(file_names):
    plt.text(entropy_bef[i], entropy_aft[i], file_name, size = 'xx-small')
plt.scatter(entropy_bef, entropy_aft, c = range(n), cmap = plt.cm.jet, s = 5 * np.asarray(grammar_lengths), alpha = 0.5)
plt.title("Entropy inflation (blob size prop. to number of rules)")
plt.xlabel('Entropy before compression')
plt.ylabel('Entropy after compression')
plt.savefig('Plots/Entropy_results_seq.png', dpi = 300)
plt.close()

