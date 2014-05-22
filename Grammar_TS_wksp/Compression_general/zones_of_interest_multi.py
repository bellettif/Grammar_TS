'''
Created on 18 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import re

import load_data

data_set = 'achu'

def get_bare_lk(seq, s_freqs, k):
    seq = filter(lambda x : x != 'g', seq)
    seq = re.subn("a+", "a", ''.join(seq))[0]
    seq = list(seq)
    N = len(seq)
    log_lks = {}
    powers = {}
    for i in range(0, N - k):
        current_sequence = ''.join(seq[i:i+k])
        log_lks[current_sequence] = np.sum(np.log(np.asarray([s_freqs[x] for x in seq[i:i+k]], dtype = np.double)))
        if current_sequence not in powers:
            powers[current_sequence] = 0
        powers[current_sequence] += 1
    for key, item in log_lks.iteritems():
        log_lks[key] = - item * powers[key] / sum(powers.values())
    return log_lks, powers

if data_set == 'achu':
    file_contents = load_data.achu_file_contents
    folder_name = 'zones_of_interest_achu/'
elif data_set == 'oldo':
    file_contents = load_data.oldo_file_contents
    folder_name = 'zones_of_interest_oldo/'
else:
    raise Exception('Invalid data set name')

sequences = file_contents.values()

flattened = []
for sequence in sequences:
    flattened.extend(sequence)

all_symbols = list(set(flattened))

symbol_frequences = {}
for s in all_symbols:
    symbol_frequences[s] = float(flattened.count(s)) / float(len(flattened))
    
labels = []
values = []
for key, item in symbol_frequences.iteritems():
    labels.append(key)
    values.append(item)

sequences = [''.join(x) for x in file_contents.values()]

print len(sequences)

def plot_zones_of_interest(sub_seq_length, plot_folder):
    packs = [get_bare_lk(x, symbol_frequences, sub_seq_length) for x in sequences]
    bare_lks = [x[0] for x in packs]
    reduced_string_lks = {}
    reduced_powers = {}
    for string_lks in bare_lks:
        for key, item in string_lks.iteritems():
            if key not in reduced_string_lks:
                reduced_string_lks[key] = 0
                reduced_powers[key] = 0
            reduced_string_lks[key] += item
            reduced_powers[key] += 1
    list_of_pairs = reduced_string_lks.items()
    list_of_pairs.sort(key = (lambda x : -x[1]))
    list_of_pairs = list_of_pairs[:50]
    labels = [x[0] for x in list_of_pairs]
    values = [x[1] for x in list_of_pairs]
    for i in xrange(len(labels)):
        labels[i] = labels[i] + '_' + str(reduced_powers[labels[i]])    
    if data_set == 'achu':
        plt.bar(range(len(labels)), values, align = 'center', color = 'r')
    else:
        plt.bar(range(len(labels)), values, align = 'center', color = 'b')
    plt.xticks(range(len(labels)), labels, rotation = 'vertical', fontsize = 6)
    plt.savefig(plot_folder + ('sub_seq_length_%d.png' % sub_seq_length), dpi = 300)
    plt.close()
    
    
    
for length in range(10):
    plot_zones_of_interest(length, folder_name)