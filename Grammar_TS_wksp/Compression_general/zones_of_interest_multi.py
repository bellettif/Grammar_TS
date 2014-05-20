'''
Created on 18 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import re

import load_data

def get_bare_lk(seq, s_freqs, k):
    seq = filter(lambda x : x != 'g', seq)
    seq = re.subn("a+", "a", ''.join(seq))[0]
    seq = list(seq)
    N = len(seq)
    lks = {}
    powers = {}
    for i in range(0, N - k):
        current_sequence = ''.join(seq[i:i+k])
        if 'aa' in current_sequence:
            print ''.join(seq)
        lks[current_sequence] = np.prod(np.asarray([s_freqs[x] for x in seq[i:i+k]], dtype = np.double))
        if current_sequence not in powers:
            powers[current_sequence] = 0
        powers[current_sequence] += 1
    for key, item in lks.iteritems():
        lks[key] = np.power(item, powers[key])
    return lks

file_contents = load_data.file_contents

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

bare_lks = [get_bare_lk(x, symbol_frequences, 5) for x in sequences]

reduced_string_lks = {}
for string_lks in bare_lks:
    for key, item in string_lks.iteritems():
        if key not in reduced_string_lks:
            reduced_string_lks[key] = 1
        reduced_string_lks[key] *= item
  
list_of_pairs = reduced_string_lks.items()
list_of_pairs.sort(key = (lambda x : -x[1]))
list_of_pairs = list_of_pairs[:20]

labels = [x[0] for x in list_of_pairs]
values = [x[1] for x in list_of_pairs]

plt.bar(range(len(labels)), values, align = 'center')
plt.xticks(range(len(labels)), labels, rotation = 'vertical')
plt.show()