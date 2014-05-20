'''
Created on 18 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

import load_data

file_contents = load_data.file_contents

sequence = file_contents['achuSeq_1.csv']
sequence = filter(lambda x : x != '1', sequence)

all_symbols = list(set(sequence))

symbol_frequences = {}
for s in all_symbols:
    symbol_frequences[s] = float(sequence.count(s)) / float(len(sequence))
    
labels = []
values = []
for key, item in symbol_frequences.iteritems():
    labels.append(key)
    values.append(item)

def get_bare_lk(sequence, k):
    N = len(sequence)
    lks = {}
    powers = {}
    for i in range(0, N - k):
        current_sequence = ''.join(sequence[i:i+k])
        lks[current_sequence] = np.cumprod(np.asarray([symbol_frequences[x] for x in sequence[i:i+k]], dtype = np.double))[-1]
        if current_sequence not in powers:
            powers[current_sequence] = 0
        powers[current_sequence] += 1
    for key, item in lks.iteritems():
        lks[key] = np.power(item, powers[key])
    return lks

string_freqs = get_bare_lk(sequence, 3)
labels = []
values = []
for key, item in string_freqs.iteritems():
    labels.append(key)
    values.append(item)
    
list_of_pairs = string_freqs.items()
list_of_pairs.sort(key = (lambda x : -x[1]))

labels = [x[0] for x in list_of_pairs]
values = [np.log(x[1]) for x in list_of_pairs]

plt.bar(range(len(labels)), values, align='center')
plt.xticks(range(len(labels)), labels, rotation = 'vertical')
plt.show()
    