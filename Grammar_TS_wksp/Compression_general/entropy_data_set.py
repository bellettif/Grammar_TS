'''
Created on 22 mai 2014

@author: francois
'''

import numpy as np
import re
import string
from matplotlib import pyplot as plt


import load_data

data_set = 'achu'
if data_set == 'achu':
    file_contents = load_data.achu_file_contents
    folder_name = 'entropy_achu/'
elif data_set == 'oldo':
    file_contents = load_data.oldo_file_contents
    folder_name = 'entropy_oldo/'
else:
    raise Exception('Invalid data set name')

def measure_probas_of_sequence(seq, k):
    seq = filter(lambda x : x != 'g', seq)
    seq = re.subn("a+", "a", ''.join(seq))[0]
    seq = list(seq)
    N = len(seq)
    counts = {}
    for i in range(0, N - k):
        current_sequence = ''.join(seq[i:i+k])
        if current_sequence not in counts:
            counts[current_sequence] = 0
        counts[current_sequence] += 1
    #print counts
    total_counts = float(sum(counts.values()))
    probas = {}
    for key, value in counts.iteritems():
        probas[key] = float(counts[key]) / total_counts
    return probas

def measure_probas_of_sequences(seqs, k):
    counts = {}
    for seq in seqs:
        N = len(seq)
        for key, value in measure_probas_of_sequence(seq, k).iteritems():
            if key not in counts:
                counts[key] = 0
            counts[key] += value * N
    probas = {}
    total_counts = float(sum(counts.values()))
    for key, value in counts.iteritems():
        probas[key] = float(counts[key]) / total_counts
    return probas

def measure_entropy_of_sequences(seqs, k):
    probas = measure_probas_of_sequences(seqs, k).values()
    probas = np.asarray(probas, dtype = np.double)
    return -np.sum(probas * np.log2(probas))

"""
characters = list(string.ascii_lowercase)[:7]
N = len(characters)

def build_sequence(sequence_length):
    result = np.random.choice(characters, size = sequence_length)
    return result

sequences = [build_sequence(1000) for i in xrange(10)]

print measure_probas_of_sequence(sequences[1], 2)

print measure_entropy_of_sequences(sequences, 2)
print measure_entropy_of_sequences(sequences, 3)
print measure_entropy_of_sequences(sequences, 4)
print measure_entropy_of_sequences(sequences, 5)
print measure_entropy_of_sequences(sequences, 6)
"""

for length in xrange(2,10):
    entropy = measure_entropy_of_sequences(file_contents.values(), length)
    probas = measure_probas_of_sequences(file_contents.values(), length)
    key_value_pairs = probas.items()
    key_value_pairs.sort(key = lambda x : -x[1])
    key_value_pairs = key_value_pairs[:50]
    labels = [x[0] for x in key_value_pairs]
    values = [x[1] for x in key_value_pairs]
    if data_set == 'achu':
        plt.bar(range(len(labels)), values, align = 'center', color = 'r')
    else:
        plt.bar(range(len(labels)), values, align = 'center', color = 'b')
    plt.title('Entropy = %f' % entropy)
    plt.xticks(range(len(labels)), labels, rotation = 'vertical', fontsize = 6)
    plt.savefig(folder_name + ('sub_seq_length_%d.png' % length), dpi = 300)
    plt.close()


    
    