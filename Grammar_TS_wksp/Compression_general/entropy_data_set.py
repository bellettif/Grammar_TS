'''
Created on 22 mai 2014

@author: francois
'''

import numpy as np
import re
import string
from matplotlib import pyplot as plt


import load_data

entropies = {}

repetitions = False
all_data_sets = ['oldo', 'achu', 'resh_oldo', 'resh_achu']

for data_set in all_data_sets:

    if data_set == 'achu':
        file_contents = load_data.achu_file_contents
        folder_name = 'entropy_achu/'
    elif data_set == 'oldo':
        file_contents = load_data.oldo_file_contents
        folder_name = 'entropy_oldo/'
    elif data_set == 'resh_oldo':
        file_contents = load_data.oldo_file_contents
        for key, value in file_contents.iteritems():
            if not repetitions:
                all_symbols = set(value)
                seq = ''.join(value)
                for symbol in all_symbols:
                    seq = re.subn(symbol + '+', symbol, ''.join(seq))[0]
                file_contents[key] = list(seq)
            np.random.shuffle(file_contents[key])
        folder_name = 'entropy_oldo_resh/'
    elif data_set == 'resh_achu':
        file_contents = load_data.achu_file_contents
        for key, value in file_contents.iteritems():
            if not repetitions:
                all_symbols = set(value)
                seq = ''.join(value)
                for symbol in all_symbols:
                    seq = re.subn(symbol + '+', symbol, ''.join(seq))[0]
                file_contents[key] = list(seq)
            np.random.shuffle(file_contents[key])
        folder_name = 'entropy_achu_resh/'
    else:
        raise Exception('Invalid data set name')
    
    def measure_probas_of_sequence(seq, k):
        all_symbols = set(seq)
        seq = ''.join(seq)
        if not repetitions:
            for symbol in all_symbols:
                seq = re.subn(symbol + '+', symbol, ''.join(seq))[0]
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
        all_symbols = []
        for seq in seqs:
            all_symbols.extend(seq)
        all_symbols = set(all_symbols)
        n_symbols = len(all_symbols)
        probas = measure_probas_of_sequences(seqs, k).values()
        probas = np.asarray(probas, dtype = np.double)
        return -np.sum(probas * np.log(probas) / np.log(n_symbols))
    
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
    
    entropies[data_set] = []
    for length in xrange(1,10):
        entropy = measure_entropy_of_sequences(file_contents.values(), length)
        entropies[data_set].append(entropy)
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

entropy_achu = np.asarray(entropies['achu']) / np.asarray(xrange(1, 10))
entropy_oldo = np.asarray(entropies['oldo']) / np.asarray(xrange(1, 10))
entropy_achu_resh = np.asarray(entropies['resh_achu']) / np.asarray(xrange(1, 10))
entropy_oldo_resh = np.asarray(entropies['resh_oldo']) / np.asarray(xrange(1, 10))

if repetitions:
    plt.title('Entropies with repetitions')
    plt.plot(xrange(1, 10), entropy_achu, marker = 'o', color = 'r')
    plt.plot(xrange(1, 10), entropy_oldo, marker = 'o', color = 'b')
    plt.plot(xrange(1, 10), entropy_achu_resh, marker = 'x', linestyle = '--', color = 'r')
    plt.plot(xrange(1, 10), entropy_oldo_resh, marker = 'x', linestyle = '--', color = 'b')
    plt.xlabel('Length of strings')
    plt.legend(('Achu', 'Oldo', 'Achu_resh', 'Oldo_resh'), 'upper right')
    plt.savefig('Entropies_with_repetitions.png', dpi = 300)
    plt.close()
else:
    plt.title('Entropies without repetitions')
    plt.plot(xrange(1, 10), entropy_achu, marker = 'o', color = 'r')
    plt.plot(xrange(1, 10), entropy_oldo, marker = 'o', color = 'b')
    plt.plot(xrange(1, 10), entropy_achu_resh, marker = 'x', linestyle = '--', color = 'r')
    plt.plot(xrange(1, 10), entropy_oldo_resh, marker = 'x', linestyle = '--', color = 'b')
    plt.xlabel('Length of strings')
    plt.legend(('Achu', 'Oldo', 'Achu_resh', 'Oldo_resh'), 'upper right')
    plt.savefig('Entropies_without_repetitions.png', dpi = 300)
    plt.close()