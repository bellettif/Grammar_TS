'''
Created on 22 mai 2014

@author: francois
'''

import numpy as np
import re
from matplotlib import pyplot as plt
import string
import cPickle as pickle

import load_data

def reduce_counts(list_of_dicts):
    reduced_counts = {}
    for current_dict in list_of_dicts:
        for key, value in current_dict.iteritems():
            if key not in reduced_counts:
                reduced_counts[key] = 0
            reduced_counts[key] += value
    return reduced_counts

def atom_counts(sequence):
    symbols = set(sequence.split(' '))
    symbols = filter(lambda x : x != ' ', symbols)
    counts = {}
    for symbol in symbols:
        counts[symbol] = len(re.findall(symbol, sequence))
    return counts

def atom_counts_multi(sequences):
    list_of_counts = [atom_counts(x) for x in sequences]
    return reduce_counts(list_of_counts)

def pair_counts(sequence, candidates):
    all_pairs = [x + ' ' + y if x != y else None for x in candidates for y in candidates]
    all_pairs.extend(x + ' _ ' + y if x != y else None for x in candidates for y in candidates)
    all_pairs = filter(lambda x : x != None, all_pairs)
    counts = {}
    for pair in all_pairs:
        symbol = pair
        symbol = re.subn(' ', '-', pair)[0]
        pattern = re.subn('_', '.', pair)[0]
        counts[symbol] = len(re.findall(pattern, sequence))
    to_delete = []
    for key, value in counts.iteritems():
        if value == 0:
            to_delete.append(key)
    for key in to_delete:
        del counts[key]
    return counts

def pair_counts_multi(sequences, candidates):
    list_of_counts = [pair_counts(x, candidates) for x in sequences]
    return reduce_counts(list_of_counts)

def init_barelk(sequences):
    counts = atom_counts_multi(sequences)
    total = float(sum(counts.values()))
    barelk = {}
    for key, value in counts.iteritems():
        barelk[key] = value / total
    return barelk
    
def compute_barelk(symbol, barelk_table):
    left_symbol = symbol.split('-')[0]
    right_symbol = symbol.split('-')[-1]
    barelk = barelk_table[left_symbol] * barelk_table[right_symbol]
    barelk_table[symbol] = barelk
    return barelk

def length_of_symbol(s):
    return len(filter(lambda x : x != '_' and x != '-', s))

def compute_pair_divergence(sequences, candidates, barelk_table):
    pair_counts = pair_counts_multi(sequences, candidates)
    total = float(sum(pair_counts.values()))
    pair_probas = {}
    for key, value in pair_counts.iteritems():
        pair_probas[key] = value / total
        compute_barelk(key, barelk_table)
    divergences = {}
    total_chars = 0
    for seq in sequences:
        total_chars += len(seq)
    total_chars = float(total_chars)
    for key in pair_probas:
        divergences[key] = pair_probas[key] / float(length_of_symbol(key)) \
                            * np.log2(pair_probas[key] / 
                                      (barelk_table[key]))
    return divergences

def substitute(sequences, symbols):
    for symbol in symbols:
        pattern = re.subn('\-', ' ', symbol)[0]
        pattern = re.subn('_', '.', pattern)[0]
        pattern = string.lower(pattern)
        for i, sequence in enumerate(sequences):
            sequences[i] = re.subn(pattern, symbol, sequence)[0]
    return sequences

"""
target_sequences = load_data.achu_file_contents.values()

list_of_best_symbols = []
for i in xrange(5):
    target_chars = []
    for sequence in target_sequences:
        target_chars.extend(sequence.split(' '))
    target_chars = set(target_chars)
    print target_chars
    target_chars = filter(lambda x : x!= ' ', target_chars)
    barelk_table = init_barelk(target_sequences)
    pair_divergence = compute_pair_divergence(target_sequences,
                                              target_chars,
                                              barelk_table)
    items = pair_divergence.items()
    items.sort(key = (lambda x : -x[1]))
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    best_symbols = labels[:6]
    best_symbols = [string.upper(x) for x in best_symbols]
    list_of_best_symbols.append(best_symbols)
    print best_symbols
    target_sequences = substitute(target_sequences, best_symbols)
    plt.bar(range(len(labels)), values, align = 'center', color = 'b')
    plt.xticks(range(len(labels)), labels, rotation = 'vertical', fontsize = 3)
    plt.show()
    for seq in target_sequences:
        print seq
    print ''
    for i, seq in enumerate(target_sequences):
        new_seq = seq.split(' ')
        new_seq = filter(lambda x : '-' in x, new_seq)
        new_seq = ' '.join(new_seq)
        new_seq = re.sub('\-', '', new_seq)
        new_seq = string.lower(new_seq)
        target_sequences[i] = new_seq
    print target_sequences
    print len(target_sequences)
    print '\n-----------------------------\n'
    
    
pickle.dump(list_of_best_symbols, open('best_symbols_achu.pi', 'wb'))
pickle.dump(target_sequences, open('last_parses_achu.pi', 'wb'))

"""