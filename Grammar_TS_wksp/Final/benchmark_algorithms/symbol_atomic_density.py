'''
Created on 1 juil. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

def compute_symbol_density(input_sentences):
    input_sentences = [x.split(' ') 
                       for x in input_sentences]
    all_symbols = []
    for input_sentence in input_sentences:
        all_symbols.extend(input_sentence)
    symbol_set = set(all_symbols)
    symbol_freqs = {}
    for symbol in symbol_set:
        symbol_freqs[symbol] = float(len(filter(lambda x : x == symbol,
                                 all_symbols))) \
                        / float(len(all_symbols))
    freq_items = symbol_freqs.items()
    freq_items.sort(key = (lambda x : x[0]))
    return freq_items

def plot_frequencies(distrib_items,
                     filepath):
    n_bars = len(distrib_items)
    width = 0.8
    x_pos = np.arange(1, n_bars + 1)
    plt.bar(x_pos, [x[1] for x in distrib_items],
            width = width)
    plt.xticks(x_pos + 0.5 * width, 
               [x[0] for x in distrib_items])
    plt.ylabel('Symbol frequency')
    plt.title('Individual symbol frequency')
    plt.savefig(filepath, dpi = 300)
    plt.close()