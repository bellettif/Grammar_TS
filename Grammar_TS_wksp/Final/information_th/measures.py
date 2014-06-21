'''
Created on 21 juin 2014

@author: francois
'''

import entropy_stats as infth

import numpy as np

def compute_entropy(symbols):
    symbol_set = list(set(symbols))
    sym_to_int = dict(zip(symbol_set, range(len(symbol_set))))
    translated_symbols = [sym_to_int[x] for x in symbols]
    translated_symbols = np.asarray(translated_symbols, dtype = np.int32)
    return infth.compute_entropy(translated_symbols)

def compute_n_gram_entropy(symbols, n):
    symbol_set = list(set(symbols))
    sym_to_int = dict(zip(symbol_set, range(len(symbol_set))))
    translated_symbols = [sym_to_int[x] for x in symbols]
    translated_symbols = np.asarray(translated_symbols, dtype = np.int32)
    return infth.compute_string_entropy(translated_symbols, n) / float(n)

#
#    k is the length of the rolling window
#
def compute_rolling_entropy(symbols, k):
    symbol_set = list(set(symbols))
    sym_to_int = dict(zip(symbol_set, range(len(symbol_set))))
    translated_symbols = [sym_to_int[x] for x in symbols]
    translated_symbols = np.asarray(translated_symbols, dtype = np.int32)
    return infth.compute_rolling_entropy(translated_symbols, k)
