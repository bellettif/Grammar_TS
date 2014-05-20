'''
Created on 16 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

import load_data

def compute_cor_lag(sequence, lag, target_i, target_j):
    if lag < 0:
        return compute_cor_lag(sequence, -lag, target_j, target_i)
    i_sequence = np.asarray([1 if x == target_i else 0 for x in sequence],
                            dtype = np.double)
    j_sequence = np.asarray([1 if x == target_j else 0 for x in sequence],
                            dtype = np.double)
    """
    if lag != 0:
        print i_sequence[lag:]
        print j_sequence[:-lag]
    else:
        print i_sequence
        print j_sequence
    """
    if lag == 0:
        """
        print np.mean(i_sequence * j_sequence)
        print np.mean(i_sequence) * np.mean(j_sequence)
        """
        return (np.mean(i_sequence * j_sequence) - np.mean(i_sequence) * np.mean(j_sequence)) / \
                    np.sqrt(np.var(i_sequence) * np.var(j_sequence))
    return np.mean(i_sequence[:-lag] * j_sequence[lag:]) - np.mean(i_sequence[:-lag]) * np.mean(j_sequence[lag:]) / \
                    np.sqrt(np.var(i_sequence[:-lag]) * np.var(j_sequence[lag:]))

def compute_cor_lags(sequence, max_lag, target_i, target_j):
    lags = np.arange(-max_lag, max_lag + 1)
    #print lags
    return {'lags' : lags,
            'cors' : [compute_cor_lag(sequence, l, target_i, target_j) for l in lags]}

def compute_cor_occurences(sequence, max_lag):
    all_symbols = set(sequence)
    result = {}
    for i in all_symbols:
        result[i] = {}
        for j in all_symbols:
            result[i][j] = compute_cor_lags(sequence, max_lag, i, j)
    return result

def plot_cor_occurences(sequence, max_lag, output_folder):
    all_symbols = list(set(sequence))
    n_symbols = len(all_symbols)
    result_dict = compute_cor_occurences(sequence, max_lag)
    for i in range(n_symbols):
        for j in range(n_symbols):
            current_i_symbol = all_symbols[i]
            current_j_symbol = all_symbols[j]
            current_lags = result_dict[current_i_symbol][current_j_symbol]['lags']
            current_cors = result_dict[current_i_symbol][current_j_symbol]['cors']
            print current_lags
            print current_cors
            plt.plot(current_lags, current_cors)
            plt.xlabel('lags')
            plt.ylabel('cors')
            plt.ylim((-1, 1))
            plt.title('Cross_cor %s %s' %(str(current_i_symbol), str(current_j_symbol)))
            plt.savefig(output_folder + 'Cross_cor_%s_%s.png' %(str(current_i_symbol), str(current_j_symbol)), dpi = 300)
            plt.close()


print load_data.file_contents
sequence = load_data.file_contents['achuSeq_1.csv']
result_dict = plot_cor_occurences(sequence, 10, 'co_occurence_plots/')