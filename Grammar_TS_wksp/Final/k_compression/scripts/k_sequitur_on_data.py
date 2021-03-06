'''
Created on 21 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

import load_data

import plot_convention.colors as colors

from load_data import achu_file_names, \
                        oldo_file_names

from k_compression.k_sequitur import k_Sequitur

repetition_data_set = [(x, load_data.achu_file_contents[x])
                       for x in achu_file_names] + \
                        [(x, load_data.oldo_file_contents[x])
                         for x in oldo_file_names]
no_repetition_data_set = [(x,load_data.no_rep_achu_file_contents[x])
                          for x in achu_file_names] + \
                          [(x, load_data.no_g_no_rep_oldo_file_contents[x])
                           for x in oldo_file_names]
                          
def run_seq(sequence, k):
    sequi = k_Sequitur(sequence.split(' '), k)
    sequi.run()
    return float(len(sequi.compressed_sequence)), \
            float(len(sequi.input_sequence)), \
            len(sequi.grammar), \
            sequi.hashed_freqs
            
            
for k in [2, 4, 6, 8, 10, 12]:
    repetition_results = [(x[0], run_seq(x[1], k)) 
                          for x in repetition_data_set]
    no_repetition_results = [(x[0], run_seq(x[1], k))
                             for x in no_repetition_data_set]
    print repetition_results
    plt.scatter(x = [x[1][1] for x in repetition_results], 
                y = [x[1][0] for x in repetition_results], 
                s = [2 * x[1][2] for x in repetition_results], 
                c = [colors.all_colors[x[0]] for x in repetition_results],
                marker = 'o',
                alpha = 0.6)
    for x in repetition_results:
        plt.text(x[1][1] + 8, x[1][0] + 8, s = x[0][:4], fontsize = 8)
    plt.ylabel('Post compression length')
    plt.xlabel('Pre compression length')
    plt.title('k-Sequitur (k = %d) (repetitions of a) (size prop. to n. of rules)' % k)
    plt.savefig('Repetitions_length_seq_%d.png' % k, dpi = 600)
    plt.close()
    
    plt.scatter(x = [x[1][1] for x in no_repetition_results], 
                y = [x[1][0] for x in no_repetition_results], 
                s = [2 * x[1][2] for x in no_repetition_results], 
                c = [colors.all_colors[x[0]] for x in no_repetition_results],
                marker = 'o',
                alpha = 0.6)
    for x in no_repetition_results:
        plt.text(x[1][1] + 8, x[1][0] + 8, s = x[0][:4], fontsize = 8)
    plt.ylabel('Post compression length')
    plt.xlabel('Pre compression length')
    plt.title('k-Sequitur (k = %d) (no repetitions of a) (size prop. to number of rules)' % k)
    plt.savefig('No_repetitions_length_seq_%d.png' % k, dpi = 600)
    plt.close()
