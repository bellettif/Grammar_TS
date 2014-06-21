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

from compression.sequitur import Sequitur
from information_th.measures import compute_entropy

repetition_data_set = [(x, load_data.achu_file_contents[x])
                       for x in achu_file_names] + \
                        [(x, load_data.oldo_file_contents[x])
                         for x in oldo_file_names]
no_repetition_data_set = [(x,load_data.no_rep_achu_file_contents[x])
                          for x in achu_file_names] + \
                          [(x, load_data.no_g_no_rep_oldo_file_contents[x])
                           for x in oldo_file_names]
                          
def run_seq(sequence):
    sequi = Sequitur(sequence)
    sequi.run()
    return float(len(sequi.compressed_sequence)), \
            float(len(sequi.input_sequence)), \
            len(sequi.grammar), \
            sequi.hashed_freqs
            
repetition_results = [(x[0], run_seq(x[1])) 
                      for x in repetition_data_set]
no_repetition_results = [(x[0], run_seq(x[1]))
                         for x in no_repetition_data_set]

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
plt.title('Sequitur compression (repetitions of a) (size prop. to n. of rules)')
plt.savefig('Repetitions_length_seq.png', dpi = 600)
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
plt.title('Sequitur compression (no repetitions of a) (size prop. to number of rules)')
plt.savefig('No_repetitions_length_seq.png', dpi = 600)
plt.close()
