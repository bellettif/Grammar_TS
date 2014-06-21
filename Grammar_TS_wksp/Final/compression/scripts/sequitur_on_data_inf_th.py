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
    sequi = Sequitur(sequence.split(' '))
    sequi.run()
    return compute_entropy(sequi.compressed_sequence), \
            compute_entropy(sequi.input_sequence), \
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
x_min = min([x[1][1] for x in repetition_results])
x_max = max([x[1][1] for x in repetition_results])
y_min = min([x[1][0] for x in repetition_results])
y_max = max([x[1][0] for x in repetition_results])
plt.plot(np.linspace(x_min, x_max, 100, True),
         np.linspace(y_min, y_max, 100, True),
         c = 'k')
for x in repetition_results:
    plt.text(x[1][1], x[1][0], s = x[0][:4], fontsize = 8)
plt.ylabel('Post compression entropy')
plt.xlabel('Pre compression entropy')
plt.title('Sequitur compression (repetitions of a) (size prop. to n. of rules)')
plt.savefig('Repetitions_entropy_seq.png', dpi = 600)
plt.close()

plt.scatter(x = [x[1][1] for x in no_repetition_results], 
            y = [x[1][0] for x in no_repetition_results], 
            s = [2 * x[1][2] for x in no_repetition_results], 
            c = [colors.all_colors[x[0]] for x in no_repetition_results],
            marker = 'o',
            alpha = 0.6)
x_min = min([x[1][1] for x in no_repetition_results])
x_max = max([x[1][1] for x in no_repetition_results])
y_min = min([x[1][0] for x in no_repetition_results])
y_max = max([x[1][0] for x in no_repetition_results])
plt.plot(np.linspace(x_min, x_max, 100, True),
         np.linspace(y_min, y_max, 100, True),
         c = 'k')
for x in no_repetition_results:
    plt.text(x[1][1], x[1][0], s = x[0][:4], fontsize = 8)
plt.ylabel('Post compression entropy')
plt.xlabel('Pre compression entropy')
plt.title('Sequitur compression (no repetitions of a) (size prop. to number of rules)')
plt.savefig('No_repetitions_entropy_seq.png', dpi = 600)
plt.close()
