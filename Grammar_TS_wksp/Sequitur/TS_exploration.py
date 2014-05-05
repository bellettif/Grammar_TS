'''
Created on 17 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

file_path = "Sequitur_lk/data/achuSeq_3.csv"

sequence = []
with open(file_path, 'rb') as input_file:
    csv_reader = csv.reader(input_file)
    for line in csv_reader:
        sequence = line
        break
    
sequence = np.asarray(sequence, dtype = np.int32)

def measure_entropy(input):
    values = {}
    for x in input:
        if x not in values:
            values[x] = 0
        values[x] += 1
    N = float(len(input))
    probas = []
    for x, n_x in values.iteritems():
        probas.append(float(n_x) / N)
    probas = np.asarray(probas, dtype = np.float)
    return - np.sum(probas * np.log2(probas))

print measure_entropy(sequence)
plt.hist(sequence)
plt.show()
