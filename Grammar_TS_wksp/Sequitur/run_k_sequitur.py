'''
Created on 14 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv
import os
import cPickle as pickle

import k_compression.k_sequitur as sequitur

folder_path = "compression/Sequitur_lk/data/"
filenames = os.listdir(folder_path)
filenames = filter(lambda x : '.csv' in x, filenames)
filenames = [folder_path + x for x in filenames]

output_folder = 'Observed_grammars/'

def build_grammar(file_path, k = 10, q = 10):
    print 'Extracting grammar from %s' % file_path
    sequence = []
    with open(file_path, 'rb') as input_file:
        csv_reader = csv.reader(input_file)
        for line in csv_reader:
            sequence = line
            break
    sequence = np.asarray(sequence, dtype = np.int32)
    grammar = sequitur.k_seq_compress(sequence, k, q)
    pickle.dump(grammar, open(output_folder + file_path.split('.')[0].split('/')[-1] + '.pi', 'wb'))
    print 'Done'
  
for filename in filenames:
    build_grammar(filename)
