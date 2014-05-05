'''
Created on 18 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

def load_sequence(file_path):
    sequence = []
    with open(file_path, 'rb') as input_file:
        csv_reader = csv.reader(input_file)
        for line in csv_reader:
            sequence = line
            break
    return np.asarray(sequence, dtype = np.int32)