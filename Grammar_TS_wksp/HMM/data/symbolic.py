'''
Created on 5 avr. 2014

@author: francois
'''

import os
import csv
import numpy as np
from matplotlib import pyplot as plt

data_folder = '/Users/francois/Dropbox/MSc Distributed Systems/Grammar of Time Series/Data/ConvertedData/'

file_list = os.listdir(data_folder)

oldo_set = {}
achu_set = {}

for file_name in file_list:
    if 'oldo' in file_name:
        target_set = oldo_set
    elif 'achu' in file_name:
        target_set = achu_set
    else:
        continue
    index = int(file_name[-5])
    with open(data_folder + file_name, 'rb') as input_file:
        csv_reader = csv.reader(input_file)
        for line in csv_reader:
            target_set[index] = np.asarray(line, dtype = np.int)
            
print 'Oldo set:'
print oldo_set

print '\n'
print 'Achu set:'
print achu_set

data_sets = {'Oldo' : oldo_set, 'Achu' : achu_set}

'''for data_set_name, target_set in data_sets.iteritems():
    n_points = {}
    n_symbols = {}
    for index, ts in target_set.iteritems():
        n_points[index] = len(ts)
        n_symbols[index] = len(set(ts))
    plt.subplot(211)
    plt.hist(n_points.values())
    plt.title('Number of points in time series of %s' % data_set_name)
    plt.subplot(212)
    plt.hist(n_symbols.values())
    plt.title('Number of symbols in time series of %s' % data_set_name)
    plt.savefig('Basis_stats_%s.png' % data_set_name, dpi = 300)
    plt.close()
'''    
