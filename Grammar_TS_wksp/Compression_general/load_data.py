'''
Created on 16 mai 2014

@author: francois
'''

'''
Created on 13 mai 2014

@author: francois
'''

import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import string

alphabet = string.ascii_lowercase

conversion = {'1' : 'a',
              '2' : 'b',
              '3' : 'c',
              '4' : 'd',
              '5' : 'e',
              '6' : 'f',
              '7' : 'g'}

data_folder = '../Sequitur/data/'

list_of_files = os.listdir(data_folder)
list_of_files = filter(lambda x : '.csv' in x, list_of_files)

file_contents = {} 

for x in list_of_files:
    with open(data_folder + x, 'rb') as input_file:
        csv_reader = csv.reader(input_file)
        for line in csv_reader:
            file_contents[x] = [conversion[y] for y in line]
            break

"""
for file_name, file_content in file_contents.iteritems():
    print file_name
    print file_content
"""