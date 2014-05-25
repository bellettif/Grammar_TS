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
import re

alphabet = string.ascii_lowercase

conversion = {'1' : 'a',
              '2' : 'b',
              '3' : 'c',
              '4' : 'd',
              '5' : 'e',
              '6' : 'f',
              '7' : 'g'}

data_folder = '../../data/'

list_of_files = os.listdir(data_folder)
list_of_files = filter(lambda x : '.csv' in x, list_of_files)

file_contents = {}

def prefilter(seq):
    seq = re.subn('g', '', ''.join(seq))[0]
    for target_char in ['a', 'b', 'c', 'd', 'e', 'f']:
        seq = re.subn(target_char + '+', target_char, ''.join(seq))[0]
    seq = ' '.join(list(seq))
    return seq

for x in list_of_files:
    with open(data_folder + x, 'rb') as input_file:
        csv_reader = csv.reader(input_file)
        for line in csv_reader:
            file_contents[x] = prefilter([conversion[y] for y in line])
            break

achu_file_contents = {}
oldo_file_contents = {}

for key, value in file_contents.iteritems():
    if('achu' in key):
        achu_file_contents[key] = value
    if('oldo' in key):
        oldo_file_contents[key] = value