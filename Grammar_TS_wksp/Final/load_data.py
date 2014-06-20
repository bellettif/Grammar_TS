'''
Created on 16 mai 2014

This script loads the sequences from files.

It converts the integer symbols into the letters a, b, c, d, e, f, g

It creates six data sets (dictionary filename -> string):
    achu_file_contents: raw achu data
    no_rep_achu_file_contents: achu data without repetition of symbol a (1)

    no_g_achu_file_contents: achu data where the g (7) symbol has been erased
    no_g_no_rep_achu_file_contents: achu data where the g (7) symbol 
                                    and repetitions of symbol a (1) have been erased

    oldo_file_contents: raw oldo data
    no_rep_oldo_file_contents: oldo data where the repetitions of symbol a (1)
                                have been erased
                                
@author: francois belletti
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

int_to_char = ['a',
               'b',
               'c',
               'd',
               'e',
               'f',
               'g']

char_to_int = dict(zip(int_to_char, range(len(int_to_char))))

data_folder = '/Users/francois/Grammar_TS/data/'

list_of_files = os.listdir(data_folder)
list_of_files = filter(lambda x : '.csv' in x, list_of_files)

file_contents = {}

def eliminate_repetitions_of_a(seq):
    seq = re.subn('a+', 'a', ''.join(seq))[0]
    return seq

def eliminate_symbol_g(seq):
    seq = re.subn('g', '', ''.join(seq))[0]
    return seq

for x in list_of_files:
    with open(data_folder + x, 'rb') as input_file:
        csv_reader = csv.reader(input_file)
        for line in csv_reader:
            file_contents[x] = ''.join([conversion[y] for y in line])
            break

achu_file_contents = {}
no_rep_achu_file_contents = {}

no_g_achu_file_contents = {}
no_g_no_rep_achu_file_contents = {}

oldo_file_contents = {}
no_rep_oldo_file_contents = {}

no_g_oldo_file_contents = {}
no_g_no_rep_oldo_file_contents = {}

for key, value in file_contents.iteritems():
    key = key.split('.')[0]
    if('achu' in key):
        achu_file_contents[key] = ' '.join(value)
        no_rep_achu_file_contents[key] = ' '.join(eliminate_repetitions_of_a(value))
        no_g_achu_file_contents[key] = ' '.join(eliminate_symbol_g(value))
        no_g_no_rep_achu_file_contents[key] = ' '.join(eliminate_symbol_g(eliminate_repetitions_of_a(value)))
    if('oldo' in key):
        oldo_file_contents[key] = ' '.join(value)
        no_rep_oldo_file_contents[key] = ' '.join(eliminate_repetitions_of_a(value))
        no_g_oldo_file_contents[key] = ' '.join(eliminate_symbol_g(value))
        no_g_no_rep_oldo_file_contents[key] = ' '.join(eliminate_symbol_g(eliminate_repetitions_of_a(value)))
        
achu_file_names = achu_file_contents.keys()
achu_file_names.sort()

oldo_file_names = oldo_file_contents.keys()
oldo_file_names.sort()