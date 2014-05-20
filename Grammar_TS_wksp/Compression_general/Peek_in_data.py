'''
Created on 13 mai 2014

@author: francois
'''

import os
import csv
import numpy as np
from matplotlib import pyplot as plt

data_folder = '../Sequitur/data/'

list_of_files = os.listdir(data_folder)
list_of_files = filter(lambda x : '.csv' in x, list_of_files)

file_contents = {} 

for x in list_of_files:
    with open(data_folder + x, 'rb') as input_file:
        csv_reader = csv.reader(input_file)
        for line in csv_reader:
            file_contents[x] = line
            break
        
for file_name, file_content in file_contents.iteritems():
    print file_name
    print file_content
    
string_frequences = {}
all_frequences = []
for length in range(1, 10):
    for file_content in file_contents.values():
        for i in xrange(len(file_content) - length):
            current_string = ''.join(set(file_content[i:i + length]))
            print current_string
            if current_string not in string_frequences:
                string_frequences[current_string] = 0
            string_frequences[current_string] += 1       
            
for current_string, current_count in  string_frequences.iteritems():
    string_frequences[current_string] = np.log(current_count)
    all_frequences.append(np.log(current_count))

print all_frequences
all_frequences = np.asarray(all_frequences)
most_frequent = np.percentile(all_frequences, 60)

print most_frequent
all_keys = string_frequences.keys()
for key in all_keys:
    if string_frequences[key] < most_frequent:
        del string_frequences[key]
        
all_pairs = []
for key, value in string_frequences.iteritems():
    all_pairs.append([key, value])
    
all_pairs.sort(key = lambda x : -x[1])

labels = [x[0] for x in all_pairs]
values = [x[1] for x in all_pairs]

print len(string_frequences)
plt.bar(range(len(labels)), values, align='center')
plt.xticks(range(len(labels)), labels, rotation = 'vertical')
plt.show()