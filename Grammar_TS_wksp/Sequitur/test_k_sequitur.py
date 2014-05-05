'''
Created on 14 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

import k_compression.k_sequitur as sequitur

sequence_name = 'achuSeq_3'
file_path = "compression/Sequitur_lk/data/%s.csv" % sequence_name

sequence = []
with open(file_path, 'rb') as input_file:
    csv_reader = csv.reader(input_file)
    for line in csv_reader:
        sequence = line
        break
    
sequence = np.asarray(sequence, dtype = np.int32)

grammar = sequitur.k_seq_compress(sequence, 10, 10)

dictio = {}
dictio.keys()

grammar_keys = grammar.keys()
grammar_keys = sorted(grammar_keys)
for lhs in grammar_keys:
    temp = grammar[lhs]
    print lhs + ' (' + temp['barcode'][:10] + (') %d refs, %d pop -> ' % (temp['refs'], temp['pop_count'])) + ' '.join(temp['rhs'])

root = grammar['0']['rhs']

result = ''
for s in root:
    if s in grammar:
        result += grammar[s]['barcode']
    else:
        result += s   
    result += ''
    
lhs_as_int = [int(x) for x in grammar.keys()]

lhs_as_int.sort()

pop_counts = {}
for x in grammar['0']['rhs']:
    if x in grammar:
        if x not in pop_counts:
            pop_counts[x] = 0
        pop_counts[x] += 1

for i in lhs_as_int:
    if i == 0: continue
    local_counts = {}
    for y in grammar[str(i)]['rhs']:
        if(y in grammar):
            if y not in local_counts:
                local_counts[y] = 0
            local_counts[y] += 1
    for x, count in local_counts.iteritems():
        if x not in pop_counts:
            pop_counts[x] = 0
        pop_counts[x] += pop_counts[str(i)] * count

print '\n'
                
for i, count in pop_counts.iteritems():
    print 'Lhs %s, counts %d' % (i, count)

print '\n'

print 'Initial sequence length: %d' % len(sequence)
print 'Post compression sequence length: %d' % len(grammar['0']['rhs'])
print 'Number of rules: %d' % (len(grammar.keys()) - 1)

