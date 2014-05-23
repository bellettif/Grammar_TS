'''
Created on 14 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

import compression.sequitur as sequitur

sequence_name = 'achuSeq_6'
file_path = "../../data/%s.csv" % sequence_name

sequence = []
with open(file_path, 'rb') as input_file:
    csv_reader = csv.reader(input_file)
    for line in csv_reader:
        sequence = line
        break
    
sequence = np.asarray(sequence, dtype = np.int32)

grammar = sequitur.run(sequence, 1000)

print 'Initial sequence length: %d' % len(sequence)
print 'Post compression sequence length: %d' % len(grammar[0][0])
print 'Number of rules: %d' % (len(grammar.keys()) - 1)
'''
all_refs = []

for lhs, [rhs, ref_count] in grammar.iteritems():
    if lhs == 0: continue
    all_refs.append({'lhs' : lhs, 'ref_count' : ref_count, 'rhs' : rhs})
    
sorted_refs = sorted(all_refs, key = (lambda x : -x['ref_count']))
    
plt.hist([x['ref_count'] for x in all_refs])
plt.title('Ref count frequency %s' % sequence_name)
plt.savefig('Ref_counts_%s.png' % sequence_name, dpi = 300)
'''
