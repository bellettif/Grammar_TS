'''
Created on 1 juil. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle
import copy

from compression.sequitur import Sequitur
from k_compression.k_sequitur import k_Sequitur
from proba_sequitur.proba_sequitur import Proba_sequitur

s_g = pickle.load(open('surrogate_grammar.pi', 'rb'))

n_roots = 128
n_wildcards = 32
n_sentences = 18

def filter_hashcodes(hashcodes):
    hashcodes = copy.deepcopy(set(hashcodes))
    hashcodes = [x.replace('-', '') for x in hashcodes]
    hashcodes = [x.replace('>', '') for x in hashcodes]
    hashcodes = [x.replace('<', '') for x in hashcodes]
    hashcodes = set(hashcodes)
    return hashcodes

input_sentences = s_g.produce_sentences(n_roots,
                                        n_wildcards,
                                        n_sentences)

input_sentences = [x.split(' ') 
                   for x in input_sentences]

sequitur_hashcodes = []
for input_sentence in input_sentences:
    seq = k_Sequitur(copy.deepcopy(input_sentence))
    seq.run()
    sequitur_hashcodes.extend(seq.hashcode_to_rule.keys())
    
s_g_hashcodes = filter_hashcodes(s_g.hashcodes.values())
sequitur_hashcodes = filter_hashcodes(sequitur_hashcodes)

common_hashcodes = s_g_hashcodes.intersection(sequitur_hashcodes)
all_hashcodes = s_g_hashcodes.union(sequitur_hashcodes)

print len(all_hashcodes)
print len(s_g_hashcodes)
print len(sequitur_hashcodes)
print len(common_hashcodes)
print np.mean([len(x) for x in common_hashcodes])

k_sequitur_hashcodes = []
for input_sentence in input_sentences:
    k_seq = k_Sequitur(copy.deepcopy(input_sentence),
                       k = 6)
    k_seq.run()
    k_sequitur_hashcodes.extend(k_seq.hashcode_to_rule.keys())
    
s_g_hashcodes = filter_hashcodes(s_g.hashcodes.values())
k_sequitur_hashcodes = filter_hashcodes(k_sequitur_hashcodes)

common_hashcodes = s_g_hashcodes.intersection(k_sequitur_hashcodes)
all_hashcodes = s_g_hashcodes.union(k_sequitur_hashcodes)

print '\n'
print common_hashcodes

print '\n'
print len(all_hashcodes)
print len(s_g_hashcodes)
print len(k_sequitur_hashcodes)
print len(common_hashcodes)
print np.mean([len(x) for x in common_hashcodes])


proba_sequitur_hashcodes = []
k = 12
proba_seq = Proba_sequitur(copy.deepcopy(input_sentences),
                           copy.deepcopy(input_sentences),
                           k = k,
                           max_rules = 5 * k,
                           random = False)
proba_seq.run()
proba_sequitur_hashcodes.extend(proba_seq.hashcode_to_rule.keys())
       
s_g_hashcodes = filter_hashcodes(s_g_hashcodes)
proba_sequitur_hashcodes = filter_hashcodes(proba_sequitur_hashcodes)

common_hashcodes = s_g_hashcodes.intersection(proba_sequitur_hashcodes)
all_hashcodes = s_g_hashcodes.union(proba_sequitur_hashcodes)

print '\n'
print common_hashcodes

print '\n'
print len(all_hashcodes)
print len(s_g_hashcodes)
print len(proba_sequitur_hashcodes)
print len(common_hashcodes)
print np.mean([len(x) for x in common_hashcodes])
