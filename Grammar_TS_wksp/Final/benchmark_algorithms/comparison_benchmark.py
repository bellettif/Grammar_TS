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

def get_scores(actual, detected):
    actual = set(actual)
    detected = set(detected)
    true_positive = actual.intersection(detected)
    precision = float(len(true_positive)) / float(len(detected))
    recall = float(len(true_positive)) / float(len(actual))
    F_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F_score

input_sentences = s_g.produce_sentences(n_roots,
                                        n_wildcards,
                                        n_sentences)

input_sentences = [x.split(' ') 
                   for x in input_sentences]

def run_benchmark(k, n_rounds):
    #
    #    Test sequitur
    #
    sequitur_hashcodes = []
    for input_sentence in input_sentences:
        seq = k_Sequitur(copy.deepcopy(input_sentence))
        seq.run()
        sequitur_hashcodes.extend(seq.hashcode_to_rule.keys())
    #   
    s_g_hashcodes = filter_hashcodes(s_g.hashcodes.values())
    sequitur_hashcodes = filter_hashcodes(sequitur_hashcodes)
    #
    common_hashcodes = s_g_hashcodes.intersection(sequitur_hashcodes)
    all_hashcodes = s_g_hashcodes.union(sequitur_hashcodes)
    #
    precision, recall, F_score = get_scores(s_g_hashcodes,
                                            sequitur_hashcodes)
    avg_depth = np.mean([len(x) for x in common_hashcodes])
    print 'Sequitur:\n\tprecision = %f, recall = %f, F_score = %f, avg depth = %f' \
            % (precision, recall, F_score, avg_depth)
            
            
    #
    #    Test k-sequitur
    #
    k_sequitur_hashcodes = []
    for input_sentence in input_sentences:
        k_seq = k_Sequitur(copy.deepcopy(input_sentence),
                           k = k)
        k_seq.run()
        k_sequitur_hashcodes.extend(k_seq.hashcode_to_rule.keys())
    #    
    s_g_hashcodes = filter_hashcodes(s_g.hashcodes.values())
    k_sequitur_hashcodes = filter_hashcodes(k_sequitur_hashcodes)
    #
    common_hashcodes = s_g_hashcodes.intersection(k_sequitur_hashcodes)
    all_hashcodes = s_g_hashcodes.union(k_sequitur_hashcodes)
    #
    precision, recall, F_score = get_scores(s_g_hashcodes,
                                            k_sequitur_hashcodes)
    avg_depth = np.mean([len(x) for x in common_hashcodes])
    print 'k-Sequitur (k = %d):\n\tprecision = %f, recall = %f, F_score = %f, avg depth = %f' \
            % (k, precision, recall, F_score, avg_depth)
            
    #
    #    Test proba-sequitur
    #
    proba_sequitur_hashcodes = []
    max_rules = n_rounds * k
    proba_seq = Proba_sequitur(copy.deepcopy(input_sentences),
                               copy.deepcopy(input_sentences),
                               k = k,
                               max_rules = max_rules,
                               random = False)
    proba_seq.run()
    proba_sequitur_hashcodes.extend(proba_seq.hashcode_to_rule.keys())
    #
    s_g_hashcodes = filter_hashcodes(s_g_hashcodes)
    proba_sequitur_hashcodes = filter_hashcodes(proba_sequitur_hashcodes)
    #
    common_hashcodes = s_g_hashcodes.intersection(proba_sequitur_hashcodes)
    all_hashcodes = s_g_hashcodes.union(proba_sequitur_hashcodes)
    #
    precision, recall, F_score = get_scores(s_g_hashcodes,
                                            proba_sequitur_hashcodes)
    avg_depth = np.mean([len(x) for x in common_hashcodes])
    print 'Proba_sequitur (k = %d, max_rules = %d):\n\tprecision = %f, recall = %f, F_score = %f, avg depth = %f' \
            % (k, max_rules, precision, recall, F_score, avg_depth)
            
for k in xrange(2, 12):
    for n_rounds in xrange(1, 6):
        run_benchmark(k, n_rounds)
        print ''
    print '\n'
