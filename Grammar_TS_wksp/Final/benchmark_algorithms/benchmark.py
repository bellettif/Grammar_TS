'''
Created on 1 juil. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle
import copy
import string

from compression.sequitur import Sequitur
from k_compression.k_sequitur import k_Sequitur
from proba_sequitur.proba_sequitur import Proba_sequitur
from plot_convention.colors import algo_colors
from surrogate_grammar import Surrogate_grammar

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

def run_benchmark(k, n_rounds, input_sentences, s_g):
    input_sentences = [x.split(' ') 
                   for x in input_sentences]
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
    sequitur_scores = {'precision' : precision,
                       'recall' : recall,
                       'F_score' : F_score,
                       'avg_depth' : avg_depth}
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
    k_sequitur_scores = {'precision' : precision,
                       'recall' : recall,
                       'F_score' : F_score,
                       'avg_depth' : avg_depth}
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
    proba_sequitur_scores = {'precision' : precision,
                            'recall' : recall,
                            'F_score' : F_score,
                            'avg_depth' : avg_depth}
    print 'Proba_sequitur (k = %d, max_rules = %d):\n\tprecision = %f, recall = %f, F_score = %f, avg depth = %f' \
            % (k, max_rules, precision, recall, F_score, avg_depth)      
    #
    #    Test proba-sequitur randomized
    #
    proba_sequitur_rand_hashcodes = []
    max_rules = n_rounds * k
    for i in xrange(len(input_sentences)):
        proba_seq = Proba_sequitur(copy.deepcopy(input_sentences),
                                   copy.deepcopy(input_sentences),
                                   k = k,
                                   max_rules = max_rules,
                                   random = True,
                                   init_T = 0.05,
                                   T_decay = 0.05)
        proba_seq.run()
        proba_sequitur_rand_hashcodes.extend(proba_seq.hashcode_to_rule.keys())
    #
    s_g_hashcodes = filter_hashcodes(s_g_hashcodes)
    proba_sequitur_rand_hashcodes = filter_hashcodes(proba_sequitur_rand_hashcodes)
    #
    common_hashcodes = s_g_hashcodes.intersection(proba_sequitur_rand_hashcodes)
    all_hashcodes = s_g_hashcodes.union(proba_sequitur_rand_hashcodes)
    #
    precision, recall, F_score = get_scores(s_g_hashcodes,
                                            proba_sequitur_rand_hashcodes)
    avg_depth = np.mean([len(x) for x in common_hashcodes])
    proba_sequitur_rand_scores = {'precision' : precision,
                                   'recall' : recall,
                                   'F_score' : F_score,
                                   'avg_depth' : avg_depth}
    print 'Proba_sequitur_rand (k = %d, max_rules = %d):\n\tprecision = %f, recall = %f, F_score = %f, avg depth = %f' \
            % (k, max_rules, precision, recall, F_score, avg_depth)
    return sequitur_scores, k_sequitur_scores, \
            proba_sequitur_scores, proba_sequitur_rand_scores

def run_complete_benchmark(k_set,
                           n_rounds,
                           input_sequences,
                           filename,
                           surrogate_grammar):
    all_sequitur_scores = []
    all_k_sequitur_scores = []
    all_proba_sequitur_scores = []
    all_proba_sequitur_rand_scores = []
    #
    #    Run benchmark over input sequences for all values of k
    #        in k_set
    #
    for k in k_set:
        sequitur_scores, k_sequitur_scores, \
            proba_sequitur_scores, proba_sequitur_rand_scores \
            = run_benchmark(k, n_rounds, input_sequences, surrogate_grammar)
        all_sequitur_scores.append(sequitur_scores)
        all_k_sequitur_scores.append(k_sequitur_scores)
        all_proba_sequitur_scores.append(proba_sequitur_scores)
        all_proba_sequitur_rand_scores.append(proba_sequitur_rand_scores)
        print '\n'
    #
    sequitur_precision = [x['precision'] for x in all_sequitur_scores]
    sequitur_recall = [x['recall'] for x in all_sequitur_scores]
    sequitur_f_score = [x['F_score'] for x in all_sequitur_scores]
    sequitur_avg_depth = [x['avg_depth'] for x in all_sequitur_scores]
    #
    k_sequitur_precision = [x['precision'] for x in all_k_sequitur_scores]
    k_sequitur_recall = [x['recall'] for x in all_k_sequitur_scores]
    k_sequitur_f_score = [x['F_score'] for x in all_k_sequitur_scores]
    k_sequitur_avg_depth = [x['avg_depth'] for x in all_k_sequitur_scores]
    #
    proba_sequitur_precision = [x['precision'] for x in all_proba_sequitur_scores]
    proba_sequitur_recall = [x['recall'] for x in all_proba_sequitur_scores]
    proba_sequitur_f_score = [x['F_score'] for x in all_proba_sequitur_scores]
    proba_sequitur_avg_depth = [x['avg_depth'] for x in all_proba_sequitur_scores]
    #
    proba_sequitur_rand_precision = [x['precision'] for x in all_proba_sequitur_rand_scores]
    proba_sequitur_rand_recall = [x['recall'] for x in all_proba_sequitur_rand_scores]
    proba_sequitur_rand_f_score = [x['F_score'] for x in all_proba_sequitur_rand_scores]
    proba_sequitur_rand_avg_depth = [x['avg_depth'] for x in all_proba_sequitur_rand_scores]
    #
    #    Plotting precision
    #
    plt.subplot(221)
    plt.title('Precision')
    plt.plot(k_set, sequitur_precision, 
             lw = 2, 
             marker = 'o', 
             c = algo_colors['sequitur'])
    plt.plot(k_set, k_sequitur_precision, 
             lw = 2, 
             marker = 'x',
             c = algo_colors['k_sequitur'])
    plt.plot(k_set, proba_sequitur_precision, 
             lw = 2, 
             marker = '^',
             c = algo_colors['proba_sequitur'])
    plt.plot(k_set, proba_sequitur_rand_precision, 
             lw = 2, 
             marker = 'v',
             c = algo_colors['proba_sequitur_rand'])
    plt.legend(('Sequitur', 'K sequitur', 'Proba sequitur', 'Proba sequitur rand'),
               'upper right')
    plt.ylabel('Precision ratio')
    plt.ylim((0, 1.0))
    plt.xlabel('K')
    #
    #    Plotting recall
    #
    plt.subplot(222)
    plt.title('Recall')
    plt.plot(k_set, sequitur_recall, 
             lw = 2, 
             marker = 'o', 
             c = algo_colors['sequitur'])
    plt.plot(k_set, k_sequitur_recall, 
             lw = 2, 
             marker = 'x',
             c = algo_colors['k_sequitur'])
    plt.plot(k_set, proba_sequitur_recall, 
             lw = 2, 
             marker = '^',
             c = algo_colors['proba_sequitur'])
    plt.plot(k_set, proba_sequitur_rand_recall, 
             lw = 2, 
             marker = 'v',
             c = algo_colors['proba_sequitur_rand'])
    plt.legend(('Sequitur', 'K sequitur', 'Proba sequitur', 'Proba sequitur rand'),
               'upper right')
    plt.ylabel('Recall ratio')
    plt.ylim((0, 1.0))
    plt.xlabel('K')
    #
    #    Plotting F score
    #
    plt.subplot(223)
    plt.title('F score')
    plt.plot(k_set, sequitur_f_score, 
             lw = 2, 
             marker = 'o', 
             c = algo_colors['sequitur'])
    plt.plot(k_set, k_sequitur_f_score, 
             lw = 2, 
             marker = 'x',
             c = algo_colors['k_sequitur'])
    plt.plot(k_set, proba_sequitur_f_score, 
             lw = 2, 
             marker = '^',
             c = algo_colors['proba_sequitur'])
    plt.plot(k_set, proba_sequitur_rand_f_score, 
             lw = 2, 
             marker = 'v',
             c = algo_colors['proba_sequitur_rand'])
    plt.legend(('Sequitur', 'K sequitur', 'Proba sequitur', 'Proba sequitur rand'),
               'upper right')
    plt.ylabel('F score')
    plt.ylim((0, 1.0))
    plt.xlabel('K')
    #
    #    Plotting avg depth
    #
    plt.subplot(224)
    plt.title('Avg depth')
    plt.plot(k_set, sequitur_avg_depth, 
             lw = 2, 
             marker = 'o', 
             c = algo_colors['sequitur'])
    plt.plot(k_set, k_sequitur_avg_depth, 
             lw = 2, 
             marker = 'x',
             c = algo_colors['k_sequitur'])
    plt.plot(k_set, proba_sequitur_avg_depth, 
             lw = 2, 
             marker = '^',
             c = algo_colors['proba_sequitur'])
    plt.plot(k_set, proba_sequitur_rand_avg_depth, 
             lw = 2, 
             marker = 'v',
             c = algo_colors['proba_sequitur_rand'])
    plt.legend(('Sequitur', 'K sequitur', 'Proba sequitur', 'Proba sequitur rand'),
               'upper right')
    plt.ylabel('Avg rule depth')
    plt.ylim((0, 8))
    plt.xlabel('K')
    #
    #    Saving to file
    #
    fig = plt.gcf()
    fig.set_size_inches((15, 15))
    plt.savefig(filename, dpi = 600)
    plt.close()





