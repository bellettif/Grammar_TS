'''
Created on 21 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

import load_data

import plot_convention.colors as colors

from load_data import achu_file_names, \
                        oldo_file_names

from k_compression.k_sequitur import k_Sequitur

repetition_data_set = [(x, load_data.achu_file_contents[x])
                       for x in achu_file_names] + \
                        [(x, load_data.oldo_file_contents[x])
                         for x in oldo_file_names]
no_repetition_data_set = [(x,load_data.no_rep_achu_file_contents[x])
                          for x in achu_file_names] + \
                          [(x, load_data.no_g_no_rep_oldo_file_contents[x])
                           for x in oldo_file_names]
                          
def run_seq(sequence, k):
    sequi = k_Sequitur(sequence.split(' '), k)
    sequi.run()
    return float(len(sequi.compressed_sequence)), \
            float(len(sequi.input_sequence)), \
            len(sequi.grammar), \
            sequi.hashed_freqs, \
            sequi.hashed_ref_counts
            
def merge_results(result_dict):
    all_rules = {}
    total_freqs = {}
    for file_name, result in result_dict.iteritems():
        rule_freq_dict = result[3]
        for rule_name, freq in rule_freq_dict.iteritems():
            if rule_name not in total_freqs:
                total_freqs[rule_name] = 0
                all_rules[rule_name] = {}
            all_rules[rule_name][file_name] = freq
            total_freqs[rule_name] += freq / float(len(result_dict.keys()))
    return all_rules, total_freqs
        
for k in [2, 4, 6, 8, 10, 12]:   
    repetition_results = dict([(x[0], run_seq(x[1], k)) 
                               for x in repetition_data_set])
    no_repetition_results = dict([(x[0], run_seq(x[1], k))
                                  for x in no_repetition_data_set])
    
    merged_rep_results, score_rep = merge_results(repetition_results)
    merged_no_rep_results, score_no_rep = merge_results(no_repetition_results)
    
    score_rep_items = score_rep.items()
    score_rep_items.sort(key = (lambda x : -x[1]))
    
    score_no_rep_items = score_no_rep.items()
    score_no_rep_items.sort(key = (lambda x : -x[1])) 
    
    plt.subplot(121)
    plt.title('Average frequency of rules with repetitions of a, k = %d' % k)
    plt.plot([x[1] for x in score_rep_items],
             lw = 2.0)
    current_max = max([x[1] for x in score_rep_items])
    current_min = min([x[1] for x in score_rep_items])
    #plt.vlines(25, current_min, current_max, 
    #           linestyles = '--', colors = 'orange', lw = 2)
    plt.yscale('log')
    plt.ylabel('Average frequency use of rule (log scale)')
    plt.xlabel('Rules')
    plt.subplot(122)
    plt.title('Average frequency of rules without repetitions of a, k = %d' % k)
    plt.plot([x[1] for x in score_no_rep_items],
             lw = 2.0)
    current_max = max([x[1] for x in score_no_rep_items])
    current_min = min([x[1] for x in score_no_rep_items])
    #plt.vlines(25, current_min, current_max, 
    #           linestyles = '--', colors = 'orange', lw = 2)
    plt.yscale('log')
    plt.ylabel('Average frequency use of rule (log scale)')
    plt.xlabel('Rules')
    fig = plt.gcf()
    fig.set_size_inches((16, 8))
    plt.savefig('Avg_freqs_%d.png' % k, dpi = 600)
    plt.close()
    
    n_represented = 25
    for i, x in enumerate(score_rep_items[:n_represented]):
        rule_hashcode = x[0]
        current_dict = merged_rep_results[x[0]]
        for file_name in (achu_file_names + oldo_file_names):
            if file_name in current_dict:
                freq = current_dict[file_name]
            else:
                freq = 0
            if file_name in achu_file_names:
                plt.scatter(i+1, freq, c = colors.all_colors[file_name],
                         alpha = 0.2, marker = 'o', s = 80)
            else:
                plt.scatter(i+1, freq, c = colors.all_colors[file_name],
                         alpha = 0.2, marker = 'p', s = 80)
    plt.title('Rule freq. across sequence, k = %d' % k)
    plt.ylabel('Rule freq.')
    plt.xlim((0, n_represented + 1))
    plt.ylim((0, 0.3))
    plt.xticks(range(1, n_represented + 1), 
               [x[0] for x in score_rep_items[:n_represented]], 
               fontsize = 4, rotation = 'vertical')
    plt.legend((achu_file_names + oldo_file_names), 'upper center', ncol = 3, fontsize = 8)
    plt.savefig('Freqs_rep_%d.png' % k, dpi = 600)
    plt.close()
    
    n_represented = 25
    for i, x in enumerate(score_no_rep_items[:n_represented]):
        rule_hashcode = x[0]
        current_dict = merged_no_rep_results[x[0]]
        for file_name in (achu_file_names + oldo_file_names):
            if file_name in current_dict:
                freq = current_dict[file_name]
            else:
                freq = 0
            if file_name in achu_file_names:
                plt.scatter(i+1, freq, c = colors.all_colors[file_name],
                         alpha = 0.2, marker = 'o', s = 80)
            else:
                plt.scatter(i+1, freq, c = colors.all_colors[file_name],
                         alpha = 0.2, marker = 'p', s = 80)
    plt.title('Rule freq. across sequence (no rep. of a), k = %d' % k)
    plt.ylabel('Rule freq.')
    plt.xlim((0, n_represented + 1))
    plt.ylim((0, 0.3))
    plt.xticks(range(1, n_represented + 1), 
               [x[0] for x in score_no_rep_items[:n_represented]], 
               fontsize = 4, rotation = 'vertical')
    plt.legend((achu_file_names + oldo_file_names), 'upper center', ncol = 3, fontsize = 8)
    plt.savefig('Freqs_no_rep_%d.png' % k, dpi = 600)
    plt.close()

print 'Done'

