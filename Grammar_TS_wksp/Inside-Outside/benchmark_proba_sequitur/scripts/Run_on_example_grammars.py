'''
Created on 10 juin 2014

@author: francois
'''

'''
Created on 4 juin 2014

@author: francois
'''

from matplotlib import pyplot as plt

import numpy as np
import time

import cPickle as pickle


'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import copy
import cPickle as pickle
import os

from proba_sequitur.Proba_sequitur import Proba_sequitur

from proba_sequitur import load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.filtered_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.filtered_oldo_file_contents.values()

from assess_proba_seq_for_counts import compute_counts

import multiprocessing as multi

repetition_options = 'rep'
loss_options = 'lossless'

degree_set = [6, 12]
max_rules_set = [30, 60]
T_set = [0.1, 0.5, 1.0]
T_decay = 0.1
p_deletion = 0.05

n_Trials = 50

filter_option_set = [('sto_not_filtered', achu_data_set, oldo_data_set),
                     ('sto_filtered', f_achu_data_set, f_oldo_data_set)]

folder_path = 'Merged_stochastic_results/'
pickled_files = os.listdir(folder_path)

print ''
print pickled_files
print ''

achu_set = range(9)
oldo_set = range(9, 18)


def merge_data(proba_seq,
               target_indices,
               merged_relative_counts,
               merged_counts,
               merged_levels,
               merged_rules,
               merged_rhs):
    hashed_counts = proba_seq.hashed_counts
    hashed_levels = proba_seq.hashed_levels
    hashed_relative_counts = proba_seq.hashed_relative_counts
    hashcodes = proba_seq.hashcode_to_rule
    hashed_rules = proba_seq.hashed_rules
    for hashcode in hashcodes:
        if hashcode not in merged_relative_counts:
            merged_relative_counts[hashcode] = {}
            merged_counts[hashcode] = {}
            merged_levels[hashcode] = hashed_levels[hashcode]
        if hashcode not in merged_rules:
            new_rule_name = 'r%d' % (len(merged_rules) + 1)
            merged_rules[hashcode] = new_rule_name
            merged_rhs[hashcode] = hashed_rules[hashcode]
        for i in target_indices:
            if i not in merged_relative_counts[hashcode]:
                merged_relative_counts[hashcode][i] = 0
                merged_counts[hashcode][i] = 0
            if i in hashed_relative_counts[hashcode]:
                merged_relative_counts[hashcode][i] += \
                    hashed_relative_counts[hashcode][i]
                merged_counts[hashcode][i] += \
                    hashed_counts[hashcode][i]  
    
def compute_results_tuple(input_tuple):
    #print input_tuple
    compute_results(input_tuple[0],
                    input_tuple[1],
                    input_tuple[2],
                    input_tuple[3],
                    input_tuple[4],
                    input_tuple[5])   

def compute_results(degree,
                    max_rules,
                    filter_option,
                    selected_achu_data_set,
                    selected_oldo_data_set,
                    T):
    #
    #    All data sets
    #
    merged_rules = {}
    merged_rhs = {}
    #
    #    Inference on achu and oldo
    #
    merged_achu_oldo_relative_counts = {}
    merged_achu_oldo_counts = {}
    merged_achu_oldo_levels = {}
    #
    #    Inference on achu
    #    
    merged_achu_relative_counts = {}
    merged_achu_counts = {}
    merged_achu_levels = {}
    #
    #    Inference on oldo
    #
    merged_oldo_relative_counts = {}
    merged_oldo_counts = {}
    merged_oldo_levels = {}
    #
    #
    keep_data = 'keep_data'
    keep_data_bool = (keep_data == 'keep_data')
    print "\tDoing degree = %d, max_rules = %d, filter_option = %s, T = %f" % \
                (degree, max_rules, filter_option, T)
    target_file_name = ('Results_merged_%s_%d_%s_%d_%f.pi' % 
                        (filter_option, degree, keep_data, max_rules, T))
    if target_file_name in pickled_files:
        print 'Reading from pickled results'
        pickled_data = pickle.load(open(folder_path + target_file_name, 'rb'))
        merged_rules = pickled_data['merged_rules']
        merged_rhs = pickled_data['merged_rhs']
        merged_achu_oldo_relative_counts = pickled_data['merged_achu_oldo_relative_counts']
        merged_achu_oldo_counts = pickled_data['merged_achu_oldo_counts']
        merged_achu_oldo_levels = pickled_data['merged_achu_oldo_levels']
        merged_achu_relative_counts = pickled_data['merged_achu_relative_counts']
        merged_achu_counts = pickled_data['merged_achu_counts']
        merged_achu_levels = pickled_data['merged_achu_levels']
        merged_oldo_relative_counts = pickled_data['merged_oldo_relative_counts']
        merged_oldo_counts = pickled_data['merged_oldo_counts']
        merged_oldo_levels = pickled_data['merged_oldo_levels']
    else:
        for i_trial in range(n_Trials):
            print "\t\tTrial = %d" % i_trial
            
            both_data_sets = copy.deepcopy(selected_achu_data_set) + \
                             copy.deepcopy(selected_oldo_data_set)
            #
            #    Proceeding with inference on both data sets
            #
            ps = Proba_sequitur(build_samples = both_data_sets,
                                count_samples = both_data_sets,
                                repetitions = True,
                                keep_data = keep_data_bool,
                                degree = degree,
                                max_rules = max_rules,
                                stochastic = True,
                                init_T = T,
                                T_decay = T_decay,
                                p_deletion = p_deletion)
            ps.infer_grammar()
            #
            #    Merging data
            #
            merge_data(ps,
                       achu_set + oldo_set,
                       merged_achu_oldo_relative_counts,
                       merged_achu_oldo_counts,
                       merged_achu_oldo_levels,
                       merged_rules,
                       merged_rhs)
            #
            #    Proceeding with inference on oldo data set
            #
            ps = Proba_sequitur(build_samples = selected_oldo_data_set,
                                count_samples = both_data_sets,
                                repetitions = True,
                                keep_data = keep_data_bool,
                                degree = degree,
                                max_rules = max_rules,
                                stochastic = True,
                                init_T = T,
                                T_decay = T_decay,
                                p_deletion = p_deletion)
            ps.infer_grammar()
            #
            #    Merging data
            #
            merge_data(ps, 
                       achu_set + oldo_set,
                       merged_oldo_relative_counts,
                       merged_oldo_counts,
                       merged_oldo_levels,
                       merged_rules,
                       merged_rhs)
            #
            #    Proceeding with inference on achu data set
            #
            ps = Proba_sequitur(build_samples = selected_achu_data_set,
                                count_samples = both_data_sets,
                                repetitions = True,
                                keep_data = keep_data_bool,
                                degree = degree,
                                max_rules = max_rules,
                                stochastic = True,
                                init_T = T,
                                T_decay = T_decay,
                                p_deletion = p_deletion)
            ps.infer_grammar()
            #
            #    Merging data
            #
            merge_data(ps,
                       achu_set + oldo_set,
                       merged_achu_relative_counts,
                       merged_achu_counts,
                       merged_achu_levels,
                       merged_rules,
                       merged_rhs)
    #
    #    Plotting merged data
    #
    all_hashcodes = merged_rules.keys()
    #
    #    Filling with zeros and sorting rules
    #
    total_counts = []
    total_counts_achu = {}
    total_counts_oldo = {}
    total_counts_achu_oldo = {}
    for hashcode in all_hashcodes:
        if hashcode not in merged_achu_oldo_counts:
            merged_achu_oldo_levels[hashcode] = 0
            merged_achu_oldo_counts[hashcode] = \
                dict(zip(achu_set + oldo_set,
                         np.zeros(len(achu_set + oldo_set))))
            merged_achu_oldo_relative_counts[hashcode] = \
                dict(zip(achu_set + oldo_set,
                         np.zeros(len(achu_set + oldo_set))))
        if hashcode not in merged_oldo_counts:
            merged_oldo_levels[hashcode] = 0
            merged_oldo_counts[hashcode] = \
                dict(zip(achu_set + oldo_set,
                         np.zeros(len(achu_set + oldo_set))))
            merged_oldo_relative_counts[hashcode] = \
                dict(zip(achu_set + oldo_set,
                         np.zeros(len(achu_set + oldo_set))))
        if hashcode not in merged_achu_counts:
            merged_achu_levels[hashcode] = 0
            merged_achu_counts[hashcode] = \
                dict(zip(achu_set + oldo_set,
                         np.zeros(len(achu_set + oldo_set))))
            merged_achu_relative_counts[hashcode] = \
                dict(zip(achu_set + oldo_set,
                         np.zeros(len(achu_set + oldo_set))))
        total_counts.append([hashcode,
                             sum(merged_oldo_relative_counts[hashcode].values()) *
                             merged_oldo_levels[hashcode]
                             + 
                             sum(merged_achu_relative_counts[hashcode].values()) *
                             merged_achu_levels[hashcode]
                             +
                             sum(merged_achu_oldo_relative_counts[hashcode].values()) *
                             merged_achu_oldo_levels[hashcode]])
        total_counts_achu[hashcode] = sum(merged_achu_relative_counts[hashcode].values())
        total_counts_oldo[hashcode] = sum(merged_oldo_relative_counts[hashcode].values())
        total_counts_achu_oldo[hashcode] = sum(merged_achu_oldo_relative_counts[hashcode].values())
    total_counts.sort(key = (lambda x : -x[1]))
    sorted_hashcodes = [x[0] for x in total_counts]
    sorted_hashcodes = sorted_hashcodes[:max_rules * 4]
    #
    #    Preparing plots
    #               
    achu_boxes = []
    oldo_boxes = []
    box_names = []
    #
    names = []
    #
    achu_medians_inferred_achu = []
    oldo_medians_inferred_achu = []
    levels_inferred_achu = []
    #
    achu_medians_inferred_oldo = []
    oldo_medians_inferred_oldo = []
    levels_inferred_oldo = []
    #
    achu_medians_inferred_both = []
    oldo_medians_inferred_both = []
    levels_inferred_both = []
    #
    for hashcode in sorted_hashcodes:
        rule_name = merged_rules[hashcode]
        left, right = merged_rhs[hashcode]
        names.append(rule_name)
        if left in merged_rules:
            left_converted = merged_rules[left]
        else:
            left_converted = left
        if right in merged_rules:
            right_converted = merged_rules[right]
        else:
            right_converted = right
        rhs = left_converted + '-' + right_converted
        achu_boxes.append([])
        oldo_boxes.append([])
        box_names.append('')
        #
        #
        #
        achu_boxes.append([merged_achu_relative_counts[hashcode][j] for j in achu_set])
        oldo_boxes.append([merged_achu_relative_counts[hashcode][j] for j in oldo_set])
        #
        achu_medians_inferred_achu.append(
                      np.median([merged_achu_relative_counts[hashcode][j] for j in achu_set]))
        oldo_medians_inferred_achu.append(
                      np.median([merged_achu_relative_counts[hashcode][j] for j in oldo_set]))
        levels_inferred_achu.append(merged_achu_levels[hashcode])
        #
        box_names.append('achu %d %.2f %s -> %s' 
                         % (merged_achu_levels[hashcode],
                            total_counts_achu[hashcode], 
                            rule_name,
                            rhs))
        #
        #
        #
        achu_boxes.append([merged_achu_oldo_relative_counts[hashcode][j] for j in achu_set])
        oldo_boxes.append([merged_achu_oldo_relative_counts[hashcode][j] for j in oldo_set])
        #
        achu_medians_inferred_both.append(
                      np.median([merged_achu_oldo_relative_counts[hashcode][j] for j in achu_set]))
        oldo_medians_inferred_both.append(
                      np.median([merged_achu_oldo_relative_counts[hashcode][j] for j in oldo_set]))
        levels_inferred_both.append(merged_achu_oldo_levels[hashcode])
        #
        box_names.append('both %d %.2f %s -> %s' 
                         % (merged_achu_oldo_levels[hashcode],
                            total_counts_achu_oldo[hashcode],
                            rule_name,
                            rhs))
        #
        #
        #
        achu_boxes.append([merged_oldo_relative_counts[hashcode][j] for j in achu_set])
        oldo_boxes.append([merged_oldo_relative_counts[hashcode][j] for j in oldo_set])
        #
        achu_medians_inferred_oldo.append(
                      np.median([merged_oldo_relative_counts[hashcode][j] for j in achu_set]))
        oldo_medians_inferred_oldo.append(
                      np.median([merged_oldo_relative_counts[hashcode][j] for j in oldo_set]))
        levels_inferred_oldo.append(merged_oldo_levels[hashcode])
        #
        box_names.append('oldo %d %.2f %s -> %s' 
                         % (merged_oldo_levels[hashcode],
                            total_counts_oldo[hashcode],
                            rule_name,
                            rhs))
        #
        achu_boxes.append([])
        box_names.append('')
        oldo_boxes.append([])
    #
    #    Plotting achu
    #
    bp = plt.boxplot(achu_boxes,
                     notch=0,
                     sym='+',
                     vert=1,
                     whis=1.5,
                     patch_artist = True)
    plt.setp(bp['boxes'], color = 'r', facecolor = 'r', alpha = 0.25)
    plt.setp(bp['whiskers'], color='r')
    plt.setp(bp['fliers'], color='r', marker='+')
    #
    #    Plotting oldo
    #
    bp = plt.boxplot(oldo_boxes,
                     notch=0, 
                     sym='+',
                     vert=1, 
                     whis=1.5, 
                     patch_artist = True)
    plt.setp(bp['boxes'], color='b', facecolor = 'b', alpha = 0.25)
    plt.setp(bp['whiskers'], color='b')
    plt.setp(bp['fliers'], color='b', marker='+')
    plt.xticks(range(1, len(box_names) + 1), 
               box_names,
               rotation = 'vertical', fontsize = 3)
    plt.yscale('log')
    plt.ylabel('Relative counts (log)')
    fig = plt.gcf()
    fig.set_size_inches((40, 8))
    plt.savefig(folder_path + ('Freqs_log_merged_%s_%d_%s_%d_%f.png' % 
                (filter_option, degree, keep_data, max_rules, T)), dpi = 600)
    plt.close()
    #
    #
    #
    print np.asarray(levels_inferred_achu)
    plt.scatter(achu_medians_inferred_achu, 
                 oldo_medians_inferred_achu,
                 c = np.asarray(levels_inferred_achu),
                 marker = 'x',
                 alpha = 0.25,
                 cmap=mpl.cm.gray
                 )
    plt.title('Relative counts inferred on achu')
    plt.xlabel('Relative counts on achu (log)')
    plt.ylabel('Relative counts on oldo (log)')
    """
    for i, rule_name in enumerate(names):
        plt.text(x = achu_medians_inferred_achu[i], 
                 y = oldo_medians_inferred_achu[i],
                 s = rule_name, fontsize = 4)
    """
    plt.savefig(folder_path + ('Inferred_achu_%s_%d_%s_%d_%f.png' % 
                (filter_option, degree, keep_data, max_rules, T)), dpi = 300)
    plt.close()
    #
    #
    #
    print np.asarray(levels_inferred_oldo)
    plt.scatter(achu_medians_inferred_oldo, 
                 oldo_medians_inferred_oldo,
                 c = np.asarray(levels_inferred_oldo),
                 marker = 'x',
                 alpha = 0.25,
                 cmap=mpl.cm.gray
                 )
    plt.title('Relative counts inferred on oldo')
    plt.xlabel('Relative counts on achu (log)')
    plt.ylabel('Relative counts on oldo (log)')
    """
    for i, rule_name in enumerate(names):
        plt.text(x = achu_medians_inferred_oldo[i], 
                 y = oldo_medians_inferred_oldo[i],
                 s = rule_name, fontsize = 4)
    """
    plt.savefig(folder_path + ('Inferred_oldo_%s_%d_%s_%d_%f.png' % 
                (filter_option, degree, keep_data, max_rules, T)), dpi = 300)
    plt.close()
    #
    #
    #
    print np.asarray(levels_inferred_both)
    plt.scatter(achu_medians_inferred_both, 
                 oldo_medians_inferred_both,
                 c = np.asarray(levels_inferred_both),
                 alpha = 0.25,
                 marker = 'x',
                 cmap=mpl.cm.gray
                 )
    plt.title('Relative counts inferred on both')
    plt.xlabel('Relative counts on achu (log)')
    plt.ylabel('Relative counts on oldo (log)')
    """
    for i, rule_name in enumerate(names):
        plt.text(x = achu_medians_inferred_both[i], 
                 y = oldo_medians_inferred_both[i],
                 s = rule_name, fontsize = 4)
    """
    plt.savefig(folder_path + ('Inferred_both_%s_%d_%s_%d_%f.png' % 
                (filter_option, degree, keep_data, max_rules, T)), dpi = 300)
    plt.close()
    #
    #
    #
    pickle.dump({'merged_rules' : merged_rules,
                 'merged_rhs' : merged_rhs,
                 'merged_achu_oldo_relative_counts' : merged_achu_oldo_relative_counts,
                 'merged_achu_oldo_counts' : merged_achu_oldo_counts,
                 'merged_achu_oldo_levels' : merged_achu_oldo_levels,
                 'merged_achu_relative_counts' : merged_achu_relative_counts,
                 'merged_achu_counts' : merged_achu_counts,
                 'merged_achu_levels' : merged_achu_levels,
                 'merged_oldo_relative_counts' : merged_oldo_relative_counts,
                 'merged_oldo_counts' : merged_oldo_counts,
                 'merged_oldo_levels' : merged_oldo_levels},
                open(folder_path + ('Results_merged_%s_%d_%s_%d_%f.pi' % 
                     (filter_option, degree, keep_data, max_rules, T)),
                     'wb'))
    print 'Done\n'


#
#    Run in parallel
#
instruction_set = [(degree, max_rules, x[0], x[1], x[2], T)
                   for degree in degree_set
                   for max_rules in max_rules_set
                   for x in filter_option_set
                   for T in T_set]

p=multi.Pool(processes = 6)
p.map(compute_results_tuple, instruction_set)



"""
#
#    Run
#
for degree in degree_set:
    for max_rules in max_rules_set:
        for filter_option, selected_achu_data_set, selected_oldo_data_set in filter_option_set:
            for T in T_set:
                compute_results(degree,
                                max_rules,
                                filter_option,
                                selected_achu_data_set,
                                selected_oldo_data_set,
                                T)
"""

from SCFG.grammar_distance import Grammar_distance

from Grammar_examples import palindrom_grammar_1, \
                             palindrom_grammar_2, \
                             palindrom_grammar_3, \
                             repetition_grammar_1, \
                             repetition_grammar_2, \
                             embedding_grammar_central_1, \
                             embedding_grammar_central_2, \
                             embedding_grammar_left_right_1, \
                             embedding_grammar_left_right_2, \
                             name_grammar_1, \
                             name_grammar_2, \
                             action_grammar_1, \
                             action_grammar_2

all_grammars = {'palindrom_grammar_1' : palindrom_grammar_1,
               'palindrom_grammar_2' : palindrom_grammar_2,
               'palindrom_grammar_3' : palindrom_grammar_3,
               'repetition_grammar_1' : repetition_grammar_1,
               'repetition_grammar_2' : repetition_grammar_2,
               'embedding_grammar_central_1' : embedding_grammar_central_1,
               'embedding_grammar_central_2' : embedding_grammar_central_2,
               'embedding_grammar_left_right_1' : embedding_grammar_left_right_1,
               'embedding_grammar_left_right_2' : embedding_grammar_left_right_2,
               'name_grammar_1' : name_grammar_1,
               'name_grammar_2' : name_grammar_2,
               'action_grammar_1' : action_grammar_1,
               'action_grammar_2' : action_grammar_2}

for grammar_name, grammar in all_grammars.iteritems():
    print 'Drawing %s' % grammar_name
    grammar.draw_grammar('Grammar_graphs/%s.png' % grammar_name)
    print 'Done\n'
    