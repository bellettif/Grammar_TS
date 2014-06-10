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
import os
import copy
import cPickle as pickle
import multiprocessing as multi

from proba_sequitur.Proba_sequitur import Proba_sequitur

from SCFG.grammar_distance import Grammar_distance

from Grammar_examples.Grammar_examples import palindrom_grammar_1, \
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

all_grammars = {'palindrom_grammar_1' :             palindrom_grammar_1,
               'palindrom_grammar_2' :              palindrom_grammar_2,
               'palindrom_grammar_3' :              palindrom_grammar_3,
               #'repetition_grammar_1' :             repetition_grammar_1,
               #'repetition_grammar_2' :             repetition_grammar_2,
               'embedding_grammar_central_1' :      embedding_grammar_central_1,
               'embedding_grammar_central_2' :      embedding_grammar_central_2,
               'embedding_grammar_left_right_1' :   embedding_grammar_left_right_1,
               'embedding_grammar_left_right_2' :   embedding_grammar_left_right_2,
               'name_grammar_1' :                   name_grammar_1,
               'name_grammar_2' :                   name_grammar_2,
               'action_grammar_1' :                 action_grammar_1,
               'action_grammar_2' :                 action_grammar_2}

repetition_options = 'rep'
loss_options = 'lossless'

degree_set = [6, 12]
max_rules_set = [30, 60]
T_set = [0.1, 0.5, 1.0]
T_decay = 0.1
p_deletion = 0.05

n_big_trials = 10
n_Trials = 50

grammar_set = all_grammars.items()

n_samples = 50
all_target_indices = range(n_samples)

folder_path = 'Seq_on_examples/'

already_computed = os.listdir(folder_path)

already_computed = filter(lambda x : '.pi' in x, already_computed)

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

def generate_samples_and_evaluate_ps(grammar_name,
                                     grammar,
                                     degree_set,
                                     max_rules_set,
                                     T_set):
    instruction_set = [(degree,
                        max_rules,
                        grammar_name,
                        [' '.join(x) for x in grammar.produce_sentences(n_samples)],
                        i,
                        T)
                        for degree in degree_set
                        for max_rules in max_rules_set
                        for i in range(n_big_trials)
                        for T in T_set]
    for instruction in instruction_set:
        evaluate_ps_tuple(instruction)
    """
    p=multi.Pool(processes = 6)
    p.map(evaluate_ps_tuple, instruction_set)
    """
    

def evaluate_ps_tuple(arg_tuple):
    evaluate_ps(arg_tuple[0],
                arg_tuple[1],
                arg_tuple[2],
                arg_tuple[3],
                arg_tuple[4],
                arg_tuple[5])

def evaluate_ps(degree,
                max_rules,
                grammar_name,
                grammar_samples,
                sample_id,
                T):
    #
    #    All data sets
    #
    merged_rules = {}
    merged_rhs = {}
    #
    #    Inference on achu and oldo
    #
    merged_relative_counts = {}
    merged_counts = {}
    merged_levels = {}
    #
    #
    #
    target_file_name = ('Results_merged_%s_%d_%d_%d_%f.pi' % 
                     (grammar_name, sample_id, degree, max_rules, T))
    if target_file_name in already_computed:
        print "\tGrammar: %s, sample_id: %d" % (grammar_name, sample_id)
        print "\t\t degree = %d, max_rules = %d, T = %f" % \
                (degree, max_rules, T)
        print 'ALREADY COMPUTED'
        return
    print "\tGrammar: %s, sample_id: %d" % (grammar_name, sample_id)
    print "\t\tDoing degree = %d, max_rules = %d, T = %f" % \
                (degree, max_rules, T)
    for i_trial in range(n_Trials):
        print "\t\t\tTrial = %d" % i_trial
        data_set = copy.deepcopy(grammar_samples)
        #
        #    Proceeding with inference on both data sets
        #
        ps = Proba_sequitur(build_samples = data_set,
                            count_samples = data_set,
                            repetitions = True,
                            keep_data = True,
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
                   all_target_indices,
                   merged_relative_counts,
                   merged_counts,
                   merged_levels,
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
    total_count_dict = {}
    for hashcode in all_hashcodes:
        total_counts.append([hashcode,
                             sum(merged_relative_counts[hashcode].values()) *
                             merged_levels[hashcode]])
        total_count_dict[hashcode] = sum(merged_relative_counts[hashcode].values())
    total_counts.sort(key = (lambda x : -x[1]))
    sorted_hashcodes = [x[0] for x in total_counts]
    sorted_hashcodes = sorted_hashcodes[:max_rules * 4]
    #
    #    Preparing plots
    #               
    boxes = []
    box_names = []
    #
    for hashcode in sorted_hashcodes:
        rule_name = merged_rules[hashcode]
        left, right = merged_rhs[hashcode]
        if left in merged_rules:
            left_converted = merged_rules[left]
        else:
            left_converted = left
        if right in merged_rules:
            right_converted = merged_rules[right]
        else:
            right_converted = right
        rhs = left_converted + '-' + right_converted
        boxes.append([])
        box_names.append('')
        #
        boxes.append([merged_relative_counts[hashcode][j] for j in all_target_indices])
        box_names.append('%d %.2f %s -> %s' 
                         % (merged_levels[hashcode],
                            total_count_dict[hashcode], 
                            rule_name,
                            rhs))
        #
        boxes.append([])
        box_names.append('')
    print boxes
    #
    #    Plotting boxes
    #
    bp = plt.boxplot(boxes,
                     notch=0,
                     sym='+',
                     vert=1,
                     whis=1.5,
                     patch_artist = True)
    plt.setp(bp['boxes'], color = 'b', facecolor = 'b', alpha = 0.25)
    plt.setp(bp['whiskers'], color='b')
    plt.setp(bp['fliers'], color='b', marker='+')
    #
    plt.xticks(range(1, len(box_names) + 1), 
               box_names,
               rotation = 'vertical', fontsize = 3)
    plt.yscale('log')
    plt.ylabel('Relative counts (log)')
    fig = plt.gcf()
    fig.set_size_inches((40, 8))
    plt.savefig(folder_path + ('Freqs_log_merged_%s_%d_%d_%d_%f.png' % 
                (grammar_name, sample_id, degree, max_rules, T)), dpi = 600)
    plt.close()
    #
    #
    #
    pickle.dump({'merged_rules' : merged_rules,
                 'merged_rhs' : merged_rhs,
                 'merged_relative_counts' : merged_relative_counts,
                 'merged_counts' : merged_counts,
                 'merged_levels' : merged_levels},
                 open(folder_path + ('Results_merged_%s_%d_%d_%d_%f.pi' % 
                     (grammar_name, sample_id, degree, max_rules, T)),
                     'wb'))
    print 'Done\n'
    
    
    
#
#    Run
#
for grammar_name, grammar in all_grammars.iteritems():
    print 'EVALUATING GRAMMAR ' + grammar_name
    generate_samples_and_evaluate_ps(grammar_name,
                                     grammar,
                                     degree_set,
                                     max_rules_set,
                                     T_set)
    print 'DONE\n\n\n'
    
