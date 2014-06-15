'''
Created on 14 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import time
import multiprocessing as multi
import copy

from proba_sequitur.Proba_seq_merger import Proba_seq_merger
from proba_sequitur.Proba_sequitur import Proba_sequitur
import proba_sequitur.load_data as load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.filtered_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.filtered_oldo_file_contents.values()

n_trials = 1000

max_represented = 400

achu_indices = range(9)
oldo_indices = range(9, 18)

degree_set = [6, 8, 12]
max_rules_set = [30, 60, 80]
T_set = [0.0, 0.1, 0.5, 1.0]
T_decay_set = [0.0, 0.1, 0.2]
p_deletion = 0.05
filter_option_set = [('not_filtered', achu_data_set, oldo_data_set),
                     ('filtered', f_achu_data_set, f_oldo_data_set)]

target_depths = range(2, 10)

def compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      max_rules,
                      random,
                      init_T,
                      T_decay,
                      p_deletion):
    prefix = '%s_%s_%d_%d_%f_%f_%f' % (filter_name,
                                        inference_name,
                                        degree,
                                        max_rules,
                                        init_T,
                                        T_decay,
                                        p_deletion)
    print '\tDoing %s' % prefix
    reducer = Proba_seq_merger()
    begin = time.clock()
    for i in xrange(n_trials):
        ps = Proba_sequitur(inference_content,
                            count_content,
                            degree,
                            max_rules,
                            random,
                            init_T,
                            T_decay,
                            p_deletion)
        ps.run()
        reducer.merge_with(ps)
    print '\tProba Sequitur done'
    print '\tComputation took %f second' % (time.clock() - begin)
    reducer.compare_data_sets(achu_indices, 
                              oldo_indices, 
                              prefix, 
                              max_represented,
                              target_depths)

def run_algo(filter_name,
        selected_achu_data_set,
        selected_oldo_data_set,
        degree,
        max_rules,
        init_T,
        T_decay,
        p_deletion): 
    count_content = selected_achu_data_set + selected_oldo_data_set
    random = True
    #
    inference_name = 'achu_oldo'
    inference_content = selected_achu_data_set + selected_oldo_data_set
    compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      max_rules,
                      random,
                      init_T,
                      T_decay,
                      p_deletion)
    #
    inference_name = 'achu'
    inference_content = selected_achu_data_set
    compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      max_rules,
                      random,
                      init_T,
                      T_decay,
                      p_deletion)
    #
    inference_name = 'oldo'
    inference_content = selected_oldo_data_set
    compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      max_rules,
                      random,
                      init_T,
                      T_decay,
                      p_deletion)

def run_algo_tuple(input_instruction):
    run_algo(input_instruction[0],
             input_instruction[1],
             input_instruction[2],
             input_instruction[3],
             input_instruction[4],
             input_instruction[5],
             input_instruction[6],
             input_instruction[7])

instruction_set = [(x[0], x[1], x[2],
                   degree, max_rules,
                   T, T_decay,
                   p_deletion)
                   for x in filter_option_set
                   for degree in degree_set
                   for max_rules in max_rules_set
                   for T in T_set
                   for T_decay in T_decay_set]

p = multi.Pool(processes = 8)
p.map(run_algo_tuple, instruction_set)