'''
Created on 14 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import time
import multiprocessing as multi
import copy

from proba_sequitur.proba_seq_merger import Proba_seq_merger
from proba_sequitur.proba_sequitur import Proba_sequitur
import load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.no_rep_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.no_rep_oldo_file_contents.values()

MAX_PROCESSES = 6

n_trials = 100

max_represented = 400

achu_indices = range(9)
oldo_indices = range(9, 18)

degree_set = [6, 8, 10]
n_round_set = [4, 5]
#T_set = [0.1, 0.2]
#T_decay_set = [0.1, 0.2]
p_deletion = 0.01
filter_option_set = [('not-filtered', achu_data_set, oldo_data_set),
                     ('filtered', f_achu_data_set, f_oldo_data_set)]

target_depths = range(2, 10)

def compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      n_rounds,
                      random,
                      p_deletion):
    prefix = '%s_%s_%d_%d_%f' % (filter_name,
                                        inference_name,
                                        degree,
                                        n_rounds,
                                        p_deletion)
    print '\tDoing %s' % prefix
    reducer = Proba_seq_merger()
    begin = time.clock()
    for i in xrange(n_trials):
        ps = Proba_sequitur(inference_content,
                            count_content,
                            degree,
                            degree * n_rounds,
                            random,
                            p_deletion = p_deletion)
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
        n_rounds,
        p_deletion): 
    count_content = selected_achu_data_set + selected_oldo_data_set
    random = False
    #
    inference_name = 'achu_oldo'
    inference_content = selected_achu_data_set + selected_oldo_data_set
    compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      degree * n_rounds,
                      random,
                      p_deletion)
    #
    inference_name = 'achu'
    inference_content = selected_achu_data_set
    compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      degree * n_rounds,
                      random,
                      p_deletion)
    #
    inference_name = 'oldo'
    inference_content = selected_oldo_data_set
    compare_achu_oldo(inference_content,
                      count_content,
                      filter_name,
                      inference_name,
                      degree,
                      degree * n_rounds,
                      random,
                      p_deletion)

def run_algo_tuple(input_instruction):
    run_algo(input_instruction[0],
             input_instruction[1],
             input_instruction[2],
             input_instruction[3],
             input_instruction[4],
             input_instruction[5])

instruction_set = [(x[0], x[1], x[2],
                   degree, n_rounds,
                   p_deletion)
                   for x in filter_option_set
                   for degree in degree_set
                   for n_rounds in n_round_set]

p = multi.Pool(processes = MAX_PROCESSES)
p.map(run_algo_tuple, instruction_set)
