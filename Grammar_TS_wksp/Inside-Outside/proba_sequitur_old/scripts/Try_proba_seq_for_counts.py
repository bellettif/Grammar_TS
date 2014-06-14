'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from Proba_sequitur_for_counts import Proba_sequitur

import load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.filtered_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.filtered_oldo_file_contents.values()

k_set = [3, 6, 9, 12]

from assess_proba_seq_for_counts import compute_plots

repetition_options = ['rep', 'no_rep']
loss_options = ['lossless', 'forgetful']
filter_options = ['filtered', 'not_filtered']

task_list = [(achu_data_set + oldo_data_set, achu_data_set, oldo_data_set, 'achu_and_oldo', 'not_filtered'),
             (achu_data_set, achu_data_set, oldo_data_set, 'achu', 'not_filtered'),
             (oldo_data_set, achu_data_set, oldo_data_set, 'oldo', 'not_filtered'),
             (f_achu_data_set + f_oldo_data_set, f_achu_data_set, f_oldo_data_set, 'achu_and_oldo', 'filtered'),
             (f_achu_data_set, f_achu_data_set, f_oldo_data_set, 'achu', 'filtered'),
             (f_oldo_data_set, f_achu_data_set, f_oldo_data_set, 'oldo', 'filtered')]

for repetition_option in repetition_options:
    for loss_option in loss_options:
        for build_data_set, achu_set, oldo_set, title, filter_option in task_list:
            compute_plots(repetition_option,
                          loss_option,
                          filter_option,
                          title,
                          build_data_set,
                          achu_set,
                          oldo_set,
                          k_set)