'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy

from proba_sequitur.Proba_sequitur import Proba_sequitur

from proba_sequitur import load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.filtered_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.filtered_oldo_file_contents.values()


from assess_proba_seq_for_counts import compute_counts

repetition_options = 'rep'
loss_options = 'lossless'

degree = 6
keep_data = 'keep_data'
max_rules = 40

keep_data_bool = (keep_data == 'keep_data')

selected_achu_data_set = achu_data_set
selected_oldo_data_set = oldo_data_set

both_data_sets = copy.deepcopy(selected_achu_data_set) + \
                 copy.deepcopy(selected_oldo_data_set)

ps = Proba_sequitur(build_samples = both_data_sets,
                    count_samples = both_data_sets,
                    repetitions = True,
                    keep_data = keep_data_bool,
                    degree = degree,
                    max_rules = max_rules)
ps.infer_grammar()

for term_char in ps.terminal_chars:
    print '%s : %f' % (term_char, ps.barelk_table[term_char])

