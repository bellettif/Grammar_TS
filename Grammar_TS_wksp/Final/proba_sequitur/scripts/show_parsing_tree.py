'''
Created on 30 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy
import pydot

from proba_sequitur.Proba_sequitur import Proba_sequitur
import load_data
from plot_convention.colors import blues, oranges
from proba_sequitur.grammar_graph import grammar_to_graph

achu_file_names = load_data.achu_file_names
oldo_file_names = load_data.oldo_file_names

achu_data_set = [load_data.achu_file_contents[x] 
                 for x in achu_file_names]
oldo_data_set = [load_data.oldo_file_contents[x]
                 for x in oldo_file_names]

achu_data_set_no_rep = [load_data.no_rep_achu_file_contents[x]
                        for x in achu_file_names]
oldo_data_set_no_rep = [load_data.no_rep_oldo_file_contents[x]
                        for x in oldo_file_names]

inference_content = achu_data_set_no_rep + oldo_data_set_no_rep
count_content = copy.deepcopy(inference_content)

n_rounds = 10
k = 8

ps = Proba_sequitur([x.split(' ') for x in inference_content],
                    [x.split(' ') for x in count_content],
                    k,
                    n_rounds * k,
                    False,
                    0.0,
                    0.0,
                    0.0)
ps.run()

print ' '.join([ps.rule_names[x] if x in ps.rule_names else x for x in ps.inference_parsed[0]])

print ' '.join([str(ps.levels[x]) if x in ps.levels else '0' for x in ps.inference_parsed[0]])

all_levels = list(set(ps.levels.values()))

level_items = ps.levels.items()

for level in all_levels:
    print [ps.rule_names[x[0]] 
           for x in 
           filter(lambda x : x[1] == level, level_items)]

ps.draw_graph('Parsing_tree_proba_seq_%s.png' % achu_file_names[0], 
                 0,
                 cut = 10)

