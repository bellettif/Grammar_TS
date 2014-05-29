'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from Proba_sequitur_for_counts import Proba_sequitur

import load_data

achu_data_set = load_data.achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()

build_data_set = oldo_data_set
count_data_set = achu_data_set + oldo_data_set
                    
proba_sequitur = Proba_sequitur(build_data_set,
                                count_data_set,
                                False)

proba_sequitur.infer_grammar(6)

print proba_sequitur.all_counts

n_achu = len(achu_data_set)
n_oldo = len(oldo_data_set)

achu_sub_set = range(n_achu)
oldo_sub_set = range(n_achu, n_achu + n_oldo)

rule_names = [proba_sequitur.rules[x] for x in proba_sequitur.all_counts.keys()]
rule_counts = [x.values() for x in proba_sequitur.all_counts.values()]
rule_counts = np.asanyarray(rule_counts)

for i in achu_sub_set:
    plt.plot(rule_counts[:,i], linestyle = 'None', marker = 'o', color = 'r')
for i in oldo_sub_set:
    plt.plot(rule_counts[:,i], linestyle = 'None', marker = 'o', color = 'b')
plt.xticks(range(len(rule_names)), rule_names, rotation = 'vertical', fontsize = 8)
plt.show()
