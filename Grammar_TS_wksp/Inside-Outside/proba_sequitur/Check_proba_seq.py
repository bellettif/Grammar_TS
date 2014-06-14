'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy
import time

from proba_sequitur.Proba_sequitur import Proba_sequitur

from proba_sequitur import load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.filtered_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.filtered_oldo_file_contents.values()

from Proba_sequitur_linear_c import run_proba_sequitur as proba_seq

inference_content = [x.split(' ') for x in achu_data_set] + \
                    [x.split(' ') for x in oldo_data_set]
count_content = copy.deepcopy(inference_content)

begin = time.clock()
result = proba_seq(inference_content,
                   count_content,
                   6,
                   40,
                   True,
                   0.0,
                   0.1,
                   0.10)
print time.clock() - begin


rule_names = result['rule_names']

for terminal_parse in result['count_parsed']:
    print ' '.join([rule_names[x] if x in rule_names else x for x in terminal_parse])
    print ''

plt.plot(result['divergences'])
plt.show()

"""
for key, count_dict in result['relative_counts'].iteritems():
    print "Counts of " + key
    for file_index, rela_count in count_dict.iteritems():
        print "\t" + str(file_index) + ": " + str(rela_count)
"""