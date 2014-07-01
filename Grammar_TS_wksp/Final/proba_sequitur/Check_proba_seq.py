'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy
import time

from proba_sequitur import Proba_sequitur
import load_data

achu_data_set = load_data.achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()

achu_data_set_no_rep = load_data.no_rep_achu_file_contents.values()
oldo_data_set_no_rep = load_data.no_rep_oldo_file_contents.values()

inference_content = achu_data_set_no_rep + oldo_data_set_no_rep
count_content = copy.deepcopy(inference_content)

begin = time.clock()
ps = Proba_sequitur([x.split(' ') for x in inference_content],
                    [x.split(' ') for x in count_content],
                    12,
                    120,
                    False,
                    0.0,
                    0.1,
                    0.0)
ps.run()
print time.clock() - begin


rule_names = ps.hashcode_to_rule

for terminal_parse in ps.count_parsed:
    print ' '.join([rule_names[x] if x in rule_names else x for x in terminal_parse])
    print ''

plt.plot(ps.divergences)
plt.show()