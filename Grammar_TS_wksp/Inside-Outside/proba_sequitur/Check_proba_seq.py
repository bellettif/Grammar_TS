'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy
import time

from Proba_sequitur import Proba_sequitur
import load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.filtered_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.filtered_oldo_file_contents.values()

inference_content = achu_data_set + oldo_data_set
count_content = copy.deepcopy(inference_content)

ps = Proba_sequitur(inference_content,
                    count_content,
                    6,
                    40,
                    True,
                    0.0,
                    0.1,
                    0.1)

begin = time.clock()
ps.run()
print time.clock() - begin


rule_names = ps.rule_names

for terminal_parse in ps.count_parsed:
    print ' '.join([rule_names[x] if x in rule_names else x for x in terminal_parse])
    print ''

plt.plot(ps.divergences)
plt.show()