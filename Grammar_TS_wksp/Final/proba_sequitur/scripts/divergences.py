'''
Created on 30 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy

from proba_sequitur.Proba_sequitur import Proba_sequitur
import load_data
from plot_convention.colors import blues, oranges

achu_data_set = load_data.achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()

achu_data_set_no_rep = load_data.no_rep_achu_file_contents.values()
oldo_data_set_no_rep = load_data.no_rep_oldo_file_contents.values()

inference_content = achu_data_set_no_rep + oldo_data_set_no_rep
count_content = copy.deepcopy(inference_content)

n_rounds = 10
k_set = np.arange(4, 12)
colormap = (k_set - min(k_set)) / float(max(k_set) - min(k_set) + 2)
all_divergences = []

for k in k_set:
    ps = Proba_sequitur([x.split(' ') for x in inference_content],
                        [x.split(' ') for x in count_content],
                        k,
                        n_rounds * k,
                        False,
                        0.0,
                        0.0,
                        0.0)
    ps.run()
    all_divergences.append(ps.divergences)

for i, divs in enumerate(all_divergences):
    plt.plot(divs / k_set[i], c = (colormap[i], colormap[i], colormap[i]),
             lw = k_set[i] / (0.5 * max(k_set)))
plt.legend((['k = %d' % x for x in k_set]), 'upper right')
plt.xlabel('Round index')
plt.ylabel('Average divergence')
plt.title('Average contribution to KL divergence of best pairs')
plt.savefig('Divergence_values_deterministic.png', dpi = 300)