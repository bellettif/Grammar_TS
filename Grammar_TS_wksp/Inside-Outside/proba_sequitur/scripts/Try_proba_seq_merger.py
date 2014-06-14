'''
Created on 14 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import time

from proba_sequitur.Proba_seq_merger import Proba_seq_merger
from proba_sequitur.Proba_sequitur import Proba_sequitur
import proba_sequitur.load_data as load_data

achu_data_set = load_data.achu_file_contents.values()
f_achu_data_set = load_data.filtered_achu_file_contents.values()
oldo_data_set = load_data.oldo_file_contents.values()
f_oldo_data_set = load_data.filtered_oldo_file_contents.values()


n_trials = 1000

achu_indices = range(9)
oldo_indices = range(9, 18)

degree_set = [6, 12]
max_rules_set = [30, 60]
T_set = [0.01, 0.1, 0.5, 1.0]
T_decay_set = [0.01, 0.1, 0.2]
p_deletion_set = [0.01, 0.05, 0.10]
filter_option_set = [('sto_not_filtered', achu_data_set, oldo_data_set),
                     ('sto_filtered', f_achu_data_set, f_oldo_data_set)]


inference_content = f_achu_data_set + f_oldo_data_set
count_content = inference_content



reducer_achu = Proba_seq_merger()
reducer_achu_oldo = Proba_seq_merger()

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
    
achu_counts_dict, oldo_counts_dict, depths_dict, sorted_rule_names = \
    reducer.get_rel_count_distribs(achu_indices,
                                   oldo_indices)
print 'Computation took %f second' % (time.clock() - begin)  
achu_counts = [achu_counts_dict[x] for x in sorted_rule_names]
oldo_counts = [oldo_counts_dict[x] for x in sorted_rule_names]
max_represented = 200
x_ticks = []
for x in sorted_rule_names:
    x_ticks.append(x + 
                   (' (%d)' % depths_dict[reducer.hashcode_to_rule[x]]) )
achu_counts = achu_counts[:max_represented]
oldo_counts = oldo_counts[:max_represented]
x_ticks = x_ticks[:max_represented]
bp = plt.boxplot(achu_counts,
                 notch=0,
                 sym='+',
                 vert=1,
                 whis=1.5,
                 patch_artist = True)
plt.setp(bp['boxes'], color = 'r', facecolor = 'r', alpha = 0.25)
plt.setp(bp['whiskers'], color='r')
plt.setp(bp['fliers'], color='r', marker='+')
bp = plt.boxplot(oldo_counts,
                 notch=0,
                 sym='+',
                 vert=1,
                 whis=1.5,
                 patch_artist = True)
plt.setp(bp['boxes'], color = 'b', facecolor = 'b', alpha = 0.25)
plt.setp(bp['whiskers'], color='b')
plt.setp(bp['fliers'], color='b', marker='+')

plt.xticks(range(1, len(x_ticks) + 1),
           x_ticks,
           rotation = 'vertical',
           fontsize = 4)
plt.ylabel('Relative counts')
plt.yscale('log')

fig = plt.gcf()
fig.set_size_inches((40, 8))
plt.savefig('Merged_results.png', dpi = 600)
plt.close()