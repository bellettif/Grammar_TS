'''
Created on 21 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy

import load_data
from load_data import achu_file_names, oldo_file_names

from information_th import measures as infth

from plot_convention import colors

#
#    Eliminate p percents on average
#
def random_mask(seq, p):
    selection = np.random.binomial(1, 1.0 - p, len(seq))
    return filter(lambda x : x != None, [seq[i] if selection[i] == 1
            else None for i in range(len(selection))])

def shuffle(seq):
    seq_copy = copy.deepcopy(seq)
    np.random.shuffle(seq_copy)
    return seq_copy

p = 0.05
n_bootstrap = 10
n_gram_targets = range(2, 18)

rep_data_set = dict(load_data.achu_file_contents.items() +
                    load_data.oldo_file_contents.items())
no_rep_data_set = dict(load_data.no_rep_achu_file_contents.items() +
                       load_data.no_rep_oldo_file_contents.items())
no_g_data_set = dict(load_data.no_g_achu_file_contents.items() +
                     load_data.no_g_oldo_file_contents.items())
no_g_no_rep_data_set = dict(load_data.no_g_no_rep_achu_file_contents.items() +
                            load_data.no_g_no_rep_oldo_file_contents.items())

for repetitions in [True, False]:
    for g_present in [True, False]:
        #
        if repetitions:
            if g_present:
                selected_data_set = rep_data_set
            else:
                selected_data_set = no_g_data_set
        else:
            if g_present:
                selected_data_set = no_rep_data_set
            else:
                selected_data_set = no_g_no_rep_data_set
        #                       
        selected_files = achu_file_names + oldo_file_names
        #
        bootstrapped_data = {}
        bootstrapped_shuffled_data = {}
        #
        for file_name in selected_files:
            bootstrapped_data[file_name] = [random_mask(selected_data_set[file_name], p)
                                            for i in xrange(n_bootstrap)]
            bootstrapped_shuffled_data[file_name] = [shuffle(x) for x in
                                                     bootstrapped_data[file_name]]
        #
        def compute_entropy_distrib(target_file):
            targets = bootstrapped_data[target_file]
            entropy_grams = [[infth.compute_n_gram_entropy(x, n)
                              for n in n_gram_targets]
                             for x in targets]
            entropy_grams = np.asanyarray(entropy_grams, 
                                          dtype = np.double)
            median_entropy = np.median(entropy_grams, 
                                       axis = 0)
            perc_95_entropy = np.percentile(entropy_grams, 
                                            q = 95, 
                                            axis = 0)
            perc_05_entropy = np.percentile(entropy_grams,
                                            q = 5, 
                                            axis = 0)
            return perc_05_entropy, median_entropy, perc_95_entropy
        #
        def compute_entropy_distrib_shuffled(target_file):
            targets = bootstrapped_shuffled_data[target_file]
            entropy_grams = [[infth.compute_n_gram_entropy(x, n)
                              for n in n_gram_targets]
                             for x in targets]
            entropy_grams = np.asanyarray(entropy_grams, 
                                          dtype = np.double)
            median_entropy = np.median(entropy_grams, 
                                       axis = 0)
            perc_95_entropy = np.percentile(entropy_grams, 
                                            q = 95, 
                                            axis = 0)
            perc_05_entropy = np.percentile(entropy_grams,
                                            q = 5, 
                                            axis = 0)
            return perc_05_entropy, median_entropy, perc_95_entropy
        #
        dict_distribs = {}
        dict_distribs_shuffled = {}
        #
        for file_name in selected_files:
            dict_distribs[file_name] = compute_entropy_distrib(file_name)
            dict_distribs_shuffled[file_name] = compute_entropy_distrib_shuffled(file_name)
            
        min_value = min(np.min(x[1]) for x in 
                        dict_distribs.values() 
                        + dict_distribs_shuffled.values())
        max_value = max(np.max(x[1]) for x in 
                        dict_distribs.values()
                         + dict_distribs_shuffled.values())
        #
        plt.subplot(211)
        for file_name in achu_file_names:
            plt.plot(n_gram_targets,
                     dict_distribs[file_name][1],
                     lw = 2,
                     alpha = 0.8,
                     c = colors.all_colors[file_name])
            plt.plot(n_gram_targets,
                     dict_distribs_shuffled[file_name][1],
                     linestyle = '--',
                     lw = 2,
                     alpha = 0.8,
                     c = colors.all_colors[file_name])
            plt.ylim((min_value, max_value))
            data_name = 'Achu'
            if repetitions and g_present:
                plt.title('n-gram entropy %s' % data_name)
            if (not repetitions) and g_present:
                plt.title('n-gram entropy %s no rep.' % data_name)
            if repetitions and (not g_present):
                plt.title('n-gram entropy %s no g symbol' % data_name)
            if (not repetitions) and (not g_present):
                plt.title('n-gram entropy %s no rep., no g symbol' % data_name)
            plt.xlabel('n-gram length')
            plt.ylabel('Length normalized entropy')
        plt.subplot(212)
        for file_name in oldo_file_names:
            plt.plot(n_gram_targets,
                     dict_distribs[file_name][1],
                     lw = 2,
                     alpha = 0.8,
                     c = colors.all_colors[file_name])
            plt.plot(n_gram_targets,
                     dict_distribs_shuffled[file_name][1],
                     linestyle = '--',
                     lw = 2,
                     alpha = 0.8,
                     c = colors.all_colors[file_name])
            plt.ylim((min_value, max_value))
            data_name = 'Oldo'
            if repetitions and g_present:
                plt.title('n-gram entropy %s' % data_name)
            if (not repetitions) and g_present:
                plt.title('n-gram entropy %s no rep.' % data_name)
            if repetitions and (not g_present):
                plt.title('n-gram entropy %s no g symbol' % data_name)
            if (not repetitions) and (not g_present):
                plt.title('n-gram entropy %s no rep., no g symbol' % data_name)
            plt.xlabel('n-gram length')
            plt.ylabel('Length normalized entropy')
        fig = plt.gcf()
        fig.set_size_inches((8, 12))
        if repetitions:
            if g_present:
                plt.savefig('Entropy_measurements_rep.png', dpi = 600)
            else:
                plt.savefig('Entropy_measurements_rep_no_g.png', dpi = 600)
        else:
            if g_present:
                plt.savefig('Entropy_measurements_no_rep.png', dpi = 600)
            else:
                plt.savefig('Entropy_measurement_no_rep_no_g.png', dpi = 600)
        plt.close()
    
    


