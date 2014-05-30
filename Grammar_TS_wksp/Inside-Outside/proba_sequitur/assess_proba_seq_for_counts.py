'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from Proba_sequitur_for_counts import Proba_sequitur

MAX_N_RULES_PLOTTED = 20

def compute_counts(build_data_set,
                   count_data_set,
                   repetitions,
                   keep_data,
                   k,
                   achu_data_set,
                   oldo_data_set):
    proba_sequitur = Proba_sequitur(build_data_set,
                                    count_data_set,
                                    repetitions,
                                    keep_data)
    proba_sequitur.infer_grammar(k)
    print '\t\t%d rules inferred' % len(proba_sequitur.rules)
    rule_names = [proba_sequitur.rules[x] for x in proba_sequitur.all_counts.keys()]
    rule_counts = [x.values() for x in proba_sequitur.all_counts.values()]
    rule_counts = np.asanyarray(rule_counts)
    counts_sum = np.sum(rule_counts, axis = 1)
    indices = range(len(counts_sum))
    indices.sort(key = (lambda i : - counts_sum[i]))
    indices = indices[:MAX_N_RULES_PLOTTED]
    rule_names = [rule_names[i] for i in indices]
    rule_counts = [rule_counts[i] for i in indices]
    rule_counts = np.asanyarray(rule_counts)
    return rule_names, rule_counts

def draw_plot(rule_counts, rule_names,
              oldo_sub_set, achu_sub_set,
              sub_plot_index,
              title,
              k,
              loss_option,
              filter_option,
              repetition_option):
    plt.subplot(sub_plot_index)
    bp = plt.boxplot([rule_counts[i,achu_sub_set] for i in xrange(len(rule_names))],
                     notch=0, sym='+', vert=1, whis=1.5, patch_artist = True)
    plt.setp(bp['boxes'], color = 'r', facecolor = 'r', alpha = 0.25)
    plt.setp(bp['whiskers'], color='r')
    plt.setp(bp['fliers'], color='r', marker='+')
    bp = plt.boxplot([rule_counts[i,oldo_sub_set] for i in xrange(len(rule_names))],
                     notch=0, sym='+', vert=1, whis=1.5, patch_artist = True)
    plt.setp(bp['boxes'], color='b', facecolor = 'b', alpha = 0.25)
    plt.setp(bp['whiskers'], color='b')
    plt.setp(bp['fliers'], color='b', marker='+')
    plt.xticks(range(1, len(rule_names) + 1), rule_names, rotation = 'vertical', fontsize = 8)
    plt.ylabel('Red for achu, blue for oldo')
    plt.title('Learn on %s %d %s %s %s' % (title, k, loss_option, filter_option, repetition_option))

#
#    repetition_option in 'rep' or 'no_rep'
#    loss_option in 'lossless' or forgetful
#    filter_option in 'filtered' or 'not_filtered'
#
def compute_plots(repetition_option,
                  loss_option,
                  filter_option,
                  title,
                  build_data_set,
                  achu_data_set,
                  oldo_data_set,
                  k_set):
    print 'Computing plots with options:'
    print '\tRepetition: ' + repetition_option
    print '\tLoss: ' + loss_option
    print '\tTitle: ' + title
    print '\tK_set: ' + ' '.join(map(str, k_set))
    repetitions = (repetition_option == 'rep')
    keep_data = (loss_option == 'lossless')
    count_data_set = achu_data_set + oldo_data_set
    n = len(achu_data_set)
    p = len(oldo_data_set)
    achu_sub_set = range(n)
    oldo_sub_set = range(n, n + p)                 
    #
    #    First value of k
    #
    #
    #        Run k sequitur
    #
    k = k_set[0]
    print '\t\tRunning proba sequitur with k = %d' % k
    rule_names, rule_counts = compute_counts(build_data_set,
                                             count_data_set,
                                             repetitions,
                                             keep_data,
                                             k,
                                             achu_data_set,
                                             oldo_data_set)
    #
    #        Boxplot
    #
    print '\t\tDone running proba sequitur with k = %d, plotting' % k
    draw_plot(rule_counts,
              rule_names,
              oldo_sub_set,
              achu_sub_set,
              411,
              title,
              k,
              loss_option,
              filter_option,
              repetition_option)
    #
    #    Second value of k
    #
    #
    #        Run k sequitur
    #
    k = k_set[1]
    print '\t\tRunning k sequitur with k = %d' % k
    rule_names, rule_counts = compute_counts(build_data_set,
                                             count_data_set,
                                             repetitions,
                                             keep_data,
                                             k,
                                             achu_data_set,
                                             oldo_data_set)
    #
    #        Box plot
    #
    print '\t\tDone running proba sequitur with k = %d, plotting' % k
    draw_plot(rule_counts,
              rule_names,
              oldo_sub_set,
              achu_sub_set,
              412,
              title,
              k,
              loss_option,
              filter_option,
              repetition_option)
    #
    #    Second value of k
    #
    #
    #        Run k sequitur
    #
    k = k_set[2]
    print '\t\tRunning k sequitur with k = %d' % k
    rule_names, rule_counts = compute_counts(build_data_set,
                                             count_data_set,
                                             repetitions,
                                             keep_data,
                                             k,
                                             achu_data_set,
                                             oldo_data_set)
    #
    #        Box plot
    #
    print '\t\tDone running proba sequitur with k = %d, plotting' % k
    draw_plot(rule_counts,
              rule_names,
              oldo_sub_set,
              achu_sub_set,
              413,
              title,
              k,
              loss_option,
              filter_option,
              repetition_option)
    #
    #    Third value of k
    #
    #
    #        Run k sequitur
    #
    k = k_set[3]
    print '\t\tRunning k sequitur with k = %d' % k
    rule_names, rule_counts = compute_counts(build_data_set,
                                             count_data_set,
                                             repetitions,
                                             keep_data,
                                             k,
                                             achu_data_set,
                                             oldo_data_set)
    #
    #        Box plot
    #
    print '\t\tDone running proba sequitur with k = %d, plotting' % k
    draw_plot(rule_counts,
              rule_names,
              oldo_sub_set,
              achu_sub_set,
              414,
              title,
              k,
              loss_option,
              filter_option,
              repetition_option)
    #
    #    Saving plot
    #
    fig = plt.gcf()
    fig.set_size_inches((8, 26))
    file_path = 'Proba_seq_%s/Learnt_on_%s_%s_%s.png' % (loss_option, title, filter_option, repetition_option)
    plt.savefig(file_path,
                dpi = 300)
    plt.close()
    print '\tSaved figure to ' + file_path
    print 'Done'
    print '\n'