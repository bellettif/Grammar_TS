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

degree_set = [3, 6, 9, 12]
max_rules_set = [30, 40, 50, 60]
filter_option_set = [('not_filtered', achu_data_set, oldo_data_set),
                     ('filtered', f_achu_data_set, f_oldo_data_set)]


for degree in degree_set:
    for max_rules in max_rules_set:
        for filter_option, selected_achu_data_set, selected_oldo_data_set in filter_option_set:
            
            keep_data = 'keep_data'
            keep_data_bool = (keep_data == 'keep_data')
            
            selected_achu_data_set = achu_data_set
            selected_oldo_data_set = oldo_data_set
            
            both_data_sets = copy.deepcopy(selected_achu_data_set) + \
                             copy.deepcopy(selected_oldo_data_set)
            
            merged_rules = {}
            merged_rules_to_hashcodes = {}
            merged_hashcodes_to_rules = {}
            merged_achu_oldo_counts = {}
            merged_achu_counts = {}
            merged_oldo_counts = {}
            merged_achu_oldo_levels = {}
            merged_achu_levels = {}
            merged_oldo_levels = {}
            current_index = [1]
            
            def merge_data(proba_seq, target, target_level):
                rules = ps.rules
                all_counts = ps.all_counts
                rule_to_hashcode = ps.hash_codes
                hashed_rules = ps.hashed_rules
                terms = ps.terminal_chars
                levels = ps.rule_levels
                level_to_rules = {}
                for rule, level in levels.iteritems():
                    if level not in level_to_rules:
                        level_to_rules[level] = []
                    level_to_rules[level].append(rule)
                for level, rules_of_level in level_to_rules.iteritems():
                    for lhs in rules_of_level:
                        hashcode = rule_to_hashcode[lhs]
                        left_hash_code, right_hash_code = hashed_rules[hashcode].split('_')
                        if hashcode not in merged_hashcodes_to_rules:
                            if left_hash_code not in terms:
                                merged_left_hash_code = merged_hashcodes_to_rules[left_hash_code]
                            else:
                                merged_left_hash_code = left_hash_code
                            if right_hash_code not in terms:
                                merged_right_hash_code = merged_hashcodes_to_rules[right_hash_code]
                            else:
                                merged_right_hash_code = right_hash_code
                            new_rule_name = 'r%d_' % current_index[0]
                            merged_hashcodes_to_rules[hashcode] = new_rule_name
                            merged_rules_to_hashcodes[new_rule_name] = hashcode
                            merged_rules[new_rule_name] = merged_left_hash_code + \
                                                            '-' + \
                                                          merged_right_hash_code
                            target[new_rule_name] = all_counts[lhs]
                            target_level[new_rule_name] = levels[lhs]
                            current_index[0] += 1
                        else:
                            rule_name = merged_hashcodes_to_rules[hashcode]
                            target[rule_name] = all_counts[lhs]
                            target_level[rule_name] = levels[lhs]
            
            #
            #
            #
            ps = Proba_sequitur(build_samples = both_data_sets,
                                count_samples = both_data_sets,
                                repetitions = True,
                                keep_data = keep_data_bool,
                                degree = degree,
                                max_rules = max_rules)
            ps.infer_grammar()
            ps.plot_stats('achu_oldo_%s_%d_%s_%d' % (filter_option, degree, keep_data, max_rules))
            merge_data(ps, merged_achu_oldo_counts, merged_achu_oldo_levels)
            
            ps = Proba_sequitur(build_samples = selected_oldo_data_set,
                                count_samples = both_data_sets,
                                repetitions = True,
                                keep_data = keep_data_bool,
                                degree = degree,
                                max_rules = max_rules)
            ps.infer_grammar()
            ps.plot_stats('oldo_%s_%d_%s_%d' % (filter_option, degree, keep_data, max_rules))
            merge_data(ps, merged_oldo_counts, merged_oldo_levels)
            
            ps = Proba_sequitur(build_samples = selected_achu_data_set,
                                count_samples = both_data_sets,
                                repetitions = True,
                                keep_data = keep_data_bool,
                                degree = degree,
                                max_rules = max_rules)
            ps.infer_grammar()
            ps.plot_stats('achu_%s_%d_%s_%d' % (filter_option, degree, keep_data, max_rules))
            merge_data(ps, merged_achu_counts, merged_achu_levels)
            
            all_rules_counted = list(set(merged_oldo_counts.keys() + merged_achu_counts.keys() + merged_achu_oldo_counts.keys()))
            
            print sorted(all_rules_counted)
            
            total_counts = []
            for rule_name in all_rules_counted:
                if rule_name not in merged_oldo_counts:
                    merged_oldo_levels[rule_name] = 'NA'
                    merged_oldo_counts[rule_name] = {}
                    for i in xrange(18):
                        merged_oldo_counts[rule_name][i] = 0.0
                if rule_name not in merged_achu_counts:
                    merged_achu_levels[rule_name] = 'NA'
                    merged_achu_counts[rule_name] = {}
                    for i in xrange(18):
                        merged_achu_counts[rule_name][i] = 0.0
                if rule_name not in merged_achu_oldo_counts:
                    merged_achu_oldo_levels[rule_name] = 'NA'
                    merged_achu_oldo_counts[rule_name] = {}
                    for i in xrange(18):
                        merged_achu_oldo_counts[rule_name][i] = 0.0
                total_counts.append([rule_name, 
                                     sum(merged_oldo_counts[rule_name].values()) + 
                                     sum(merged_achu_counts[rule_name].values()) +
                                     sum(merged_achu_oldo_counts[rule_name].values())])
            print total_counts
            total_counts.sort(key = (lambda x : -x[1]))
            
            rule_names = [x[0] for x in total_counts]
            
            achu_sub_set = range(9)
            oldo_sub_set = range(9, 18)
            
            achu_boxes = []
            oldo_boxes = []
            box_names = []
            for rule_name in rule_names:
                achu_boxes.append([])
                oldo_boxes.append([])
                box_names.append('')
                achu_boxes.append([merged_achu_counts[rule_name].values()[j] for j in achu_sub_set])
                oldo_boxes.append([merged_achu_counts[rule_name].values()[j] for j in oldo_sub_set])
                box_names.append('achu ' + str(merged_achu_levels[rule_name]) + ' ' + 
                                 rule_name + '->' + merged_rules[rule_name])
                achu_boxes.append([merged_achu_oldo_counts[rule_name].values()[j] for j in achu_sub_set])
                oldo_boxes.append([merged_achu_oldo_counts[rule_name].values()[j] for j in oldo_sub_set])
                box_names.append('both ' + str(merged_achu_oldo_levels[rule_name]) + ' ' +
                                 rule_name + '->' + merged_rules[rule_name])
                achu_boxes.append([merged_oldo_counts[rule_name].values()[j] for j in achu_sub_set])
                oldo_boxes.append([merged_oldo_counts[rule_name].values()[j] for j in oldo_sub_set])
                box_names.append('oldo ' + str(merged_oldo_levels[rule_name]) + ' '+ 
                                 rule_name + '->' + merged_rules[rule_name])
                achu_boxes.append([])
                box_names.append('')
                oldo_boxes.append([])
                 
            bp = plt.boxplot(achu_boxes,
                             notch=0,
                             sym='+',
                             vert=1,
                             whis=1.5,
                             patch_artist = True)
            plt.setp(bp['boxes'], color = 'r', facecolor = 'r', alpha = 0.25)
            plt.setp(bp['whiskers'], color='r')
            plt.setp(bp['fliers'], color='r', marker='+')
            
            bp = plt.boxplot(oldo_boxes,
                             notch=0, 
                             sym='+',
                             vert=1, 
                             whis=1.5, 
                             patch_artist = True)
            plt.setp(bp['boxes'], color='b', facecolor = 'b', alpha = 0.25)
            plt.setp(bp['whiskers'], color='b')
            plt.setp(bp['fliers'], color='b', marker='+')
            plt.xticks(range(1, len(box_names) + 1), 
                       box_names,
                       rotation = 'vertical', fontsize = 4)
            fig = plt.gcf()
            fig.set_size_inches((40, 8))
            plt.savefig('Freqs_merged_%s_%d_%s_%d.png' % (filter_option, degree, keep_data, max_rules), dpi = 600)
            plt.close()
        