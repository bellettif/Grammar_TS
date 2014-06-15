'''
Created on 14 juin 2014

@author: francois
'''

import copy
import numpy as np
from matplotlib import pyplot as plt
import os

class Proba_seq_merger:
    
    def __init__(self):
        #
        self.rules = {}
        self.terminal_chars = []
        #
        self.rule_to_hashcode = {}
        self.hashcode_to_rule = {}
        #
        self.relative_counts = {}
        self.absolute_counts = {}
        self.tot_relative_counts = {}
        #
        self.levels = {}
        self.depths = {}
        #
        self.scores_computed = False
        
    def merge_with(self, ps):
        for lhs_hash, (left_hash, right_hash) in ps.rules.iteritems():
            if lhs_hash not in self.hashcode_to_rule:
                lhs_name = self.create_new_rule(lhs_hash, ps.depths[lhs_hash])
                if left_hash not in self.hashcode_to_rule:
                    if left_hash in ps.rules:
                        left_name = self.create_new_rule(left_hash, 
                                                         ps.depths[left_hash])
                    else:
                        left_name = left_hash
                        self.terminal_chars.append(left_hash)
                else:
                    left_name = self.hashcode_to_rule[left_hash]
                if right_hash not in self.hashcode_to_rule:
                    if right_hash in ps.rules:
                        right_name = self.create_new_rule(right_hash, 
                                                          ps.depths[right_hash])
                    else:
                        right_name = right_hash
                        self.terminal_chars.append(right_hash)
                else:
                    right_name = self.hashcode_to_rule[right_hash]
                self.rules[lhs_name] = [left_name, right_name]
            lhs_name = self.hashcode_to_rule[lhs_hash]
            self.relative_counts[lhs_name].append(ps.relative_counts[lhs_hash])        
            self.absolute_counts[lhs_name].append(ps.absolute_counts[lhs_hash])
            self.tot_relative_counts[lhs_name] += sum(ps.absolute_counts[lhs_hash])
            self.levels[lhs_name].append(ps.levels[lhs_hash])
            
    def create_new_rule(self, hashcode, depth):
        name = 'r%d' % (len(self.rule_to_hashcode) + 1)
        self.rule_to_hashcode[name] = hashcode
        self.hashcode_to_rule[hashcode] = name
        self.relative_counts[name] = []
        self.absolute_counts[name]=  []
        self.tot_relative_counts[name] = 0
        self.levels[name] = []
        self.depths[name] = depth
        return name
    
    def get_rel_count_distribs(self, first_indices, 
                                     second_indices,
                                     target_depth):
        if not self.scores_computed:
            for key, value in self.tot_relative_counts.iteritems():
                self.tot_relative_counts[key] = value * self.depths[key]
            self.scores_computed = True
        first_count_distribs = {}
        second_count_distribs = {}
        tot_rel_counts_items = self.tot_relative_counts.items()
        tot_rel_counts_items = filter(lambda x : self.depths[x[0]] == target_depth,
                                      tot_rel_counts_items)
        tot_rel_counts_items.sort(key = (lambda x : -x[1]))
        sorted_rule_names = [x[0] for x in tot_rel_counts_items]
        sorted_hashcodes = [self.rule_to_hashcode[x[0]] for x in tot_rel_counts_items]
        for rule_name in sorted_rule_names:
            count_list = self.relative_counts[rule_name]
            first_distrib = []
            second_distrib = []
            for counts in count_list:
                first_distrib.extend(filter(lambda x : x != 0, [counts[i] for i in first_indices]))
                second_distrib.extend(filter(lambda x : x != 0, [counts[i] for i in second_indices]))
            first_count_distribs[self.rule_to_hashcode[rule_name]] = first_distrib
            second_count_distribs[self.rule_to_hashcode[rule_name]] = second_distrib
        return first_count_distribs, second_count_distribs, copy.deepcopy(self.depths), sorted_hashcodes

    def get_rel_count_avgs(self, first_indices, 
                           second_indices,
                           target_depth):
        if not self.scores_computed:
            for key, value in self.tot_relative_counts.iteritems():
                self.tot_relative_counts[key] = value * self.depths[key]
            self.scores_computed = True
        first_count_distribs = {}
        second_count_distribs = {}
        tot_rel_counts_items = self.tot_relative_counts.items()
        tot_rel_counts_items = filter(lambda x : self.depths[x[0]] == target_depth,
                                      tot_rel_counts_items)
        tot_rel_counts_items.sort(key = (lambda x : -x[1]))
        sorted_rule_names = [x[0] for x in tot_rel_counts_items]
        sorted_hashcodes = [self.rule_to_hashcode[x[0]] for x in tot_rel_counts_items]
        for rule_name in sorted_rule_names:
            count_list = self.relative_counts[rule_name]
            first_distrib = []
            second_distrib = []
            for counts in count_list:
                first_distrib.append([counts[i] for i in first_indices])
                second_distrib.append([counts[i] for i in second_indices])
            first_distrib = np.asanyarray(first_distrib, dtype = np.double)
            second_distrib = np.asanyarray(second_distrib, dtype = np.double)
            first_count_distribs[self.rule_to_hashcode[rule_name]] = np.mean(first_distrib, axis = 1)
            second_count_distribs[self.rule_to_hashcode[rule_name]] = np.mean(second_distrib, axis = 1)
        return first_count_distribs, second_count_distribs, copy.deepcopy(self.depths), sorted_hashcodes
        
    def compare_data_sets(self, 
                          achu_indices,
                          oldo_indices,
                          prefix,
                          max_represented,
                          target_depths):
        os.mkdir(prefix)
        print "Comparing data sets with parameters %s" % prefix
        for target_depth in target_depths:
            achu_counts_dict, oldo_counts_dict, depths_dict, sorted_rule_names = \
                self.get_rel_count_avgs(achu_indices,
                                           oldo_indices,
                                           target_depth)
            achu_counts = []
            oldo_counts = []
            if len(achu_counts) == 0 or len(oldo_counts) == 0: continue
            for x in sorted_rule_names:
                achu_counts.append(achu_counts_dict[x])
                achu_counts.append([])
                oldo_counts.append(oldo_counts_dict[x])
                oldo_counts.append([])
            x_ticks = []
            for x in sorted_rule_names:
                x_ticks.append(x + 
                               (' (%d)' % depths_dict[self.hashcode_to_rule[x]]) )
                x_ticks.append('')
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
            plt.title('Relative count averages (Achu in red, Oldo in blue)')
            fig = plt.gcf()
            fig.set_size_inches((60, 8))
            plt.savefig('%s/Merged_results_%s_avg_%d.png' % 
                        (prefix, prefix, target_depth), dpi = 300)
            plt.close()
            #
            achu_counts_dict, oldo_counts_dict, depths_dict, sorted_rule_names = \
                self.get_rel_count_distribs(achu_indices,
                                               oldo_indices,
                                               target_depth)
            achu_counts = []
            oldo_counts = []
            for x in sorted_rule_names:
                achu_counts.append(achu_counts_dict[x])
                achu_counts.append([])
                oldo_counts.append(oldo_counts_dict[x])
                oldo_counts.append([])
            x_ticks = []
            for x in sorted_rule_names:
                x_ticks.append(x + 
                               (' (%d)' % depths_dict[self.hashcode_to_rule[x]]) )
                x_ticks.append('')
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
            plt.title('Relative count distributions (Achu in red, Oldo in blue)')
            fig = plt.gcf()
            fig.set_size_inches((60, 8))
            plt.savefig('%s/Merged_results_%s_distribs_%d.png' 
                        % (prefix, prefix, target_depth), dpi = 300)
            plt.close()
        
        