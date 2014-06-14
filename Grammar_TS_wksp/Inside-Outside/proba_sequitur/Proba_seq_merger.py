'''
Created on 14 juin 2014

@author: francois
'''

import copy

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
            self.tot_relative_counts[lhs_name] += sum(ps.relative_counts[lhs_hash])
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
    
    def get_rel_count_distribs(self, first_indices, second_indices):
        first_count_distribs = {}
        second_count_distribs = {}
        tot_rel_counts_items = self.tot_relative_counts.items()
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
            
            
        
                
                