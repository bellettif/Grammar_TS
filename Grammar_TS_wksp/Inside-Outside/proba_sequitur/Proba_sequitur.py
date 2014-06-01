'''
Created on 22 mai 2014

@author: francois
'''

import numpy as np
import re
from matplotlib import pyplot as plt
import string
import cPickle as pickle
import copy
import time

import load_data

from surrogate.sto_grammar import SCFG
from surrogate.sto_rule import Sto_rule
from surrogate import sto_grammar


class Proba_sequitur:

    def __init__(self,
                 samples,
                 repetitions,
                 keep_data):
        self.current_rule_index = 1
        self.samples = samples
        self.terminal_parsing = {}
        self.rules = {}
        self.level = 0
        self.repetitions = repetitions
        self.keep_data = keep_data
        self.counts = {}
        self.terminal_chars = []
        self.next_rule_name = 0
        self.barelk_table = {}
        self.reconstructed = {}
        self.reconstructed_length = {}
        self.reconstructed_ratio = 0
        self.index_to_non_term = []
        self.non_term_to_index = {}  

    def reduce_counts(self,
                      list_of_dicts):
        reduced_counts = {}
        for current_dict in list_of_dicts:
            for key, value in current_dict.iteritems():
                if key not in reduced_counts:
                    reduced_counts[key] = 0
                reduced_counts[key] += value
        return reduced_counts

    def atom_counts(self,
                    sequence):
        symbols = set(sequence.split(' '))
        symbols = filter(lambda x : x != ' ', symbols)
        counts = {}
        for symbol in symbols:
            counts[symbol] = len(re.findall(symbol, sequence))
        return counts

    def atom_counts_multi(self,
                          sequences):
        list_of_counts = [self.atom_counts(x) for x in sequences]
        return self.reduce_counts(list_of_counts)

    def pair_counts(self,
                    sequence,
                    candidates):
        if not self.repetitions:
            all_pairs = [x + ' ' + y if x != y else None for x in candidates for y in candidates]
        else:
            all_pairs = [x + ' ' + y for x in candidates for y in candidates]
        #all_pairs.extend(x + ' _ ' + y if x != y else None for x in candidates for y in candidates)
        all_pairs = filter(lambda x : x != None, all_pairs)
        counts = {}
        for pair in all_pairs:
            symbol = pair
            symbol = re.subn(' ', '-', pair)[0]
            pattern = re.subn('_', '.', pair)[0]
            counts[symbol] = len(re.findall(pattern, sequence))
        to_delete = []
        for key, value in counts.iteritems():
            if value == 0:
                to_delete.append(key)
        for key in to_delete:
            del counts[key]
        return counts

    def pair_counts_multi(self,
                          sequences,
                          candidates):
        list_of_counts = [self.pair_counts(x, candidates) for x in sequences]
        return self.reduce_counts(list_of_counts)

    def init_barelk(self,
                    sequences):
        counts = self.atom_counts_multi(sequences)
        total = float(sum(counts.values()))
        barelk = {}
        for key, value in counts.iteritems():
            barelk[key] = value / total
        return barelk
    
    def compute_barelk(self,
                       symbol, barelk_table):
        left_symbol = symbol.split('-')[0]
        right_symbol = symbol.split('-')[-1]
        barelk = barelk_table[left_symbol] * barelk_table[right_symbol]
        barelk_table[symbol] = barelk
        return barelk

    def length_of_symbol(self,
                         s):
        return len(filter(lambda x : x != '_' and x != '-', s))

    def compute_pair_divergence(self,
                                sequences,
                                candidates,
                                barelk_table):
        pair_counts = self.pair_counts_multi(sequences, candidates)
        total = float(sum(pair_counts.values()))
        pair_probas = {}
        for key, value in pair_counts.iteritems():
            pair_probas[key] = value / total
            self.compute_barelk(key, barelk_table)
        divergences = {}
        total_chars = 0
        for seq in sequences:
            total_chars += len(seq)
        total_chars = float(total_chars)
        for key in pair_probas:
            divergences[key] = pair_probas[key] / float(self.length_of_symbol(key)) \
                                * np.log2(pair_probas[key] / 
                                          (barelk_table[key]))
        return divergences

    def substitute(self,
                   sequences,
                   symbols,
                   rule_names):
        temp_counts = {}
        for k, symbol in enumerate(symbols):
            pattern = re.subn('\-', ' ', symbol)[0]
            pattern = re.subn('_', '.', pattern)[0]
            #pattern = string.lower(pattern)
            temp_counts[string.lower(rule_names[k])] = 0
            for i, sequence in enumerate(sequences):
                #print re.subn(pattern, rule_names[k], sequence)
                sequences[i], c = re.subn(pattern, rule_names[k], sequence)
                temp_counts[string.lower(rule_names[k])] += c
        return sequences, temp_counts

    def reconstruct(self, terminal_parse):
        to_parse = copy.deepcopy(filter(lambda x : 'rule' in x, terminal_parse.split(' ')))
        to_parse = ' '.join(to_parse)
        next_to_parse = []
        reconstructed = []
        while len(to_parse) > 0:
            for elt in to_parse.split(' '):
                if elt not in self.rules:
                    reconstructed.append(elt)
                    continue
                rhs = self.rules[elt]
                left_member, right_member = rhs.split('-')
                if left_member in self.rules:
                    next_to_parse.append(left_member)
                else:
                    reconstructed.append(left_member)
                if right_member in self.rules:
                    next_to_parse.append(right_member)
                else:
                    reconstructed.append(right_member)
            to_parse = ' '.join(next_to_parse)
            next_to_parse = []
        reconstructed = ' '.join(reconstructed)
        return reconstructed

    def infer_grammar(self, degree):
        print self.samples
        self.level = 0
        self.current_rule_index = 1
        target_sequences = copy.deepcopy(self.samples)
        list_of_best_symbols = []
        list_of_rules = []
        self.counts = {}
        for sequence in target_sequences:
            self.terminal_chars.extend(sequence.split(' '))
        self.terminal_chars = set(self.terminal_chars)
        self.barelk_table = self.init_barelk(target_sequences)
        while len(target_sequences) > 0:
            self.level += 1
            print self.level
            print [len(x) for x in target_sequences]
            target_chars = []
            for sequence in target_sequences:
                target_chars.extend(sequence.split(' '))
            target_chars = set(target_chars)
            target_chars = filter(lambda x : x!= ' ', target_chars)
            barelk_table = self.init_barelk(target_sequences)
            pair_divergence = self.compute_pair_divergence(target_sequences,
                                                           target_chars,
                                                           barelk_table)
            items = pair_divergence.items()
            items.sort(key = (lambda x : -x[1]))
            labels = [x[0] for x in items]
            values = [x[1] for x in items]
            if degree != 0:
                best_symbols = labels[:degree]
            else:
                best_symbols_index = filter(lambda i : values[i] > 0, range(len(values)))
                best_symbols = labels[:len(best_symbols_index)]
            list_of_best_symbols.append(best_symbols)
            rule_names = []
            for best_symbol in best_symbols:
                self.rules['rule%d' % self.current_rule_index] = best_symbol
                rule_names.append('Rule%d' % self.current_rule_index)
                self.current_rule_index += 1
            list_of_rules.append(rule_names)
            target_sequences, temp_counts = self.substitute(target_sequences, best_symbols, rule_names)
            for key, value in temp_counts.iteritems():
                if key not in self.counts:
                    self.counts[key] = 0
                self.counts[key] += value
            temp_target_sequences = []
            for i, seq in enumerate(target_sequences):
                new_seq = seq.split(' ')
                if not self.keep_data:
                    new_seq = filter(lambda x : 'Rule' in x, new_seq)
                else:
                    n_not_rule = len(filter(lambda x : 'Rule' in x, new_seq))
                new_seq = ' '.join(new_seq)
                new_seq = re.sub('\-', '', new_seq)
                new_seq = string.lower(new_seq)
                if not self.keep_data:
                    if self.level == 1:
                        self.reconstructed[i] = [x if 'Rule' in x else '*' for x in seq.split(' ')]
                        self.reconstructed_length[i] = 2 * len(filter(lambda x : 'Rule' in x, seq.split(' ')))
                if not self.keep_data:
                    if len(new_seq) == 0:
                        self.terminal_parsing[i] = seq
                    else:
                        temp_target_sequences.append(new_seq)
                else:
                    if n_not_rule == 0:
                        self.terminal_parsing[i] = seq
                    else:
                        temp_target_sequences.append(new_seq)
            target_sequences = temp_target_sequences
            """
            print '\nIteration %d, %d rules\n' % (self.level, len(self.rules))
            print len(target_sequences)
            print '\n-----------------------------\n'
            """
        if self.keep_data:
            total_length = float(sum([len(x) for x in self.samples]))
            total_reconstructed_length = 0
            for key, value in self.terminal_parsing.iteritems():
                total_reconstructed_length += len(self.samples[key]) - len(filter(lambda x : 'rule' not in x, value.split(' ')))
            self.reconstructed_ratio = float(total_reconstructed_length) / total_length    
            print 'Total reconstructured length = ' + str(total_reconstructed_length)
            print 'Total length = ' + str(total_length)
        to_delete = []
        for key, value in self.counts.iteritems():
            if value == 0:
                to_delete.append(key)
        for key in to_delete:
            del self.counts[key]
            del self.rules[key]
            
    
    def print_result(self):
        rule_counts = self.counts.items()
        rule_items = self.rules.items()
        self.rule_ranking = []
        for i in xrange(len(rule_counts)):
            self.rule_ranking.append([rule_counts[i][0], rule_counts[i][1], rule_items[i][1]])
        self.rule_ranking.sort(key = (lambda x : -x[1]))
        print 'Terminal parsing:'
        print self.terminal_parsing
        print 'Rules (%d), %d levels:' % (len(self.rule_ranking), self.level)
        print self.rule_ranking
        print 'Rules:'
        print self.rules
        print 'Reconstructed ratio:'
        print self.reconstructed_ratio
        print ''
     
    def map_rules(self):
        self.index_to_non_term = {}
        self.index_to_score = {}
        self.non_term_to_index = {}
        self.index_to_non_term[0] = 0
        self.non_term_to_index[0] = 0
        n = 1
        for rule_lhs in self.rules:
            rule_number = n
            self.index_to_non_term[rule_number] = rule_lhs
            self.index_to_score[rule_number] = self.counts[rule_lhs]
            self.non_term_to_index[rule_lhs] = rule_number
            n += 1
        self.preterminal_rules = {}
        for i, term in enumerate(self.terminal_chars):
            weights = np.ones(len(self.terminal_chars)) * 0.01
            weights[i] = 1.0
            self.preterminal_rules[term] = \
                Sto_rule(int(n),
                         [],
                         [],
                         weights,
                         self.terminal_chars)
            self.index_to_non_term[n] = string.upper(term)
            self.non_term_to_index[string.upper(term)] = n
            n += 1
        list_of_keys = self.index_to_non_term.keys()
        list_of_keys.sort()
        temp = []
        temp_2 = {}
        for i, key in enumerate(list_of_keys):
            temp.append(self.index_to_non_term[key])
            if key in self.index_to_score:
                temp_2[i] = self.index_to_score[key]
        self.index_to_non_term = temp
        self.index_to_score = temp_2
        
    def create_root_rule(self):
        self.map_rules()
        terminal_parsing_counts = {}
        for value in self.terminal_parsing.values():
            for rule in value.split(' '):
                if rule not in self.rules:
                    continue
                if rule not in terminal_parsing_counts:
                    terminal_parsing_counts[rule] = 0
                terminal_parsing_counts[rule] += 1
        total_weight = float(sum(terminal_parsing_counts.values()))
        weight_list = []
        rule_list = []
        for key, value in terminal_parsing_counts.iteritems():
            left_rule, right_rule = self.rules[key].split('-')
            if (left_rule in self.rules) and (right_rule in self.rules):
                left_rule = self.non_term_to_index[left_rule]
                right_rule = self.non_term_to_index[right_rule]
                rule_list.append([left_rule, right_rule])
                weight_list.append(value / total_weight)
        total_weight = sum(weight_list)
        for i, w in enumerate(weight_list):
            weight_list[i] = w / total_weight
        self.root_rule = Sto_rule(0,
                                  weight_list,
                                  rule_list,
                                  [],
                                  [])
        
    def create_grammar(self):
        list_of_rules = []
        list_of_rules.append(self.root_rule)
        for rule_name, rule_content in self.preterminal_rules.iteritems():
            list_of_rules.append(rule_content)
        for rule_name, rule_content in self.rules.iteritems():
            left_rule, right_rule = rule_content.split('-')
            if left_rule not in self.rules:
                left_rule = self.non_term_to_index[string.upper(left_rule)]
            else:
                left_rule = self.non_term_to_index[left_rule]
            if right_rule not in self.rules:
                right_rule = self.non_term_to_index[string.upper(right_rule)]
            else:
                right_rule = self.non_term_to_index[right_rule]
            rule_name = self.non_term_to_index[rule_name]
            list_of_rules.append(Sto_rule(int(rule_name),
                                          [1.0],
                                          [[left_rule, right_rule]],
                                          [],
                                          []))
        self.grammar = SCFG(list_of_rules, 0)
        self.grammar.blurr_A()
                            
        
        
        