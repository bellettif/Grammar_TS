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
                 repetitions):
        self.current_rule_index = 1
        self.samples = samples
        self.terminal_parsing = {}
        self.rules = {}
        self.level = 0
        self.repetitions = repetitions
        self.counts = {}
        self.terminal_chars = []
        self.next_rule_name = 0
        self.barelk_table = {}

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

    def infer_grammar(self, degree):
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
            """
            print best_symbols
            rule_names = []
            print ''
            """
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
            """
            plt.bar(range(len(labels)), values, align = 'center', color = 'b')
            plt.xticks(range(len(labels)), labels, rotation = 'vertical', fontsize = 3)
            plt.show()
            print ''
            for seq in target_sequences:
                print seq + '/'
            print ''
            """
            temp_target_sequences = []
            for i, seq in enumerate(target_sequences):
                new_seq = seq.split(' ')
                new_seq = filter(lambda x : 'Rule' in x, new_seq)
                #new_seq = filter(lambda x : x in list_of_rules[-1], new_seq)
                if (self.level > 1):
                    pass
                                     #or x in list_of_rules[-2], new_seq)
                #else:
                #    new_seq = filter(lambda x : x in list_of_rules[-1], new_seq)
                new_seq = ' '.join(new_seq)
                new_seq = re.sub('\-', '', new_seq)
                new_seq = string.lower(new_seq)
                if len(new_seq) == 0:
                    self.terminal_parsing[i] = seq
                else:
                    temp_target_sequences.append(new_seq)
            target_sequences = temp_target_sequences
            """
            print len(target_sequences)
            print '\n-----------------------------\n'
            """
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
        rule_ranking = []
        for i in xrange(len(rule_counts)):
            rule_ranking.append([rule_counts[i][0], rule_counts[i][1], rule_items[i][1]])
        rule_ranking.sort(key = (lambda x : -x[1]))
        print 'Terminal parsing:'
        print self.terminal_parsing
        print 'Rules (%d), %d levels:' % (len(rule_ranking), self.level)
        print rule_ranking
        print 'Rules:'
        print self.rules
        print ''
  
    def create_root_rule(self):
        self.root_rule = {}
        terminal_weights = []
        for key in self.terminal_chars:
            terminal_weights.append(self.barelk_table[key])
        self.terminal_weights = np.asarray(terminal_weights)
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
            weight_list.append(value / total_weight)
            left_rule, right_rule = self.rules[key].split('-')
            if (left_rule in self.rules) and (right_rule in self.rules):
                left_rule = list(left_rule)[4:]
                right_rule = list(right_rule)[4:]
                left_rule = int(''.join(left_rule))
                right_rule = int(''.join(right_rule))
                rule_list.append([left_rule, right_rule])
        print rule_list
        self.root_rule = Sto_rule(0,
                                  weight_list,
                                  rule_list,
                                  self.terminal_weights * 0.01,
                                  self.terminal_chars)
    
    def create_basic_stochastic_rules(self):
        self.basic_rules = {}
        terminal_weights = []
        for key in self.terminal_chars:
            terminal_weights.append(self.barelk_table[key])
        self.terminal_weights = np.asarray(terminal_weights)
        self.basic_rules = {}
        for rule, value in self.rules.iteritems():
            left_rule, right_rule = value.split('-')
            if (left_rule in self.rules) and (right_rule in self.rules):
                left_rule = list(left_rule)[4:]
                right_rule = list(right_rule)[4:]
                left_rule = int(''.join(left_rule))
                right_rule = int(''.join(right_rule))
                rule_number = int(''.join(list(rule)[4:]))
                self.basic_rules[rule_number] = Sto_rule(rule_number,
                                                         [1.0],
                                                         [[left_rule, right_rule]],
                                                         self.terminal_weights,
                                                         self.terminal_chars)
                
    def complete_rules(self):
        self.other_rules = {}
        for key, value in self.basic_rules.iteritems():
            for non_terms in value.non_term_s:
                if non_terms[0] not in self.basic_rules:
                    rule_number = non_terms[0]
                    self.other_rules[non_terms[0]] = Sto_rule(rule_number,
                                                              [],
                                                              [],
                                                              self.terminal_weights,
                                                              self.terminal_chars)
                if non_terms[1] not in self.basic_rules:
                    rule_number = non_terms[1]
                    self.other_rules[non_terms[1]] = Sto_rule(rule_number,
                                                              [],
                                                              [],
                                                              self.terminal_weights,
                                                              self.terminal_chars)
        for non_terms in self.root_rule.non_term_s:
            if non_terms[0] not in self.basic_rules:
                rule_number = non_terms[0]
                self.other_rules[non_terms[0]] = Sto_rule(rule_number,
                                                          [],
                                                          [],
                                                          self.terminal_weights,
                                                          self.terminal_chars)
            if non_terms[1] not in self.basic_rules:
                rule_number = non_terms[1]
                self.other_rules[non_terms[1]] = Sto_rule(rule_number,
                                                          [],
                                                          [],
                                                          self.terminal_weights,
                                                          self.terminal_chars)
                
    def create_sto_grammar(self):
        print self.terminal_chars
        self.create_root_rule()
        self.create_basic_stochastic_rules()
        self.complete_rules()
        all_rules = [self.root_rule]
        all_rules.extend(self.basic_rules.values())
        all_rules.extend(self.other_rules.values())
        print set([x.rule_name for x in all_rules])
        self.grammar = SCFG(all_rules, 0)
        
        
        