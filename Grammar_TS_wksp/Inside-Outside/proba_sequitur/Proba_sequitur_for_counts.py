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
                 build_samples,
                 count_samples,
                 repetitions):
        self.current_rule_index = 1
        #
        self.build_samples = build_samples
        self.count_samples = count_samples
        self.all_counts = {}
        self.hashcode_to_rule = {}
        self.rule_to_hashcode = {}
        #
        self.terminal_parsing = {}
        self.rules = {}
        self.level = 0
        self.repetitions = repetitions
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
            temp_counts[string.lower(rule_names[k])] = {}
            for i, sequence in enumerate(sequences):
                sequences[i], c = re.subn(pattern, rule_names[k], sequence)
                temp_counts[string.lower(rule_names[k])][i] = c
        return sequences, temp_counts

    def infer_grammar(self, degree):
        self.level = 0
        self.current_rule_index = 1
        target_sequences = copy.deepcopy(self.build_samples)
        target_for_counts = copy.deepcopy(self.count_samples)
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
            rule_names = []
            for best_symbol in best_symbols:
                new_rule_name = 'rule%d' % self.current_rule_index
                self.rules[new_rule_name] = best_symbol
                #
                #
                #
                left_member, right_member = best_symbol.split('-')
                hash_code = ''
                if left_member in self.rule_to_hashcode:
                    hash_code += self.rule_to_hashcode[left_member]
                else:
                    hash_code += left_member
                hash_code += '_'
                if right_member in self.rule_to_hashcode:
                    hash_code += self.rule_to_hashcode[right_member]
                else:
                    hash_code += right_member
                self.rule_to_hashcode[new_rule_name] = hash_code
                self.hashcode_to_rule[hash_code] = new_rule_name
                #
                #
                #
                rule_names.append('Rule%d' % self.current_rule_index)
                self.current_rule_index += 1
            list_of_rules.append(rule_names)
            #
            #
            #
            target_for_counts, target_counts = self.substitute(target_for_counts, best_symbols, rule_names)
            for key, count_dict in target_counts.iteritems():
                for i, count in count_dict.iteritems():
                    if len(target_for_counts[i]) == 0:
                        continue
                    target_counts[key][i] = float(count) / float(len(target_for_counts[i]))
            for key, value in target_counts.iteritems():
                if key in self.all_counts:
                    print 'Key already in counts'
                self.all_counts[key] = value
            for i, seq in enumerate(target_for_counts):
                new_seq = seq.split(' ')
                new_seq = filter(lambda x : 'Rule' in x, new_seq)
                new_seq = ' '.join(new_seq)
                new_seq = re.sub('\-', '', new_seq)
                new_seq = string.lower(new_seq)
                target_for_counts[i] = new_seq
            #
            #
            #
            target_sequences, temp_counts = self.substitute(target_sequences, best_symbols, rule_names)
            for key, value in temp_counts.iteritems():
                if key not in self.counts:
                    self.counts[key] = 0
                self.counts[key] += sum(value.values())
            temp_target_sequences = []
            for i, seq in enumerate(target_sequences):
                new_seq = seq.split(' ')
                new_seq = filter(lambda x : 'Rule' in x, new_seq)
                new_seq = ' '.join(new_seq)
                new_seq = re.sub('\-', '', new_seq)
                new_seq = string.lower(new_seq)
                if self.level == 1:
                    self.reconstructed[i] = [x if 'Rule' in x else '*' for x in seq.split(' ')]
                    self.reconstructed_length[i] = 2 * len(filter(lambda x : 'Rule' in x, seq.split(' ')))
                if len(new_seq) == 0:
                    self.terminal_parsing[i] = seq
                else:
                    temp_target_sequences.append(new_seq)
            if self.level == 1:
                total_reconstructed_length = sum(self.reconstructed_length.values())
                total_length = sum([len(x) for x in self.build_samples])
                print 'Total reconstructured length = ' + str(total_reconstructed_length)
                print 'Total length = ' + str(total_length)
                self.reconstructed_ratio = float(total_reconstructed_length) / float(total_length)
            target_sequences = temp_target_sequences
        to_delete = []
        for key, value in self.counts.iteritems():
            if value == 0:
                to_delete.append(key)
        for key in to_delete:
            del self.counts[key]
            del self.rules[key]
            del self.all_counts[key]
            
    
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
        