'''
Created on 22 mai 2014

@author: francois
'''

import numpy as np
import matplotlib.pyplot as plt
import re
import string
import copy
import time

MAX_RULES = 30

class Proba_sequitur:

    def __init__(self,
                 build_samples,
                 count_samples,
                 repetitions,
                 keep_data,
                 degree,
                 max_rules = MAX_RULES):
        self.current_rule_index = 1
        #
        self.build_samples = build_samples
        self.count_samples = count_samples
        self.all_counts = {}
        self.cumulated_counts = {}
        #
        self.terminal_parsing = {}
        self.rules = {}
        self.level = 0
        self.repetitions = repetitions
        self.counts = {}
        self.terminal_chars = []
        self.next_rule_name = 0
        self.barelk_table = {}
        #
        self.index_to_non_term = []
        self.non_term_to_index = {}
        #
        self.keep_data = keep_data
        self.degree = degree
        self.max_rules = max_rules
        #
        self.hash_codes = {}
        self.hashed_rules = {}
        self.hashed_counts = {}
        #
        self.total_divs = []
        self.lengths = []
        self.rule_divs = {}
        #
        self.rule_levels = {}

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
                    all_pairs):
        counts = {}
        for pair in all_pairs:
            symbol = pair
            pattern = re.sub('-', ' ', pair)
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
        if not self.repetitions:
            all_pairs = [x + '-' + y if x != y else None for x in candidates for y in candidates]
        else:
            all_pairs = [x + '-' + y for x in candidates for y in candidates]
        all_pairs = filter(lambda x : x != None, all_pairs)
        list_of_counts = []
        for sequence in sequences:
            if sequence == '': continue
            list_of_counts.append(self.pair_counts(sequence, all_pairs))
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
            temp_counts[string.lower(rule_names[k])] = {}
            for i, sequence in enumerate(sequences):
                if sequence == '': continue
                sequences[i], c = re.subn(pattern, rule_names[k], sequence)
                temp_counts[string.lower(rule_names[k])][i] = c
        return sequences, temp_counts
    
    def compute_hash_code(self,
                          symbol):
        left, right = symbol.split('-')
        if left in self.terminal_chars:
            left_hash = left
        else:
            left_hash = self.hash_codes[left]
        if right in self.terminal_chars:
            right_hash = right
        else:
            right_hash = self.hash_codes[right]
        return '>' + left_hash + '-' + right_hash + '<'

    def infer_grammar(self):
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
        print '\nStarting grammar inference, degree = %d' % (self.degree)
        self.lengths.append([len(x.split(' ')) for x in target_for_counts])
        while len(target_sequences) > 0 and len(self.rules) < self.max_rules:
            self.level += 1
            begin_time = time.clock()
            print '\tIteration %d, %d rules, %d words for build, %d words for count' % (self.level,
                                                                                        len(self.rules),
                                                                                        sum([len(x.split(' ')) for x in target_sequences]),
                                                                                        sum([len(x.split(' ')) for x in target_for_counts]))
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
            if self.degree != 0:
                best_symbols = labels[:self.degree]
            else:
                best_symbols_index = filter(lambda i : values[i] > 0, range(len(values)))
                best_symbols = labels[:len(best_symbols_index)]
            self.total_divs.append(sum(values[:self.degree]))
            list_of_best_symbols.append(best_symbols)
            rule_names = []
            for i, best_symbol in enumerate(best_symbols):
                new_rule_name = 'r%d_' % self.current_rule_index
                self.rules[new_rule_name] = best_symbol
                self.rule_levels[new_rule_name] = self.level
                hashcode = self.compute_hash_code(best_symbol)
                left, right = best_symbol.split('-')
                self.hash_codes[new_rule_name] = hashcode
                if left in self.terminal_chars:
                    left_hash_code = left
                else:
                    left_hash_code = self.hash_codes[left]
                if right in self.terminal_chars:
                    right_hash_code = right
                else:
                    right_hash_code = self.hash_codes[right]
                self.hashed_rules[hashcode] = left_hash_code + '_' + right_hash_code
                rule_names.append('r%d_' % self.current_rule_index)
                self.current_rule_index += 1
                self.rule_divs[new_rule_name] = values[i]
            list_of_rules.append(rule_names)
            #
            #
            #
            target_for_counts, target_counts = self.substitute(target_for_counts, best_symbols, rule_names)
            for key, count_dict in target_counts.iteritems():
                for i, count in count_dict.iteritems():
                    if len(target_for_counts[i]) == 0:
                        continue
                    target_counts[key][i] = float(count) / float(len(target_for_counts[i].split(' ')))
            for key, value in target_counts.iteritems():
                if key in self.all_counts:
                    print 'Key already in counts'
                self.all_counts[key] = value
                self.hashed_counts[key] = value
            for i, seq in enumerate(target_for_counts):
                if seq == '': continue
                to_delete = (len(filter(lambda x : x[0] == 'r' and x[-1] == '_', seq.split(' '))) == 0)
                if not self.keep_data:
                    new_seq = seq.split(' ')
                    new_seq = filter(lambda x : x[0] == 'r' and x[-1] == '_', new_seq)
                    new_seq = ' '.join(new_seq)
                else:
                    new_seq = seq
                if not to_delete:
                    target_for_counts[i] = new_seq
                else:
                    target_for_counts[i] = ''
            #
            #
            #
            target_sequences, temp_counts = self.substitute(target_sequences, best_symbols, rule_names)
            for key, value in temp_counts.iteritems():
                if key not in self.counts:
                    self.counts[key] = 0
                self.counts[key] += sum(value.values())
            for i, seq in enumerate(target_sequences):
                if seq == '': continue
                to_delete = (len(filter(lambda x : x[0] == 'r' and x[-1] == '_', seq.split(' '))) == 0)
                if not self.keep_data:
                    new_seq = seq.split(' ')
                    new_seq = filter(lambda x : x[0] == 'r' and x[-1] == '_', new_seq)
                    new_seq = ' '.join(new_seq)
                else:
                    new_seq = seq
                if to_delete:
                    self.terminal_parsing[i] = seq
                    target_sequences[i] = ''
                else:
                    target_sequences[i] = new_seq
            to_delete = []
            for key, value in self.counts.iteritems():
                if value == 0:
                    to_delete.append(key)
            for key in to_delete:
                del self.counts[key]
                del self.rules[key]
                del self.rule_divs[key]
                del self.all_counts[key]
                del self.rule_levels[key]
                del self.hashed_rules[self.hash_codes[key]]
                del self.hash_codes[key]
            self.lengths.append([len(x.split(' ')) for x in target_for_counts])
            print '\tTook %f seconds' % (time.clock() - begin_time)
        print 'Grammar inference done'
        for i, seq in enumerate(target_sequences):
            if seq == '': continue
            self.terminal_parsing[i] = seq
        for rule_name, count_dict in self.all_counts.iteritems():
            self.cumulated_counts[rule_name] = sum(count_dict.values())
        if len(target_sequences) == 0:
            print '\tNo target sequence remaining'
        else:
            print '\tMaximum number of rules (%d) reached' % self.max_rules
        print '\n'
            
    
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
        print ''
        
    def plot_stats(self, filename):
        achu_sub_set = range(9)
        oldo_sub_set = range(9, 18)
        #
        count_items = self.all_counts.items()
        count_items.sort(key = (lambda x : -sum(x[1].values())))
        #
        n_iterations = len(self.total_divs)
        n_rules = len(self.rule_divs)
        rule_div_items = self.rule_divs.items()
        rule_div_items.sort(key = (lambda x : -x[1]))
        #
        #    Plotting cumulated divergence
        #
        plt.subplot(311)
        plt.title('Divergence wrt iteration')
        plt.plot(range(1, n_iterations + 1),
                 self.total_divs)
        plt.ylabel('Cumulated divergence')
        #
        #    Plotting_lengths
        #
        plt.subplot(312)
        plt.title('Length wrt iteration')
        bp = plt.boxplot([[x[j] for j in achu_sub_set] for x in self.lengths],
                             notch=0, sym='+', vert=1, whis=1.5, patch_artist = True)
        plt.setp(bp['boxes'], color = 'r', facecolor = 'r', alpha = 0.25)
        plt.setp(bp['whiskers'], color='r')
        plt.setp(bp['fliers'], color='r', marker='+')
        bp = plt.boxplot([[x[j] for j in oldo_sub_set] for x in self.lengths],
                             notch=0, sym='+', vert=1, whis=1.5, patch_artist = True)
        plt.setp(bp['boxes'], color='b', facecolor = 'b', alpha = 0.25)
        plt.setp(bp['whiskers'], color='b')
        plt.setp(bp['fliers'], color='b', marker='+')
        plt.xticks(range(1, (len(self.lengths)) + 1), 
                   map(str, range(1, (len(self.lengths)) + 1)),
                   rotation = 'vertical', fontsize = 6)
        plt.ylabel('Length of parse')
        #
        #    Plotting individual contributions to divergence
        #
        plt.subplot(313)
        plt.title('Individual contributions to divergence')
        plt.plot(range(n_rules),
                 [x[1] for x in rule_div_items])
        plt.xticks(range(n_rules), 
                   [str(self.rule_levels[x[0]]) + ' ' + x[0] + '->' + self.rules[x[0]] for x in rule_div_items],
                   rotation = 'vertical',
                   fontsize = 6)
        plt.ylabel('Contribution to divergence')
        #
        #    Saving plot
        #
        fig = plt.gcf()
        fig.set_size_inches((16, 10))
        plt.savefig('Stats_%s.png' % filename, dpi = 300)
        plt.close()
        #
        #    Plotting relative use frequency
        #
        plt.title('Relative use frequency')
        bp = plt.boxplot([[count_items[i][1].values()[j] for j in achu_sub_set] for i in xrange(n_rules)],
                             notch=0, sym='+', vert=1, whis=1.5, patch_artist = True)
        plt.setp(bp['boxes'], color = 'r', facecolor = 'r', alpha = 0.25)
        plt.setp(bp['whiskers'], color='r')
        plt.setp(bp['fliers'], color='r', marker='+')
        bp = plt.boxplot([[count_items[i][1].values()[j] for j in oldo_sub_set] for i in xrange(n_rules)],
                             notch=0, sym='+', vert=1, whis=1.5, patch_artist = True)
        plt.setp(bp['boxes'], color='b', facecolor = 'b', alpha = 0.25)
        plt.setp(bp['whiskers'], color='b')
        plt.setp(bp['fliers'], color='b', marker='+')
        plt.xticks(range(1, n_rules + 1), [str(self.rule_levels[x[0]]) + ' ' + x[0] + '->' + self.rules[x[0]] for x in count_items], rotation = 'vertical', fontsize = 6)
        plt.ylabel('Frequencies, red for achu, blue for oldo')
        #
        #    Saving plot
        #
        fig = plt.gcf()
        fig.set_size_inches((16, 10))
        plt.savefig('Freqs_%s.png' % filename, dpi = 300)
        plt.close()
        