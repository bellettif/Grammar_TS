'''
Created on 5 mai 2014

@author: francois
'''

import numpy as np
import os
import cPickle as pickle
from matplotlib import pyplot as plt

import data_colors

target_folder = 'Observed_grammars/'

def build_grammar_dict(folder, filter_crit):
    list_of_files = filter(lambda x : filter_crit in x,
                           os.listdir(target_folder))
    grammar_dict = {}
    for file_name in list_of_files:
        complete_path = target_folder + file_name
        grammar = pickle.load(open(complete_path, 'rb'))
        grammar_dict[file_name.split('.')[0]] = grammar
    return grammar_dict

def reduce_rules(grammar_dict):
    rules = {}
    for current_dict in grammar_dict.values():
        for lhs, rule in current_dict.iteritems():
            if lhs == '0': continue
            rhs = rule['rhs']
            pop_count = rule['pop_count']
            refs = rule['refs']
            if lhs not in rules:
                rules[lhs] = {'lhs' : lhs,
                              'rhs' : rhs,
                              'pop_count' : pop_count,
                              'refs' : refs}
            else:
                rules[lhs]['pop_count'] += pop_count
                rules[lhs]['refs'] += refs
    return rules

def plot_rules(dict_of_rules, title, target_file):
    flat_rules = sorted(dict_of_rules.values(), key = (lambda x : -x['pop_count']))
    pop_counts = [[x['lhs'], x['pop_count']] for x in flat_rules]
    pops = [x[1] for x in pop_counts]
    labels = [filter(lambda i : (i != '<') and (i != '>'), x[0]) for x in pop_counts]
    plt.bar(np.arange(len(pops)), pops, align = 'center', color = 'lightblue')
    plt.xticks(np.arange(len(pops)), labels, rotation = 70, fontsize = 4)
    plt.ylabel('Popularity of rule')
    plt.title(title)
    plt.savefig(target_file, dpi = 300)
    plt.close()
    
def plot_multiple_rules(dict_of_grammar, colors, 
                        data_set_name, target_file):
    array_of_labels = [[x['lhs'] for x in y.values()] for y in dict_of_grammar.values()]
    all_labels = []
    current_index = 0
    for array in array_of_labels:
        all_labels.extend(array)
    all_labels = filter(lambda x : x != '0', all_labels)
    all_labels = list(set(all_labels))
    cum_count_labels = {}
    for label in all_labels:
        cum_count_labels[label] = 0
        for grammar in dict_of_grammar.values():
            if label in grammar:
                cum_count_labels[label] += grammar[label]['pop_count']
    all_labels = sorted(all_labels, key = (lambda x : -cum_count_labels[x]))
    bottom_counts = np.zeros(len(all_labels))
    for name, grammar in dict_of_grammar.iteritems():
        pop_counts = []
        for current_label in all_labels:
            if current_label not in grammar:
                pop_counts.append(0)
            else:
                pop_counts.append(grammar[current_label]['pop_count'])
        pop_counts = np.asarray(pop_counts)
        plt.bar(np.arange(len(all_labels)),
                pop_counts,
                bottom = bottom_counts,
                align = 'center',
                color = colors[name])
        bottom_counts += pop_counts
    all_labels = [filter(lambda i : (i != '>') and (i != '<'), x) for x in all_labels]
    plt.xticks(np.arange(len(all_labels)), all_labels, rotation = 70, fontsize = 4)
    plt.ylabel('Popularity of rule')
    plt.title('Rule popularity in %s' % data_set_name)
    plt.legend(dict_of_grammar.keys())
    plt.savefig(target_file, dpi = 300)
    plt.close()
'''
achu_grammars = build_grammar_dict(target_folder, 'achu')
oldo_grammars = build_grammar_dict(target_folder, 'oldo')

achu_rules = reduce_rules(achu_grammars)
oldo_rules = reduce_rules(oldo_grammars)
     
plot_rules(achu_rules, 'achuSeq reduced', 'Achu_grammars_reduced.png')
plot_rules(oldo_rules, 'oldoSeq reduced', 'Oldo_grammars_reduced.png')

plot_multiple_rules(achu_grammars, data_colors.achu_colors, 'achuSeq set', 'Achu_grammars.png')
plot_multiple_rules(oldo_grammars, data_colors.oldo_colors, 'oldoSeq set', 'Oldo_grammars.png')
'''
