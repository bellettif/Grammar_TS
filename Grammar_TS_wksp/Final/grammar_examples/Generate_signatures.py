'''
Created on 9 juin 2014

@author: francois
'''

from matplotlib import pyplot as plt

import numpy as np

import cPickle as pickle


from SCFG.grammar_distance import Grammar_distance

from Grammar_examples import palindrom_grammar_1, \
                             palindrom_grammar_2, \
                             palindrom_grammar_3, \
                             repetition_grammar_1, \
                             repetition_grammar_2, \
                             embedding_grammar_central_1, \
                             embedding_grammar_central_2, \
                             embedding_grammar_left_right_1, \
                             embedding_grammar_left_right_2, \
                             name_grammar_1, \
                             name_grammar_2, \
                             action_grammar_1, \
                             action_grammar_2

all_grammars = {'palindrom_grammar_1' : palindrom_grammar_1,
               'palindrom_grammar_2' : palindrom_grammar_2,
               'palindrom_grammar_3' : palindrom_grammar_3,
               'repetition_grammar_1' : repetition_grammar_1,
               'repetition_grammar_2' : repetition_grammar_2,
               'embedding_grammar_central_1' : embedding_grammar_central_1,
               'embedding_grammar_central_2' : embedding_grammar_central_2,
               'embedding_grammar_left_right_1' : embedding_grammar_left_right_1,
               'embedding_grammar_left_right_2' : embedding_grammar_left_right_2,
               'name_grammar_1' : name_grammar_1,
               'name_grammar_2' : name_grammar_2,
               'action_grammar_1' : action_grammar_1,
               'action_grammar_2' : action_grammar_2}

n_samples = 100000
max_represented = 100


def generate_signature(grammar):
    sentences = [' '.join(x) for x in
                 grammar.produce_sentences(n_samples)]
    sentence_set = set(sentences)
    counts = []
    for s in sentence_set:
        counts.append([s, len(filter(lambda x : x == s, sentences))])
    counts.sort(key = (lambda x : -x[1]))
    plt.subplot(211)
    plt.bar(range(len(counts)), [x[1] for x in counts])
    plt.xticks(range(len(counts)), [x[0] for x in counts], rotation = 'vertical', fontsize = 4)
    plt.subplot(212)
    plt.bar(range(len(counts)), [len(x[0].split(' ')) for x in counts])
    plt.xticks(range(len(counts)), [x[0] for x in counts], rotation = 'vertical', fontsize = 4)
    plt.show()
    
generate_signature(name_grammar_1)
    
    
    