'''
Created on 26 mai 2014

@author: francois
'''

from matplotlib import pyplot as plt
import numpy as np

from Proba_sequitur_for_counts import Proba_sequitur

from benchmarks.grammar_examples import *
from benchmarks.learning_rate_analyst import Learning_rate_analyst

import load_data

import cPickle as pickle

def extract_grammar(sequences, sequences_for_counts, repetitions, degree, title):
    proba_seq = Proba_sequitur(sequences, sequences_for_counts, repetitions)
    proba_seq.infer_grammar(degree)
    hashcode_to_rule = proba_seq.hashcode_to_rule
    rule_to_hashcode = proba_seq.rule_to_hashcode
    rules = proba_seq.rules
    if repetitions:
        file_path = 'Achu_oldo_grammars/' + title + '_rep_' + str(degree) + '.pi'
    else:
        file_path = 'Achu_oldo_grammars/' + title + '_no_rep_' + str(degree) + '.pi'
    pickle.dump({'hashcode_to_rule' : hashcode_to_rule,
                 'rule_to_hashcode' : rule_to_hashcode,
                 'rules': rules,
                 'counts' : proba_seq.all_counts},
                 open(file_path, 'wb'))

title_to_data = {'achu' : load_data.achu_file_contents.values(),
                 'achu_filtered' : load_data.filtered_achu_file_contents.values(),
                 'oldo' : load_data.oldo_file_contents.values(),
                 'oldo_filtered' : load_data.filtered_oldo_file_contents.values(),
                 'achu_and_oldo' : load_data.achu_file_contents.values() \
                                  + load_data.oldo_file_contents.values(),
                 'achu_and_oldo_filtered' : load_data.filtered_achu_file_contents.values() \
                                            + load_data.filtered_oldo_file_contents.values()}

for repetitions in [True, False]:
    for degree in range(6, 9):
        for key_1, value_1 in title_to_data.iteritems():
            for key_2, value_2 in title_to_data.iteritems():
                print str(key_1) + str(key_2)
                print degree
                if repetitions:
                    print 'With repetitions'
                else:
                    print 'Without repetitions'
                extract_grammar(value_1, value_2, repetitions, degree, str(key_1) + '_count_' + str(key_2))