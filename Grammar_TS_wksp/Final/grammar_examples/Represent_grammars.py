'''
Created on 4 juin 2014

@author: francois
'''

from matplotlib import pyplot as plt

import numpy as np
import time

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

for grammar_name, grammar in all_grammars.iteritems():
    print 'Drawing %s' % grammar_name
    grammar.draw_grammar('Grammar_graphs/%s.png' % grammar_name)
    print 'Done\n'
    