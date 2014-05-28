'''
Created on 26 mai 2014

@author: francois
'''

from matplotlib import pyplot as plt
import numpy as np

from Proba_sequitur import Proba_sequitur

from benchmarks.grammar_examples import *
from benchmarks.learning_rate_analyst import Learning_rate_analyst

grammar_1_sentences = grammar_1.produce_sentences(100)
grammar_1_sentences = [' '.join(x) for x in grammar_1_sentences]
grammar_1_sentences = filter(lambda x : len(x) > 2, grammar_1_sentences)

palindrom_sentences = palindrom_grammar.produce_sentences(100)
palindrom_sentences = [' '.join(x) for x in palindrom_sentences]
palindrom_sentences = filter(lambda x : len(x) > 2, palindrom_sentences)

action_sentences = action_grammar.produce_sentences(100)
action_sentences = [' '.join(x) for x in action_sentences]
action_sentences = filter(lambda x : len(x) > 2, action_sentences)

proba_seq = Proba_sequitur(palindrom_sentences, True)
proba_seq.infer_grammar(6)
proba_seq.print_result()

proba_seq.create_root_rule()
proba_seq.create_grammar()
proba_seq.grammar.blurr_A()

print proba_seq.grammar.index_to_non_term
print proba_seq.index_to_score
print proba_seq.grammar.non_term_to_index
print proba_seq.grammar.index_to_term
print proba_seq.grammar.term_to_index

lr_analyst = Learning_rate_analyst(proba_seq.grammar,
                                   200,
                                   [x.split(' ') for x in palindrom_sentences])

print 'Coucou'

log_lks, A, B = lr_analyst.compute_learning_rate('proposal',
                                                 50,
                                                 1.0,
                                                 proba_seq.grammar.A,
                                                 proba_seq.grammar.B)

plt.plot(log_lks)
plt.show()


