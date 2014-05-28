'''
Created on 26 mai 2014

@author: francois
'''

from matplotlib import pyplot as plt
import numpy as np

from proba_sequitur.Proba_sequitur import Proba_sequitur
from Grammar_folder import Grammar_folder

from benchmarks.grammar_examples import *
from benchmarks.learning_rate_analyst import Learning_rate_analyst

selected_grammar = palindrom_grammar
n_samples = 100

sentences = selected_grammar.produce_sentences(n_samples)
sentences = [' '.join(x) for x in sentences]
sentences = filter(lambda x : len(x) > 2, sentences)

selected_grammar = sentences

proba_seq = Proba_sequitur(selected_grammar, True)
proba_seq.infer_grammar(10)
proba_seq.print_result()

proba_seq.create_root_rule()
proba_seq.create_grammar()
proba_seq.grammar.blurr_A()

print proba_seq.grammar.index_to_non_term
print proba_seq.index_to_score
print proba_seq.grammar.non_term_to_index
print proba_seq.grammar.index_to_term
print proba_seq.grammar.term_to_index

samples = [x.split(' ') for x in sentences]

grammar_folder = Grammar_folder(proba_seq.grammar,
                                samples,
                                5)

grammar_folder.iterate()

grammar_folder.to_merge = [3, 5]
grammar_folder.merge()

grammar_folder.iterate()

all_log_lks = np.vstack(grammar_folder.all_lks)

print all_log_lks

plt.plot(all_log_lks)
plt.show()


"""
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
"""

