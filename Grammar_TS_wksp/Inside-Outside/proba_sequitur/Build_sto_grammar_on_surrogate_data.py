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

proba_seq = Proba_sequitur(action_sentences, True)
proba_seq.infer_grammar(10)
proba_seq.print_result()
#proba_seq.create_sto_grammar()
print '\n'

"""
my_lr_analyst = Learning_rate_analyst(proba_seq.grammar,
                                      100)

log_lks, A, B = my_lr_analyst.compute_learning_rate('perturbated',
                                                    100,
                                                    1.0)

plt.plot(np.average(log_lks, axis = 1))
plt.show()
"""