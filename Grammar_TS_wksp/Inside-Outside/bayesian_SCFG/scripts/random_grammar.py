'''
Created on 16 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
from Grammar_examples.Grammar_examples import word_grammar

from SCFG.sto_grammar import SCFG

sentences = word_grammar.produce_sentences(100)

N = word_grammar.N + word_grammar.M
M = len(word_grammar.term_chars)

print word_grammar.term_chars

new_A, new_B, likelihoods = word_grammar.estimate_A_B(samples = sentences, 
                                                      n_iterations = 50, 
                                                      init_option = 'random',
                                                      A_proposal = np.zeros((N, N, N)),
                                                      B_proposal = np.zeros((N, M)),
                                                      term_chars = word_grammar.term_chars,
                                                      noise_source_A = 0,
                                                      param_1_A = 0.1,
                                                      param_2_A = 0.1,
                                                      epsilon_A = 0,
                                                      noise_source_B = 0,
                                                      param_1_B = 0.1,
                                                      param_2_B = 0,
                                                      epsilon_B = 0)
word_grammar_estim = SCFG()
word_grammar_estim.init_from_A_B(new_A, new_B, word_grammar.term_chars)

estim_sentences = word_grammar.produce_sentences(100)

for sentence in estim_sentences:
    print sentence

"""
plt.plot(likelihoods)
plt.show()
"""