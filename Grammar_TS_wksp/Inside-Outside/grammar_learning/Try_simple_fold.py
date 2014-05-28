'''
Created on 28 mai 2014

@author: francois
'''

from benchmarks.grammar_examples import *

from surrogate.sto_grammar import SCFG

selected_grammar = action_grammar

sentence = selected_grammar.produce_sentences(1)[0]

A = selected_grammar.A
B = selected_grammar.B

print len(A)

to_merge = [3, 4]
term_chars = selected_grammar.index_to_term

merged_grammar = SCFG(A = A,
                      B = B,
                      to_merge = to_merge,
                      term_chars = term_chars)

print sentence

print selected_grammar.compute_inside_outside(sentence,
                                              selected_grammar.A,
                                              selected_grammar.B)

print merged_grammar.compute_inside_outside(sentence,
                                            merged_grammar.A,
                                            merged_grammar.B)

