'''
Created on 26 mai 2014

@author: francois
'''

from Proba_sequitur import Proba_sequitur

from benchmarks.grammar_examples import *

grammar_1_sentences = grammar_1.produce_sentences(500)
grammar_1_sentences = [' '.join(x) for x in grammar_1_sentences]
grammar_1_sentences = filter(lambda x : len(x) > 2, grammar_1_sentences)

palindrom_sentences = palindrom_grammar.produce_sentences(500)
palindrom_sentences = [' '.join(x) for x in palindrom_sentences]
palindrom_sentences = filter(lambda x : len(x) > 2, palindrom_sentences)

action_sentences = action_grammar.produce_sentences(500)
action_sentences = [' '.join(x) for x in action_sentences]
action_sentences = filter(lambda x : len(x) > 2, action_sentences)

print 'Grammar_1 with repetitions'
proba_seq = Proba_sequitur(grammar_1_sentences, True)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Grammar_1 without repetitions'
proba_seq = Proba_sequitur(grammar_1_sentences, False)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Palindrom sentences with repetitions'
proba_seq = Proba_sequitur(palindrom_sentences, True)
proba_seq.infer_grammar(9)
proba_seq.print_result()
print '\n'

print 'Palindrom sentences without repetitions'
proba_seq = Proba_sequitur(palindrom_sentences, False)
proba_seq.infer_grammar(9)
proba_seq.print_result()
print '\n'

print 'Action grammar sentences with repetitions'
proba_seq = Proba_sequitur(action_sentences, True)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Action grammar sentences without repetitions'
proba_seq = Proba_sequitur(action_sentences, False)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'