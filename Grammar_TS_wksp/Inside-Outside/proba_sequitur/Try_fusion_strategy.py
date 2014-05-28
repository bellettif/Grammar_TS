'''
Created on 28 mai 2014

@author: francois
'''


from benchmarks.grammar_examples import *

grammar_1_sentences = grammar_1.produce_sentences(500)
grammar_1_sentences = [' '.join(x) for x in grammar_1_sentences]

palindrom_sentences = palindrom_grammar.produce_sentences(500)
palindrom_sentences = [' '.join(x) for x in palindrom_sentences]

action_sentences = action_grammar.produce_sentences(500)
action_sentences = [' '.join(x) for x in action_sentences]

all_terms_in_grammar_1 = []
for sentence in grammar_1_sentences:
    all_terms_in_grammar_1.extend(sentence.split(' '))
    
term_set = set(all_terms_in_grammar_1)
term_set_frequencies = {}
for term in term_set:
    term_set_frequencies[term] = len(filter(lambda x : x == term, all_terms_in_grammar_1))

term_probas = {}
for term, counts in term_set_frequencies.iteritems():
    term_probas[term] = counts / float(sum(term_set_frequencies.values()))

print all_terms_in_grammar_1
print term_set
print term_set_frequencies
print term_probas