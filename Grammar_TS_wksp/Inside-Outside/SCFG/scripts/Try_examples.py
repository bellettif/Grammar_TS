'''
Created on 3 juin 2014

@author: francois
'''

from Grammar_examples import repetition_grammar
from Grammar_examples import embedding_grammar_central
from Grammar_examples import embedding_grammar_left_right
from Grammar_examples import CSExample

sentences = CSExample.produce_sentences(100)

for sentence in sentences:
    print sentence