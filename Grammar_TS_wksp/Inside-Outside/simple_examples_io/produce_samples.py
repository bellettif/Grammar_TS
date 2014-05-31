'''
Created on 30 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG

from grammar_examples import *

target_grammar = palindrom_grammar
A = target_grammar.A
B = target_grammar.B

sentences = target_grammar.produce_sentences(100)
sentences = filter(lambda x : len(x) < 5, sentences)

def produce_random_A_and_B(A_shape, B_shape, option):
    if option == 'perturbated':
        A_proposal = A + np.random.normal(0.0, 0.1, A_shape)
        B_proposal = B + np.random.normal(0.0, 0.1, B_shape)
    elif option == 'random':
        A_proposal = np.random.uniform(0.0, 1.0, A_shape)
        B_proposal = np.random.uniform(0.0, 1.0, B_shape)
    A_proposal = np.maximum(A_proposal, 0.01 * np.ones(A_shape))
    B_proposal = np.maximum(B_proposal, 0.01 * np.ones(B_shape))
    for i in xrange(A_shape[0]):
        total = np.sum(A_proposal[i, :, :]) + np.sum(B_proposal[i, :])
        A_proposal[i, :, :] /= total
        B_proposal[i, :] /= total
    return A_proposal, B_proposal

A_shape = target_grammar.A.shape
B_shape = target_grammar.B.shape



A_B_proposals = [produce_random_A_and_B(A_shape,
                                        B_shape, 'random')
                 for i in xrange(20)]

all_lks = []
for i, (A_proposal, B_proposal) in enumerate(A_B_proposals):
    print i
    print A_proposal
    print B_proposal
    print '\n'
    all_lks.append(np.log(target_grammar.compute_probas_proposal(sentences,
                                                          A_proposal,
                                                          B_proposal)))
    
all_lks = np.asanyarray(all_lks)
print all_lks.shape

for i in range(len(all_lks)):
    plt.plot(all_lks[i,:])
plt.show()





