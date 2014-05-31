'''
Created on 30 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG
from stochastic_grammar_wrapper import SCFG_c as SCFG_c

from grammar_examples import *

import time

target_grammar = palindrom_grammar
A = target_grammar.A
B = target_grammar.B

N_sentences = 100
N_grammars = 100

min_y = -70

terminals = list(palindrom_grammar.index_to_term)

sentences = target_grammar.produce_sentences(N_sentences)

#plt.hist([len(x) for x in sentences])
#plt.show()

sentences = filter(lambda x : len(x) < 22, sentences)

print len(sentences)

time.sleep(1)

def produce_random_A_and_B(A_shape, B_shape, option, sigma = 0):
    if option == 'perturbated':
        A_proposal = A + np.random.normal(0.0, sigma, A_shape)
        B_proposal = B + np.random.normal(0.0, sigma, B_shape)
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

N = target_grammar.A.shape[0]
M = target_grammar.B.shape[1]
A_shape = target_grammar.A.shape
B_shape = target_grammar.B.shape



A_B_proposals = [produce_random_A_and_B(A_shape,
                                        B_shape, 'perturbated', 0.01)
                 for i in xrange(N_grammars)]
all_lks_random = []
for i, (A_proposal, B_proposal) in enumerate(A_B_proposals):
    print i
    all_lks_random.append(np.log(SCFG_c.estimate_likelihoods(A_proposal,
                                                  B_proposal,
                                                  terminals,
                                                  sentences)))
all_lks_random = np.asanyarray(all_lks_random)
plt.subplot(221)
plt.title("0")
plt.ylim((min_y, 0))
plt.plot(all_lks_random)

reduced_A_shape = (N - 1, N - 1, N - 1)
reduced_B_shape = (N - 1, M)

A_B_proposals = [produce_random_A_and_B(reduced_A_shape,
                                        reduced_B_shape, 'random')
                 for i in xrange(N_grammars)]
all_lks_perturbated = []
for i, (A_proposal, B_proposal) in enumerate(A_B_proposals):
    print i
    all_lks_perturbated.append(np.log(SCFG_c.estimate_likelihoods(A_proposal,
                                                  B_proposal,
                                                  terminals,
                                                  sentences)))
all_lks_perturbated = np.asanyarray(all_lks_perturbated)
plt.subplot(222)
plt.title("-1")
plt.ylim((min_y, 0))
plt.plot(all_lks_perturbated)

reduced_A_shape = (N + 1, N + 1, N + 1)
reduced_B_shape = (N + 1, M)
A_B_proposals = [produce_random_A_and_B(reduced_A_shape,
                                        reduced_B_shape, 'random')
                 for i in xrange(N_grammars)]
all_lks_perturbated = []
for i, (A_proposal, B_proposal) in enumerate(A_B_proposals):
    print i
    all_lks_perturbated.append(np.log(SCFG_c.estimate_likelihoods(A_proposal,
                                                  B_proposal,
                                                  terminals,
                                                  sentences)))
all_lks_perturbated = np.asanyarray(all_lks_perturbated)
plt.subplot(223)
plt.title("+1")
plt.ylim((min_y, 0))
plt.plot(all_lks_perturbated)

reduced_A_shape = (N, N , N)
reduced_B_shape = (N, M)
A_B_proposals = [produce_random_A_and_B(reduced_A_shape,
                                        reduced_B_shape,
                                        'perturbated',
                                        0.1)
                 for i in xrange(N_grammars)]
all_lks_perturbated = []
for i, (A_proposal, B_proposal) in enumerate(A_B_proposals):
    print i
    all_lks_perturbated.append(np.log(SCFG_c.estimate_likelihoods(A_proposal,
                                                  B_proposal,
                                                  terminals,
                                                  sentences)))
all_lks_perturbated = np.asanyarray(all_lks_perturbated)
plt.subplot(224)
plt.title("0 perturbated 0.1")
plt.ylim((min_y, 0))
plt.plot(all_lks_perturbated)
plt.show()
