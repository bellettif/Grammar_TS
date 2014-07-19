'''
Created on 18 juil. 2014

@author: francois
'''

import numpy as np
import cPickle as pickle

from SCFG.sto_grammar import SCFG, normalize_slices
from grammar_examples.grammar_examples import produce_word_grammar

from matplotlib import pyplot as plt

def compute_signature(target_grammar, grammar_name, n_samples):
    sentences = target_grammar.produce_sentences(n_samples)
    sentences = [' '.join(x) for x in sentences]
    # Empirical frequency
    count_dict = {}
    for sentence in sentences:
        if sentence not in count_dict:
            count_dict[sentence] = 0
        count_dict[sentence] += 1
    total_count = float(sum(count_dict.values()))
    for key, value in count_dict.iteritems():
        count_dict[key] = float(value) / total_count
    histogram = count_dict.items()
    histogram.sort(key = (lambda x : -x[1]))
    histogram = histogram[:100]
    width = 0.8
    plt.bar(range(len(histogram)), [x[1] for x in histogram],
            width = width)
    plt.xticks(width*0.5 + np.arange(len(histogram)),
               [x[0] for x in histogram],
               rotation = 'vertical',
               fontsize = 4)
    plt.ylabel('Sentence frequency')
    plt.title('%s frequentist signature' % grammar_name)
    plt.savefig('%s frequentist signature.png' % grammar_name, 
                dpi = 600)
    plt.close()
    #Likelihood
    sentence_set = list(set(sentences))
    sentence_set = [x.split(' ') for x in sentence_set]
    lks = target_grammar.estimate_likelihoods(sentence_set)
    sentence_set = [' '.join(x) for x in sentence_set]
    histogram = [(sentence_set[i], lks[i]) for i in xrange(len(lks))]
    histogram.sort(key = (lambda x : -x[1]))
    histogram = histogram[:100]
    width = 0.8
    plt.bar(range(len(histogram)), [x[1] for x in histogram],
            width = width)
    plt.xticks(width*0.5 + np.arange(len(histogram)),
               [x[0] for x in histogram],
               rotation = 'vertical',
               fontsize = 4)
    plt.ylabel('Sentence likelihood')
    plt.title('%s bayesian signature' % grammar_name)
    plt.savefig('%s bayesian signature lk.png' % grammar_name,
                dpi = 600)
    plt.close()

n_samples = 1e4
probas = [0.5, 0.5, 0.5, 0.5]
probas = np.asarray(probas)
print probas
target_grammar, rule_nick_names = produce_word_grammar(probas[0],
                                                       probas[1],
                                                       probas[2],
                                                       probas[3])
compute_signature(target_grammar, 'Word grammar 1', n_samples)

probas = [0.35, 0.65, 0.65, 0.35]
probas = np.asarray(probas)
print probas
target_grammar, rule_nick_names = produce_word_grammar(probas[0],
                                                       probas[1],
                                                       probas[2],
                                                       probas[3])
compute_signature(target_grammar, 'Word grammar 2', n_samples)

probas = [0.65, 0.35, 0.35, 0.65]
probas = np.asarray(probas)
print probas
target_grammar, rule_nick_names = produce_word_grammar(probas[0],
                                                       probas[1],
                                                       probas[2],
                                                       probas[3])
compute_signature(target_grammar, 'Word grammar 3', n_samples)

probas = [0.49, 0.51, 0.49, 0.51]
probas = np.asarray(probas)
print probas
target_grammar, rule_nick_names = produce_word_grammar(probas[0],
                                                       probas[1],
                                                       probas[2],
                                                       probas[3])
compute_signature(target_grammar, 'Word grammar 4', n_samples)

n_samples = 1e4
probas = [0.5, 0.5, 0.5, 0.5]
probas = np.asarray(probas)
print probas
target_grammar, rule_nick_names = produce_word_grammar(probas[0],
                                                       probas[1],
                                                       probas[2],
                                                       probas[3])
target_grammar.rotate(0)
compute_signature(target_grammar, 'Word grammar root', n_samples)
target_grammar.rotate(1)
compute_signature(target_grammar, 'Word grammar NP', n_samples)
target_grammar.rotate(5)
compute_signature(target_grammar, 'Word grammar JJNN', n_samples)

