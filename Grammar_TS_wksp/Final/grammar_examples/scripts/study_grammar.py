'''
Created on 16 juil. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import re
import copy

from SCFG.grammar_distance import compute_distance, \
                                    compute_distance_MC, \
                                    compute_distance_matrix, \
                                    compute_distance_matrix_MC

def study_grammar(grammar_name, grammar_ex, 
                  rule_nick_names = {},
                  n_sentences = 10000,
                  n_samples = 10000):
    #
    #    Extract grammar parameters
    #
    term_chars = grammar_ex.term_chars
    n_term_chars = len(term_chars)
    grammar_ex.map_rules()
    rules = grammar_ex.rules
    n_rules = len(rules.keys())
    #
    #    Print example of sentences
    #
    example_sentences = grammar_ex.produce_sentences(n_sentences)
    all_pairs = [x + y for x in term_chars for y in term_chars]
    sentences = [''.join(x) for x in example_sentences]
    print '\r'.join(sentences[:20])
    #
    #    Draw grammar graph
    #
    grammar_ex.draw_grammar('%s.png' % grammar_name)
    #
    #    Print co-occurrence graph
    #
    co_occurrence_dict = {}
    for pair in all_pairs:
        if pair not in co_occurrence_dict:
            co_occurrence_dict[pair] = 0
        for sentence in sentences:
            co_occurrence_dict[pair] += len(re.findall(pair, sentence))
    #
    total = float(sum(co_occurrence_dict.values()))
    #
    co_oc_matrix = np.zeros((n_term_chars, n_term_chars))
    for i, left in enumerate(term_chars):
        for j, right in enumerate(term_chars):
            co_oc_matrix[i, j] = co_occurrence_dict[left + right] / total
    #              
    plt.matshow(co_oc_matrix, cmap = 'PuOr')
    plt.yticks(range(n_term_chars), term_chars)
    plt.xticks(range(n_term_chars), term_chars)
    plt.title('Co occurrences of pairs of symbols')
    plt.ylabel('Predecessor')
    plt.xlabel('Successor')
    plt.colorbar()
    plt.savefig('Co-occurrences %s example.png' % grammar_name,
                dpi = 600)
    plt.close()
    #
    #    Plot distance matrices
    #
    internal_distance_matrix = \
        grammar_ex.compute_internal_distance_matrix(n_samples = n_samples,
                                                           symmetric = False)
    
    internal_distance_matrix_sym = \
        grammar_ex.compute_internal_distance_matrix(n_samples = n_samples,
                                                           symmetric = True)
    #
    rule_names = [rule_nick_names[i] for i in xrange(n_rules)]
    #
    plt.matshow(internal_distance_matrix,
               cmap = 'PuOr')
    plt.xticks(range(n_rules), rule_names)
    plt.yticks(range(n_rules), rule_names)
    plt.title('Internal distance matrix KL %s' % grammar_name)
    plt.colorbar(orientation = 'horizontal')
    plt.savefig('Internal_dist_%s_KL.png' % grammar_name)
    plt.close()
    #
    plt.matshow(internal_distance_matrix_sym,
               cmap = 'PuOr')
    plt.xticks(range(n_rules), rule_names)
    plt.yticks(range(n_rules), rule_names)
    plt.title('Internal distance matrix JS %s' % grammar_name)
    plt.colorbar(orientation = 'horizontal')
    plt.savefig('Internal_dist_%s_JS.png' % grammar_name)
    plt.close()


