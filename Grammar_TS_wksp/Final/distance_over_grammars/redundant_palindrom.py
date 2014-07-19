'''
Created on 19 juil. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from SCFG.grammar_distance import compute_distance_matrix

from grammar_examples.grammar_examples import palindrom_grammar, \
                                                palindrom_rule_nick_names, \
                                                redundant_palindrom_grammar, \
                                                redundant_palindrom_rule_nick_names

n_samples = 1e5
                                         
target_grammar = palindrom_grammar
target_red_grammar = redundant_palindrom_grammar

target_grammar_name = 'Palindrom'
target_grammar_red_name = 'Redundant palindrom'

target_grammar_rule_names = palindrom_rule_nick_names
target_grammar_red_rule_names = redundant_palindrom_rule_nick_names

def plot_signature(target_grammar, grammar_name, n_samples):
    sentences = target_grammar.produce_sentences(n_samples)
    sentences = [''.join(x) for x in sentences]
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
    histogram = histogram[:25]
    width = 0.8
    plt.bar(range(len(histogram)), [x[1] for x in histogram],
            width = width)
    plt.xticks(width*0.5 + np.arange(len(histogram)),
               [x[0] for x in histogram],
               rotation = 'vertical',
               fontsize = 10)
    plt.ylabel('Sentence frequency')
    plt.title('%s frequentist signature' % grammar_name)
    plt.savefig('%s frequentist signature.png' % grammar_name)
    plt.close()
    #Likelihood
    sentence_set = list(set(sentences))
    lks = target_grammar.estimate_likelihoods(sentence_set)
    histogram = [(sentence_set[i], lks[i]) for i in xrange(len(lks))]
    histogram.sort(key = (lambda x : -x[1]))
    histogram = histogram[:25]
    width = 0.8
    plt.bar(range(len(histogram)), [x[1] for x in histogram],
            width = width)
    plt.xticks(width*0.5 + np.arange(len(histogram)),
               [x[0] for x in histogram],
               rotation = 'vertical',
               fontsize = 10)
    plt.ylabel('Sentence likelihood')
    plt.title('%s bayesian signature' % grammar_name)
    plt.savefig('%s bayesian signature lk.png' % grammar_name)
    plt.close()
    
def plot_dist_matrices(left_grammar,
                     left_grammar_name,
                     left_grammar_rule_names,
                     right_grammar,
                     right_grammar_name,
                     right_grammar_rule_names,
                     n_samples):
    dist_matrix_KL = compute_distance_matrix(left_grammar, 
                                              right_grammar, 
                                              n_samples, 
                                              JS = False)
    dist_matrix_JS = compute_distance_matrix(left_grammar,
                                             right_grammar,
                                             n_samples,
                                             JS = True)
    plt.imshow(dist_matrix_KL, cmap = 'PuOr',
               interpolation = 'None')
    plt.title('KL dist matrix %s vs %s' 
              % (left_grammar_name,
                 right_grammar_name))
    plt.xticks(range(dist_matrix_KL.shape[1]),
               right_grammar_rule_names,
               rotation = 'vertical',
               fontsize = 8)
    plt.yticks(range(dist_matrix_KL.shape[0]),
               left_grammar_rule_names,
               fontsize = 8)
    plt.colorbar()
    plt.savefig('KL_dist_%s_%s.png' % (left_grammar_name, right_grammar_name),
                dpi = 600)
    plt.close()
    #
    plt.imshow(dist_matrix_JS, cmap = 'PuOr',
               interpolation = 'None')
    plt.title('JS dist matrix %s vs %s' 
              % (left_grammar_name,
                 right_grammar_name))
    plt.xticks(range(dist_matrix_JS.shape[1]),
               right_grammar_rule_names,
               rotation = 'vertical',
               fontsize = 8)
    plt.yticks(range(dist_matrix_JS.shape[0]),
               left_grammar_rule_names,
               fontsize = 8)
    plt.colorbar()
    plt.savefig('JS_dist_%s_%s.png' % (left_grammar_name, right_grammar_name),
                dpi = 600)
    plt.close()
    
plot_signature(target_grammar,
               target_grammar_name,
               n_samples)

plot_signature(target_red_grammar,
               target_grammar_red_name,
               n_samples)

plot_dist_matrices(target_grammar,
                   target_grammar_name,
                   target_grammar_rule_names,
                   target_grammar,
                   target_grammar_name,
                   target_grammar_rule_names,
                   n_samples)

plot_dist_matrices(target_red_grammar,
                   target_grammar_red_name,
                   target_grammar_red_rule_names,
                   target_red_grammar,
                   target_grammar_red_name,
                   target_grammar_red_rule_names,
                   n_samples)

plot_dist_matrices(target_grammar,
                   target_grammar_name,
                   target_grammar_rule_names,
                   target_red_grammar,
                   target_grammar_red_name,
                   target_grammar_red_rule_names,
                   n_samples)