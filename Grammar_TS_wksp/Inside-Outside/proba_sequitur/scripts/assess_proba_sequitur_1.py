'''
Created on 26 mai 2014

@author: francois
'''

from matplotlib import pyplot as plt
import numpy as np

from Old_Proba_sequitur import Proba_sequitur

from benchmarks.grammar_examples import *
from benchmarks.learning_rate_analyst import Learning_rate_analyst

import load_data

def assess_proba_sequitur(input_sentences, title):
    for repetitions in [True, False]:
        reconstructed_ratios = []
        n_rules = []
        for k in range(2, 15):
            proba_seq = Proba_sequitur(input_sentences, repetitions)
            proba_seq.infer_grammar(k)
            print proba_seq.rules
            reconstructed_ratios.append(proba_seq.reconstructed_ratio)
            n_rules.append(len(proba_seq.rules))
        if repetitions:
            my_color = 'b'
        else:
            my_color = 'r'
        """
        plt.plot(reconstructed_ratios, n_rules, linestyle = '--', marker = 'o', color =  my_color)
        for i, k in enumerate(range(2, 15)):
            plt.text(reconstructed_ratios[i] - 0.5 * np.std(reconstructed_ratios), n_rules[i], 'k= ' + str(k), color = my_color)
        """
    """
    plt.title(str(title) + ' (k is max number of distinct rules)')
    plt.xlabel('Percentage of strings explained')
    plt.ylabel('Number of rules')
    plt.legend(('With repetitions', 'Without repetitions'), 'upper left')
    plt.savefig('../Benchmark_results/Proba_sequitur/%s.png' % title, dpi = 300)
    plt.close()
    """
        

grammar_1_sentences = grammar_1.produce_sentences(100)
grammar_1_sentences = [' '.join(x) for x in grammar_1_sentences]
grammar_1_sentences = filter(lambda x : len(x) > 2, grammar_1_sentences)

palindrom_sentences = palindrom_grammar.produce_sentences(100)
palindrom_sentences = [' '.join(x) for x in palindrom_sentences]
palindrom_sentences = filter(lambda x : len(x) > 2, palindrom_sentences)

action_sentences = action_grammar.produce_sentences(100)
action_sentences = [' '.join(x) for x in action_sentences]
action_sentences = filter(lambda x : len(x) > 2, action_sentences)

assess_proba_sequitur(action_sentences, 'Action sentences')

assess_proba_sequitur(palindrom_sentences, 'Palindrom sentences')

assess_proba_sequitur(grammar_1_sentences, 'Grammar_1 sentences')

assess_proba_sequitur(load_data.achu_file_contents.values(), 'Achu files')

assess_proba_sequitur(load_data.filtered_achu_file_contents.values(), 'Filtered achu files')

assess_proba_sequitur(load_data.oldo_file_contents.values(), 'Oldo files')

assess_proba_sequitur(load_data.filtered_oldo_file_contents.values(), 'Filtered oldo files')
