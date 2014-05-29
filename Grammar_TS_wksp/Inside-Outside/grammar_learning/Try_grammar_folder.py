'''
Created on 26 mai 2014

@author: francois
'''

from matplotlib import pyplot as plt
import numpy as np

from proba_sequitur.Proba_sequitur import Proba_sequitur
from Grammar_folder import Grammar_folder
import proba_sequitur.load_data as load_data

from benchmarks.grammar_examples import *
from benchmarks.learning_rate_analyst import Learning_rate_analyst

selected_grammar = palindrom_grammar
n_samples = 50

sentences = selected_grammar.produce_sentences(n_samples)
sentences = [' '.join(x) for x in sentences]
sentences = filter(lambda x : len(x) > 2, sentences)

selected_samples = sentences

proba_seq = Proba_sequitur(load_data.oldo_file_contents.values(), True)
proba_seq.infer_grammar(6)
proba_seq.print_result()

proba_seq.create_root_rule()
proba_seq.create_grammar()
proba_seq.grammar.blurr_A()

print proba_seq.grammar.index_to_non_term
print proba_seq.index_to_score
print proba_seq.grammar.non_term_to_index
print proba_seq.grammar.index_to_term
print proba_seq.grammar.term_to_index

samples = [x.split(' ') for x in sentences]

step_size = 20

title = 'Oldo_data_set'

grammar_folder = Grammar_folder(proba_seq.grammar,
                                samples,
                                step_size)

max_size = len(proba_seq.grammar.grammar)

for i in xrange(max_size):
    grammar_folder.iterate()
    print grammar_folder.to_merge
    grammar_folder.merge()
    all_log_lks = np.vstack(grammar_folder.all_lks)
    min_y = np.min(all_log_lks)
    max_y = np.max(all_log_lks)
    print all_log_lks
    n = len(grammar_folder.sto_grammar.grammar)
    plt.plot(all_log_lks)
    plt.vlines(step_size * np.asarray(range(i + 1)), min_y, max_y)
    plt.xlabel('Iterations')
    plt.ylabel('Log lks')
    plt.title('Estimated %s grammar (%d rules)' % (title, n))
    plt.savefig("Mergin_strategy_%s_%d.png" % (title, i))
    plt.close()


"""
lr_analyst = Learning_rate_analyst(proba_seq.grammar,
                                   200,
                                   [x.split(' ') for x in palindrom_sentences])

print 'Coucou'

log_lks, A, B = lr_analyst.compute_learning_rate('proposal',
                                                 50,
                                                 1.0,
                                                 proba_seq.grammar.A,
                                                 proba_seq.grammar.B)

plt.plot(log_lks)
plt.show()
"""

