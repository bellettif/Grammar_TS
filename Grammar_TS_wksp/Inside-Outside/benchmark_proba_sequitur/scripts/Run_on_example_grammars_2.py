'''
Created on 17 juin 2014

@author: francois
'''

import numpy as np

from proba_sequitur.Proba_sequitur import Proba_sequitur
from proba_sequitur.Proba_seq_merger import Proba_seq_merger
from Grammar_examples.Grammar_examples import word_grammar, word_grammar_noisy

samples = word_grammar_noisy.produce_sentences(1000)

for sample in samples:
    print sample
    
print [len(x) for x in samples]

n_trials = 1000

psm = Proba_seq_merger()

for i in xrange(n_trials):
    ps = Proba_sequitur(inference_samples = samples,
                        count_samples = samples, 
                        degree = 6, 
                        max_rules = 30, 
                        random = True, 
                        init_T = 0.01, 
                        T_decay = 0.1, 
                        p_deletion = 0.05)
    ps.run()
    psm.merge_with(ps)


absolute_counts = psm.absolute_counts
absolute_count_items = absolute_counts.items()

depths = psm.depths

absolute_count_items.sort(key = (lambda x : -sum([sum(y.values()) for y in x[1]])))

for depth in range(2, 10):
    for x in filter(lambda x : depths[x[0]] == depth, absolute_count_items):
        print psm.rule_to_hashcode[x[0]] + ' ' + str(sum([sum(y.values()) for y in x[1]]))
    print '\n'