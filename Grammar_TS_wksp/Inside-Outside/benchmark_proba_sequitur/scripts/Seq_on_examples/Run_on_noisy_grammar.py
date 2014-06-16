'''
Created on 16 juin 2014

@author: francois
'''

import numpy as np

from proba_sequitur.Proba_sequitur import Proba_sequitur
from Grammar_examples.Grammar_examples import noisy_grammar


samples = noisy_grammar.produce_sentences(1000)
samples = filter(lambda x : len(x) > 600, samples)

print len(samples)
print [len(x) for x in samples]


ps = Proba_sequitur(samples,
                    samples,
                    6,
                    30,
                    True,
                    0.0,
                    0.0,
                    0.05)

ps.run()

absolute_counts = ps.relative_counts
absolute_counts_items = absolute_counts.items()
absolute_counts_items.sort(key = (lambda x : -sum(x[1].values())))

print absolute_counts

for x in absolute_counts_items:
    print 'Lhs %.2f %s' % (sum(x[1].values()), x[0])