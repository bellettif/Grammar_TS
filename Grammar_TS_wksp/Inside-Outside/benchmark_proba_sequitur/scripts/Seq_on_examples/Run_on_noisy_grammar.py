'''
Created on 16 juin 2014

@author: francois
'''

from proba_sequitur.Proba_sequitur import Proba_sequitur
from Grammar_examples.Grammar_examples import noisy_grammar


samples = noisy_grammar.produce_sentences(100)

ps = Proba_sequitur(samples,
                    samples,
                    6,
                    30,
                    True,
                    0.1,
                    0.1,
                    0.05)

ps.run()

absolute_counts = ps.absolute_counts
absolute_counts_items = absolute_counts.items()
absolute_counts_items.sort(key = (lambda x : -sum(x[1])))

for x in absolute_counts_items:
    print 'Lhs ' + x[0]