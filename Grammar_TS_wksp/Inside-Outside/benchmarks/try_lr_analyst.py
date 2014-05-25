'''
Created on 25 mai 2014

@author: francois
'''

import stochastic_grammar_wrapper.SCFG_c

import numpy as np
from matplotlib import pyplot as plt

from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG
from learning_rate_analyst import Learning_rate_analyst

rule_1 = Sto_rule(1,
                  [0.2, 0.3, 0.1, 0.4],
                  [[1, 1], [1, 2], [2, 1], [2, 2]],
                  [2.4, 0.1, 0.1],
                  ['Bernadette', 'Colin', 'Michel'])

rule_2 = Sto_rule(2,
                  [0.2, 0.3, 0.4, 0.1],
                  [[2, 2], [2, 1], [1, 2], [1, 1]],
                  [0.1, 2.0],
                  ['Pierre', 'Mathieu'])

rule_3 = Sto_rule(3,
                  [0.6, 0.5, 0.4, 0.8],
                  [[1, 1], [1, 2], [2, 1], [2, 2]],
                  [],
                  [])

main_grammar = SCFG([rule_1, rule_2, rule_3], 3)

lr_analyst = Learning_rate_analyst(main_grammar,
                                   20)

log_lks = lr_analyst.compute_learning_rate('perturbated',
                                           50,
                                           1.0)

avg_log_lks = np.average(log_lks, axis = 1)

print avg_log_lks

plt.plot(avg_log_lks, color = 'b')
plt.plot(range(len(avg_log_lks)),
         np.ones(len(avg_log_lks)) * lr_analyst.exact_lk,
         color = 'r')
plt.show()

