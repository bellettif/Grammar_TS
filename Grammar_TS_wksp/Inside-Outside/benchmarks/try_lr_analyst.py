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

from grammar_examples import *

main_grammar = palindrom_grammar

lr_analyst = Learning_rate_analyst(main_grammar,
                                   20)

log_lks, A, B = lr_analyst.compute_learning_rate('perturbated',
                                           50,
                                           1.0)

avg_log_lks = np.average(log_lks, axis = 1)

print avg_log_lks

print lr_analyst.compute_squared_diff_model(A, B)

plt.plot(avg_log_lks, color = 'b')
plt.plot(range(len(avg_log_lks)),
         np.ones(len(avg_log_lks)) * lr_analyst.exact_lk,
         color = 'r')
plt.title('Learning rate inside outside')
plt.savefig('Learning_rate_palindrom_grammar_perturbated.png', dpi = 300)
plt.ylabel('Avg Log likelood')
plt.xlabel('N iterations')
plt.close()

