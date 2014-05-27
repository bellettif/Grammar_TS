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

for main_grammar, title in [[grammar_1, 'simple_grammar'], 
                            [action_grammar, 'action_grammar'],
                            [palindrom_grammar, 'palindrom_grammar']]:
    lr_analyst = Learning_rate_analyst(main_grammar,
                                       20)
    all_log_lks = []
    for i in xrange(100):
        log_lks, A, B = lr_analyst.compute_learning_rate('random',
                                                   50,
                                                   1.0)
        all_log_lks.append(log_lks)
    log_lks = np.hstack(all_log_lks)
    avg_log_lks = np.median(log_lks, axis = 1)
    upper_log_lks = np.percentile(log_lks, 95, axis = 1)
    lower_log_lks = np.percentile(log_lks, 5, axis = 1)
    lr_analyst.compute_squared_diff_model(A, B)
    plt.plot(avg_log_lks, color = 'b')
    plt.plot(upper_log_lks, color = 'r', linestyle = '--')
    plt.plot(lower_log_lks, color = 'r', linestyle = '--')
    plt.plot(range(len(avg_log_lks)),
             np.ones(len(avg_log_lks)) * lr_analyst.exact_lk,
             color = 'k')
    plt.title('Learning rate inside outside')
    plt.legend(('Median log lk estim', '95 perc.', '5 perc.', 'Exact log lk'), 'lower right')
    plt.ylabel('Median Log likelood')
    plt.xlabel('N iterations')
    plt.savefig('Learning_rate_%s_random.png' % title, dpi = 300)
    plt.close()

