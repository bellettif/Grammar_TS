'''
Created on 17 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

import information_th.entropy_stats as entropy_stats
from test_entropy import measure_entropy
   
sequence = np.random.binomial(1, 0.5, 1000)
sequence = np.asarray(sequence, dtype = np.int32)

probas = np.linspace(0, 1.0, 100)
data = [np.asarray(np.random.binomial(1, x, 100000), dtype = np.int32) for x in probas]

entropies = [entropy_stats.compute_string_entropy(line, 5) for line in data]
ginis = [entropy_stats.compute_string_gini(line, 5) for line in data]

entropies = [entropy_stats.compute_entropy(line) for line in data]
ginis = [entropy_stats.compute_gini(line) for line in data]

plt.title("Test information theory package (5 elt strings)")
plt.plot(probas, entropies, c ='g')
plt.plot(probas, ginis, c = 'b')
plt.xlabel('Bernoulli parameter')
plt.legend(('Entropy', 'Gini'))
plt.savefig('Plots/test_entropy_gini_5_strings.png', dpi = 300)
plt.close()
'''
data_1 = list(np.random.binomial(1, 0.7, 50000))
data_2 = list(np.random.binomial(1, 0.2, 25000))
data_3 = list(np.random.binomial(1, 0.5, 25000))
data = data_1 + data_2 + data_3
data = np.asarray(data, dtype = np.int32)
rolling_entropy = entropy_stats.compute_rolling_entropy(data, 100)

plt.vlines([50000, 75000], 0, 1, colors = 'red')
plt.plot(rolling_entropy)
plt.title('Rolling entropy, Bernoulli p = 0.7, 0.2, 0.7')
plt.ylabel('Rolling entropy (100 elt window)')
plt.xlabel('Index in time series')
plt.savefig('Plots/test_rolling_entropy.png', dpi = 300)
plt.close()
'''
#plt.plot(entropy_stats.compute_rolling_entropy(sequence, 8))
#plt.show()
