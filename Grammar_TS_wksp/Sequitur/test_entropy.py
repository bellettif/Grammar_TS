'''
Created on 17 avr. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import csv


def measure_entropy(input):
    values = {}
    for x in input:
        if x not in values:
            values[x] = 0
        values[x] += 1
    N = float(len(input))
    probas = []
    for x, n_x in values.iteritems():
        probas.append(float(n_x) / N)
    probas = np.asarray(probas, dtype = np.float)
    return - np.sum(probas * np.log2(probas))

def measure_gini(input):
    values = {}
    for x in input:
        if x not in values:
            values[x] = 0
        values[x] += 1
    N = float(len(input))
    probas = []
    for x, n_x in values.iteritems():
        probas.append(float(n_x) / N)
    probas = np.asarray(probas, dtype = np.float)
    return np.sum(probas * (1.0 - probas))

def measures(input):
    values = {}
    for x in input:
        if x not in values:
            values[x] = 0
        values[x] += 1
    N = float(len(input))
    probas = []
    for x, n_x in values.iteritems():
        probas.append(float(n_x) / N)
    probas = np.asarray(probas, dtype = np.float)
    return {'gini' : np.sum(probas * (1.0 - probas)),
            'entropy' : - np.sum(probas * np.log2(probas))} 

N = 10000

probas = np.linspace(0, 1, 100)
samples = [np.random.binomial(1, x, N) for x in probas]
measures = [measures(x) for x in samples]

plt.plot(probas, [x['gini'] for x in measures], c = 'b')
plt.plot(probas, [x['entropy'] for x in measures] , c = 'g')
plt.xlabel('Bernouilli p')
plt.ylabel('Entropy and Gini')
plt.legend(('Gini', 'Entropy'), 'upper right')
plt.savefig('Entropy and Gini bernouilli.png')
plt.close()

