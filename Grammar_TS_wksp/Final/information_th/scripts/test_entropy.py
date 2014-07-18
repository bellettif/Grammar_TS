'''
Created on 21 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

import information_th.measures as inf_th
                            
n_symbols = 10000

def build_random_bits(length,
                      p):
    return np.asarray(np.random.choice([0, 1], length, p = [1.0 - p, p]),
                      dtype = np.int32)

def build_random_words(length,
                       words = ['car', 'house', 'office'],
                       p_scheme = np.ones(3)):
    p_scheme = np.asarray(p_scheme, dtype = np.double)
    p_scheme /= np.sum(p_scheme)
    return np.random.choice(words, length, p = p_scheme)

print build_random_bits(1000, 0.8)
print build_random_words(1000)

print inf_th.compute_entropy(build_random_bits(1000, 0.8))

p_values = np.linspace(0, 
                       1.0, 
                       100, 
                       endpoint = True)

#
#    Compute entropy of sequences of bernoulli observations
#        with different values of p
#
atom_entropy_values = [inf_th.compute_entropy(
                          build_random_bits(n_symbols, x)
                          )
                       for x in p_values]
plt.title('Entropy measurements for digrams')
plt.plot(p_values, atom_entropy_values)
plt.xlabel('Value of p')
plt.ylabel('Entropy')
plt.show()

#
#    Compute entropy of consecutive digrams in sequences of bernoulli
#        generated observations with different values of p
#
digram_entropy_values = [inf_th.compute_n_gram_entropy(
                           build_random_bits(n_symbols, x), 2
                           )
                         for x in p_values]
plt.title('Entropy measurements for digrams')
plt.plot(p_values, digram_entropy_values)
plt.xlabel('Value of p')
plt.ylabel('Entropy')
plt.show()

#
#    Compute rolling entropy in a sequence of berbouilli generated
#        observations where p varies
#
sequence = np.asarray(list(build_random_bits(10000, 0.5)) +
                      list(build_random_bits(10000, 0.2)) +
                      list(build_random_bits(10000, 0.5)),
                      dtype = np.int32)
rolling_entropy_values = inf_th.compute_rolling_entropy(sequence, k = 100)
plt.plot(rolling_entropy_values)
plt.show()

print inf_th.compute_entropy(build_random_words(1000,
                                                ['bla', 'abl', 'bla'],
                                                [0.25, 0.5, 0.25]))