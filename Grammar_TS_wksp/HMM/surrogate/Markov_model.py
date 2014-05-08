'''
Created on 8 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from Obs_generator import Obs_generator

class Markov_model:
    
    def __init__(self, initial, A, B, alphabet):
        self.initial = np.asarray(initial, dtype = np.double)
        self.initial /= np.sum(self.initial)
        self.A = np.asanyarray(A, dtype = np.double)
        sums = np.sum(self.A, axis = 1)
        for i, sum in enumerate(sums):
            self.A[i] /= sum
        #map(lambda x : x / np.sum(x), self.A)
        self.B = B
        map(lambda x : x / np.sum(x), self.B)
        self.generators = [Obs_generator(alphabet, weights) for weights in B]
        self.states = range(len(self.generators))
        self.current_state = np.random.choice(self.states, p = self.initial)
        
    def gen_obs(self):
        return self.generators[self.current_state].gen_obs()
    
    def transition(self):
        old_state = self.current_state
        self.current_state = np.random.choice(self.states, p = self.A[self.current_state])
        return self.current_state
        
        
initial = [0.4, 0.8, 0.9]
A = [[0.1, 0.8, 0.1], [0.1, 0.1, 0.9], [0.9, 0.1 , 0.1]]
alphabet = [-1, -2, -3]
B = [[1.0, 0.5, 0.2], [0.2, 1.0, 0.5], [0.5, 0.2, 1.0]]
        
my_markov_model = Markov_model(initial, A, B, alphabet)


results = {}
for i in range(1000):
    current_state = my_markov_model.current_state
    if current_state not in results:
        results[current_state] = []
    results[current_state].append(my_markov_model.gen_obs())
    my_markov_model.transition()

for latent_state, current_obs in results.iteritems():
    plt.hist(current_obs)
    plt.title(latent_state)
    plt.show()
    

            
        
        