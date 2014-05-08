'''
Created on 8 mai 2014

@author: francois
'''

import numpy as np
from Obs_generator import Obs_generator

class Markov_model:
    
    def __init__(self, initial, A, B, alphabet):
        self.initial = np.asarray(initial, dtype = np.double)
        self.initial /= np.sum(self.initial)
        self.A = np.asanyarray(A, dtype = np.double)
        map(lambda x : x / np.sum(x), self.A)
        self.B = B
        map(lambda x : x / np.sum(x), self.B)
        self.generators = [Obs_generator(alphabet, weights) for weights in B]
        self.states = range(len(self.generators))
        self.current_state = np.random.choice(self.states, p = self.initial)
        
    def gen_obs(self):
        print 'Generating from ' + str(self.current_state)
        return self.generators[self.current_state].gen_obs()
        
        
initial = [0.4, 0.8, 0.9]
A = [[0.3, 0.3, 0.3], [0.9, 0.1, 0.1], [0.1, 0.2 , 0.6]]
alphabet = ['a', 'b', 'c']
B = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        
my_markov_model = Markov_model(initial, A, B, alphabet)

for i in range(10):
    print my_markov_model.gen_obs()

print 'Cool'
        
            
        
        