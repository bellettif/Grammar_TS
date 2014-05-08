'''
Created on 8 mai 2014

@author: francois
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from surrogate.Markov_model import Markov_model

class Proba_computer:
    
    def __init__(self, initial, A, B, alphabet):
        self.initial = np.asarray(initial, dtype = np.double)
        self.initial /= np.sum(self.initial)
        self.A = np.asanyarray(A, dtype = np.double)
        sums = np.sum(self.A, axis = 1)
        for i, sum in enumerate(sums):
            self.A[i] /= sum
        self.B = np.asanyarray(B)
        sums = [np.sum(x) for x in self.B]
        for i, sum in enumerate(sums):
            self.B[i] /= sum
        self.alphabet = alphabet
        self.reversed_alphabet = {}
        for i, x in enumerate(alphabet):
            self.reversed_alphabet[x] = i
        self.n_states = len(self.initial)
        self.n_letters = len(self.alphabet)
            
    def compute_b(self, obs):
        return self.B[:,self.reversed_alphabet[obs]]
            
    def compute_forward_probas(self, data):
        alphas = np.zeros((self.n_states, len(data)))
        alphas[:,0] = self.initial * self.compute_b(data[0])
        alphas[:,0] /= np.sum(alphas[:,0])
        for t, datum in enumerate(data):
            if t == 0: continue
            alphas[:,t] = np.dot(alphas[:,t-1].T, self.A) * self.compute_b(datum)
            alphas[:,t] /= np.sum(alphas[:,t])
        return alphas
    
    def compute_backward_probas(self, data):
        betas = np.zeros((self.n_states, len(data)))
        betas[:,-1] = np.ones(self.n_states)
        betas[:,-1] /= np.sum(betas[:,-1])
        for t, datum in enumerate(data[::-1]):
            if t == (len(data) - 1): break
            betas[:,-(t+2)] = np.dot(self.A, betas[:,-(t+1)] * self.compute_b(datum))
            betas[:,-(t+2)] /= np.sum(betas[:,-(t+2)])
        return betas
    
    def compute_probas(self, data):
        gammas = self.compute_backward_probas(data) * self.compute_forward_probas(data)
        for i in xrange(gammas.shape[1]):
            gammas[:,i] /= np.sum(gammas[:,i])
        return gammas
        
        
        
        
