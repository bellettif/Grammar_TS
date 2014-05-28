'''
Created on 28 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from surrogate.sto_grammar import SCFG
from benchmarks.learning_rate_analyst import Learning_rate_analyst

class Grammar_folder:
    
    def __init__(self,
                 sto_grammar,
                 samples,
                 n_its_step):
        self.sto_grammar = sto_grammar
        self.samples = samples
        self.A = self.sto_grammar.A
        self.B = self.sto_grammar.B
        self.scores = []
        self.n_its_step = n_its_step
        self.lr_analyst = Learning_rate_analyst(sto_grammar,
                                                samples = samples)
        self.all_lks = []
        self.preterminal_indices = filter(lambda i : np.sum(self.A[i, :, :]) == 0, range(self.A.shape[0]))
        self.non_preterminal_indices = filter(lambda i : np.sum(self.A[i, :, :]) > 0, range(self.A.shape[0]))
                
    def iterate(self):
        print self.A
        mass = [np.sum(self.A[:,:,i] + self.A[:,i,:]) for i in xrange(len(self.A))]
        """
        plt.plot(mass)
        plt.savefig('Initial_grammar/mass.png')
        plt.close()
        """
        for i in self.non_preterminal_indices:
            first_matrix = np.copy(self.A[i, :])
            first_matrix -= np.min(self.A)
            first_matrix /= (np.max(self.A) - np.min(self.A))
            """
            plt.matshow(first_matrix)
            plt.savefig('Initial_grammar/outgoing_sym_%d.png' % i)
            plt.close()
            """
            second_matrix = self.A[:, i, :] + self.A[:, :, i]
            second_matrix -= np.min(self.A)
            second_matrix /= (np.max(self.A) - np.min(self.A))
            """
            plt.matshow(second_matrix)
            plt.savefig('Initial_grammar/incoming_sym_%d.png' % i)
            plt.close()
            """
        log_lks, self.A, self.B = self.lr_analyst.compute_learning_rate('proposal',
                                                                      self.n_its_step,
                                                                      sigma = 1.0, 
                                                                      A = self.A, 
                                                                      B = self.B)
        self.all_lks.append(log_lks)
        mass = [np.sum(self.A[:,:,i] + self.A[:,i,:]) for i in xrange(len(self.A))]
        """
        plt.plot(mass)
        plt.savefig('Estimated_grammar/mass.png')
        plt.close()
        """
        for i in self.non_preterminal_indices:
            first_matrix = np.copy(self.A[i, :])
            first_matrix -= np.min(self.A)
            first_matrix /= (np.max(self.A) - np.min(self.A))
            """
            plt.matshow(first_matrix)
            plt.savefig('Estimated_grammar/outgoing_sym_%d.png' % i)
            plt.close()
            """
            second_matrix = self.A[:, i, :] + self.A[:, :, i]
            second_matrix -= np.min(self.A)
            second_matrix /= (np.max(self.A) - np.min(self.A))
            """
            plt.matshow(second_matrix)
            plt.savefig('Estimated_grammar/incoming_sym_%d.png' % i)
            plt.close()
            """
        n = len(self.non_preterminal_indices)
        distances = np.zeros((n - 1, n - 1))
        for i in xrange(1, n):
            for j in  xrange(i + 1, n):
                left_index = self.non_preterminal_indices[i]
                right_index = self.non_preterminal_indices[j]
                distances[i - 1, j - 1] = np.average((self.A[left_index, :] - self.A[right_index, :]) ** 2)
        for i in xrange(distances.shape[0]):
            for j in xrange(distances.shape[1]):
                if distances[i, j] == 0:
                    distances[i, j] = np.inf
        """
        print distances
        plt.matshow(distances)
        plt.savefig('Distance.png')
        plt.close()
        """
        row_mins = np.argmin(distances, axis = 1)
        min_values = np.min(distances, axis = 1)
        col_min = np.argmin(min_values)
        self.to_merge = (row_mins[col_min], col_min)
        self.to_merge = [3, 4]
        
    def merge(self):
        self.sto_grammar = SCFG(A = self.A, B = self.B,
                                to_merge = self.to_merge, 
                                term_chars = self.sto_grammar.index_to_term)
        self.A = self.sto_grammar.A
        self.B = self.sto_grammar.B
        self.lr_analyst = Learning_rate_analyst(self.sto_grammar,
                                                samples = self.samples)
        self.preterminal_indices = filter(lambda i : np.sum(self.A[i, :, :]) == 0, range(self.A.shape[0]))
        self.non_preterminal_indices = filter(lambda i : np.sum(self.A[i, :, :]) > 0, range(self.A.shape[0]))
        
        
        