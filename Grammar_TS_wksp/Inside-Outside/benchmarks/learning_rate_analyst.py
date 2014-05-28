'''
Created on 25 mai 2014

@author: francois
'''

import stochastic_grammar_wrapper.SCFG_c

import numpy as np
from matplotlib import pyplot as plt

from surrogate.sto_rule import Sto_rule
from surrogate.sto_grammar import SCFG

class Learning_rate_analyst:
    
    def __init__(self,
                 grammar,
                 n_samples = 0,
                 samples = 0):
        self.grammar = grammar
        self.n_samples = n_samples
        if samples == 0:
            self.samples = grammar.produce_sentences(self.n_samples)
        else:
            self.samples = samples
            self.n_samples = len(samples)
        self.exact_lk = np.average(self.compute_all_lks('actual'))
        self.model_A = np.copy(grammar.A)
        self.model_B = np.copy(grammar.B)
        
    # Proposal_type in ['actual', 'flat', 'perturbated', 'matrix']
    def compute_all_lks(self,
                        proposal_type,
                        sigma = 1.0,
                        A_proposal = 0,
                        B_proposal = 0):
        log_lks = []
        if (proposal_type == 'actual'):
            for sample in self.samples:
                E, F, log_lk = self.grammar.compute_inside_outside(sample,
                                                                   self.grammar.A,
                                                                   self.grammar.B)
                log_lks.append(log_lk)
        elif (proposal_type == 'random'):
            A = np.ones(self.grammar.A.shape) + np.random.normal(0.0, 1.0, self.grammar.A.shape)
            A = np.maximum(A, np.zeros(self.grammar.A.shape))
            B = np.ones(self.grammar.B.shape) + np.random.normal(0.0, 1.0, self.grammar.B.shape)
            B = np.maximum(B, np.zeros(self.grammar.B.shape))
            for i in xrange(A.shape[0]):
                total = np.sum(A[i, :, :]) + np.sum(B[i, :])
                A[i, :, :] /= total
                B[i, :] /= total
            for sample in self.samples:
                E, F, log_lk = self.grammar.compute_inside_outside(sample,
                                                                   A,
                                                                   B)
                log_lks.append(log_lk)
        elif (proposal_type == 'perturbated'):
            A = np.copy(self.grammar.A)
            B = np.copy(self.grammar.B)
            A += np.random.normal(0.0, sigma, A.shape)
            B += np.random.normal(0.0, sigma, B.shape)
            A = np.maximum(A, np.ones(A.shape) * sigma * 1e-5)
            B = np.maximum(B, np.ones(B.shape) * sigma * 1e-5)
            for i in xrange(A.shape[0]):
                total = np.sum(A[i, :, :]) + np.sum(B[i, :])
                A[i, :, :] /= total
                B[i, :] /= total
            for sample in self.samples:
                E, F, log_lk = self.grammar.compute_inside_outside(sample,
                                                                   A,
                                                                   B)
                log_lks.append(log_lk)
        elif (proposal_type == 'matrix'):
            A_shapes = self.grammar.A.shape
            B_shapes = self.grammar.B.shape
            assert (A_shapes[0] == A_proposal.shape[0])
            assert (A_shapes[1] == A_proposal.shape[1])
            assert (A_shapes[2] == A_proposal.shape[2])
            assert (B_shapes[0] == B_proposal.shape[0])
            assert (B_shapes[1] == B_proposal.shape[1])
            for sample in self.samples:
                E, F, log_lk = self.grammar.compute_inside_outside(sample,
                                                                   A_proposal,
                                                                   B_proposal)
                log_lks.append(log_lk)
        else:
            raise Exception('Invalid proposal_type, must be actual, flat, perturbated, or matrix')
        return np.asarray(log_lks, dtype = np.double)
    
    #init_type in ['actual', 'flat', 'perturbated']
    def compute_learning_rate(self,
                              init_type,
                              n_iterations,
                              sigma = 0,
                              A = 0,
                              B = 0):
        if (init_type == 'actual'):
            A = np.copy(self.grammar.A)
            B = np.copy(self.grammar.B)
        elif (init_type == 'random'):
            A = np.ones(self.grammar.A.shape) + np.random.normal(0.0, 1.0, self.grammar.A.shape)
            A = np.maximum(A, np.zeros(self.grammar.A.shape))
            B = np.ones(self.grammar.B.shape) + np.random.normal(0.0, 1.0, self.grammar.B.shape)
            B = np.maximum(B, np.zeros(self.grammar.B.shape))
            for i in xrange(A.shape[0]):
                total = np.sum(A[i, :, :]) + np.sum(B[i, :])
                A[i, :, :] /= total
                B[i, :] /= total
        elif (init_type == 'perturbated'):
            A = np.copy(self.grammar.A)
            B = np.copy(self.grammar.B)
            A += np.random.normal(0.0, sigma, A.shape)
            B += np.random.normal(0.0, sigma, B.shape)
            A = np.maximum(A, np.ones(A.shape) * sigma * 1e-5)
            B = np.maximum(B, np.ones(B.shape) * sigma * 1e-5)
            for i in xrange(A.shape[0]):
                total = np.sum(A[i, :, :]) + np.sum(B[i, :])
                A[i, :, :] /= total
                B[i, :] /= total
        elif (init_type == 'proposal'):
            pass
        else:
            raise Exception('Invalid init type, must be either actual, flat or perturbated')
        log_lks = np.zeros((n_iterations, self.n_samples))
        for i in xrange(n_iterations):
            print 'Iteration ' + str(i)
            log_lks[i, :] = self.compute_all_lks('matrix',
                                                  0,
                                                  A, B)
            A, B = self.grammar.estimate_model(self.samples, 1, A, B)
        return log_lks, A, B
    
    def compute_squared_diff_model(self,
                           A,
                           B):
        A_shapes = A.shape
        B_shapes = B.shape
        first_weight = float(A_shapes[0] * A_shapes[1] * A_shapes[2])
        second_weight = float(B_shapes[0] * B_shapes[1])
        total_weight = first_weight + second_weight
        num = np.average((A - self.model_A) ** 2) * first_weight
        num += np.average((B - self.model_B) ** 2) * second_weight
        return num / total_weight
