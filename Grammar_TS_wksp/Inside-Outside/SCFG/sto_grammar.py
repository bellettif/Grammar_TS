'''
Created on 20 mai 2014

@author: francois
'''

import SCFG_c
import numpy as np


def normalize_slices(A, B):
    assert(A.ndim == 3)
    assert(B.ndim == 2)
    assert(A.shape[0] == A.shape[1] == A.shape[2] == B.shape[0])
    for i in xrange(A.shape[0]):
        total = np.sum(A[i,:,:]) + np.sum(B[i,:])
        A[i,:,:] /= total
        B[i,:] /= total
    return A, B


class SCFG:
    
    def __init__(self):
        self.term_chars = []
        self.term_char_to_index = {}
        self.A = np.zeros((0, 0, 0))
        self.B = np.zeros((0, 0))
        self.N = 0
        self.M = 0
        #
        self.rules_mapped = False
        self.rules = {}
        
    # Rule_dict[int] = (list of int pairs, list of weights, list of terms, list_of_weights)
    def init_from_rule_dict(self, rule_dict):
        self.N = len(rule_dict)
        assert all(rule_dict.keys().sort() == range(self.N))
        #
        #    Grabbing all terminal characters
        #
        all_terms = []
        for list_of_pairs, list_of_weights, list_of_terms, list_of_term_weights in rule_dict.values():
            all_terms.extend(list_of_terms)
            for left, right in list_of_pairs:
                assert(left in rule_dict)
                assert(right in rule_dict)
        self.term_chars = list(set(all_terms))
        assert(len(self.term_chars) > 0)
        self.M = len(self.term_chars)
        for i, term_char in enumerate(self.term_chars):
            self.term_char_to_index[term_char] = i
        #
        #    Building A and B matrices
        #
        self.A = np.zeros((self.N, self.N, self.N))
        self.B = np.zeros((self.N, self.M))
        for i in xrange(self.N):
            list_of_pairs, list_of_weights, list_of_terms, list_of_term_weights = rule_dict[i]
            assert(len(list_of_pairs) == len(list_of_weights))
            assert(len(list_of_terms) == len(list_of_term_weights))
            for l in xrange(len(list_of_pairs)):
                left, right = list_of_pairs[l]
                weight = list_of_weights[l]
                self.A[i, left, right] = weight
            for l in xrange(len(list_of_terms)):
                term = list_of_terms[l]
                weight = list_of_term_weights[l]
                self.B[i, self.term_char_to_index[term]] = weight               
        
    def init_from_A_B(self, A, B, term_chars):
        assert(len(term_chars) > 0)
        assert(A.ndim == 3)
        assert(B.ndim == 2)
        assert(A.shape[0] == A.shape[1] == A.shape[2] == B.shape[0])
        assert(B.shape[1] == len(term_chars))
        self.N = A.shape[0]
        self.M = B.shape[1]
        self.A = np.copy(A)
        self.B = np.copy(B)
        self.term_chars = term_chars
        for i, term in enumerate(self.term_chars):
            self.term_char_to_index[term] = i
            
    def produce_sentences(self, n_sentences, max_length = 0):
        if max_length == 0:
            return SCFG_c.produce_sentences(self.A,
                                            self.B,
                                            self.term_chars,
                                            n_sentences)
        else:
            return filter(lambda x : len(x) < max_length,
                          SCFG_c.produce_sentences(self.A,
                                            self.B,
                                            self.term_chars,
                                            n_sentences))
            
    def estimate_likelihoods(self,
                             samples,
                             A_proposal = 0,
                             B_proposal = 0,
                             term_chars = []):
        assert(len(samples) > 0)
        if(A_proposal == 0 or B_proposal == 0 or len(term_chars) == 0):
            assert(A_proposal == 0 and B_proposal == 0 and len(term_chars) == 0)
            return SCGF_c.estimate_likelihoods(self.A,
                                        self.B,
                                        self.term_chars,
                                        samples)
        else:
            assert(A_proposal.ndim == 3)
            assert(B_proposal.ndim == 2)
            assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
            assert(B_proposal.shape[1] == len(term_chars))
            return SCGF_c.estimate_likelihoods(A_proposal,
                                               B_proposal,
                                               term_chars,
                                               samples)
         
    #    
    #    Init option = exact, perturbated (need to give perturbation options),
    #            perturbated_keep_zeros, explicit
    #    Returns c_new_A, c_new_B, likelihoods
    #
    def estimate_A_B(self,
                     samples,
                     n_iterations,
                     init_option = 'exact',
                     A_proposal = 0,
                     B_proposal = 0,
                     term_chars = [],
                     noise_source_A = 0,
                     param_1_A = 0,
                     param_2_A = 0,
                     epsilon_A = 0,
                     noise_source_B = 0,
                     param_1_B = 0,
                     param_2_B = 0,
                     epsilon_B = 0):
        assert(init_option in ['exact', 'perturbated', 'perturbated_keep_zeros', 'explicit'])
        assert(n_iterations > 0)
        assert(len(samples) > 0)
        if init_option == 'exact':
            return SCFG_c.iterate_estimation(self.A,
                                             self.B,
                                             self.term_chars,
                                             samples,
                                             n_iterations)
        if init_option == 'perturbated':
            A_proposal = self.A + noise_source_A(param_1_A, param_2_A, (self.N, self.N, self.N))
            A_proposal = np.maximum(A_proposal, epsilon_A * np.ones((self.N, self.N, self.N)))
            B_proposal = self.B + noise_source_B(param_1_B, param_2_B, (self.N, self.M))
            B_proposal = np.maximum(B_proposal, epsilon_B * np.ones((self.N, self.M)))
            normalize_slices(A_proposal, B_proposal)
            return SCFG_c.iterate_estimation(A_proposal,
                                             B_proposal,
                                             self.term_chars,
                                             samples,
                                             n_iterations)
        if init_option == 'perturbated_keep_zeros':
            A_proposal = self.A + noise_source_A(param_1_A, param_2_A, (self.N, self.N, self.N))
            A_proposal = np.maximum(A_proposal, epsilon_A * np.ones((self.N, self.N, self.N)))
            B_proposal = self.B + noise_source_B(param_1_B, param_2_B, (self.N, self.M))
            B_proposal = np.maximum(B_proposal, epsilon_B * np.ones((self.N, self.M)))
            A_proposal[np.where(self.A == 0)] = 0
            B_proposal[np.where(self.B == 0)] = 0
            normalize_slices(A_proposal, B_proposal)
            return SCFG_c.iterate_estimation(A_proposal,
                                             B_proposal,
                                             self.term_chars,
                                             samples,
                                             n_iterations)
        if init_option == 'explicit':
            if len(term_chars) == 0:
                term_chars = self.term_chars
            assert(A_proposal.ndim == 3)
            assert(B_proposal.ndim == 2)
            assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
            assert(B_proposal.shape[1] == len(term_chars))
            return SCFG_c.iterate_estimation(A_proposal,
                                             B_proposal,
                                             term_chars,
                                             samples,
                                             n_iterations)
            