'''
Created on 20 mai 2014

@author: francois
'''

import SCFG_c

import numpy as np
from matplotlib import pyplot as plt

import os
import shutil
import string

import time

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
        assert sorted(rule_dict.keys()) == range(self.N)
        #
        #    Grabbing all terminal characters
        #
        all_terms = []
        for list_of_pairs, list_of_weights, list_of_terms, list_of_term_weights in rule_dict.values():
            all_terms.extend(list_of_terms)
            for left, right in list_of_pairs:
                assert(left in rule_dict)
                assert(right in rule_dict)
        self.term_chars = sorted(list(set(all_terms)))
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
        normalize_slices(self.A, self.B)          
        
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
            return SCFG_c.estimate_likelihoods(self.A,
                                               self.B,
                                               self.term_chars,
                                               samples)
        else:
            assert(A_proposal.ndim == 3)
            assert(B_proposal.ndim == 2)
            assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
            assert(B_proposal.shape[1] == len(term_chars))
            return SCFG_c.estimate_likelihoods(A_proposal,
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
        assert(init_option in ['exact',
                               'perturbated',
                               'perturbated_keep_zeros',
                               'explicit',
                               'explicit_keep_zeros'])
        assert(n_iterations > 0)
        assert(len(samples) > 0)
        if init_option == 'exact':
            return SCFG_c.iterate_estimation(self.A,
                                             self.B,
                                             self.term_chars,
                                             samples,
                                             n_iterations)
        if init_option == 'perturbated':
            assert(A_proposal == 0 and B_proposal == 0 and len(term_chars) == 0)
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
            assert(A_proposal == 0 and B_proposal == 0 and len(term_chars) == 0)
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
            assert(noise_source_A == 0 and
                   param_1_A == 0 and
                   param_2_A == 0 and
                   epsilon_A == 0 and
                   noise_source_B == 0 and
                   param_1_B == 0 and
                   param_2_B == 0 and
                   epsilon_B == 0)
            if len(term_chars) == 0:
                term_chars = self.term_chars
            assert(A_proposal.ndim == 3)
            assert(B_proposal.ndim == 2)
            assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
            assert(B_proposal.shape[1] == len(term_chars))
            normalize_slices(A_proposal, B_proposal)
            return SCFG_c.iterate_estimation(A_proposal,
                                             B_proposal,
                                             term_chars,
                                             samples,
                                             n_iterations)
        if init_option == 'explicit_keep_zeros':
            assert(noise_source_A == 0 and
                   param_1_A == 0 and
                   param_2_A == 0 and
                   epsilon_A == 0 and
                   noise_source_B == 0 and
                   param_1_B == 0 and
                   param_2_B == 0 and
                   epsilon_B == 0)
            if len(term_chars) == 0:
                term_chars = self.term_chars
            assert(A_proposal.ndim == 3)
            assert(B_proposal.ndim == 2)
            assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
            assert(B_proposal.shape[1] == len(term_chars))
            A_proposal[np.where(self.A == 0)] = 0
            B_proposal[np.where(self.B == 0)] = 0
            normalize_slices(A_proposal, B_proposal)
            return SCFG_c.iterate_estimation(A_proposal,
                                             B_proposal,
                                             term_chars,
                                             samples,
                                             n_iterations)
            
    def plot_grammar_matrices(self,
                              folder_path,
                              folder_name,
                              A_matrix = np.zeros(0),
                              B_matrix = np.zeros(0)):
        if folder_name in os.listdir(folder_path):
            shutil.rmtree(folder_path + '/' + folder_name,
                          True)
        os.mkdir(folder_path + '/' + folder_name)
        if(len(A_matrix) == 0):
            A_matrix = self.A
        if(len(B_matrix) == 0):
            B_matrix = self.B
        assert(A_matrix.shape[0] == A_matrix.shape[1] == A_matrix.shape[2] == B_matrix.shape[0])
        N = A_matrix.shape[0]
        for i in xrange(N):
            plt.subplot(211)
            plt.title('A %d' % i)
            plt.imshow(A_matrix[i])
            plt.clim(0, 1.0)
            plt.subplot(212)
            plt.plot(range(len(B_matrix[i])), B_matrix[i], linestyle = 'None', marker = 'o')
            plt.ylim(-0.2, 1.0)
            plt.xlim(-1, len(B_matrix[i]))
            plt.title('B %d' % i)
            plt.savefig(folder_path + '/' + folder_name + '/' + string.lower(folder_name) + '_rule_' + str(i) + '.png', dpi = 300)
            plt.close()
        
    def compare_grammar_matrices_3(self,
                                 folder_path,
                                 folder_name,
                                 A_1_matrix = np.zeros(0),
                                 B_1_matrix = np.zeros(0),
                                 A_2_matrix = np.zeros(0),
                                 B_2_matrix = np.zeros(0),
                                 A_3_matrix = np.zeros(0),
                                 B_3_matrix = np.zeros(0)):
        if folder_name in os.listdir(folder_path):
            shutil.rmtree(folder_path + '/' + folder_name,
                          True)
        os.mkdir(folder_path + '/' + folder_name)
        if(len(A_3_matrix) == 0):
            A_3_matrix = self.A
        if(len(B_3_matrix) == 0):
            B_3_matrix = self.B
        assert(A_1_matrix.shape[0] == A_1_matrix.shape[1] == A_1_matrix.shape[2] == B_1_matrix.shape[0])
        assert(A_2_matrix.shape[0] == A_2_matrix.shape[1] == A_2_matrix.shape[2] == B_2_matrix.shape[0])
        assert(A_3_matrix.shape[0] == A_3_matrix.shape[1] == A_3_matrix.shape[2] == B_3_matrix.shape[0])
        N = A_1_matrix.shape[0]
        for i in xrange(N):
            plt.subplot(231)
            plt.title('First A matrix %d' % i)
            plt.imshow(A_1_matrix[i])
            plt.clim(0, 1.0)
            plt.subplot(232)
            plt.title('Second A matrix %d' % i)
            plt.imshow(A_2_matrix[i])
            plt.clim(0, 1.0)
            plt.subplot(233)
            plt.title('Third A matrix %d' % i)
            plt.imshow(A_3_matrix[i])
            plt.clim(0, 1.0)
            plt.subplot(234)
            plt.plot(range(len(B_1_matrix[i])), B_1_matrix[i], linestyle = 'None', marker = 'o')
            plt.ylim(-0.2, 1.0)
            plt.xlim(-1, len(B_1_matrix[i]))
            plt.title('First B matrix %d' % i)
            plt.subplot(235)          
            plt.plot(range(len(B_2_matrix[i])), B_2_matrix[i], linestyle = 'None', marker = 'o')
            plt.ylim(-0.2, 1.0)
            plt.xlim(-1, len(B_2_matrix[i]))
            plt.title('Second B matrix %d' % i)
            plt.subplot(236)          
            plt.plot(range(len(B_3_matrix[i])), B_3_matrix[i], linestyle = 'None', marker = 'o')
            plt.ylim(-0.2, 1.0)
            plt.xlim(-1, len(B_3_matrix[i]))
            plt.title('Third B matrix %d' % i)
            plt.savefig(folder_path + '/' + folder_name + '/' + string.lower(folder_name) + '_compare_rule_' + str(i) + '.png', dpi = 300)
            plt.close()
            
    def compare_grammar_matrices(self,
                                 folder_path,
                                 folder_name,
                                 A_1_matrix = np.zeros(0),
                                 B_1_matrix = np.zeros(0),
                                 A_2_matrix = np.zeros(0),
                                 B_2_matrix = np.zeros(0)):
        if folder_name in os.listdir(folder_path):
            shutil.rmtree(folder_path + '/' + folder_name,
                          True)
        os.mkdir(folder_path + '/' + folder_name)
        if(len(A_2_matrix) == 0):
            A_2_matrix = self.A
        if(len(B_2_matrix) == 0):
            B_2_matrix = self.B
        assert(A_1_matrix.shape[0] == A_1_matrix.shape[1] == A_1_matrix.shape[2] == B_1_matrix.shape[0])
        assert(A_2_matrix.shape[0] == A_2_matrix.shape[1] == A_2_matrix.shape[2] == B_2_matrix.shape[0])
        N = A_1_matrix.shape[0]
        for i in xrange(N):
            plt.subplot(221)
            plt.title('First A matrix %d' % i)
            plt.imshow(A_1_matrix[i])
            plt.clim(0, 1.0)
            plt.subplot(222)
            plt.title('Second A matrix %d' % i)
            plt.imshow(A_2_matrix[i])
            plt.clim(0, 1.0)  
            plt.subplot(223)          
            plt.plot(range(len(B_1_matrix[i])), B_1_matrix[i], linestyle = 'None', marker = 'o')
            plt.ylim(-0.2, 1.0)
            plt.xlim(-1, len(B_1_matrix[i]))
            plt.title('First B matrix %d' % i)
            plt.subplot(224)          
            plt.plot(range(len(B_2_matrix[i])), B_2_matrix[i], linestyle = 'None', marker = 'o')
            plt.ylim(-0.2, 1.0)
            plt.xlim(-1, len(B_2_matrix[i]))
            plt.title('Second B matrix %d' % i)
            plt.savefig(folder_path + '/' + folder_name + '/' + string.lower(folder_name) + '_compare_rule_' + str(i) + '.png', dpi = 300)
            plt.close()
            
    def plot_stats(self, n_samples, max_length = 0, max_represented = 0):
        first_time = time.clock()
        freqs, strings = SCFG_c.compute_stats(self.A,
                                              self.B,
                                              self.term_chars,
                                              n_samples,
                                              max_length)
        print time.clock() - first_time
        freqs = np.asarray(freqs, dtype = np.double)
        total = float(np.sum(freqs))
        print 'Number of sentences %f' % total
        freqs /= total
        entropy = -np.sum(freqs * np.log(freqs))
        indices = range(len(freqs))
        indices.sort(key = (lambda i : -freqs[i]))
        freqs = [freqs[i] for i in indices]
        strings = [strings[i] for i in indices]
        if max_represented != 0:
            freqs = freqs[:max_represented]
            strings = strings[:max_represented]
        plt.bar(np.arange(len(strings)), np.log(freqs), align = 'center')
        plt.xticks(np.arange(len(strings)), strings, rotation = 'vertical', fontsize = 4)
        plt.title('Frequences (%d sequences, %f entropy)' % (int(total), entropy))
        plt.show()
        
        
        
            