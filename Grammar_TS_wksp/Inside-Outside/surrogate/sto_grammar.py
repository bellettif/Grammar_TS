'''
Created on 20 mai 2014

@author: francois
'''

import stochastic_grammar_wrapper.SCFG_c as SCFG_c

import numpy as np
from surrogate.sto_rule import Sto_rule

class SCFG:
    
    # List of rules is a list of sto_rules
    # Root symbol is an int
    def __init__(self,
                 list_of_rules = [],
                 root_symbol = [],
                 A = np.zeros((0, 0, 0)),
                 B = np.zeros((0, 0)),
                 to_merge = [],
                 term_chars = []): # Root is always the first rule
        self.grammar = {}
        self.A = np.copy(A)
        self.B = np.copy(B)
        if len(list_of_rules) > 0:
            all_non_terms = []
            all_terms = []
            for rule in list_of_rules:
                self.grammar[rule.rule_name] = rule
                flattened_non_term = []
                for non_term_pair in rule.non_term_s:
                    flattened_non_term.append(non_term_pair[0])
                    flattened_non_term.append(non_term_pair[1])
                all_non_terms.extend(flattened_non_term)
                all_terms.extend(rule.term_s)
            self.root_symbol = root_symbol
            self.index_to_non_term = []
            self.index_to_term = []
            self.non_term_to_index = {}
            self.term_to_index = {}
            self.compute_parameters()
        elif len(to_merge) > 0:
            self.root_symbol = 0
            first_index = to_merge[0]
            second_index = to_merge[1]
            sub_selection = range(A.shape[0])
            sub_selection = filter(lambda x : x != second_index, sub_selection)
            old_A = np.copy(A)
            old_A[first_index, :, :] = 0.5 * old_A[first_index, :, :] + 0.5 * old_A[second_index, :, :]
            old_A[:, first_index, :] = 0.5 * old_A[:, first_index, :] + 0.5 * old_A[:, second_index, :]
            old_A[:, :, first_index] = 0.5 * old_A[:, :, first_index] + 0.5 * old_A[:, :, second_index]
            self.A = old_A[np.ix_(sub_selection, sub_selection, sub_selection)]
            self.B = np.copy(B[sub_selection])
            self.B[first_index] = 0.5 * B[first_index] + 0.5 * B[second_index]
            for i in xrange(self.A.shape[0]):
                total_weight = np.sum(self.A[i]) + np.sum(self.B[i])
                self.A[i,:,:] /= total_weight
                self.B[i,:] /= total_weight
            self.index_to_non_term = range(len(sub_selection))
            self.non_term_to_index = {}
            for i, non_term in enumerate(self.index_to_non_term):
                self.non_term_to_index[non_term] = i
            self.index_to_term = term_chars
            self.term_to_index = {}
            for i, term in enumerate(self.index_to_term):
                self.term_to_index[term] = i
            self.grammar = {}
            N = self.A.shape[0]
            M = self.B.shape[1]
            for i in xrange(N):
                self.grammar[i] = Sto_rule(i,
                                           [A[i,j,k] for j in xrange(N) for k in xrange(N)],
                                           [[j, k] for j in xrange(N) for k in xrange(N)],
                                           [B[i, j] for j in xrange(M)],
                                           term_chars)
                
    def blurr_A(self):
        for i in xrange(self.A.shape[0]):
            if np.sum(self.A[i,:,:]) == 0:
                continue
            self.A[i] += 0.1 * np.ones((self.A.shape[0], self.A.shape[1]))
        self.recompute_grammar()
        self.A, self.B, self.index_to_non_term, self.index_to_term = \
                SCFG_c.compute_parameters(self)
                       
    def recompute_grammar(self):
        for i in range(self.A.shape[0]):
            rule_name = self.index_to_non_term[i]
            rule = self.grammar[rule_name]
            weights = []
            symbols = []
            for j in range(self.A.shape[1]):
                for k in range(self.A.shape[2]):
                    weights.append(self.A[i,j,k])
                    symbols.append([self.index_to_non_term[j], self.index_to_non_term[k]])
            rule.non_term_w = np.asarray(weights, dtype = np.double)
            rule.n_non_term = len(rule.non_term_w)
            rule.non_term_s = symbols
        
    def compute_parameters(self):
        self.A, self.B, self.index_to_non_term, self.index_to_term = \
                SCFG_c.compute_parameters(self)
        for index, non_term in enumerate(self.index_to_non_term):
            self.non_term_to_index[non_term] = index
        for index, term in enumerate(self.index_to_term):
            self.term_to_index[term] = index
            
    def print_parameters(self):
        print 'A:'
        print self.A
        print '\n'
        print 'B:'
        print self.B
        print '\n'
        print 'Index to non term:'
        print self.index_to_non_term
        print '\n'
        print 'Non term to index:'
        print self.non_term_to_index
        print '\n'
        print 'Index to term:'
        print self.index_to_term
        print '\n'
        print 'Term to index:'
        print self.term_to_index
        print '\n'
        
    # Return a list of list of strings
    def produce_sentences(self,
                           n_samples):
        return SCFG_c.compute_derivations(self.grammar[self.root_symbol],
                                          self,
                                          n_samples)
        
    # Returns A_estim, B_estim
    def estimate_model(self,
                       sentences,
                       n_iterations,
                       proposal_A,
                       proposal_B):
        return SCFG_c.estimate_model(self,
                                     sentences,
                                     proposal_A,
                                     proposal_B,
                                     n_iterations)
        
    # Returns E_esitm, F_Estim, log_lk
    def compute_inside_outside(self,
                               sentence,
                               proposal_A,
                               proposal_B):
        E, F = SCFG_c.compute_inside_outside(self,
                                             sentence,
                                             proposal_A,
                                             proposal_B)
        log_lk = np.log(E[self.non_term_to_index[self.root_symbol], 0, -1])
        return E, F, log_lk
    
    def compute_probas(self,
                       sentences):
        probas = []
        for sentence in sentences:
            E, F = SCFG_c.compute_inside_outside(self,
                                                 sentence,
                                                 self.A,
                                                 self.B)
            probas.append(E[self.non_term_to_index[self.root_symbol], 0, -1])
        return np.asarray(probas, dtype = np.double)
        
    def compute_probas_proposal(self,
                                sentences,
                                A_proposal,
                                B_proposal):
        probas = []
        for sentence in sentences:
            E, F = SCFG_c.compute_inside_outside(self,
                                                 sentence,
                                                 A_proposal,
                                                 B_proposal)
            probas.append(E[self.non_term_to_index[self.root_symbol], 0, -1])
        return np.asarray(probas, dtype = np.double)    
        
        