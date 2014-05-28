'''
Created on 20 mai 2014

@author: francois
'''

import stochastic_grammar_wrapper.SCFG_c as SCFG_c

import numpy as np

class SCFG:
    
    # List of rules is a list of sto_rules
    # Root symbol is an int
    def __init__(self,
                 list_of_rules,
                 root_symbol,
                 A = np.zeros((0, 0, 0)),
                 B = np.zeros((0, 0)),
                 list_of_rule_names = [],
                 index_to_term = []): # Root is always the first rule
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
        else:
            self.index_to_term = index_to_term
            self.term_to_index = {}
            self.index_to_non_term = []
            for i, term in enumerate(self.index_to_term):
                self.term_to_index[term]= i
            self.root_symbol = list_of_rule_names[0]
            for i, rule_name in enumerate(list_of_rule_names):
                self.index_to_non_term.append(rule_name)
                self.non_term_to_index[rule_name]= i
                
    def blurr_A(self):
        self.A += 0.005 * np.ones(self.A.shape)
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