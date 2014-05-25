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
                 root_symbol):
        self.grammar = {}
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
        self.A = np.zeros((0, 0))
        self.B = np.zeros((0, 0))
        self.index_to_non_term = []
        self.index_to_term = []
        self.non_term_to_index = {}
        self.term_to_index = {}
        self.compute_parameters()
        
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
        
    def producte_sentences(self,
                           n_samples):
        return SCFG_c.compute_derivations(self.grammar[self.root_symbol],
                                          self,
                                          n_samples)
        
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
        
    def compute_inside_outside(self,
                               sentence,
                               proposal_A,
                               proposal_B):
        return SCFG_c.compute_inside_outside(self,
                                             sentence,
                                             proposal_A,
                                             proposal_B)