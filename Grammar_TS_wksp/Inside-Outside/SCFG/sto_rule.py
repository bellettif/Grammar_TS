'''
Created on 20 mai 2014

@author: francois
'''

import numpy as np

class Sto_rule:
    
    #
    #    Rule name, integer
    #    Non term w, list of doubles
    #    Non term s, list of pairs of integers
    #    Term w, list of doubles
    #    Term s, list of chars
    #
    def __init__(self,
                 rule_name,
                 non_term_w,
                 non_term_s,
                 term_w,
                 term_s):
        self.rule_name = rule_name
        self.non_term_w = np.asarray(non_term_w, dtype = np.double)
        self.n_non_term = len(self.non_term_w)
        self.tot_non_term_w = np.sum(self.non_term_w)
        self.non_term_s = non_term_s
        self.term_w = np.asarray(term_w, dtype = np.double)
        self.n_term = len(self.term_w)
        self.tot_term_w = np.sum(self.term_w)
        self.term_s = term_s
        #
        tot_weight = self.tot_non_term_w + self.tot_term_w
        self.non_term_w /= tot_weight
        self.term_w /= tot_weight
        
    def print_state(self):
        print 'Printing rule ' + str(self.rule_name) + ':'
        print 'Non terminal symbols:'
        for i in xrange(len(self.non_term_w)):
            print '\t' + str(self.rule_name) + (' (%f) ' % (self.non_term_w[i])) + ' -> ' + \
                        str(self.non_term_s[i][0]) + ' ' + str(self.non_term_s[i][1])
        print 'Terminal symbols:'
        for i in xrange(len(self.term_w)):
            print '\t' + str(self.rule_name) + (' (%f) ' % (self.term_w[i])) + ' -> ' + \
                        str(self.term_s[i])
        print ''
    
    