'''
Created on 5 avr. 2014

@author: francois
'''

import string

class Simple_sequitur:
    
    def __init__(self, target_string):
        self.target_string = target_string
        self.current_index = 0
        self.terminals = string.ascii_lowercase
        self.non_terminals = string.ascii_uppercase
        self.current_non_terminal_index = 0
        self.S = ''
        self.rules = {}                 ## LHS -> RHS
        self.rules_ref_count = {}       ## LHS ref count
        self.digrams = {}               ## digram -> LHS
        
    ## Apply the grammatical substitution predicates to target_RHS
    def apply_rules(self, target_RHS):
        for LHS, RHS in self.rules.iteritems():
            if RHS in target_RHS:
                self.rules_ref_count[LHS] += 1
                target_RHS = string.replace(target_RHS, RHS, LHS)   
        
    ## What are the new digrams in S ?
    def discover_new_digrams(self):
        if len(self.S) < 2:
            print 'S too short for digrams to be found'
            return False
        for i in range(len(self.S) - 1):
            current_digram = self.S[i] + self.S[i+1]
            if current_digram not in self.digrams:
                ## A new rule needs to be created
                non_terminal = self.non_terminals[self.current_non_terminal_index]
                self.current_non_terminal_index += 1
                self.rules[non_terminal] = current_digram
                self.rules_ref_count[non_terminal] = 1
        
    def next(self):
        self.S += self.target_string[self.current_index]
        
        
        
        self.current_index += 1
        if self.current_index == len(self.target_string):
            print 'Done'
            return True             ## Is done == True
        else:
            return False            ## Is done == False