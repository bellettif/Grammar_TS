'''
Created on 22 mai 2014

@author: francois
'''

import numpy as np
import matplotlib.pyplot as plt
import re
import string
import copy
import time

#
#    Import c++ code wrapped with cython
#
from Proba_sequitur_linear_c import run_proba_sequitur as proba_seq
from grammar_graph import grammar_to_graph

class Proba_sequitur:

    #
    #    Inference_samples: the grammar will be built on these samples
    #    Count_samples: the occurrence counts will be computed with respect
    #                    to these samples
    #    Degree: k, the degree of coalescence of the parsing tree
    #    Max_rules: maximum number of rules that will be found
    #    Random: whether the algorithm is stochastic or not
    #    Init_T: initial boltzmann temperature
    #    T_decay: temperature decay rate (per coalescence round)
    #    p_deletion: probability of deletion in the input sequence
    #
    def __init__(self,
                 inference_samples,
                 count_samples,
                 k,
                 max_rules,
                 random,
                 init_T = 0,
                 T_decay = 0,
                 p_deletion = 0):
        #
        self.inference_samples = inference_samples
        self.count_samples = count_samples
        #
        self.k = k
        self.max_rules = max_rules
        self.random = random
        #
        self.init_T = max(init_T, 0.05) if random else init_T
        self.T_decay = T_decay
        #
        self.p_deletion = p_deletion
        #
        #    Final parses with rules written as hashcodes
        #
        self.inference_parsed = []
        self.count_parsed = []
        #
        #    Final parses with rules written with their names
        #
        self.hashed_inference_parsed = []
        self.hashed_count_parsed = []
        #
        #    Grammar
        #
        self.grammar = {}
        self.hashcode_to_rule = {}
        self.rule_to_hashcode = {}
        self.hashed_grammar = {}
        #
        #    Hashed frequency results
        #
        self.relative_counts = {}
        self.absolute_counts = {}
        self.levels = {}
        self.depths = {}
        #
        self.divergences = {}
        #
        
    #
    #    Execute the c++ algorithm (see python wrapper)
    #
    def run(self):
        result = proba_seq(self.inference_samples,
                           self.count_samples,
                           self.k,
                           self.max_rules,
                           self.random,
                           self.init_T,
                           self.T_decay,
                           self.p_deletion)
        self.hashed_inference_parsed = result['inference_parsed']
        self.hashed_count_parsed = result['count_parsed']
        self.hashed_grammar = result['rules']
        self.hashcode_to_rule = result['rule_names']
        hashcode_to_rule_items = self.hashcode_to_rule.items()
        hashcodes = [x[0] for x in hashcode_to_rule_items]
        rulenames = [x[1] for x in hashcode_to_rule_items]
        self.rule_to_hashcode = dict(zip(rulenames, hashcodes))
        self.relative_counts = result['relative_counts']
        self.absolute_counts = result['absolute_counts']
        self.levels = result['levels']
        self.depths = result['depths']
        self.divergences = result['divergences']
        for lhs_hash, (left_hash, right_hash) in self.hashed_grammar.iteritems():
            lhs_converted = self.hashcode_to_rule[lhs_hash] \
                            if lhs_hash in self.hashcode_to_rule \
                            else lhs_hash
            left_converted = self.hashcode_to_rule[left_hash] \
                                if left_hash in self.hashcode_to_rule \
                                else left_hash
            right_converted = self.hashcode_to_rule[right_hash] \
                                if right_hash in self.hashcode_to_rule \
                                else right_hash
            self.grammar[lhs_converted] = (left_converted, 
                                           right_converted)
        self.inference_parsed = \
                [[self.hashcode_to_rule[x] if x in self.hashcode_to_rule 
                  else x for x in seq]
                  for seq in self.hashed_inference_parsed]
        self.count_parsed = \
                [[self.hashcode_to_rule[x] if x in self.hashcode_to_rule 
                      else x for x in seq]
                      for seq in self.hashed_count_parsed]
            
    def draw_graph(self, file_path, 
                   seq_index = 0,
                   cut = 10):
        grammar_to_graph(file_path,
                         self.converted_grammar,
                         self.converted_inference_parsed[seq_index],
                         cut)
        
        