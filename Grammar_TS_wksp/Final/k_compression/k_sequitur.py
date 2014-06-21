'''
Created on 21 juin 2014

@author: francois
'''

import k_sequitur_c
import numpy as np

from grammar_graph import grammar_to_graph

class k_Sequitur():
    
    #
    #    Input sequence must be a list of chars
    #
    def __init__(self, input_sequence):
        if(len(input_sequence) == 0):
            raise Exception('Input sequence length in sequitur is 0')
        #
        #    Store input and output
        #
        self.input_sequence = input_sequence
        self.compressed_sequence = []
        self.initial_length = len(input_sequence)
        self.post_compression_length = 0
        #
        #    Conversion dictionaries
        #
        self.int_to_char = list(set(input_sequence))
        self.char_to_int = dict(zip(self.int_to_char, range(len(self.int_to_char))))
        self.int_to_char = dict(zip(range(len(self.int_to_char)), self.int_to_char))
        #
        #    Grammar
        #
        self.grammar = {}
        self.ref_counts = {}
        self.hashed_ref_counts = {}
        self.hashed_freqs = {}
        self.hashcode_to_rule = {}
        self.rule_to_hashcode = {}
        
    def run(self):
        temp_grammar = k_sequitur_c.run(np.asarray([self.char_to_int[x] 
                                                  for x in self.input_sequence],
                                                 dtype = np.int32))
        rule_indices = temp_grammar.keys()
        rule_indices.sort(key = (lambda x : -x))
        for lhs in rule_indices:
            [rhs_array, ref_count] = temp_grammar[lhs]
            if lhs == 0:
                self.compressed_sequence = [self.int_to_char[x] 
                                            if x in self.int_to_char
                                            else 'r%d' % (-x)
                                            for x in rhs_array]
                self.post_compression_length = len(self.compressed_sequence)
                continue
            rhs_transformed = [self.int_to_char[x] if x in self.int_to_char
                                else 'r%d' % (-x)
                                for x in rhs_array]
            rhs_hashcode = [self.int_to_char[x] if x in self.int_to_char
                            else
                            self.rule_to_hashcode['r%d' % (-x)]
                            for x in rhs_array]
            lhs_hashcode = '>' + '-'.join(rhs_hashcode) + '<'
            rule_name = 'r%d' % (-lhs)
            self.grammar[rule_name] = rhs_transformed
            self.ref_counts[rule_name] = ref_count
            self.hashed_ref_counts[lhs_hashcode] = ref_count
            self.hashed_freqs[lhs_hashcode] = float(ref_count) / float(len(self.input_sequence))
            self.hashcode_to_rule[lhs_hashcode] = rule_name
            self.rule_to_hashcode[rule_name] = lhs_hashcode
            
    def draw_graph(self, file_path, cut = 10):
        grammar_to_graph(file_path, 
                         self.grammar, 
                         self.compressed_sequence,
                         cut)