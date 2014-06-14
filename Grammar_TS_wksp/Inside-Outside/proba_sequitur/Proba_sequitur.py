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

from Proba_sequitur_linear_c import run_proba_sequitur as proba_seq

MAX_RULES = 30

class Proba_sequitur:

    def __init__(self,
                 inference_samples,
                 count_samples,
                 degree,
                 max_rules,
                 random,
                 init_T = 0,
                 T_decay = 0,
                 p_deletion = 0):
        self.inference_samples = inference_samples
        self.count_samples = count_samples
        #
        self.degree = degree
        self.max_rules = max_rules
        self.random = random
        #
        self.init_T = init_T
        self.T_decay = T_decay
        #
        self.p_deletion = p_deletion
        #
        #
        self.inference_parsed = []
        self.count_parsed = []
        self.rules = {}
        self.rule_names = {}
        self.relative_counts = {}
        self.absolute_counts = {}
        self.levels = {}
        self.depths = {}
        self.divergences = {}
        
    def run(self):
        result = proba_seq(self.inference_samples,
                           self.count_samples,
                           self.degree,
                           self.max_rules,
                           self.random,
                           self.init_T,
                           self.T_decay,
                           self.p_deletion)
        self.inference_parsed = result['inference_parsed']
        self.count_parsed = result['count_parsed']
        self.rules = result['rules']
        self.rule_names = result['rule_names']
        self.relative_counts = result['relative_counts']
        self.absolute_counts = result['absolute_counts']
        self.levels = result['levels']
        self.depths = result['depths']
        self.divergences = result['divergences']
        
        
        