'''
Created on 20 mai 2014

@author: francois
'''

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