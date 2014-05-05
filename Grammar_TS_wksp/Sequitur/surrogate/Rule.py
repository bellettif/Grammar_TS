'''
Created on 5 mai 2014

@author: francois
'''

import time

class Rule:
    
    def __init__(self, lhs, rhs):
        self.rhs = rhs
        self.lhs = lhs
        self.barcode_computed = False
        
    def to_string(self):
        return str(self.lhs) + (' (%s) -> ' % self.barcode) + ' '.join(map(str, self.rhs))
    
    def compute_barcode(self, grammar):
        self.barcode = ''.join([x if x not in grammar
                                else (grammar[x].barcode if grammar[x].barcode_computed 
                                else grammar[x].compute_barcode(grammar))
                                for x in self.rhs])
        self.barcode_computed = True
        return self.barcode
        
def gen_framing_rule(lhs, framing, middle, n, m):
    rhs = ([str(framing)] * n) + [str(middle)] + ([str(framing)] * n)
    return Rule(lhs, rhs) 

def gen_power_rule(lhs, first, second, n):
    rhs = [str(first), str(second)]
    rhs *= n
    return Rule(lhs, rhs)       

grammar = {'B' : Rule('B', ['A', 'c']),
           'A' : Rule('A', ['a', 'b']),
           'C' : gen_framing_rule('C', 'A', 'B', 2, 3),
           'D' : gen_power_rule('D', 'C', 'B', 3)}

for lhs, rule in grammar.iteritems():
    rule.compute_barcode(grammar)
    
print [x.to_string() for x in grammar.values()]