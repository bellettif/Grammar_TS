'''
Created on 5 mai 2014

@author: francois
'''

import time

def gen_framing_rule(lhs, framing, middle, n, m):
    rhs = ([str(framing)] * n) + [str(middle)] + ([str(framing)] * n)
    return Rule(lhs, rhs) 

def gen_power_rule(lhs, first, second, n):
    rhs = [str(first), str(second)]
    rhs *= n
    return Rule(lhs, rhs)

class Rule:
    
    def __init__(self, lhs, rhs):
        self.rhs = map(str, rhs)
        self.lhs = str(lhs)
        self.barcode_computed = False
        self.barcode = []
        
    def to_string(self):
        return str(self.lhs) + (' (%s) -> ' % self.barcode) + ' '.join(map(str, self.rhs))
    
    def compute_barcode(self, grammar):
        for x in self.rhs:
            if x not in grammar:
                self.barcode.append(x)
            else:
                if grammar[x].barcode_computed:
                    self.barcode.extend(grammar[x].barcode)
                else:
                    self.barcode.extend(grammar[x].compute_barcode(grammar))
        self.barcode_computed = True
        return self.barcode
        


