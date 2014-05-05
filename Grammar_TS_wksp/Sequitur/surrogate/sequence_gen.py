'''
Created on 5 mai 2014

@author: francois
'''

class CFG:
    
    def __init__(self, rules):
        self.rules = rules
        
    def print_grammar(self):
        print self.rules
        


print gen_framing_rule(1, 2, 2, 3)
print gen_power_rule(2, 3, 10)