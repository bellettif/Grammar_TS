'''
Created on 20 juin 2014

@author: francois
'''

class CSGrammar:
    
    def __init__(self,
                N, M = 1):
        self.N = N
        self.M = M
        
    def produce_sentences(self, n_samples):
        return [self.produce_sentence() for i in xrange(n_samples)]
    
    def produce_sentence(self):
        return (['a'] * self.N) + (['b'] * self.M) + (['a'] * self.N)
        

CSExample = CSGrammar(3, 2)