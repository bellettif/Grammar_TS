'''
Created on 7 juil. 2014

@author: francois
'''

import numpy as np
import time

from SCFG.sto_grammar import SCFG
from SCFG.grammar_distance import Grammar_distance

N = 7
M = 3

A_1 = np.zeros((N, N, N), dtype = np.double)
B_1 = np.zeros((N, M), dtype = np.double)

A_2 = np.zeros((N, N, N), dtype = np.double)
B_2 = np.zeros((N, M), dtype = np.double)

A_3 = np.zeros((N, N, N), dtype = np.double)
B_3 = np.zeros((N, M), dtype = np.double)

A_1[0, 1, 2] = 0.3;
A_1[0, 3, 4] = 0.3;
A_1[0, 5, 6] = 0.3;

A_1[0, 1, 1] = 0.2;
A_1[0, 3, 3] = 0.2;
A_1[0, 5, 5] = 0.2;

A_1[2, 0, 1] = 1.0;
A_1[4, 0, 3] = 1.0;
A_1[6, 0, 5] = 1.0;

B_1[1, 0] = 1.0;
B_1[3, 1] = 1.0;
B_1[5, 2] = 1.0;

A_2[0, 1, 2] = 0.23;
A_2[0, 3, 4] = 0.23;
A_2[0, 5, 6] = 0.23;

A_2[0, 1, 1] = 0.2;
A_2[0, 3, 3] = 0.2;
A_2[0, 5, 5] = 0.2;

A_2[2, 0, 1] = 1.0;
A_2[4, 0, 3] = 1.0;
A_2[6, 0, 5] = 1.0;

B_2[1, 0] = 1.0;
B_2[3, 1] = 1.0;
B_2[5, 2] = 1.0;

A_3[0, 1, 2] = 0.25;
A_3[0, 3, 4] = 0.25;
A_3[0, 5, 6] = 0.25;

A_3[0, 1, 1] = 0.2;
A_3[0, 3, 3] = 0.2;
A_3[0, 5, 5] = 0.2;

A_3[2, 0, 1] = 1.0;
A_3[4, 0, 3] = 1.0;
A_3[6, 0, 5] = 1.0;

B_3[1, 0] = 1.0;
B_3[3, 1] = 1.0;
B_3[5, 2] = 1.0;

grammar_1 = SCFG()
grammar_1.init_from_A_B(A_1, B_1, ['0', '1', '2'])

grammar_1.draw_grammar('palindrom.png')

grammar_2 = SCFG()
grammar_2.init_from_A_B(A_2, B_2, ['0', '1', '2'])

grammar_3 = SCFG()
grammar_3.init_from_A_B(A_3, B_3, ['0', '1', '2'])


grammar_dist_1_1 = Grammar_distance(grammar_1, grammar_1)
grammar_dist_1_2 = Grammar_distance(grammar_1, grammar_2)
grammar_dist_1_3 = Grammar_distance(grammar_1, grammar_3)

grammar_dist_2_1 = Grammar_distance(grammar_2, grammar_1)
grammar_dist_2_2 = Grammar_distance(grammar_2, grammar_2)
grammar_dist_2_3 = Grammar_distance(grammar_2, grammar_3)

grammar_dist_3_1 = Grammar_distance(grammar_3, grammar_1)
grammar_dist_3_2 = Grammar_distance(grammar_3, grammar_2)
grammar_dist_3_3 = Grammar_distance(grammar_3, grammar_3)

n_samples = 10000

dist_1_1 = grammar_dist_1_1.compute_distance(n_samples)
dist_1_2 = grammar_dist_1_2.compute_distance(n_samples)
dist_1_3 = grammar_dist_1_3.compute_distance(n_samples)

dist_2_1 = grammar_dist_2_1.compute_distance(n_samples)
dist_2_2 = grammar_dist_2_2.compute_distance(n_samples)
dist_2_3 = grammar_dist_2_3.compute_distance(n_samples)

dist_3_1 = grammar_dist_3_1.compute_distance(n_samples)
dist_3_2 = grammar_dist_3_2.compute_distance(n_samples)
dist_3_3 = grammar_dist_3_3.compute_distance(n_samples)

print "Self distances:"
print "\t1: 1_1 = %f" % dist_1_1
print "\t2: 2_2 = %f" % dist_2_2
print "\t3: 3_3 = %f" % dist_3_3

print "Cross distances:"
print "\t1-2: 1->2 = %f, 2->1 = %f" % (dist_1_2, dist_2_1)
print "\t1-2: 1->3 = %f, 3->1 = %f" % (dist_1_3, dist_3_1)
print "\t1-2: 2->3 = %f, 3->2 = %f" % (dist_2_3, dist_3_2)

sym_1_2 = 0.5 * (dist_1_2 + dist_2_1)
sym_1_3 = 0.5 * (dist_1_3 + dist_3_1)
sym_2_3 = 0.5 * (dist_2_3 + dist_3_2)

print "Check triangle:"
print "\t sym_1_3 + sym_2_3 = %f, sym 1_2 = %f" % (sym_1_3 + sym_2_3, sym_1_2)
print "\t sym_1_2 + sym_2_3 = %f, sym 1_3 = %f" % (sym_1_2 + sym_2_3, sym_1_3)






