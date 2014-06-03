'''
Created on 3 juin 2014

@author: francois
'''

import numpy as np

N = 7

terms = ['a', 'b', 'c']

M = len(terms)

A = np.zeros((N, N, N), dtype = np.double)
B = np.zeros((N, M), dtype = np.double)

A[0, 1, 2] = 1.0
A[0, 3, 4] = 1.0
A[0, 5, 6] = 1.0
A[0, 1, 1] = 1.0
A[0, 3, 3] = 1.0
A[0, 5, 5] = 1.0

A[2, 0, 1] = 1.0

A[4, 0, 3] = 1.0

A[6, 0, 5] = 1.0

B[1, 0] = 1.0
B[3, 1] = 1.0
B[5, 2] = 1.0

for i in xrange(N):
    total = np.sum(A[i]) + np.sum(B[i])
    A[i] /= total
    B[i] /= total

first_index = 1
second_index = 2
sub_selection = range(A.shape[0])
sub_selection = filter(lambda x : x != second_index, sub_selection)
new_A = np.copy(A)
old_A = A
new_A[first_index, :, :] = 0.5 * old_A[first_index, :, :] + 0.5 * old_A[second_index, :, :]
new_A[:, first_index, :] += old_A[:, second_index, :]
new_A[:, :, first_index] += old_A[:, :, second_index]
new_A = new_A[np.ix_(sub_selection, sub_selection, sub_selection)]

old_B = np.copy(B)
new_B = old_B[sub_selection]
new_B[first_index] = 0.5 * old_B[first_index] + 0.5*old_B[second_index]

for i in xrange(new_A.shape[0]):
    print i
    print np.sum(new_A[i]) + np.sum(new_B[i])
    print ''

