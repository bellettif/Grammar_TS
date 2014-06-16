'''
Created on 16 juin 2014

@author: francois
'''


import numpy as np

N = 5

my_matrix = np.asanyarray([[i + j for i in xrange(N)] for j in xrange(N)])

target_index = 3

copy = np.zeros((N, N), dtype = np.int)
copy[0,:] = my_matrix[target_index,:]

source_indices = [target_index] + filter(lambda x : x != target_index, range(N)) 
for i in xrange(N):
    copy[i,:] = my_matrix[source_indices[i], :]

copy_ = np.zeros((N, N), dtype = np.int)
for i in xrange(N):
    copy_[:,i] = copy[:, source_indices[i]]

print my_matrix
print ''
print copy_


