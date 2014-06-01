'''
Created on 1 juin 2014

@author: francois
'''

from SCFG_c import *

import numpy as np
from matplotlib import pyplot as plt

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

samples = produce_sentences(A, B,
                            terms,
                            100)

#for sample in samples:
#    print sample
    
A_init = np.random.uniform(0.0, 1.0, (N, N, N))
B_init = np.random.uniform(0.0, 1.0, (N, M))

A_init = np.maximum(A, 0.01 * np.ones((N, N, N)))
B_init = np.maximum(B, 0.01 * np.ones((N, M)))

for i in xrange(N):
    total = np.sum(A_init[i]) + np.sum(B_init[i])
    A_init[i] /= total
    B_init[i] /= total
    
estim_A, estim_B, likelihoods = iterate_estimation(A_init,
                                                   B_init,
                                                   terms,
                                                   samples,
                                                   100)

for i in xrange(N):
    plt.subplot(221)
    plt.title('Actual A %d' % i)
    plt.imshow(A[i])
    plt.subplot(222)
    plt.title('Estimated A %d' % i)
    plt.imshow(estim_A[i])
    plt.subplot(223)
    plt.title('Actual B %d' % i)
    plt.plot(range(len(B[i])), B[i], linestyle = 'None', marker = 'o')
    plt.subplot(224)
    plt.title('Estimated B %d' % i)
    plt.plot(range(len(B[i])), estim_B[i], linestyle = 'None', marker = 'o')
    
    plt.show()
    

