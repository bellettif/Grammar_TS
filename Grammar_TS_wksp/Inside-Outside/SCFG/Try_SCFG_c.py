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
                            1000)

samples = filter(lambda x : len(x) < 10, samples)

print len(set([' '.join(x) for x in samples]))

#for sample in samples:
#    print sample
    
def estimate_with_random_init(): 
    A_init = np.random.uniform(0.0, 1.0, (N, N, N))
    B_init = np.random.uniform(0.0, 1.0, (N, M))
    A_init = np.maximum(A_init, 0.01 * np.ones((N, N, N)))
    B_init = np.maximum(B_init, 0.01 * np.ones((N, M)))
    B_init[0,:] = np.zeros(M) 
    B_init[1,:] = [1.0, 0.0, 0.0]
    A_init[1,:,:] = np.zeros((N, N))
    B_init[2,:] = np.zeros(M)
    B_init[3,:] = [0.0, 1.0, 0.0]
    A_init[3,:,:] = np.zeros((N, N))
    B_init[4,:] = np.zeros(M)
    B_init[5,:] = [0.0, 0.0, 1.0]
    A_init[5,:,:] = np.zeros((N, N))
    B_init[6,:] = np.zeros(M)
    for i in xrange(N):
        total = np.sum(A_init[i]) + np.sum(B_init[i])
        A_init[i] /= total
        B_init[i] /= total
    estim_A, estim_B, likelihoods = iterate_estimation(A_init,
                                                       B_init,
                                                       terms,
                                                       samples,
                                                       100)
    return likelihoods

def estimate_with_random_init_perturbated():
    A_init = np.random.uniform(0.0, 1.0, (N, N, N))
    B_init = np.random.uniform(0.0, 1.0, (N, M))
    A_init = np.maximum(A_init, 0.01 * np.ones((N, N, N)))
    B_init = np.maximum(B_init, 0.01 * np.ones((N, M)))
    B_init[0,:] = np.zeros(M) 
    B_init[1,:] = [1.0, 0.0, 0.0]
    A_init[1,:,:] = np.zeros((N, N))
    B_init[2,:] = np.zeros(M)
    B_init[3,:] = [0.0, 1.0, 0.0]
    A_init[3,:,:] = np.zeros((N, N))
    B_init[4,:] = np.zeros(M)
    B_init[5,:] = [0.0, 0.0, 1.0]
    A_init[5,:,:] = np.zeros((N, N))
    B_init[6,:] = np.zeros(M)
    for i in xrange(N):
        total = np.sum(A_init[i]) + np.sum(B_init[i])
        A_init[i] /= total
        B_init[i] /= total
    estim_A, estim_B, likelihoods = iterate_estimation_perturbated(A_init,
                                                                   B_init,
                                                                   terms,
                                                                   samples,
                                                                   100,
                                                                   np.random.uniform,
                                                                   0.0,
                                                                   1.0,
                                                                   0.1,
                                                                   np.random.uniform,
                                                                   0.0,
                                                                   1.0,
                                                                   0.1,
                                                                   0.5)
    return likelihoods

"""
plt.subplot(221)
plt.plot(np.log(estimate_with_random_init_perturbated()))
#
plt.subplot(222)
plt.plot(np.log(estimate_with_random_init_perturbated()))
#
plt.subplot(223)
plt.plot(np.log(estimate_with_random_init()))
#
plt.subplot(224)
plt.plot(np.log(estimate_with_random_init()))
#
plt.show()
"""

A_init = np.random.uniform(0.0, 1.0, (N, N, N))
B_init = np.random.uniform(0.0, 1.0, (N, M))
A_init = np.maximum(A_init, 0.01 * np.ones((N, N, N)))
B_init = np.maximum(B_init, 0.01 * np.ones((N, M)))
B_init[0,:] = np.zeros(M) 
B_init[1,:] = [1.0, 0.0, 0.0]
A_init[1,:,:] = np.zeros((N, N))
B_init[2,:] = np.zeros(M)
B_init[3,:] = [0.0, 1.0, 0.0]
A_init[3,:,:] = np.zeros((N, N))
B_init[4,:] = np.zeros(M)
B_init[5,:] = [0.0, 0.0, 1.0]
A_init[5,:,:] = np.zeros((N, N))
B_init[6,:] = np.zeros(M)
for i in xrange(N):
    total = np.sum(A_init[i]) + np.sum(B_init[i])
    A_init[i] /= total
    B_init[i] /= total
estim_A, estim_B, likelihoods = iterate_estimation_perturbated(A_init,
                                                               B_init,
                                                               terms,
                                                               samples,
                                                               100,
                                                               np.random.normal,
                                                               0.0,
                                                               0.1,
                                                               0.01,
                                                               np.random.uniform,
                                                               0.0,
                                                               0.0,
                                                               0.0,
                                                               0.9)

likelihoods_exact = estimate_likelihoods(A,
                                         B,
                                         terms,
                                         samples)
temp = np.zeros((100, len(samples)))
for i in range(100):
    temp[i] = likelihoods_exact
likelihoods_exact = temp

plt.plot(np.log(likelihoods))
plt.plot(np.log(likelihoods_exact), linestyle = '--')
plt.ylim((-30, 0))
plt.show()


for i in xrange(N):
    plt.subplot(231)
    plt.title('Actual A %d' % i)
    plt.imshow(A[i])
    plt.clim(0, 1.0)
    plt.subplot(232)
    plt.title('Estimated A %d' % i)
    plt.imshow(estim_A[i])
    plt.clim(0, 1.0)
    plt.subplot(233)
    plt.title('Init A %d' % i)
    plt.imshow(A_init[i])
    plt.clim(0, 1.0)
    #
    plt.subplot(234)
    plt.title('Actual B %d' % i)
    plt.plot(range(len(B[i])), B[i], linestyle = 'None', marker = 'o')
    plt.ylim(-0.2, 1.0)
    plt.xlim(-1, len(B[i]))
    plt.subplot(235)
    plt.title('Estimated B %d' % i)
    plt.plot(range(len(estim_B[i])), estim_B[i], linestyle = 'None', marker = 'o')
    plt.ylim(-0.2, 1.0)
    plt.xlim(-1, len(estim_B[i]))
    plt.subplot(236)
    plt.title('Init B %d' % i)
    plt.plot(range(len(B_init[i])), B_init[i], linestyle = 'None', marker = 'o')
    plt.ylim(-0.2, 1.0)
    plt.xlim(-1, len(B_init[i]))
    plt.show()
    

