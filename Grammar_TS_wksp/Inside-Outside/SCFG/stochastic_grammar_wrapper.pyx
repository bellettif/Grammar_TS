import numpy as np
cimport numpy as np
np.import_array()
cimport libc.stdlib
import ctypes

from datetime import datetime
from time import mktime

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.list cimport list
from cython.operator cimport dereference as deref

cdef extern from "<random>" namespace "std":
	cdef cppclass mt19937:
		mt19937 ()
		mt19937(unsigned long root)
		
cdef extern from "flat_in_out.h":
	cdef cppclass Flat_in_out:
		Flat_in_out(double* A, double* B,
	               	int N, int M,
	               	const vector[string] & terminals)
		void compute_probas_flat(const vector[vector[string]] & samples,
                           	double * probas)
		void compute_proba_flat(const vector[string] & sample)
		void compute_inside_outside_flat(double* E, double* F,
									const vector[string] & sample)
		void compute_inside_probas_flat(double* E,
								   const vector[string] & sample)
		void estimate_A_B(const vector[vector[string]] & samples,
						  double * sample_probas,
						  double * new_A,
						  double * new_B)
		vector[vector[string]] produce_sentences(int n_sentences)
	
def estimate_likelihoods(np.ndarray A_proposal,
						 np.ndarray B_proposal,
						 terminals,
						 samples):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] likelihoods = np.zeros(n_samples, dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = A_proposal
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = B_proposal
	cdef Flat_in_out* fio = new Flat_in_out(<double*> c_A_proposal.data,
											<double*> c_B_proposal.data,
	               							N, M,
	               		     				terminals)
	fio.compute_probas_flat(samples,
					   		<double*> likelihoods.data)
	del fio
	return likelihoods

def produce_sentences(np.ndarray A_proposal,
					  np.ndarray B_proposal,
					  terminals,
					  n_sentences):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = A_proposal
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = B_proposal
	cdef Flat_in_out* fio = new Flat_in_out(<double*> c_A_proposal.data,
											<double*> c_B_proposal.data,
	               							N, M,
	               		     				terminals)
	cdef vector[vector[string]] sentences = fio.produce_sentences(n_sentences)
	del fio
	return sentences

def estimate_A_B(np.ndarray A_proposal,
				 np.ndarray B_proposal,
				 terminals,
				 samples):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] likelihoods = np.zeros(n_samples, dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = A_proposal
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = B_proposal
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_new_A = np.zeros((N, N, N), dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_new_B = np.zeros((N, M), dtype = DTYPE)
	cdef Flat_in_out* fio = new Flat_in_out(<double*> c_A_proposal.data,
											<double*> c_B_proposal.data,
	               							N, M,
	               		     				terminals)
	fio.estimate_A_B(samples,
		   		     <double*> likelihoods.data,
	   				 <double*> c_new_A.data,
   		  			 <double*> c_new_B.data)
	del fio
	return c_new_A, c_new_B, likelihoods

def iterate_estimation(np.ndarray A_proposal,
					   np.ndarray B_proposal,
					   terminals,
					   samples,
					   n_iterations):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = np.copy(A_proposal)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = np.copy(B_proposal)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_new_A = np.zeros((N, N, N), dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_new_B = np.zeros((N, M), dtype = DTYPE)
	cdef Flat_in_out* fio
	cdef np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] temp_likelihoods
	likelihoods = np.zeros((n_iterations + 1, n_samples), dtype = DTYPE)
	for iter in xrange(n_iterations):
		fio = new Flat_in_out(<double*> c_A_proposal.data,
							  <double*> c_B_proposal.data,
   							  N, M,
   		     				  terminals)
		temp_likelihoods = np.zeros(n_samples, dtype = DTYPE)
		fio.estimate_A_B(samples, 
						 <double *> temp_likelihoods.data,
						 <double*> c_new_A.data,
						 <double*> c_new_B.data)
		likelihoods[iter, :] = temp_likelihoods
		del fio
		c_A_proposal = c_new_A
		c_B_proposal = c_new_B
	fio = new Flat_in_out(<double*> c_A_proposal.data,
						  <double*> c_B_proposal.data,
						  N, M,
	     				  terminals)
	temp_likelihoods = np.zeros(n_samples, dtype = DTYPE)
	fio.compute_probas_flat(samples,
					   	    <double*> temp_likelihoods.data)
	likelihoods[n_iterations, :] = temp_likelihoods
	del c_A_proposal
	del c_B_proposal
	del fio
	return c_new_A, c_new_B, likelihoods

def iterate_estimation(np.ndarray A_proposal,
					   np.ndarray B_proposal,
					   terminals,
					   samples,
					   n_iterations):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = np.copy(A_proposal)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = np.copy(B_proposal)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_new_A = np.zeros((N, N, N), dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_new_B = np.zeros((N, M), dtype = DTYPE)
	cdef Flat_in_out* fio
	cdef np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] temp_likelihoods
	likelihoods = np.zeros((n_iterations + 1, n_samples), dtype = DTYPE)
	for iter in xrange(n_iterations):
		fio = new Flat_in_out(<double*> c_A_proposal.data,
							  <double*> c_B_proposal.data,
   							  N, M,
   		     				  terminals)
		temp_likelihoods = np.zeros(n_samples, dtype = DTYPE)
		fio.estimate_A_B(samples, 
						 <double *> temp_likelihoods.data,
						 <double*> c_new_A.data,
						 <double*> c_new_B.data)
		likelihoods[iter, :] = temp_likelihoods
		del fio
		c_A_proposal = c_new_A
		c_B_proposal = c_new_B
	fio = new Flat_in_out(<double*> c_A_proposal.data,
						  <double*> c_B_proposal.data,
						  N, M,
	     				  terminals)
	temp_likelihoods = np.zeros(n_samples, dtype = DTYPE)
	fio.compute_probas_flat(samples,
					   	    <double*> temp_likelihoods.data)
	likelihoods[n_iterations, :] = temp_likelihoods
	del c_A_proposal
	del c_B_proposal
	del fio
	return c_new_A, c_new_B, likelihoods

def iterate_estimation_perturbated(np.ndarray A_proposal,
					   np.ndarray B_proposal,
					   terminals,
					   samples,
					   n_iterations,
					   noise_source_A,
					   param_1_A,
					   param_2_A,
					   epsilon_A,
					   noise_source_B,
					   param_1_B,
					   param_2_B,
					   epsilon_B,
					   dampen):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = np.copy(A_proposal)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = np.copy(B_proposal)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] c_new_A = np.zeros((N, N, N), dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] c_new_B = np.zeros((N, M), dtype = DTYPE)
	cdef Flat_in_out* fio
	cdef np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] temp_likelihoods
	likelihoods = np.zeros((n_iterations + 1, n_samples), dtype = DTYPE)
	for iter in xrange(n_iterations):
		fio = new Flat_in_out(<double*> c_A_proposal.data,
							  <double*> c_B_proposal.data,
   							  N, M,
   		     				  terminals)
		temp_likelihoods = np.zeros(n_samples, dtype = DTYPE)
		fio.estimate_A_B(samples, 
						 <double *> temp_likelihoods.data,
						 <double*> c_new_A.data,
						 <double*> c_new_B.data)
		likelihoods[iter, :] = temp_likelihoods
		del fio
		c_A_proposal = c_new_A + noise_source_A(param_1_A, param_2_A * np.power(1.0 / float(n_iterations + 1.0), dampen), (N, N, N))
		c_B_proposal = c_new_B + noise_source_B(param_1_B, param_1_A * np.power(1.0 / float(n_iterations + 1.0), dampen), (N, M))
		c_A_proposal = np.maximum(c_A_proposal, np.power(1.0 / float(n_iterations + 1.0), dampen) * epsilon_A * np.ones((N, N, N), dtype = np.double))
		c_B_proposal = np.maximum(c_B_proposal, np.power(1.0 / float(n_iterations + 1.0), dampen) * epsilon_B * np.ones((N, M), dtype = np.double))
		for i in xrange(N):
			total = np.sum(c_A_proposal[i]) + np.sum(c_B_proposal[i])
			c_A_proposal[i] /= total
			c_B_proposal[i] /= total
	fio = new Flat_in_out(<double*> c_A_proposal.data,
						  <double*> c_B_proposal.data,
						  N, M,
	     				  terminals)
	temp_likelihoods = np.zeros(n_samples, dtype = DTYPE)
	fio.compute_probas_flat(samples,
					   	    <double*> temp_likelihoods.data)
	likelihoods[n_iterations, :] = temp_likelihoods
	del c_A_proposal
	del c_B_proposal
	del fio
	return c_new_A, c_new_B, likelihoods