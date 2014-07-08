import numpy as np
cimport numpy as np
np.import_array()
cimport libc.stdlib
import ctypes

class Looping_der_except(Exception):
	def __init__(self, code):
		self.code = code
	def __str__(self):
		return repr(self.code)

from datetime import datetime
from time import mktime

DTYPE = np.double
ctypedef np.double_t DTYPE_t
FTYPE = np.float32
ctypedef np.float32_t FTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.list cimport list
from libcpp cimport bool
from cython.operator cimport dereference as deref

cdef extern from "<random>" namespace "std":
	cdef cppclass mt19937:
		mt19937 ()
		mt19937(unsigned long root)
		
cdef extern from "flat_in_out.h":
	cdef cppclass Flat_in_out:
		Flat_in_out(float* A, float* B,
	               	int N, int M,
	               	const vector[string] & terminals,
	               	int root_index)
		void compute_probas_flat(const vector[vector[string]] & samples,
                           	float * probas)
		void compute_proba_flat(const vector[string] & sample)
		void compute_inside_outside_flat(float* E, float* F,
									const vector[string] & sample)
		void compute_inside_probas_flat(float* E,
								   const vector[string] & sample)
		void estimate_A_B(const vector[vector[string]] & samples,
						  float * sample_probas,
						  float * new_A,
						  float * new_B)
		vector[vector[string]] produce_sentences(int n_sentences,
												 bool & does_not_terminate)
		void compute_frequences(int n_sentences,
                                 vector[int] & freqs,
                                 vector[string] & strings,
                                 bool & does_not_terminate,
                                 int max_length)
	
def estimate_likelihoods(np.ndarray A_proposal,
						 np.ndarray B_proposal,
						 terminals,
						 samples,
						 root_index):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[FTYPE_t, ndim = 1, mode = 'c'] likelihoods = np.zeros(n_samples, dtype = FTYPE)
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = A_proposal
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = B_proposal
	cdef Flat_in_out* fio = new Flat_in_out(<float*> c_A_proposal.data,
											<float*> c_B_proposal.data,
	               							N, M,
	               		     				terminals,
	               		     				root_index)
	fio.compute_probas_flat(samples,
					   		<float*> likelihoods.data)
	del fio
	return likelihoods

def produce_sentences(np.ndarray A_proposal,
					  np.ndarray B_proposal,
					  terminals,
					  n_sentences,
					  root_index):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = A_proposal
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = B_proposal
	cdef Flat_in_out* fio = new Flat_in_out(<float*> c_A_proposal.data,
											<float*> c_B_proposal.data,
	               							N, M,
	               		     				terminals,
	               		     				root_index)
	cdef bool does_not_terminate
	cdef vector[vector[string]] sentences = fio.produce_sentences(n_sentences,
																does_not_terminate)
	if does_not_terminate:
		raise Looping_der_except('Derivation does not terminate in produce sentences')
	del fio
	return sentences

def estimate_A_B(np.ndarray A_proposal,
				 np.ndarray B_proposal,
				 terminals,
				 samples,
				 root_index = 0):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[FTYPE_t, ndim = 1, mode = 'c'] likelihoods = np.zeros(n_samples, dtype = FTYPE)
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = A_proposal
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = B_proposal
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_new_A = np.zeros((N, N, N), dtype = FTYPE)
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_new_B = np.zeros((N, M), dtype = FTYPE)
	cdef Flat_in_out* fio = new Flat_in_out(<float*> c_A_proposal.data,
											<float*> c_B_proposal.data,
	               							N, M,
	               		     				terminals,
	               		     				root_index)
	fio.estimate_A_B(samples,
		   		     <float*> likelihoods.data,
	   				 <float*> c_new_A.data,
   		  			 <float*> c_new_B.data)
	del fio
	return c_new_A, c_new_B, likelihoods

def iterate_estimation(np.ndarray A_proposal,
					   np.ndarray B_proposal,
					   terminals,
					   samples,
					   n_iterations,
					   root_index):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = np.copy(A_proposal)
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = np.copy(B_proposal)
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_new_A = np.zeros((N, N, N), dtype = FTYPE)
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_new_B = np.zeros((N, M), dtype = FTYPE)
	cdef Flat_in_out* fio
	cdef np.ndarray[FTYPE_t, ndim = 1, mode = 'c'] temp_likelihoods
	likelihoods = np.zeros((n_iterations + 1, n_samples), dtype = FTYPE)
	for iter in xrange(n_iterations):
		fio = new Flat_in_out(<float*> c_A_proposal.data,
							  <float*> c_B_proposal.data,
   							  N, M,
   		     				  terminals,
   		     				  root_index)
		temp_likelihoods = np.zeros(n_samples, dtype = FTYPE)
		fio.estimate_A_B(samples, 
						 <float *> temp_likelihoods.data,
						 <float *> c_new_A.data,
						 <float *> c_new_B.data)
		likelihoods[iter, :] = temp_likelihoods
		del fio
		c_A_proposal = c_new_A
		c_B_proposal = c_new_B
	fio = new Flat_in_out(<float*> c_A_proposal.data,
						  <float*> c_B_proposal.data,
						  N, M,
	     				  terminals,
	     				  root_index)
	temp_likelihoods = np.zeros(n_samples, dtype = FTYPE)
	fio.compute_probas_flat(samples,
					   	    <float*> temp_likelihoods.data)
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
					   dampen,
					   root_index):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	n_samples = len(samples)
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = np.copy(A_proposal)
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = np.copy(B_proposal)
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_new_A = np.zeros((N, N, N), dtype = FTYPE)
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_new_B = np.zeros((N, M), dtype = FTYPE)
	cdef Flat_in_out* fio
	cdef np.ndarray[FTYPE_t, ndim = 1, mode = 'c'] temp_likelihoods
	likelihoods = np.zeros((n_iterations + 1, n_samples), dtype = FTYPE)
	for iter in xrange(n_iterations):
		fio = new Flat_in_out(<float*> c_A_proposal.data,
							  <float*> c_B_proposal.data,
   							  N, M,
   		     				  terminals,
   		     				  root_index)
		temp_likelihoods = np.zeros(n_samples, dtype = FTYPE)
		fio.estimate_A_B(samples, 
						 <float*> temp_likelihoods.data,
						 <float*> c_new_A.data,
						 <float*> c_new_B.data)
		likelihoods[iter, :] = temp_likelihoods
		del fio
		c_A_proposal = c_new_A + noise_source_A(param_1_A, param_2_A * np.power(1.0 / float(n_iterations + 1.0), dampen), (N, N, N))
		c_B_proposal = c_new_B + noise_source_B(param_1_B, param_1_A * np.power(1.0 / float(n_iterations + 1.0), dampen), (N, M))
		c_A_proposal = np.maximum(c_A_proposal, np.power(1.0 / float(n_iterations + 1.0), dampen) * epsilon_A * np.ones((N, N, N), dtype = FTYPE))
		c_B_proposal = np.maximum(c_B_proposal, np.power(1.0 / float(n_iterations + 1.0), dampen) * epsilon_B * np.ones((N, M), dtype = FTYPE))
		for i in xrange(N):
			total = np.sum(c_A_proposal[i]) + np.sum(c_B_proposal[i])
			c_A_proposal[i] /= total
			c_B_proposal[i] /= total
	fio = new Flat_in_out(<float*> c_A_proposal.data,
						  <float*> c_B_proposal.data,
						  N, M,
	     				  terminals,
	     				  root_index)
	temp_likelihoods = np.zeros(n_samples, dtype = FTYPE)
	fio.compute_probas_flat(samples,
					   	    <float*> temp_likelihoods.data)
	likelihoods[n_iterations, :] = temp_likelihoods
	del c_A_proposal
	del c_B_proposal
	del fio
	return c_new_A, c_new_B, likelihoods

def compute_stats(np.ndarray A_proposal,
				  np.ndarray B_proposal,
				  terminals,
				  n_sentences,
				  max_length,
				  root_index):
	assert(A_proposal.shape[0] == A_proposal.shape[1] == A_proposal.shape[2] == B_proposal.shape[0])
	assert(B_proposal.shape[1] == len(terminals))
	N = A_proposal.shape[0]
	M = B_proposal.shape[1]
	cdef np.ndarray[FTYPE_t, ndim = 3, mode = 'c'] c_A_proposal = A_proposal
	cdef np.ndarray[FTYPE_t, ndim = 2, mode = 'c'] c_B_proposal = B_proposal
	cdef Flat_in_out* fio = new Flat_in_out(<float*> c_A_proposal.data,
											<float*> c_B_proposal.data,
	               							N, M,
	               		     				terminals,
	               		     				root_index)
	cdef vector[string] strings
	cdef vector[int] freqs
	cdef bool does_not_terminate
	fio.compute_frequences(n_sentences, 
						   freqs,
						   strings, 
						   max_length,
						   does_not_terminate)
	if does_not_terminate:
		raise Looping_der_except('Derivation does not terminate in compute stats')
	del fio
	return freqs, strings
	