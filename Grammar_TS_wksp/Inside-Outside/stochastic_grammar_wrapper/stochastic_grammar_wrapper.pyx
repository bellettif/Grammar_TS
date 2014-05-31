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

cdef extern from "scfg.h":
	cdef cppclass SCFG
	cdef cppclass Stochastic_rule:
		Stochastic_rule(const int & rule_name,
						const vector[double] & non_term_w,
						const vector[pair[int, int]] & non_term_s,
						mt19937 & RNG,
						const vector[double] & term_w,
						const vector[string] & term_s)
		void print_rule()
		list[string] complete_derivation(SCFG & grammar)
	cdef cppclass SCFG:
		SCFG(const vector[Stochastic_rule] & rules,
        	 const int & root_symbol)
		Stochastic_rule & get_rule(int i)
		void print_symbols()
		void print_params()
		int get_n_non_terms()
		vector[int] get_index_to_non_term_vect()
		double*** get_A()
		int get_n_terms()
		vector[string] get_index_to_term_vect()
		double** get_B()
		
cdef extern from "in_out_proba.h":
	cdef cppclass In_out_proba:
		In_out_proba(const SCFG & grammar,
					 const vector[string] & input,
					 double * A,
					 double * B)
		void get_inside_outside(double*** & E,
                            	double*** & F,
                             	int & N,
                              	int & M,
                               	int & length)
		void check_integrity()
		double run_CYK()
		void print_A_and_B()
		
cdef extern from "model_estimator.h":
	cdef cppclass Model_estimator:
		Model_estimator(const SCFG & grammar,
						const vector[vector[string]] & inputs,
						double * initial_A_guess,
						double * initial_B_guess)
		double*** get_A_estim()
		double** get_B_estim()
		void estimate_from_inputs()
		void swap_model_estim()
		void print_estimates()
		
def print_c_rule(input_sto_rule):
	now = datetime.now()
	dt = datetime.now()
	sec_since_epoch = mktime(dt.timetuple()) + dt.microsecond/1000000.0
	millis_since_epoch = sec_since_epoch * 1000
	cdef mt19937 *rng = new mt19937(<unsigned long> millis_since_epoch)
	cdef Stochastic_rule *my_rule = new Stochastic_rule(input_sto_rule.rule_name,
														input_sto_rule.non_term_w,
														input_sto_rule.non_term_s,
														deref(rng),
														input_sto_rule.term_w,
														input_sto_rule.term_s)
	my_rule.print_rule()
	del my_rule
	del rng
	
def compute_parameters(input_grammar):
	now = datetime.now()
	dt = datetime.now()
	sec_since_epoch = mktime(dt.timetuple()) + dt.microsecond/1000000.0
	millis_since_epoch = sec_since_epoch * 1000
	cdef mt19937 *rng = new mt19937(<unsigned long> millis_since_epoch)
	cdef vector[Stochastic_rule] *c_list_of_rules = new vector[Stochastic_rule]()
	for rule in input_grammar.grammar.values():
		c_list_of_rules.push_back(Stochastic_rule(rule.rule_name,
												  rule.non_term_w,
												  rule.non_term_s,
												  deref(rng),
												  rule.term_w,
												  rule.term_s))
	cdef SCFG *grammar = new SCFG(deref(c_list_of_rules),
								input_grammar.root_symbol)
	cdef int N = grammar.get_n_non_terms()
	cdef int M = grammar.get_n_terms()
	cdef double*** A = grammar.get_A()
	cdef double** B = grammar.get_B()
	cdef vector[int] index_to_non_term = grammar.get_index_to_non_term_vect()
	cdef vector[string] index_to_term = grammar.get_index_to_term_vect()
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] A_converted = np.zeros((N, N, N), dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] B_converted = np.zeros((N, M), dtype = DTYPE)
	for i in xrange(N):
		for j in xrange(N):
			for k in xrange(N):
				A_converted[i, j, k] = A[i][j][k]
		for j in xrange(M):
			B_converted[i, j] = B[i][j]
	del grammar
	del rng
	del c_list_of_rules
	return A_converted, B_converted, index_to_non_term, index_to_term
	
def compute_inside_outside(input_grammar,
						   input_sentence,
						   np.ndarray proposal_A,
						   np.ndarray proposal_B):
	now = datetime.now()
	dt = datetime.now()
	sec_since_epoch = mktime(dt.timetuple()) + dt.microsecond/1000000.0
	millis_since_epoch = sec_since_epoch * 1000
	cdef mt19937 *rng = new mt19937(<unsigned long> millis_since_epoch)
	cdef vector[Stochastic_rule] *c_list_of_rules = new vector[Stochastic_rule]()
	for rule in input_grammar.grammar.values():
		c_list_of_rules.push_back(Stochastic_rule(rule.rule_name,
												  rule.non_term_w,
												  rule.non_term_s,
												  deref(rng),
												  rule.term_w,
												  rule.term_s))
	cdef SCFG *grammar = new SCFG(deref(c_list_of_rules),
								input_grammar.root_symbol)
	cdef In_out_proba * proba_cmpter = new In_out_proba(deref(grammar),
				 										input_sentence,
			 				 							<double *> proposal_A.data,
				 			 				 			<double *> proposal_B.data)
	cdef int length = 0
	cdef int N = 0
	cdef int M = 0
	cdef double*** E
	cdef double*** F
	proba_cmpter.get_inside_outside(E, F, N, M, length)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] E_converted = np.zeros((N, length, length),
																		   dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] F_converted = np.zeros((N, length, length),
																		   dtype = DTYPE)
	for i in xrange(N):
		for j in xrange(length):
			for k in xrange(length):
				E_converted[i, j, k] = E[i][j][k]
				F_converted[i, j, k] = F[i][j][k]
	del proba_cmpter
	del grammar
	del c_list_of_rules
	del rng
	return E_converted, F_converted
	
def compute_derivations(target_rule,
					    input_grammar,
					    n):
	now = datetime.now()
	dt = datetime.now()
	sec_since_epoch = mktime(dt.timetuple()) + dt.microsecond/1000000.0
	millis_since_epoch = sec_since_epoch * 1000
	cdef mt19937 *rng = new mt19937(<unsigned long> millis_since_epoch)
	cdef vector[Stochastic_rule] *c_list_of_rules = new vector[Stochastic_rule]()
	for rule in input_grammar.grammar.values():
		c_list_of_rules.push_back(Stochastic_rule(rule.rule_name,
												  rule.non_term_w,
												  rule.non_term_s,
												  deref(rng),
												  rule.term_w,
												  rule.term_s))
	cdef SCFG *grammar = new SCFG(deref(c_list_of_rules),
								input_grammar.root_symbol)
	cdef Stochastic_rule *c_target_rule = new Stochastic_rule(target_rule.rule_name,
															  target_rule.non_term_w,
															  target_rule.non_term_s,
															  deref(rng),
															  target_rule.term_w,
															  target_rule.term_s)
	cdef vector[list[string]] *der_results = new vector[list[string]]()
	for i in xrange(n):
		der_results.push_back(c_target_rule.complete_derivation(deref(grammar)))
	del c_target_rule
	del grammar
	del c_list_of_rules
	del rng
	return deref(der_results)
	
def estimate_model(input_grammar,
				   input_sentences,
				   np.ndarray initial_A_guess,
				   np.ndarray initial_B_guess,
				   n_its):
	now = datetime.now()
	dt = datetime.now()
	sec_since_epoch = mktime(dt.timetuple()) + dt.microsecond/1000000.0
	millis_since_epoch = sec_since_epoch * 1000
	cdef mt19937 *rng = new mt19937(<unsigned long> millis_since_epoch)
	cdef vector[Stochastic_rule] *c_list_of_rules = new vector[Stochastic_rule]()
	for rule in input_grammar.grammar.values():
		c_list_of_rules.push_back(Stochastic_rule(rule.rule_name,
												  rule.non_term_w,
												  rule.non_term_s,
												  deref(rng),
												  rule.term_w,
												  rule.term_s))
	cdef SCFG *grammar = new SCFG(deref(c_list_of_rules),
								input_grammar.root_symbol)
	cdef Model_estimator * estimator = new Model_estimator(deref(grammar),
														   input_sentences,
														   <double *> initial_A_guess.data,
														   <double *> initial_B_guess.data)
	for current_it in xrange(n_its - 1):
		estimator.estimate_from_inputs()
		estimator.swap_model_estim()
	estimator.estimate_from_inputs()
	cdef double*** A_estim = estimator.get_A_estim()
	cdef double** B_estim = estimator.get_B_estim()
	cdef int N = grammar.get_n_non_terms()
	cdef int M = grammar.get_n_terms()
	cdef np.ndarray[DTYPE_t, ndim = 3, mode = 'c'] A_estim_converted = np.zeros((N, N, N), dtype = DTYPE)
	cdef np.ndarray[DTYPE_t, ndim = 2, mode = 'c'] B_estim_converted = np.zeros((N, M), dtype = DTYPE)
	for i in xrange(N):
		for j in xrange(N):
			for k in xrange(N):
				A_estim_converted[i, j, k] = A_estim[i][j][k]
		for j in xrange(M):
			B_estim_converted[i, j] = B_estim[i][j]
	del estimator
	del grammar
	del c_list_of_rules
	del rng
	return A_estim_converted, B_estim_converted