import numpy as np
cimport numpy as np
np.import_array()
cimport libc.stdlib
import ctypes
import time

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from cython.operator cimport dereference as deref

cdef extern from "<random>" namespace "std":
	cdef cppclass mt19937:
		mt19937 ()
		mt19937(unsigned root)

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
		double*** get_A()
		int get_n_terms()
		double** get_B()
		
def create_c_rule(input_sto_rule):
	cdef mt19937 *rng = new mt19937(time.clock())
	cdef Stochastic_rule *my_rule = new Stochastic_rule(input_sto_rule.rule_name,
														input_sto_rule.non_term_w,
														input_sto_rule.non_term_s,
														deref(rng),
														input_sto_rule.term_w,
														input_sto_rule.term_s)
	my_rule.print_rule()
	del my_rule
	del rng
	
def create_c_grammar(list_of_rules,
					 root_symbol):
	cdef mt19937 *rng = new mt19937(time.clock())
	cdef vector[Stochastic_rule] *c_list_of_rules = new vector[Stochastic_rule]()
	for rule in list_of_rules:
		c_list_of_rules.push_back(Stochastic_rule(rule.rule_name,
												  rule.non_term_w,
												  rule.non_term_s,
												  deref(rng),
												  rule.term_w,
												  rule.term_s))
	cdef SCFG *grammar = new SCFG(deref(c_list_of_rules), root_symbol)
	grammar.print_params()
	cdef int N = grammar.get_n_non_terms()
	cdef int M = grammar.get_n_terms()
	cdef double*** A = grammar.get_A()
	cdef double** B = grammar.get_B()
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
	return A_converted, B_converted
	
	