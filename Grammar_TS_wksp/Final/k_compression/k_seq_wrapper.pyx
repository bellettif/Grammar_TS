import numpy as np
cimport numpy as np
np.import_array()
cimport libc.stdlib
import ctypes

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

# Linking with the c code
cdef extern from "k_seq.cpp":
	void cpp_exec_k_seq(int * raw_input,
					int input_length,
					int max_occurences,
					int rule_trimming,
					int * lhs_array,
					int & n_lhs,
					int * input_string,
					int & input_string_length,
					int * rhs_array,
					int * rhs_lengths,
					int * barcode_array,
					int * barcode_lengths,
					int * ref_count_array,
					int * pop_count_array)

# Be careful, 0 is always a reserved root symbol (this can changed)
#
#	max occurences is k, rule trimming is k
#
def run(np.ndarray symbols,
		   max_occurences = 2,
		   rule_trimming = 2):
	# Input build phase
	length = len(symbols)
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_lhs_array = np.zeros(length, dtype = ITYPE)
	cdef int n_lhs
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_input_string = np.zeros(length, dtype = ITYPE)
	cdef int input_string_length
	cdef np.ndarray[ITYPE_t, ndim = 2, mode = 'c'] c_rhs_array = np.zeros((length, length), dtype = ITYPE)
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_rhs_lengths = np.zeros(length, dtype = ITYPE)
	cdef np.ndarray[ITYPE_t, ndim = 2, mode = 'c'] c_barcode_array = np.zeros((length, length), dtype = ITYPE)
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_barcode_lengths = np.zeros(length, dtype = ITYPE)
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_ref_count_array = np.zeros(length, dtype = ITYPE)
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_pop_count_array = np.zeros(length, dtype = ITYPE)
	
	# Execution phase
	cpp_exec_k_seq(<int *> symbols.data,
				   length,
				   max_occurences,
				   rule_trimming,
				   <int *> c_lhs_array.data,
				   n_lhs,
				   <int *> c_input_string.data,
				   input_string_length,
				   <int *> c_rhs_array.data,
				   <int *> c_rhs_lengths.data,
				   <int *> c_barcode_array.data,
				   <int *> c_barcode_lengths.data,
				   <int *> c_ref_count_array.data,
				   <int *> c_pop_count_array.data)
	
	# Trimming phase
	grammar = {}	
	grammar[0] = [c_input_string[:input_string_length],
						0]
	for i in xrange(n_lhs):
		grammar[c_lhs_array[i]] = [c_rhs_array[i, :c_rhs_lengths[i]],
										c_pop_count_array[i]]
	return grammar