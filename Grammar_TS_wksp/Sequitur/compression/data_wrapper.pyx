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
cdef extern from "seq.cpp":
	void exec_sequitur(int * content,
					   int root_symbol,
					   int n_symbols,
					   int * rhs_array,
					   int max_length,
					   int & length_overflow,
					   int * final_stream,
					   int & stream_length,
					   int & stream_ref_count,
					   int * lhs,
					   int * rhs_lengths,
					   int * ref_counts,
					   int & n_rules)

# Be careful, 0 is always a reserved root symbol (this can changed)
def run(np.ndarray symbols, max_length = 10):
	# Input build phase
	cdef int c_n_symbols = len(symbols)
	cdef int c_max_length = max_length
	cdef int c_root_symbol = 0
	cdef np.ndarray[ITYPE_t, ndim = 2, mode = 'c'] c_rhs_array = np.zeros((c_n_symbols, c_max_length), dtype = ITYPE)
	cdef int c_length_overflow = 0
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_final_stream = np.zeros(c_n_symbols, dtype = ITYPE)								
	cdef int c_stream_length
	cdef int c_stream_ref_count
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_lhs = np.zeros(c_n_symbols, dtype = ITYPE)
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_rhs_lengths = np.zeros(c_n_symbols, dtype = ITYPE)
	cdef np.ndarray[ITYPE_t, ndim = 1, mode = 'c'] c_ref_counts = np.zeros(c_n_symbols, dtype = ITYPE)		
	cdef int c_n_rules
	
	# Execution phase
	exec_sequitur(<int *> symbols.data,
				  c_root_symbol,
			  	  c_n_symbols,
			  	  <int *> c_rhs_array.data, #Numpy arrays are represented as c unidimensional arrays
			  	  c_max_length,
			  	  c_length_overflow,
			  	  <int *> c_final_stream.data,
			  	  c_stream_length,
			  	  c_stream_ref_count,
			  	  <int *> c_lhs.data,
			  	  <int *> c_rhs_lengths.data,
			  	  <int *> c_ref_counts.data,
			  	  c_n_rules)
	
	# Trimming phase
	grammar = {}
	grammar[c_root_symbol] = [c_final_stream[:c_stream_length],
									c_stream_ref_count]
	for i in xrange(c_n_rules):
		grammar[c_lhs[i]] = [c_rhs_array[i, : c_rhs_lengths[i]],
								   c_ref_counts[i]]
	return grammar