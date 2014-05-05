import numpy as np
cimport numpy as np
np.import_array()
cimport libc.stdlib
import ctypes

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

STUFF = "Cython's import is buggy :)"

# Linking with the c code
cdef extern from "entropy_meas.cpp":
	double comp_entropy(int * content,
					  	int length)
	void comp_rolling_entropy(int * content,
							  int length,
							  int k,
							  double * result)
	double comp_string_entropy(int * content,
						  	   int length,
						  	   int k) # k is window length
	double comp_gini(int * content,
					 int length)
	void comp_rolling_gini(int * content,
						   int length,
						   int k,
						   double * result)
	double comp_string_gini(int * content,
							int length,
							int k)

def compute_entropy(np.ndarray symbols):
	# Execution phase
	cdef double entropy = comp_entropy(<int *> symbols.data,
				  					   		   len(symbols))
	# Trimming phase
	return entropy

def compute_rolling_entropy(np.ndarray symbols,
							k):
	cdef np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] c_result = np.zeros(len(symbols) - k + 1, dtype = DTYPE)								
	comp_rolling_entropy(<int *> symbols.data, len(symbols), k, <double *> c_result.data)
	return c_result

def compute_string_entropy(np.ndarray symbols,
							k):
	cdef double entropy = comp_string_entropy(<int *> symbols.data,
											   len(symbols), k)
	return entropy

def compute_gini(np.ndarray symbols):
	# Execution phase
	cdef double gini = comp_gini(<int *> symbols.data,
				  					   		   len(symbols))
	# Trimming phase
	return gini

def compute_rolling_gini(np.ndarray symbols,
							k):
	cdef np.ndarray[DTYPE_t, ndim = 1, mode = 'c'] c_result = np.zeros(len(symbols) - k + 1, dtype = DTYPE)								
	comp_rolling_gini(<int *> symbols.data, len(symbols), k, <double *> c_result.data)
	return c_result


def compute_string_gini(np.ndarray symbols,
						   int k):
	cdef double gini = comp_string_gini(<int *> symbols.data,
												len(symbols),
												k)
	return gini