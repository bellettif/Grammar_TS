import numpy as np
cimport numpy as np
np.import_array()
cimport libc.stdlib
import ctypes

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from cython.operator cimport dereference as deref

DTYPE = np.double
ctypedef np.double_t DTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

# Linking with the c code
cdef extern from "dummy.h":
	cdef cppclass Dummy[T]:
		Dummy(vector[T] & content)
		void modify_content(int i,
							const T & value)
		vector[T] & get_content()

def modify_list(list_of_strings,
				list_of_modifications):
	cdef Dummy[string] * c_dummy = new Dummy[string](list_of_strings)
	for key, value in list_of_modifications.iteritems():
		c_dummy.modify_content(key, value)
	list_of_strings = c_dummy.get_content()
	del c_dummy
	return list_of_strings