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
		Dummy()
		void set_content(pair[T, T] content)
		void print_content()

def print_content(content):
	cdef pair[string, string] my_pair
	my_pair.first = 'michel'
	my_pair.second = 'mathieu'
	print my_pair.first
	print my_pair.second
	cdef Dummy[string] c_dummy
	c_dummy.set_content(content)
	c_dummy.print_content()
	cdef vector[int] *vect_to_show = new vector[int]()
	for i in xrange(10):
		vect_to_show.push_back(i*i)
	print deref(vect_to_show)