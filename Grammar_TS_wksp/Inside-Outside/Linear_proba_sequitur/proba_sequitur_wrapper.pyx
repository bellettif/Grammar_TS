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
	
cdef extern from "launcher.h":
	cdef void launch_proba_sequitur(const vector[vector[string]] & inference_content,
		                        	const vector[vector[string]] & count_content,
		                        	int degree, int max_rules,
		                         	vector[vector[string]] & inference_parsed,
		                         	vector[vector[string]] & counts_parsed,
		                          	vector[string] & hashcodes,
		                           	vector[pair[string, string]] & hashed_rhs,
		                    	   	vector[vector[double]] & relative_counts,
		                         	vector[vector[int]] & absolute_counts,
		                          	vector[int] & levels,
		                           	vector[int] & depths)
	
def run_proba_sequitur(inference_content,
					   count_content,
					   deg,
					   max):
	cdef vector[vector[string]] inf_parsed
	cdef vector[vector[string]] cou_parsed
	cdef vector[string] codes
	cdef vector[pair[string, string]] rhs_codes
	cdef vector[vector[double]] rel_counts
	cdef vector[vector[int]] abs_counts
	cdef vector[int] levs
	cdef vector[int] deps
	launch_proba_sequitur(inference_content,
						  count_content,
						  deg,
						  max,
						  inf_parsed,
						  cou_parsed,
						  codes,
						  rhs_codes,
						  rel_counts,
						  abs_counts,
						  levs,
						  deps)
	count_indices = range(len(count_content))
	rules = {}
	relative_count_dict = {}
	absolute_count_dict = {}
	level_dict = {}
	depth_dict = {}
	for i, hashcode in enumerate(codes):
		rules[hashcode] = rhs_codes[i]
		relative_count_dict[hashcode] = dict(zip(count_indices, rel_counts[i]))
		absolute_count_dict[hashcode] = dict(zip(count_indices, abs_counts[i]))
		level_dict[hashcode] = levs[i]
		depth_dict[hashcode] = deps[i]
	return {'inference_parsed' : inf_parsed,
			'count_parsed' : cou_parsed,
			'rules' : rules,
			'relative_counts' : relative_count_dict,
			'absolute_counts' : absolute_count_dict,
			'levels' : level_dict,
			'depths' : depth_dict}
	
	
	
	
	