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
		                          	vector[string] & rule_names,
		                           	vector[pair[string, string]] & hashed_rhs,
		                    	   	vector[vector[double]] & relative_counts,
		                         	vector[vector[int]] & absolute_counts,
		                          	vector[int] & levels,
		                           	vector[int] & depths,
		                           	vector[double] & divergences)
	
def run_proba_sequitur(inference_content,
					   count_content,
					   deg,
					   max):
	cdef vector[vector[string]] inf_parsed
	cdef vector[vector[string]] cou_parsed
	cdef vector[string] codes
	cdef vector[string] names
	cdef vector[pair[string, string]] rhs_codes
	cdef vector[vector[double]] rel_counts
	cdef vector[vector[int]] abs_counts
	cdef vector[int] levs
	cdef vector[int] deps
	cdef vector[double] divs
	launch_proba_sequitur(inference_content,
						  count_content,
						  deg,
						  max,
						  inf_parsed,
						  cou_parsed,
						  codes,
						  names,
						  rhs_codes,
						  rel_counts,
						  abs_counts,
						  levs,
						  deps,
						  divs)
	count_indices = range(len(count_content))
	rules = {}
	rule_names = {}
	relative_count_dict = {}
	absolute_count_dict = {}
	level_dict = {}
	depth_dict = {}
	for i, hashcode in enumerate(codes):
		rules[hashcode] = rhs_codes[i]
		rule_names[hashcode] = names[i]
		relative_count_dict[hashcode] = dict(zip(count_indices, rel_counts[i]))
		absolute_count_dict[hashcode] = dict(zip(count_indices, abs_counts[i]))
		level_dict[hashcode] = levs[i]
		depth_dict[hashcode] = deps[i]
	return {'inference_parsed' : inf_parsed,
			'count_parsed' : cou_parsed,
			'rules' : rules,
			'rule_names': rule_names,
			'relative_counts' : relative_count_dict,
			'absolute_counts' : absolute_count_dict,
			'levels' : level_dict,
			'depths' : depth_dict,
			'divergences' : divs}
	
	
	
	
	