#include <iostream>
#include <string>

#include "k_sequitur_lk/k_sequitur.h"

static void cpp_exec_k_seq(int * raw_input,
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
					int * pop_count_array){
	K_sequitur<int> k_sequitur(raw_input,
							   input_length,
							   max_occurences);
	for(int i = 0; i < input_length - 1; ++i){
		k_sequitur.next();
	}
	k_sequitur.collapse_grammar(rule_trimming);
	k_sequitur.fill_with_info(lhs_array, n_lhs,
							  input_length,
							  input_string, input_string_length,
							  rhs_array, rhs_lengths,
							  barcode_array, barcode_lengths,
							  ref_count_array, pop_count_array);
}

