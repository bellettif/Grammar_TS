#include "sentence_prod.cuh"

__device__ void produce_sentence(
		curandState * state,
		int * sentence,
		int * length,
		int * error_status,
		bool * dev_todo,
		bool * dev_next_todo,
		int * dev_buffer,
		const Compact_grammar * dev_cmpct_g)
{
	*length = 0;
	int last_to_do = 1;
	bool * swap_todo_pt;
	for(int i = 1; i < MAX_LENGTH; ++i){
		dev_todo[i] = false;
	}
	int * swap_pt;
	sentence[0] = dev_cmpct_g->_root_symbol;
	dev_todo[0] = true;
	int offset, s;
	bool emit;
	int i, j, k, term;
	bool keep_going = true;
	int current_iter = 0;
	while(keep_going){
		++ current_iter;
		offset = 0;
		keep_going = false;
		for(s = 0; s < last_to_do; ++s){
			if(dev_todo[s]){
				i = sentence[s];
				if(emit_term(
						state,
						dev_cmpct_g->_dev_non_term_term_dists,
						i
						)
					){
					emit = true;
				}else{
					emit = false;
					keep_going = true;
				}
				if(! emit){
					non_term_derivation(
							state,
							dev_cmpct_g->_dev_A,
							i, j, k,
							dev_cmpct_g->_N
							);
					dev_buffer[s + offset] = j;
					dev_next_todo[s + offset] = true;
					dev_buffer[s + offset + 1] = k;
					dev_next_todo[s + offset + 1] = true;
					++ offset;
				}else{
					term_derivation(
							state,
							dev_cmpct_g->_dev_B,
							i,
							term,
							dev_cmpct_g->_M
							);
					dev_buffer[s + offset] = term;
					dev_next_todo[s + offset] = false;
				}
			}else{
				dev_buffer[s + offset] = sentence[s];
				dev_next_todo[s + offset] = false;
			}
		}
		last_to_do = s + offset;
		*length = s + offset;
		swap_pt = dev_buffer;
		dev_buffer = sentence;
		sentence = swap_pt;
		swap_todo_pt = dev_next_todo;
		dev_next_todo = dev_todo;
		dev_todo = swap_todo_pt;
		if((*length) > MAX_LENGTH){
			*error_status = 1;
			return;
		}
		if(current_iter > MAX_ITERS){
			*error_status = 2;
			return;
		}
	}
	*error_status = 0;
}

__global__ void produce_sentence_kernel(
		curandState * state_array,
		int * dev_sentences,
		int * dev_lengths,
		int * dev_error_status,
		int n_sentences,
		bool * dev_todo,
		bool * dev_next_todo,
		int * dev_buffer,
		Compact_grammar * dev_cmpct_g){
	int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(tid >= n_sentences) return;
	produce_sentence(
			state_array + tid * MAX_LENGTH,
			dev_sentences + tid * MAX_LENGTH,
			dev_lengths + tid,
			dev_error_status + tid,
			dev_todo + tid * MAX_LENGTH,
			dev_next_todo + tid * MAX_LENGTH,
			dev_buffer + tid * MAX_LENGTH,
			dev_cmpct_g
	);
}
