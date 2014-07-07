/*
 * sentence_prod.cuh
 *
 *  Created on: 6 juil. 2014
 *      Author: francois
 */

#ifndef SENTENCE_PROD_CUH_
#define SENTENCE_PROD_CUH_

#include "limitations.h"

#include "cu_choice.cuh"
#include "cuda_dimensions.h"

#include "compact_grammar.h"

__device__ void produce_sentence(
		curandState * state,
		int * sentence,
		int * length,
		int * error_status,
		bool * dev_todo,
		bool * dev_next_todo,
		int * dev_buffer,
		const Compact_grammar * dev_cmpct_g);

__global__ void produce_sentence_kernel(
		curandState * state_array,
		int * dev_sentences,
		int * dev_lengths,
		int * dev_error_status,
		int n_sentences,
		bool * dev_todo,
		bool * dev_next_todo,
		int * dev_buffer,
		Compact_grammar * dev_cmpct_g);

#endif /* SENTENCE_PROD_CUH_ */
