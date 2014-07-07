#include "inside.h"

/*
*  Run the inside part of the algorithm on subsequences on length == level
*
*  E is the linear flattened version of a 3D array of dimension
*      N * length * length
*  Level in the length of the subsequence that is currently being examined
*      in sample.
*  The results will be stored in E
*
*/
__device__ void compute_inside_level_symbol(
			int level,						// Level in E (length of substring)
			int i,							// Targetted symbol
			float * E,						// Pointer to corresponding E array
			int * sample,					// Pointer to corresponding sample
			int length,						// Length of sample
			Compact_grammar * cmpct_g		// Characteristics of the grammar
			){
	int N = cmpct_g->_N;
	int M = cmpct_g->_M;
	float * A = cmpct_g->_dev_A;
	float * B = cmpct_g->_dev_B;
	if(level == 0){
		for(int i = 0; i < N; ++i){
			for(int s = 0; s < length; ++s){
				E[i*length*length + s*length + s] = B[i*M + sample[s]];
			}
		}
	}else{
		int t;
		int s;
		int r;
		int j;
		int k;
		float temp_sum;
		float E_left;
		float E_right;
		for(s = 0; s < length - level; ++s){
			t = s + level;
			temp_sum = 0;
			for(r = s; r < t; ++r){
				E_left = E[j*length*length + s*length + r];
				E_right = E[k*length*length + (r+1)*length + t];
				for(j = 0; j < N; ++j){
					for(k = 0; k < N; ++k){
						/*
						 *  i is the index of parent the non term symbol
						 *  s is the index of the starting position in the parsed sequence
						 *  t is the index of the ending position in the parsed sequence
						 *      Here we have t - s = level as per the definition of the function
						 *  j is the index of the left child non term symbol
						 *  k is the index of the right child non term symbol
						 *
						 */
						 temp_sum +=
								A[i*N*N + j*N + k]
								*
								E_left
								*
								E_right;
					}
					E[i*length*length + s*length + t] = temp_sum;
				}
			}
		}
	}
}

__device__ void init_E(
		int i,
		int length,
		float * E,
		){
	for(int s = 0; s < length; ++s){
		for(int t = 0; t < length; ++t){
			E[i*length*length + s*length + t] = 0.0;
		}
	}
}

/*
*  Run the inside part of the algorithm on all subsequences of sample
*
*  CUDA KERNEL
*
*  The result will be stored in E whose dimension is N * length * length
*      (flattenned version of the 3D array)
*
*  Es contains the E arrays of all samples, the stride is N_symbols * MAX_LENGTH * MAX_LENGTH;
*
*  There is one thread per symbol per sample
*
*/
__global__ void compute_inside_probas_kernel(
		float* Es,
		int * samples,
		int * lengths,
		Compact_grammar * cmpct_g,
		int n_samples,
		int level,
		){
	int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(tid >= n_samples * MAX_LENGTH) return;

	int N = cmpct->_N;
	int E_stride = N * MAX_LENGTH * MAX_LENGTH;

	int sample_id = tid / MAX_LENGTH;
	int symbol_id = tid % N;

	int length = lengths[sample_id];

	if(level == -1){
		init_E(symbol_id,
			   length,
			   E + sample_id * E_stride);
	}else{
		if(level >= length) return;
		compute_inside_level_symbol(
				level,
				symbol_id,
				E + sample_id * E_stride,
				samples + sample_id + MAX_LENGTH,
				length,
				cmpct_g);
	}
}

void compute_inside_probas(
		float* Es,
		int * samples,
		int * lengths,
		Compact_grammar * cmpct_g,
		int n_samples){
	int n_blocks = ceil(((float) n_samples) / ((float) BLOCK_SIZE));
	for(int level = -1; level < MAX_LENGTH; ++level){
		compute_inside_probas_kernel<<<n_bloks, BLOCK_SIZE>>>(
				Es,
				samples,
				lengths,
				cmpct_g,
				n_samples,
				level);
	}
}
