/*
 * cu_choice.cuh
 *
 *  Created on: 5 juil. 2014
 *      Author: francois
 */

#ifndef CU_CHOICE_CUH_
#define CU_CHOICE_CUH_

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "utils.h"
#include "cuda_dimensions.h"

__device__ static int choice(
		curandState * state,
		float * weights,
		int N){
	float unif = curand_uniform(state);
	float cum_sum = 0;
	for(int i = 0; i < N; ++i){
		cum_sum += weights[i];
		if(cum_sum >= unif) return i;
	}
	return N - 1;
}

__device__ static bool emit_term(
		curandState * state,
		float * dev_non_term_term_dists,
		int i){
	return (choice(state,
			dev_non_term_term_dists + i * 2,
			2)
			== 1);
}

__device__ static void non_term_derivation(
		curandState * state,
		float * dev_A,
		int i, int & j, int & k,
		int N){
	int pos = choice(state,
			dev_A + i * N * N,
			N * N);
	j = pos / N;
	k = pos % N;
}

__device__ static void term_derivation(
		curandState * state,
		float * dev_B,
		int i, int & term,
		int M){
	term = choice(state,
			dev_B + i * M,
			M);
}

__global__ void gpu_choice_kernel(
		curandState * state_array,
		float * weights, int N,
		int * samples, int n_samples);

void gpu_choice(
		curandState * state_array,
		float * weights, int N,
		int * samples, int n_samples);

#endif /* CU_CHOICE_CUH_ */
