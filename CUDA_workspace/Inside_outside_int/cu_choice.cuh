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

__device__ int choice(
		curandState * state,
		float * weights,
		int N);

__device__ bool emit_term(
		curandState * state,
		float * dev_non_term_term_dists,
		int i);

__device__ void non_term_derivation(
		curandState * state,
		float * dev_A,
		int i, int & j, int & k,
		int N);

__device__ void term_derivation(
		curandState * state,
		float * dev_B,
		int i, int & term,
		int M);

__global__ void gpu_choice_kernel(
		curandState * state_array,
		float * weights, int N,
		int * samples, int n_samples);

void gpu_choice(
		curandState * state_array,
		float * weights, int N,
		int * samples, int n_samples);

#endif /* CU_CHOICE_CUH_ */
