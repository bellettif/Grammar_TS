/*
 * random_utils.h
 *
 *  Created on: 6 juil. 2014
 *      Author: francois
 */

#ifndef RANDOM_UTILS_H_
#define RANDOM_UTILS_H_

#include "utils.h"
#include "cuda_dimensions.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void setup_states_kernel(curandState * state_array,
	int N,
	unsigned long long seed,
	unsigned long long offset = 0);

void setup_states(curandState * state_array,
		int N,
		unsigned long long seed,
		unsigned long long offset = 0);

#endif /* RANDOM_UTILS_H_ */
