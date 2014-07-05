/*
 * matrix_utils.h
 *
 *  Created on: 5 juil. 2014
 *      Author: francois
 */

#ifndef MATRIX_UTILS_CUH_
#define MATRIX_UTILS_CUH_

#include "utils.h"

#define BLOCK_SIZE 256

#define BLOCK_SIZE_2D_1 16
#define BLOCK_SIZE_2D_2 16

__global__ void fill_with_zeros(float * dev_array, int N);

__global__ void compute_sums_kernel(float * dev_array,
		float * sum_array,
		int stride, int N_sums,
		int iter);


#endif /* MATRIX_UTILS_CUH_ */
