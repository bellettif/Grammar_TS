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

void fill_with_zeros(float * dev_array, int N);

void compute_sums_on_device(float * dev_array, float * dev_sum_array,
		int stride, int N_sums);

void compute_sums(float * array, float * sum_array,
		int stride, int N_sums);

void print_matrix_3D(float * dev_matrix,
		int N_1,
		int N_2,
		int N_3);

void print_matrix_2D(float * dev_matrix,
		int N_1,
		int N_2);

void add_vectors_on_device(float * dev_A,
		float * dev_B,
		float * dev_C,
		int N);

void add_vectors(float * A,
		float * B,
		float * C,
		int N);

#endif /* MATRIX_UTILS_CUH_ */
