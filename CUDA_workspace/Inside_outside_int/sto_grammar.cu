/*
 * sto_grammar.cu
 *
 *  Created on: 3 juil. 2014
 *      Author: francois
 */

#include <cuda_runtime.h>

#include "matrix_utils.h"
#include "sto_grammar.h"

__device__ static void emission_choice(int i,
		float * dev_non_term_term_dists){

}

__device__ inline static void non_term_derivation(int i, int & j, int & k,
		float * dev_A, int N){
	/*
	 * TODO
	 */
}

__device__ inline static void term_derivation(int i, int & term,
		float * dev_B, int N, int M){
	/*
	 * TODO
	 */
}


cu_Sto_grammar::cu_Sto_grammar(int N, int M,
			int root_symbol):
	_N(N), _M(M),
	_root_symbol(root_symbol)
{
	CUDA_CHECK(dev_alloc<float>(_dev_A, _N * _N * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_cum_A, _N * _N * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_B, _N * _M));
	CUDA_CHECK(dev_alloc<float>(_dev_cum_B, _N * _M));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_term_dists, 2 * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_tot_weights, _N));

	fill_with_zeros(_dev_A, _N * _N * _N);
	fill_with_zeros(_dev_cum_A, _N * _N * _N);
	fill_with_zeros(_dev_B, _N * _M);
	fill_with_zeros(_dev_cum_B, _N * _M);
	fill_with_zeros(_dev_non_term_weights, _N);
	fill_with_zeros(_dev_term_weights, _N);
	fill_with_zeros(_dev_non_term_term_dists, 2 * _N);
	fill_with_zeros(_dev_tot_weights, _N);
}

cu_Sto_grammar::cu_Sto_grammar(float * A, float * B,
			int N, int M,
			int root_symbol):
			_N(N), _M(M),
			_root_symbol(root_symbol)
{
	CUDA_CHECK(dev_alloc<float>(_dev_A, _N * _N * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_cum_A, _N * _N * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_B, _N * _M));
	CUDA_CHECK(dev_alloc<float>(_dev_cum_B, _N * _M));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_term_dists, 2 * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_tot_weights, _N));

	CUDA_CHECK(copy_to_device<float>(_dev_A, A, _N * _N * _N));
	CUDA_CHECK(copy_to_host<float>(_dev_B, B, _N * _M));

	compute_sums_on_device(_dev_A, _dev_non_term_weights, _N * _N, _N);
	compute_sums_on_device(_dev_B, _dev_term_weights, _M, _N);
	add_vectors_on_device(_dev_non_term_weights, _dev_term_weights,
			_dev_tot_weights, _N);
	copy_full_to_column_on_device(_dev_non_term_term_dists,
			_dev_non_term_weights, 2, 0, _N);
	copy_full_to_column_on_device(_dev_non_term_term_dists,
				_dev_term_weights, 2, 1, _N);
}

cu_Sto_grammar::~cu_Sto_grammar(){
	CUDA_CHECK(dev_free<float>(_dev_A));
	CUDA_CHECK(dev_free<float>(_dev_cum_A));
	CUDA_CHECK(dev_free<float>(_dev_B));
	CUDA_CHECK(dev_free<float>(_dev_cum_B));
	CUDA_CHECK(dev_free<float>(_dev_non_term_weights));
	CUDA_CHECK(dev_free<float>(_dev_term_weights));
	CUDA_CHECK(dev_free<float>(_dev_non_term_term_dists));
	CUDA_CHECK(dev_free<float>(_dev_tot_weights));
}

void cu_Sto_grammar::printA(){
	print_matrix_3D(_dev_A, _N, _N, _N);
}

void cu_Sto_grammar::print_cum_A(){
	print_matrix_3D(_dev_cum_A, _N, _N, _N);
}

void cu_Sto_grammar::printB(){
	print_matrix_2D(_dev_B, _N, _M);
}

void cu_Sto_grammar::print_cum_B(){
	print_matrix_2D(_dev_cum_B, _N, _M);
}

void cu_Sto_grammar::print_non_term_weights(){
	print_matrix_1D(_dev_non_term_weights, _N);
}

void cu_Sto_grammar::print_term_weights(){
	print_matrix_1D(_dev_term_weights, _N);
}

void cu_Sto_grammar::print_non_term_term_dists(){
	print_matrix_1D(_dev_non_term_term_dists, 2 * _N);
}

void cu_Sto_grammar::print_tot_weights(){
	print_matrix_1D(_dev_tot_weights, _N);
}

void cu_Sto_grammar::set_A(int i, int j, int k, float p)
{
	/*
	 * Changing probability on A table
	 */
	float * position = _dev_A + i * _N * _N + j * _N + k;
	float prev;
	CUDA_CHECK(copy_to_host<float>(& prev, position, 1));
	CUDA_CHECK(copy_to_device<float>(position, & p, 1));
	/*
	 * Changing weights
	 */
	float * weight_position = _dev_non_term_weights + i ;
	float prev_weight;
	CUDA_CHECK(copy_to_host<float>(& prev_weight, weight_position, 1));
	prev_weight = prev_weight - prev + p;
	CUDA_CHECK(copy_to_device<float>(weight_position, & prev_weight, 1));
	/*
	 * Changing non_term / term distribution
	 */
	float * dist_position = _dev_non_term_term_dists + i;
	CUDA_CHECK(copy_on_device<float>(dist_position, weight_position, 1));
	/*
	 * 	Changing tot weight
	 */
	float * tot_weight_position = _dev_tot_weights + i;
	float prev_tot_weight;
	CUDA_CHECK(copy_to_host<float>(& prev_tot_weight, tot_weight_position, 1));
	prev_tot_weight = prev_tot_weight - prev + p;
	CUDA_CHECK(copy_to_device<float>(tot_weight_position, & prev_tot_weight, 1));
}

void cu_Sto_grammar::set_B(int i, int j, float p){
	/*
	 * Changing probability on B table
	 */
	float * position = _dev_B + i * _M + j;
	float prev;;
	CUDA_CHECK(copy_to_host<float>(& prev, position, 1));
	CUDA_CHECK(copy_to_device<float>(position, & p, 1));
	/*
	 * Changing weights
	 */
	float * weight_position = _dev_term_weights + i ;
	float prev_weight;
	CUDA_CHECK(copy_to_host<float>(& prev_weight, weight_position, 1));
	prev_weight = prev_weight - prev + p;
	CUDA_CHECK(copy_to_device<float>(weight_position, & prev_weight, 1));
	/*
	 * Changing non_term / term distribution
	 */
	float * dist_position = _dev_non_term_term_dists + i + 1;
	CUDA_CHECK(copy_on_device<float>(dist_position, weight_position, 1));
	/*
	 * 	Changing tot weight
	 */
	float * tot_weight_position = _dev_tot_weights + i;
	float prev_tot_weight;
	CUDA_CHECK(copy_to_host<float>(& prev_tot_weight, tot_weight_position, 1));
	prev_tot_weight = prev_tot_weight - prev + p;
	CUDA_CHECK(copy_to_device<float>(tot_weight_position, & prev_tot_weight, 1));
}

void cu_Sto_grammar::normalize(){
	divide_by(_dev_A, _dev_tot_weights,
			_N, _N * _N);
	divide_by(_dev_B, _dev_tot_weights,
			_N, _M);
	divide_by(_dev_non_term_weights, _dev_tot_weights,
			_N, 1);
	divide_by(_dev_term_weights, _dev_tot_weights,
			_N, 1);
	divide_by(_dev_non_term_term_dists, _dev_tot_weights,
			_N, 2);
	fill_with_scalar(_dev_tot_weights, 1.0, _N);
	compute_cumsum(_dev_cum_A, _dev_A,
			_N, _N * _N);
	compute_cumsum(_dev_cum_B, _dev_B,
				_N, _M);
}

void cu_Sto_grammar::set_root_symbol(int new_root_symbol){
	_root_symbol = new_root_symbol;
}



/*
 * sentence will contain the result and is at most
 * MAX_LENGTH symbol long
 */
int cu_Sto_grammar::produce_sentence(int * sentence,
		int & length,
		int MAX_LENGTH){
	/*
	 * TODO
	 */
	return 0;
}
