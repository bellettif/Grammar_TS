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
		flaot * dev_B, int N, int M){
	/*
	 * TODO
	 */
}


cu_Sto_grammar::cu_Sto_grammar(int N, int M,
			int root_symbol = 0):
	_N(N), _M(M),
	_root_symbol(root_symbol)
{
	dev_alloc<float>(_dev_A, _N * _N * _N);
	dev_alloc<float>(_dev_B, _N * _M);
	dev_alloc<float>(_dev_non_term_weights, _N);
	dev_alloc<float>(_dev_term_weights, _N);
	dev_alloc<float>(_dev_non_term_term_dists, 2 * _N);

	fill_with_zeros(_dev_A, _N * _N * _N);
	fill_with_zeros(_dev_B, _N * _M);
	fill_with_zeros(_dev_non_term_weights, _N);
	fill_with_zeros(_dev_term_weights, _N);
	fill_with_zeros(_dev_non_term_term_dists, 2 * _N);
}

cu_Sto_grammar::cu_Sto_grammar(float * A, float * B,
			int N, int M,
			int root_symbol = 0):
			_N(N), _M(M),
			_root_symbol(root_symbol)
{
	dev_alloc<float>(_dev_A, _N * _N * _N);
	dev_alloc<float>(_dev_B, _N * _M);
	dev_alloc<float>(_dev_non_term_weights, _N);
	dev_alloc<float>(_dev_term_weights, _N);
	dev_alloc<float>(_dev_non_term_term_dists, 2 * _N);

	copy_to_device<float>(_dev_A, A, _N * _N * _N);
	copy_to_host<float>(_dev_B, B, _N * _M);
}

cu_Sto_grammar::~cu_Sto_grammar(){
	dev_free<float>(_dev_A);
	dev_free<float>(_dev_B);
	dev_free<float>(_dev_non_term_weights);
	dev_free<float>(_dev_term_weights);
	dev_free<float>(_dev_non_term_term_dists);
}

void cu_Sto_grammar::printA(){
	print_matrix_3D(_dev_A, _N, _N, _N);
}

void cu_Sto_grammar::printB(){
	print_matrix_2D(_dev_B, _N, _M);
}

void cu_Sto_grammar::set_A(int i, int j, int k, float p)
{
	/*
	 * Changing probability on A table
	 */
	float * position = _dev_A + i * _N * _N + j * _N + k;
	float prev;
	copy_to_host<float>(& prev, position, 1);
	copy_to_device<float>(position, & p, 1);
	/*
	 * Changing weights
	 */
	float * weight_position = _dev_non_term_weights + i ;
	float prev_weight;
	copy_to_host<float>(& prev_weight, weight_position, 1);
	prev_weight = prev_weight - prev + p;
	copy_to_device<float>(weight_position, & prev_weight, 1);
	/*
	 * Changing non_term / term distribution
	 */
	float * dist_position = _dev_non_term_term_dists + i;
	copy_on_device<float>(dist_position, weight_position, 1);
}

void cu_Sto_grammar::set_B(int i, int j, float p){
	/*
	 * Changing probability on B table
	 */
	float * position = _dev_B + i * _N + j;
	float prev;
	copy_to_host<float>(& prev, position, 1);
	copy_to_device<float>(position, & p, 1);
	/*
	 * Changing weights
	 */
	float * weight_position = _dev_term_weights + i ;
	float prev_weight;
	copy_to_host<float>(& prev_weight, weight_position, 1);
	prev_weight = prev_weight - prev + p;
	copy_to_device<float>(weight_position, & prev_weight, 1);
	/*
	 * Changing non_term / term distribution
	 */
	float * dist_position = _dev_non_term_term_dists + i + 1;
	copy_on_device<float>(dist_position, weight_position, 1);
}

void cu_Sto_grammar::normalize(){
	/*
	 * TODO
	 */
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
