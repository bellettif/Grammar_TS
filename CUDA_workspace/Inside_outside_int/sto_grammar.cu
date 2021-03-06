/*
 * sto_grammar.cu
 *
 *  Created on: 3 juil. 2014
 *      Author: francois
 */

#include <cuda_runtime.h>

#include "matrix_utils.h"
#include "sto_grammar.h"
#include "sentence_prod.cuh"

Sto_grammar::Sto_grammar(int N, int M,
			int root_symbol):
	_N(N), _M(M),
	_root_symbol(root_symbol)
{
	CUDA_CHECK(dev_alloc<float>(_dev_A, _N * _N * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_B, _N * _M));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_term_dists, 2 * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_tot_weights, _N));
	CUDA_CHECK(dev_alloc<Compact_grammar>(_dev_cmpct_grammar, 1));

	Compact_grammar temp(
			_N,
			_M,
			_root_symbol,
			_dev_A,
			_dev_B,
			_dev_non_term_weights,
			_dev_term_weights,
			_dev_non_term_term_dists,
			_dev_tot_weights);

	CUDA_CHECK(copy_to_device<Compact_grammar>(
			_dev_cmpct_grammar,
			& temp,
			1));

	fill_with_zeros(_dev_A, _N * _N * _N);
	fill_with_zeros(_dev_B, _N * _M);
	fill_with_zeros(_dev_non_term_weights, _N);
	fill_with_zeros(_dev_term_weights, _N);
	fill_with_zeros(_dev_non_term_term_dists, 2 * _N);
	fill_with_zeros(_dev_tot_weights, _N);
}

Sto_grammar::Sto_grammar(float * A, float * B,
			int N, int M,
			int root_symbol):
			_N(N), _M(M),
			_root_symbol(root_symbol)
{
	CUDA_CHECK(dev_alloc<float>(_dev_A, _N * _N * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_B, _N * _M));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_term_weights, _N));
	CUDA_CHECK(dev_alloc<float>(_dev_non_term_term_dists, 2 * _N));
	CUDA_CHECK(dev_alloc<float>(_dev_tot_weights, _N));
	CUDA_CHECK(dev_alloc<Compact_grammar>(_dev_cmpct_grammar, 1));

	CUDA_CHECK(copy_to_device<float>(_dev_A, A, _N * _N * _N));
	CUDA_CHECK(copy_to_device<float>(_dev_B, B, _N * _M));

	compute_sums_on_device(_dev_A, _dev_non_term_weights, _N * _N, _N);
	compute_sums_on_device(_dev_B, _dev_term_weights, _M, _N);
	add_vectors_on_device(_dev_non_term_weights, _dev_term_weights,
			_dev_tot_weights, _N);
	copy_full_to_column_on_device(_dev_non_term_term_dists,
			_dev_non_term_weights, 2, 0, _N);
	copy_full_to_column_on_device(_dev_non_term_term_dists,
				_dev_term_weights, 2, 1, _N);

	Compact_grammar temp(
				_N,
				_M,
				_root_symbol,
				_dev_A,
				_dev_B,
				_dev_non_term_weights,
				_dev_term_weights,
				_dev_non_term_term_dists,
				_dev_tot_weights);

	CUDA_CHECK(copy_to_device<Compact_grammar>(
			_dev_cmpct_grammar,
			& temp,
			1));
}

Sto_grammar::~Sto_grammar(){
	CUDA_CHECK(dev_free<float>(_dev_A));
	CUDA_CHECK(dev_free<float>(_dev_B));
	CUDA_CHECK(dev_free<float>(_dev_non_term_weights));
	CUDA_CHECK(dev_free<float>(_dev_term_weights));
	CUDA_CHECK(dev_free<float>(_dev_non_term_term_dists));
	CUDA_CHECK(dev_free<float>(_dev_tot_weights));
	CUDA_CHECK(dev_free<Compact_grammar>(_dev_cmpct_grammar));
}

Compact_grammar * Sto_grammar::get_cmpct_grammar(){
	return _dev_cmpct_grammar;
}

void Sto_grammar::printA(){
	print_matrix_3D(_dev_A, _N, _N, _N);
}

void Sto_grammar::printB(){
	print_matrix_2D(_dev_B, _N, _M);
}

void Sto_grammar::print_non_term_weights(){
	print_matrix_1D(_dev_non_term_weights, _N);
}

void Sto_grammar::print_term_weights(){
	print_matrix_1D(_dev_term_weights, _N);
}

void Sto_grammar::print_non_term_term_dists(){
	print_matrix_1D(_dev_non_term_term_dists, 2 * _N);
}

void Sto_grammar::print_tot_weights(){
	print_matrix_1D(_dev_tot_weights, _N);
}

void Sto_grammar::set_A(int i, int j, int k, float p)
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
	float * dist_position = _dev_non_term_term_dists + 2*i;
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

void Sto_grammar::set_B(int i, int j, float p){
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
	float * dist_position = _dev_non_term_term_dists + 2*i + 1;
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

void Sto_grammar::normalize(){
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
}

void Sto_grammar::set_root_symbol(int new_root_symbol){
	_root_symbol = new_root_symbol;
}

/*
 * sentence will contain the result and is at most
 * MAX_LENGTH symbol long
 * length points to the final sentence length
 */
int Sto_grammar::produce_sentences_dev(
		curandState * state_array,
		int * dev_sentences,
		int * dev_lengths,
		int * dev_error_status,
		int n_sentences){
	Compact_grammar host_cmpct (_N,
								_M,
								_root_symbol,
								_dev_A,
								_dev_B,
								_dev_non_term_weights,
								_dev_term_weights,
								_dev_non_term_term_dists,
								_dev_tot_weights);

	bool * dev_todo;
	bool * dev_next_todo;
	int * dev_buffer;

	dev_alloc<bool>(dev_todo, n_sentences * MAX_LENGTH);
	dev_alloc<bool>(dev_next_todo, n_sentences * MAX_LENGTH);
	dev_alloc<int>(dev_buffer, n_sentences * MAX_LENGTH);

	int n_blocks = ceil( ((float) n_sentences) / ((float) BLOCK_SIZE));
	produce_sentence_kernel<<<n_blocks, BLOCK_SIZE>>>(state_array,
			dev_sentences,
			dev_lengths,
			dev_error_status,
			n_sentences,
			dev_todo,
			dev_next_todo,
			dev_buffer,
			_dev_cmpct_grammar);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());

	dev_free<bool>(dev_todo);
	dev_free<bool>(dev_next_todo);
	dev_free<int>(dev_buffer);

	return 0;
}
