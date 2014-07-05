/*
 * sto_grammar.cu
 *
 *  Created on: 3 juil. 2014
 *      Author: francois
 */

#include "matrix_utils.cuh"
#include "sto_grammar.h"

#define MAX_ITERS 128


class cu_Sto_grammar{

	const int 	_N;
	const int	_M;
	int			_root_symbol;
	float * 	_dev_A;
	float * 	_dev_B;
	float *		_dev_non_term_weights;
	float *		_dev_term_weights;
	float *		_dev_non_term_term_dists;

public:

	cu_Sto_grammar(int N, int M,
			int root_symbol = 0):
		_N(N), _M(M),
		_root_symbol(root_symbol)
	{
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_A,
				_N *_N *_N * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_B,
				_N * _M * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_non_term_weights,
				_N * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_term_weights,
				_N * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_non_term_term_dists,
				2 * _N * sizeof(float)));

		int n_blocks = ceil(_N * _N * _N / BLOCK_SIZE);
		fill_with_zeros<<<n_blocks, BLOCK_SIZE>>>(_dev_A, _N * _N * _N);
		CUDA_CHECK_RETURN(cudaGetLastError());

		n_blocks = ceil(_N * _M / BLOCK_SIZE);
		fill_with_zeros<<<n_blocks, BLOCK_SIZE>>>(_dev_B, _N * _M);
		CUDA_CHECK_RETURN(cudaGetLastError());

		n_blocks = ceil(_N / BLOCK_SIZE);
		fill_with_zeros<<<n_blocks, BLOCK_SIZE>>>(_dev_non_term_weights, _N);
		CUDA_CHECK_RETURN(cudaGetLastError());

		n_blocks = ceil(_N / BLOCK_SIZE);
		fill_with_zeros<<<n_blocks, BLOCK_SIZE>>>(_dev_term_weights, _N);
		CUDA_CHECK_RETURN(cudaGetLastError());

		n_blocks = ceil(2 * _N / BLOCK_SIZE);
		fill_with_zeros<<<n_blocks, BLOCK_SIZE>>>(_dev_non_term_term_dists, 2 * _N);
		CUDA_CHECK_RETURN(cudaGetLastError());
	}

	cu_Sto_grammar(float * A, float * B,
			int N, int M,
			int root_symbol = 0):
			_N(N), _M(M),
			_root_symbol(root_symbol){
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_A,
				_N *_N *_N * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_B,
				_N * _M * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_non_term_weights,
				_N * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_term_weights,
				_N * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **) & _dev_non_term_term_dists,
				2 * _N * sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(_dev_A, A, _N *_N *_N * sizeof(float),
				cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(_dev_B, B, _N *_M * sizeof(float),
				cudaMemcpyHostToDevice));
	}

	~cu_Sto_grammar(){
		CUDA_CHECK_RETURN(cudaFree(_dev_A));
		CUDA_CHECK_RETURN(cudaFree(_dev_B));
		CUDA_CHECK_RETURN(cudaFree(_dev_non_term_weights));
		CUDA_CHECK_RETURN(cudaFree(_dev_term_weights));
		CUDA_CHECK_RETURN(cudaFree(_dev_non_term_term_dists));
	}

	void printA(){
		/*
		 * TODO
		 */
	}

	void printB(){
		/*
		 * TODO
		 */
	}

	void set_A(int i, int j, int k, float p){
		/*
		 * TODO
		 */
	}

	void set_B(int i, int j, float p){
		/*
		 * TODO
		 */
	}

	void normalize(){
		/*
		 * TODO
		 */
	}

	void set_root_symbol(int new_root_symbol){
		_root_symbol = new_root_symbol;
	}

	__device__ inline void non_term_derivation(int i, int & j, int & k){
		/*
		 * TODO
		 */
	}

	__device__ inline void term_derivation(int i, int & term){
		/*
		 * TODO
		 */
	}

	/*
	 * sentence will contain the result and is at most
	 * MAX_LENGTH symbol long
	 */
	int produce_sentence(int * sentence,
			int & length,
			int MAX_LENGTH){
		/*
		 * TODO
		 */
		return 0;
	}

};
