#include "cu_choice.cuh"

/*
__device__ int choice(
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
*/

/*
__device__ inline bool emit_term(
		curandState * state,
		float * dev_non_term_term_dists,
		int i){
	return (choice(state,
			dev_non_term_term_dists + i * 2,
			2)
			== 1);
}
*/

/*
__device__ void non_term_derivation(
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
*/

/*
__device__ void term_derivation(
		curandState * state,
		float * dev_B,
		int i, int & term,
		int M){
	term = choice(state,
			dev_B + i * M,
			M);
}
*/

/**
 * N below BLOCK_SIZE
 */

__global__ void gpu_choice_kernel(
		curandState * state_array,
		float * weights, int N,
		int * samples, int n_samples){
	extern __shared__ float shared_weights[];

	int pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(pos >= n_samples){
		return;
	}

	int tid = threadIdx.x;
	if(tid < N){
		shared_weights[tid] = weights[tid];
	}
	__syncthreads();

	samples[pos] = choice(state_array + pos, shared_weights, N);
	__syncthreads();
}

void gpu_choice(
		curandState * state_array,
		float * weights, int N,
		int * samples, int n_samples){
	int n_blocks = ceil( ((float) n_samples) / ((float) BLOCK_SIZE) );
	gpu_choice_kernel<<<n_blocks, BLOCK_SIZE, N>>>(state_array,
			weights, N, samples, n_samples);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}
