#include "random_utils.cuh"

__global__ void setup_states_kernel(curandState * state_array,
	int N,
	unsigned long long seed,
	unsigned long long offset)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	curand_init(seed, tid, offset, & state_array[tid]);
}

void setup_states(curandState * state_array,
		int N,
		unsigned long long seed,
		unsigned long long offset){
	int n_blocks = ceil( ((float) N) / ((float) BLOCK_SIZE) );
	setup_states_kernel<<<n_blocks, BLOCK_SIZE>>>(state_array,
			N, seed, offset);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}
