#include"sum_utils.h"

#define SMALL_BLOCK_SIZE 64

__global__ void KL_div_kernel(
		float * self_probas,
		float * cross_probas,
		int * dev_error_status,
		float * block_sums,
		int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= N) return;

	int bid = blockIdx.x;
	int tid = threadIdx.x;

	extern __shared__ float inner_sum[];
	if((dev_error_status[id] == 0) && (self_probas[id] != 0.0)){
		inner_sum[tid] =  log2f(self_probas[id] / cross_probas[id]) * self_probas[id];
	}else{
		inner_sum[tid] = 0;
	}
	__syncthreads();

	int coalescence = 2;
	while(coalescence <= blockDim.x){
		if(tid % coalescence == 0){
			inner_sum[tid] += inner_sum[tid + (coalescence / 2)];
		}
		__syncthreads();
		coalescence *= 2;
	}
	block_sums[bid] = inner_sum[0];

	coalescence = 2;
	while(coalescence <= gridDim.x){
		if(bid % coalescence == 0){
			if(bid + (coalescence / 2) < gridDim.x){
				block_sums[bid] += block_sums[bid + (coalescence / 2)];
			}
		}
		__syncthreads();
		coalescence *= 2;
	}
}

float KL_div(
		float * dev_self_probas,
		float * dev_cross_probas,
		int * dev_error_status,
		int N){
	int n_blocks = ceil( ((float) N) / ((float) SMALL_BLOCK_SIZE) );
	float * block_sums;
	CUDA_CHECK(dev_alloc<float>(block_sums, n_blocks));
	KL_div_kernel<<<n_blocks, SMALL_BLOCK_SIZE,
			SMALL_BLOCK_SIZE>>>(
			dev_self_probas,
			dev_cross_probas,
			dev_error_status,
			block_sums,
			N);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
	float return_value;
	CUDA_CHECK(copy_to_host(& return_value,
			block_sums, 1));
	return return_value;
}

__global__ void compute_mid_sum_kernel(
		float * dev_left,
		float * dev_right,
		float * dev_mean,
		int N){
	int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(tid >= N) return;

	dev_mean[tid] = 0.5 * (dev_left[tid] + dev_right[tid]);
}

void compute_mid_sum(
		float * dev_left,
		float * dev_right,
		float * dev_mean,
		int N){
	int n_blocks = ceil( ((float) N) / ((float) BLOCK_SIZE) );
	compute_mid_sum_kernel<<<n_blocks, BLOCK_SIZE>>>(
				dev_left,
				dev_right,
				dev_mean,
				N);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}


