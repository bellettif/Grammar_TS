#include "matrix_utils.cuh"

__global__ void fill_with_zeros(float * dev_array, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	dev_array[tid] = 0.0;
}

__global__ void compute_sums_kernel(float * dev_array,
		float * sum_array,
		int stride, int N_sums,
		int iter){
	int bid = blockIdx.x;
	if(bid >= N_sums) return;

	int tid = threadIdx.x;
	__shared__ float shared_copy[BLOCK_SIZE];

	int pos = iter * BLOCK_SIZE + tid;
	if(pos >= stride){
		shared_copy[tid] = 0;
	}else{
		shared_copy[tid] = dev_array[bid * stride + pos];
	}
	__syncthreads();

	int coalescence = 2;
	while(coalescence <= BLOCK_SIZE){
		if(tid % coalescence == 0){
			shared_copy[tid] += shared_copy[tid + (coalescence / 2)];
		}
		__syncthreads();
		coalescence *= 2;
	}

	if(iter == 0){
		sum_array[bid] = shared_copy[0];
	}else{
		sum_array[bid] += shared_copy[0];
	}
}

void compute_sums(float * array, float * sum_array,
		int stride, int N_sums){
	float * dev_input;
	float * dev_sums;

	CUDA_CHECK_RETURN(cudaMalloc((void **) & dev_input,
			N_sums * stride * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) & dev_sums,
			N_sums * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(dev_input, array,
			N_sums * stride * sizeof(float),
			cudaMemcpyHostToDevice));

	int n_iters = ceil(stride / BLOCK_SIZE);
	for(int iter = 0; iter < n_iters; ++iter){
		compute_sums_kernel<<<N_sums, BLOCK_SIZE>>>(
				dev_input,
				dev_sums,
				stride, N_sums,
				iter
				);
		CUDA_CHECK_RETURN(cudaGetLastError());
	}

	CUDA_CHECK_RETURN(cudaMemcpy(sum_array, dev_sums,
			N_sums * sizeof(float),
			cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(dev_input));
	CUDA_CHECK_RETURN(cudaFree(dev_sums));
}

void compute_sums_on_device(float * dev_array, float * dev_sum_array,
		int stride, int N_sums){
	int n_iters = ceil(stride / BLOCK_SIZE);
	for(int iter = 0; iter < n_iters; ++iter){
		compute_sums_kernel<<<N_sums, BLOCK_SIZE>>>(
				dev_array,
				dev_sum_array,
				stride, N_sums,
				iter
				);
		CUDA_CHECK_RETURN(cudaGetLastError());
	}
}
