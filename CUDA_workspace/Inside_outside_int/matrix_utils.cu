#include "matrix_utils.h"

__global__ void fill_with_scalar_kernel(float * dev_array, float scalar, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	dev_array[tid] = scalar;
}

void fill_with_scalar(float * dev_array, float scalar, int N){
	int n_blocks = ceil( ((float) N) / ((float) BLOCK_SIZE) );
	fill_with_scalar_kernel<<<n_blocks, BLOCK_SIZE>>>(dev_array, scalar, N);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void fill_with_zeros(float * dev_array, int N){
	fill_with_scalar(dev_array, 0.0, N);
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

void compute_sums_on_device(float * dev_array, float * dev_sum_array,
		int stride, int N_sums){
	int n_iters = ceil( ((float) stride) / ((float) BLOCK_SIZE) );
	for(int iter = 0; iter < n_iters; ++iter){
		compute_sums_kernel<<<N_sums, BLOCK_SIZE>>>(
				dev_array,
				dev_sum_array,
				stride, N_sums,
				iter
				);
		CUDA_CHECK(check_last_error());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}

void compute_sums(float * array, float * sum_array,
		int stride, int N_sums){
	float * dev_input;
	float * dev_sums;
	CUDA_CHECK(dev_alloc<float>(dev_input, N_sums * stride));
	CUDA_CHECK(dev_alloc<float>(dev_sums, N_sums));

	CUDA_CHECK(copy_to_device<float>(dev_input, array, N_sums * stride));

	compute_sums_on_device(dev_input, dev_sums, stride, N_sums);

	CUDA_CHECK(copy_to_host<float>(sum_array, dev_sums, N_sums));

	CUDA_CHECK(dev_free<float>(dev_input));
	CUDA_CHECK(dev_free<float>(dev_sums));
}

__global__ void print_matrix_3D_kernel(float * dev_matrix,
		int N_1,
		int N_2,
		int N_3){
	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = threadIdx.y;
	if(i >= N_1) return;
	if(j >= N_2) return;
	if(k >= N_3) return;
	int pos = i * N_2 * N_3 + j * N_3 + k;
	printf("(%d,%d,%d)=%f\n", i, j, k, dev_matrix[pos]);
}

void print_matrix_3D(float * dev_matrix,
		int N_1,
		int N_2,
		int N_3){
	dim3 block_dim(N_2, N_3);
	print_matrix_3D_kernel<<<N_1, block_dim>>>(dev_matrix,
			N_1,
			N_2,
			N_3);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void print_matrix_2D_kernel(float * dev_matrix,
		int N_1,
		int N_2){
	int i = threadIdx.x;
	int j = threadIdx.y;
	if(i >= N_1) return;
	if(j >= N_2) return;
	int pos = i * N_2 + j;
	printf("(%d,%d)=%f\n", i, j, dev_matrix[pos]);
}

void print_matrix_2D(float * dev_matrix,
		int N_1,
		int N_2){
	dim3 block_dim(N_1, N_2);
	print_matrix_2D_kernel<<<1, block_dim>>>(dev_matrix,
			N_1,
			N_2);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void print_matrix_1D_kernel(float * dev_matrix,
		int N){
	int i = threadIdx.x;
	if(i >= N) return;
	int pos = i;
	printf("(%d)=%f\n", i, dev_matrix[pos]);
}

void print_matrix_1D(float * dev_matrix,
		int N){
	dim3 block_dim(N);
	print_matrix_1D_kernel<<<1, block_dim>>>(dev_matrix,
			N);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void add_dev_vector_kernel(float * dev_A,
		float * dev_B,
		float * dev_C,
		int N){
	int pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(pos >= N) return;
	dev_C[pos] = dev_A[pos] + dev_B[pos];
}

void add_vectors_on_device(float * dev_A,
		float * dev_B,
		float * dev_C,
		int N){
	int n_blocks = ceil( ((float) N) / ((float) BLOCK_SIZE) );
	add_dev_vector_kernel<<<n_blocks, BLOCK_SIZE>>>(dev_A, dev_B, dev_C, N);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void add_vectors(float * A,
		float * B,
		float * C,
		int N){
	float * dev_A;
	float * dev_B;
	float * dev_C;

	CUDA_CHECK(dev_alloc<float>(dev_A, N));
	CUDA_CHECK(dev_alloc<float>(dev_B, N));
	CUDA_CHECK(dev_alloc<float>(dev_C, N));

	CUDA_CHECK(copy_to_device<float>(dev_A, A, N));
	CUDA_CHECK(copy_to_device<float>(dev_B, B, N));

	add_vectors_on_device(dev_A, dev_B, dev_C, N);

	CUDA_CHECK(copy_to_host<float>(C, dev_C, N));

	CUDA_CHECK(dev_free<float>(dev_A));
	CUDA_CHECK(dev_free<float>(dev_B));
	CUDA_CHECK(dev_free<float>(dev_C));
}

__global__ void divide_by_kernel(float * M, float * tot,
		int N, int stride){
	int i = blockIdx.x;
	if(i >= N) return;

	int j = threadIdx.x;
	if(j >= stride) return;

	int pos = i * stride + j;
	M[pos] /= tot[i];
}

void divide_by(float * M, float * tot,
		int N, int stride){
	divide_by_kernel<<<N, stride>>>(M, tot, N, stride);
	CUDA_CHECK(check_last_error());
	CUDA_CHECK(cudaDeviceSynchronize());
}

