#include "matrix_utils.h"

__global__ void fill_with_zeros_kernel(float * dev_array, int N){
	int tid = threadIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	dev_array[tid] = 0.0;
}

void fill_with_zeros(float * dev_array, int N){
	int n_blocks = ceil( ((float) N) / ((float) BLOCK_SIZE) );
	fill_with_zeros_kernel<<<n_blocks, BLOCK_SIZE>>>(dev_array, N);
	check_last_error();
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
		check_last_error();
	}
}

void compute_sums(float * array, float * sum_array,
		int stride, int N_sums){
	float * dev_input;
	float * dev_sums;
	dev_alloc<float>(dev_input, N_sums * stride);
	dev_alloc<float>(dev_sums, N_sums);

	copy_to_device<float>(dev_input, array, N_sums * stride);

	compute_sums_on_device(dev_input, dev_sums, stride, N_sums);

	copy_to_host<float>(sum_array, dev_sums, N_sums);

	dev_free<float>(dev_input);
	dev_free<float>(dev_sums);
}

__global__ void print_matrix_3D_kernel(float * dev_matrix,
		int N_1,
		int N_2,
		int N_3){
	int i = threadIdx.x;
	int j = threadIdx.y;
	int k = threadIdx.z;
	if(i >= N_1) return;
	if(j >= N_2) return;
	if(k >= N_3) return;
	int pos = i * N_2 * N_3 + j * N_3 + k;
	if(j == N_2 - 1){
		printf("%f\n\n", dev_matrix[pos]);
		return;
	}
	if(k == N_3 - 1){
		printf("%f\n", dev_matrix[pos]);
		return;
	}
	printf("%f ", dev_matrix[pos]);
}

void print_matrix_3D(float * dev_matrix,
		int N_1,
		int N_2,
		int N_3){
	dim3 block_dim(N_1, N_2, N_3);
	print_matrix_3D_kernel<<<1, block_dim>>>(dev_matrix,
			N_1,
			N_2,
			N_3);
	check_last_error();
}

__global__ void print_matrix_2D_kernel(float * dev_matrix,
		int N_1,
		int N_2){
	int i = threadIdx.x;
	int j = threadIdx.y;
	if(i >= N_1) return;
	if(j >= N_2) return;
	int pos = i * N_2 + j;
	if(j == N_2 - 1){
		printf("%f\n", dev_matrix[pos]);
		return;
	}
	printf("%f ", dev_matrix[pos]);
}

void print_matrix_2D(float * dev_matrix,
		int N_1,
		int N_2){
	dim3 block_dim(N_1, N_2);
	print_matrix_2D_kernel<<<1, block_dim>>>(dev_matrix,
			N_1,
			N_2);
	check_last_error();
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
	check_last_error();
}

void add_vectors(float * A,
		float * B,
		float * C,
		int N){
	float * dev_A;
	float * dev_B;
	float * dev_C;

	dev_alloc<float>(dev_A, N);
	dev_alloc<float>(dev_B, N);
	dev_alloc<float>(dev_C, N);

	copy_to_device<float>(dev_A, A, N);
	copy_to_device<float>(dev_B, B, N);

	add_vectors_on_device(dev_A, dev_B, dev_C, N);

	copy_to_host<float>(C, dev_C, N);

	dev_free<float>(dev_A);
	dev_free<float>(dev_B);
	dev_free<float>(dev_C);
}
