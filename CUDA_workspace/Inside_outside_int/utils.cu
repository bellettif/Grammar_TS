#include "utils.h"

__global__ void copy_full_to_columns_on_device_kernel(float * dest_pt,
		float * source_pt, int stride, int offset, int N){
	int pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(pos >= N) return;
	if(pos % stride == 0){
		dest_pt[pos * stride + offset] = source_pt[pos];
	}
}


void copy_full_to_column_on_device(float * dest_pt, float * source_pt,
		int stride, int offset, int N){
	int n_blocks = ceil(((float) N ) / ((float) BLOCK_SIZE));
	copy_full_to_columns_on_device_kernel<<<n_blocks, BLOCK_SIZE>>>(dest_pt,
			source_pt, stride, offset, N);
	CUDA_CHECK(check_last_error());
}
