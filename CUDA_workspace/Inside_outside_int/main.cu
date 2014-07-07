#include <iostream>
#include <string>
#include <time.h>
#include <iostream>

#include "sto_grammar.h"
#include "matrix_utils.h"
#include "utils.h"

#include "cu_choice.cuh"
#include "random_utils.cuh"

int main(void){

	CUDA_CHECK(cudaDeviceReset());
	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	int max_length = 32;
	int n_symbols = 16;

	int stride = n_symbols * max_length * max_length;

	int n_samples = 2048;

	int * dev_ptr;

	CUDA_CHECK(dev_alloc<int>(dev_ptr, n_samples * stride));

	std::cout << "Allocation done" << std::endl;

	CUDA_CHECK(dev_free<int>(dev_ptr));

	std::cout << "De-allocation done" << std::endl;


	return 0;
}
