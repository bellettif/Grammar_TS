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

	std::cout << "Start" << std::endl;

	const int N = 4;
	float weights[N] = {0.1, 0.2, 0.4, 0.3};
	float distribs[N] = {0.0, 0.0, 0.0, 0.0};

	float * dev_weights;
	dev_alloc<float>(dev_weights, N);

	copy_to_device(dev_weights, weights, N);

	const int n_samples = 1024 * 10;
	int results[n_samples];
	int * dev_results;
	dev_alloc<int>(dev_results, n_samples);

	curandState * state_array;
	dev_alloc<curandState>(state_array, n_samples);

	setup_states(state_array,
			n_samples,
			0,
			0);

	gpu_choice(state_array,
			dev_weights, N,
			dev_results, n_samples);

	copy_to_host(results, dev_results, n_samples);

	for(int i = 0; i < n_samples; ++i){
		distribs[results[i]] += 1;
	}
	for(int i = 0; i < N; ++i){
		distribs[i] /= ((float) n_samples);
		std::cout << distribs[i] << std::endl;
	}

	std::cout << "Coucou" << std::endl;

	dev_free<curandState>(state_array);
	dev_free<int>(dev_results);
	dev_free<float>(dev_weights);

	CUDA_CHECK(cudaDeviceReset());

	return 0;
}
