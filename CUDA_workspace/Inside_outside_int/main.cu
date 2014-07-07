#include <iostream>
#include <string>
#include <time.h>
#include <iostream>

#include "sto_grammar.h"
#include "matrix_utils.h"
#include "utils.h"

#include "cu_choice.cuh"
#include "random_utils.cuh"
#include "inside.h"

int main(void){

	CUDA_CHECK(cudaDeviceReset());

	std::cout << "Start" << std::endl;

	/*
	 * Grammar example Figure 7 of Lari, Young 1987
	 */
	Sto_grammar palindrom_grammar(7, 3);
	/*
	 * Non terminal symbols
	 */
	palindrom_grammar.set_A(0, 1, 2, 0.3);
	palindrom_grammar.set_A(0, 3, 4, 0.3);
	palindrom_grammar.set_A(0, 5, 6, 0.3);
	palindrom_grammar.set_A(0, 1, 1, 0.2);
	palindrom_grammar.set_A(0, 3, 3, 0.2);
	palindrom_grammar.set_A(0, 5, 5, 0.2);
	//
	palindrom_grammar.set_A(2, 0, 1, 1.0);
	palindrom_grammar.set_A(4, 0, 3, 1.0);
	palindrom_grammar.set_A(6, 0, 5, 1.0);
	/*
	 * Terminal symbols
	 */
	palindrom_grammar.set_B(1, 0, 1.0);
	palindrom_grammar.set_B(3, 1, 1.0);
	palindrom_grammar.set_B(5, 2, 1.0);
	/*
	 * Normalize
	 */
	palindrom_grammar.normalize();

	const int n_samples = 128;

	curandState * state_array;
	dev_alloc<curandState>(state_array, n_samples * MAX_LENGTH);

	setup_states(state_array,
			n_samples * MAX_LENGTH,
			0,
			0);

	int * host_sentences = (int *) malloc(n_samples * MAX_LENGTH * sizeof(int));
	int * host_lengths = (int *) malloc(n_samples * sizeof(int));
	int * host_error_status = (int *) malloc(n_samples * sizeof(int));

	int * dev_sentences;
	int * dev_lengths;
	int * dev_error_status;

	float * probas = (float *) malloc(n_samples * sizeof(float));
	float * dev_probas;

	dev_alloc<float>(dev_probas, n_samples);

	dev_alloc<int>(dev_sentences, n_samples * MAX_LENGTH);
	dev_alloc<int>(dev_lengths, n_samples);
	dev_alloc<int>(dev_error_status, n_samples);

	palindrom_grammar.produce_sentences_dev(
			state_array,
			dev_sentences,
			dev_lengths,
			dev_error_status,
			n_samples);

	std::cout << "Starting compute probas" << std::endl;

	compute_probas(dev_probas,
			dev_sentences,
			dev_lengths,
			dev_error_status,
			& palindrom_grammar,
			n_samples);

	std::cout << "Compute probas done" << std::endl;

	copy_to_host<int>(host_sentences, dev_sentences, n_samples * MAX_LENGTH);
	copy_to_host<int>(host_lengths, dev_lengths, n_samples);
	copy_to_host<int>(host_error_status, dev_error_status, n_samples);

	copy_to_host<float>(probas, dev_probas, n_samples);

	dev_free<curandState>(state_array);

	for(int i = 0; i < n_samples; ++i){
		std::cout << host_error_status[i] << " ";
		std::cout << host_lengths[i] << ": ";
		if(host_error_status[i] == 1){
			std::cout << "Over max length" << std::endl;
			continue;
		}
		if(host_error_status[i] == 2){
			std::cout << "Over max iters" << std::endl;
			continue;
		}
		std::cout << probas[i] << ": ";
		for(int j = 0; j < host_lengths[i]; ++j){
			std::cout << host_sentences[i * MAX_LENGTH + j] << " ";
		}std::cout << std::endl;
	}

	dev_free<int>(dev_sentences);
	dev_free<int>(dev_lengths);
	dev_free<int>(dev_error_status);

	dev_free<float>(dev_probas);

	free(host_sentences);
	free(host_lengths);
	free(host_error_status);

	free(probas);

	return 0;
}
