#include "launch.h"

#include"random_utils.cuh"
#include<time.h>

#define SAMPLE_BATCH 2048

void compute_probas_batch(float * A,
					float * B,
					float * probas,
					int *	error_status,
					int		n_samples,
					int N,
					int M){
	CUDA_CHECK(cudaDeviceReset());

	Sto_grammar * grammar = new Sto_grammar(A, B, N, M, 0);
	/*
	 * Normalize
	 */
	grammar->normalize();

	curandState * state_array;
	CUDA_CHECK(dev_alloc<curandState>(state_array, SAMPLE_BATCH * MAX_LENGTH));

	setup_states(state_array,
			SAMPLE_BATCH * MAX_LENGTH,
			time(NULL),
			0);

	int n_iters = ceil( ((float) n_samples) / ((float) SAMPLE_BATCH) );

	int * dev_sentences;
	int * dev_lengths;
	int * dev_error_status;

	float * dev_probas;
	float * dev_Es;

	CUDA_CHECK(dev_alloc<int>(dev_sentences, SAMPLE_BATCH * MAX_LENGTH));
	CUDA_CHECK(dev_alloc<int>(dev_lengths, SAMPLE_BATCH));
	CUDA_CHECK(dev_alloc<int>(dev_error_status, SAMPLE_BATCH));
	CUDA_CHECK(dev_alloc<float>(dev_probas, SAMPLE_BATCH));
	CUDA_CHECK(dev_alloc<float>(dev_Es, SAMPLE_BATCH * grammar->_N * MAX_LENGTH * MAX_LENGTH));

	for(int iter = 0; iter < n_iters; ++iter){
		grammar->produce_sentences_dev(
				state_array,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				SAMPLE_BATCH);

		compute_probas(dev_probas,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				grammar,
				SAMPLE_BATCH);

		if(iter < n_iters - 1){
			copy_to_host<int>(error_status + iter * SAMPLE_BATCH,
					dev_error_status, SAMPLE_BATCH);
			copy_to_host<float>(probas + iter * SAMPLE_BATCH,
					dev_probas, SAMPLE_BATCH);
		}else{
			int remainder = n_samples % SAMPLE_BATCH;
			if(remainder == 0){
				copy_to_host<int>(error_status + iter * SAMPLE_BATCH,
									dev_error_status, SAMPLE_BATCH);
				copy_to_host<float>(probas + iter * SAMPLE_BATCH,
						dev_probas, SAMPLE_BATCH);
			}else{
				copy_to_host<int>(error_status + iter * SAMPLE_BATCH,
									dev_error_status, remainder);
				copy_to_host<float>(probas + iter * SAMPLE_BATCH,
						dev_probas, remainder);
			}
		}

	}

	CUDA_CHECK(dev_free<curandState>(state_array));

	CUDA_CHECK(dev_free<int>(dev_sentences));
	CUDA_CHECK(dev_free<int>(dev_lengths));
	CUDA_CHECK(dev_free<int>(dev_error_status));

	CUDA_CHECK(dev_free<float>(dev_probas));
	CUDA_CHECK(dev_free<float>(dev_Es));

	delete grammar;

	CUDA_CHECK(cudaDeviceReset());
}

float compute_dist_batch(float * A_1,
		float * B_1,
		int N_1,
		int M_1,
		int root_1,
		float * A_2,
		float * B_2,
		int N_2,
		int M_2,
		int root_2,
		int n_samples){
	CUDA_CHECK(cudaDeviceReset());

	n_samples = ceil( ((float) n_samples) / ((float) SAMPLE_BATCH)) * SAMPLE_BATCH;

	std::cout << n_samples << std::endl;

	Sto_grammar * grammar_1 = new Sto_grammar(A_1, B_1, N_1, M_1, root_1);
	grammar_1->normalize();

	Sto_grammar * grammar_2 = new Sto_grammar(A_2, B_2, N_2, M_2, root_2);
	grammar_2->normalize();

	curandState * state_array;
	dev_alloc<curandState>(state_array, SAMPLE_BATCH * MAX_LENGTH);

	setup_states(state_array,
			SAMPLE_BATCH * MAX_LENGTH,
			time(NULL),
			0);

	int n_iters = ceil( ((float) n_samples) / ((float) SAMPLE_BATCH) );

	int * dev_sentences;
	int * dev_lengths;
	int * dev_error_status;
	float * dev_self_probas;
	float * dev_cross_probas;
	float * dev_mid_sum;

	dev_alloc<int>(dev_sentences, SAMPLE_BATCH * MAX_LENGTH);
	dev_alloc<int>(dev_lengths, SAMPLE_BATCH);
	dev_alloc<int>(dev_error_status, SAMPLE_BATCH);
	dev_alloc<float>(dev_self_probas, SAMPLE_BATCH);
	dev_alloc<float>(dev_cross_probas, SAMPLE_BATCH);
	dev_alloc<float>(dev_mid_sum, SAMPLE_BATCH);

	float distance = 0;

	for(int iter = 0; iter < n_iters; ++iter){
		grammar_1->produce_sentences_dev(
				state_array,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				SAMPLE_BATCH);
		compute_probas(dev_self_probas,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				grammar_1,
				SAMPLE_BATCH);
		compute_probas(dev_cross_probas,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				grammar_2,
				SAMPLE_BATCH);
		compute_mid_sum(dev_self_probas,
				dev_cross_probas,
				dev_mid_sum,
				SAMPLE_BATCH);
		distance += KL_div(dev_self_probas,
				dev_mid_sum,
				dev_error_status,
				SAMPLE_BATCH);

		grammar_2->produce_sentences_dev(
				state_array,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				SAMPLE_BATCH);
		compute_probas(dev_self_probas,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				grammar_2,
				SAMPLE_BATCH);
		compute_probas(dev_cross_probas,
				dev_sentences,
				dev_lengths,
				dev_error_status,
				grammar_1,
				SAMPLE_BATCH);
		compute_mid_sum(dev_self_probas,
						dev_cross_probas,
						dev_mid_sum,
						SAMPLE_BATCH);
		distance += KL_div(dev_self_probas,
				dev_mid_sum,
				dev_error_status,
				SAMPLE_BATCH);
	}

	/*
	dev_free<curandState>(state_array);

	dev_free<int>(dev_sentences);
	dev_free<int>(dev_lengths);
	dev_free<int>(dev_error_status);

	dev_free<float>(dev_self_probas);
	dev_free<float>(dev_cross_probas);
	dev_free<float>(dev_mid_sum);
	*/

	delete grammar_1;
	delete grammar_2;

	CUDA_CHECK(cudaDeviceReset());

	return distance / ((float) n_samples);
}
