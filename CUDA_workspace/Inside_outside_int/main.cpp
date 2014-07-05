#include <iostream>
#include <string>
#include <time.h>

#include "sto_grammar.h"

int main(void){

	srand(time(NULL));

	const int N_sums = 64;
	const int stride = 1024;

	float input[N_sums * stride];
	float sums[N_sums];

	int i, j;
	for(i = 0; i < N_sums; ++i){
		for(j = 0; j < stride; ++j){
			input[i * stride + j] = i * j;
		}
	}

	compute_sums(input, sums,
			stride, N_sums);

	for(int i = 0; i < N_sums; ++i){
		std::cout << sums[i] << std::endl;
	}

	return 0;
}
