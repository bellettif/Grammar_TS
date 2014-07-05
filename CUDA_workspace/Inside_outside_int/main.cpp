#include <iostream>
#include <string>
#include <time.h>
#include <iostream>

#include "sto_grammar.h"
#include "matrix_utils.h"

int main(void){

	srand(time(NULL));

	const int N = 128;
	float A[N];
	float B[N];
	float C[N];

	for(int i = 0; i < N; ++i){
		A[i] = i * i;
		B[i] = -i;
	}

	add_vectors(A, B, C, N);

	return 0;
}
