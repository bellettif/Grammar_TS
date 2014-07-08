#include <iostream>
#include <string>
#include <time.h>
#include <iostream>

#include"launch.h"

int main(void){

	int N = 16;
	int M = 4;

	float * A_1 = (float*) calloc(N * N * N, sizeof(float));
	float * B_1 = (float*) calloc(N * M, sizeof(float));

	float * A_2 = (float*) calloc(N * N * N, sizeof(float));
	float * B_2 = (float*) calloc(N * M, sizeof(float));

	A_1[0 * N * N + 1 * N + 2] = 0.3;
	A_1[0 * N * N + 3 * N + 4] = 0.3;
	A_1[0 * N * N + 5 * N + 6] = 0.3;

	A_1[0 * N * N + 1 * N + 1] = 0.2;
	A_1[0 * N * N + 3 * N + 3] = 0.2;
	A_1[0 * N * N + 5 * N + 5] = 0.2;

	A_1[2 * N * N + 0 * N + 1] = 1.0;
	A_1[4 * N * N + 0 * N + 3] = 1.0;
	A_1[6 * N * N + 0 * N + 5] = 1.0;

	B_1[1 * M + 0] = 1.0;
	B_1[3 * M + 1] = 1.0;
	B_1[5 * M + 2] = 1.0;
	for(int i = 7; i < N; ++i){
		B_1[i * M + 3] = 1.0;
	}

	A_2[0 * N * N + 1 * N + 2] = 0.23;
	A_2[0 * N * N + 3 * N + 4] = 0.23;
	A_2[0 * N * N + 5 * N + 6] = 0.23;

	A_2[0 * N * N + 1 * N + 1] = 0.2;
	A_2[0 * N * N + 3 * N + 3] = 0.2;
	A_2[0 * N * N + 5 * N + 5] = 0.2;

	A_2[2 * N * N + 0 * N + 1] = 1.0;
	A_2[4 * N * N + 0 * N + 3] = 1.0;
	A_2[6 * N * N + 0 * N + 5] = 1.0;

	B_2[1 * M + 0] = 1.0;
	B_2[3 * M + 1] = 1.0;
	B_2[5 * M + 2] = 1.0;
	for(int i = 7; i < N; ++i){
		B_2[i * M + 3] = 1.0;
	}

	const int n_samples = 4096 * 8;

	time_t timer;
	time(&timer);

	std::cout << sqrt(compute_dist_batch(A_1, B_1,
						N, M,
						0,
						A_2, B_2,
						N, M,
						0,
						n_samples)) << std::endl;

	std::cout << difftime(time(NULL), timer) << std::endl;

	free(A_1);
	free(B_1);

	free(A_2);
	free(B_2);

	return 0;
}
