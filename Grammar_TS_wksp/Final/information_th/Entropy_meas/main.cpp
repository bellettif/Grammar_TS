#include<iostream>
#include <random>

#include "Entropy_measure.h"


int main(){
	int N = 100000;
	double p = 0.1;
	int k = 5;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution d(p);

	std::vector<int> input_data(N);
	for(int i = 0; i < N; ++i){
		input_data[i] = d(gen);
	}

	std::vector<std::vector<int>> strings(N - k + 1, std::vector<int>(k));
	// TODO
	std::vector<int>::iterator begin_it = input_data.begin();
	std::vector<int>::iterator end_it = input_data.begin();
	std::advance(end_it, k);
	for(int i = 0; i < N - k + 1; ++i){
		std::copy(begin_it++, end_it++, strings[i].begin());
	}


	double mean = 0;
	for(int i = 0; i < N; ++i){
		mean += input_data[i];
	}
	mean /= (double) N;
	std::cout << "Mean: " << mean << std::endl;

	Entropy_measure<int> e_m(input_data);
	std::cout << e_m.compute_entropy() << std::endl;
	e_m.show_counts();

	Entropy_measure<std::vector<int>> e_m_rolling(strings);
	std::cout << e_m_rolling.compute_entropy() << std::endl;
	e_m_rolling.show_counts();

	return 0;
}
