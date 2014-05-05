#include <iostream>

#include "Entropy_meas/Entropy_measure.h"

static double comp_entropy(int * content,
						 int length){
	std::vector<int> input_data(content, content + length);
	Entropy_measure<int> e_m(input_data);
	return e_m.compute_entropy();
}

static double comp_string_entropy(int * content,
								   int length,
								   int k){
	std::vector<std::vector<int>> strings(length - k + 1);
	for(int i = 0; i < length - k + 1; ++i){
		strings[i].assign(content + i, content + i + k);
	}
	Entropy_measure<std::vector<int>> e_m_rolling(strings);
	return e_m_rolling.compute_entropy();
}

static void comp_rolling_entropy(int * content,
							     int length,
							     int k,
							     double * result){
	std::vector<int> window;
	Entropy_measure<int> e_m;
	for(int i = 0; i < length - k + 1; ++i){
		window.assign(content + i, content + i + k);
		e_m.reset(window);
		result[i] = e_m.compute_entropy();
		window.clear();
	}
}

static double comp_gini(int * content,
						 int length){
	std::vector<int> input_data(content, content + length);
	Entropy_measure<int> e_m(input_data);
	return e_m.compute_gini();
}

static void comp_rolling_gini(int * content,
							  int length,
							  int k,
							  double * result){
	std::vector<int> window;
	Entropy_measure<int> e_m;
	for(int i = 0; i < length - k + 1; ++i){
		window.assign(content + i, content + i + k);
		e_m.reset(window);
		result[i] = e_m.compute_gini();
		window.clear();
	}
}

static double comp_string_gini(int * content,
								   int length,
								   int k){
	std::vector<std::vector<int>> strings(length - k + 1);
	for(int i = 0; i < length - k + 1; ++i){
		strings[i].assign(content + i, content + i + k);
	}
	Entropy_measure<std::vector<int>> e_m_rolling(strings);
	return e_m_rolling.compute_gini();
}
