#include "derivation_result.cuh"

__device__ cu_Derivation_result::cu_Derivation_result():
	_is_term(false),
	_left(0),
	_right(0),
	_term(0)
{}

__device__ cu_Derivation_result::cu_Derivation_result(bool is_term,
			int left, int right, int term):
			_is_term(term),
			_left(left),
			_right(right),
			_term(term)
{}
