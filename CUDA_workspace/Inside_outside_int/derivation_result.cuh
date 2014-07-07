/*
 * derivation_result.h
 *
 *  Created on: 6 juil. 2014
 *      Author: francois
 */

#ifndef DERIVATION_RESULT_H_
#define DERIVATION_RESULT_H_

struct cu_Derivation_result{
	bool 	_is_term;
	int		_left;
	int		_right;
	int		_term;

	__device__ cu_Derivation_result();

	__device__ cu_Derivation_result(bool is_term,
			int left, int right, int term);

}der_res;


#endif /* DERIVATION_RESULT_H_ */
