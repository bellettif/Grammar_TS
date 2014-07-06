#include <iostream>
#include <string>
#include <time.h>
#include <iostream>

#include "sto_grammar.h"
#include "matrix_utils.h"
#include "utils.h"

int main(void){

	/*
	 * Grammar example Figure 4 of Lari, Young 1987
	 */
	cu_Sto_grammar simple_grammar(5, 2);
	/*
	 * Non terminal symbols
	 */
	simple_grammar.set_A(0, 1, 3, 3);
	simple_grammar.set_A(0, 2, 4, 3);
	simple_grammar.set_A(0, 1, 1, 2);
	simple_grammar.set_A(0, 2, 2, 2);
	simple_grammar.set_A(3, 0, 1, 10.0);
	simple_grammar.set_A(4, 0, 2, 10.0);
	/*
	 * Terminal symbols
	 */
	simple_grammar.set_B(1, 0, 0.1);
	simple_grammar.set_B(2, 1, 0.1);
	/*
	 * Normalize
	 */
	simple_grammar.normalize();

	std::cout << "A matrix:" << std::endl;
	simple_grammar.printA();
	std::cout << std::endl;

	std::cout << "B matrix:" << std::endl;
	simple_grammar.printB();
	std::cout << std::endl;

	std::cout << "Non term weights matrix:" << std::endl;
	simple_grammar.print_non_term_weights();
	std::cout << std::endl;

	std::cout << "Term weights matrix:" << std::endl;
	simple_grammar.print_term_weights();
	std::cout << std::endl;

	std::cout << "Tot weights matrix:" << std::endl;
	simple_grammar.print_tot_weights();
	std::cout << std::endl;

	return 0;
}
