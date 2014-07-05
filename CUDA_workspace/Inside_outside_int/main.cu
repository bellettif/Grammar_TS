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
	simple_grammar.set_A(0, 1, 3, 0.3);
	simple_grammar.set_A(0, 2, 4, 0.3);
	simple_grammar.set_A(0, 1, 1, 0.2);
	simple_grammar.set_A(0, 2, 2, 0.2);
	simple_grammar.set_A(3, 0, 1, 1.0);
	simple_grammar.set_A(4, 0, 2, 1.0);
	/*
	 * Terminal symbols
	 */
	simple_grammar.set_B(1, 0, 1.0);
	simple_grammar.set_B(2, 1, 1.0);
	/*
	 * Normalize
	 */
	simple_grammar.printB();

	return 0;
}
