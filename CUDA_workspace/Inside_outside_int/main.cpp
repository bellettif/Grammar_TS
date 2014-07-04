#include <iostream>
#include <string>
#include <time.h>

#include "sto_grammar.h"
#include "choice.h"

int main(void){

	srand(time(NULL));

	/*
	 * Grammar example Figure 4 of Lari, Young 1987
	 */
	Sto_grammar simple_grammar(5, 2);
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
	simple_grammar.normalize();

	/*
	 * Grammar example Figure 7 of Lari, Young 1987
	 */
	Sto_grammar palindrom_grammar(7, 3);
	/*
	 * Non terminal symbols
	 */
	palindrom_grammar.set_A(0, 1, 2, 0.3);
	palindrom_grammar.set_A(0, 3, 4, 0.3);
	palindrom_grammar.set_A(0, 5, 6, 0.3);
	palindrom_grammar.set_A(0, 1, 1, 0.2);
	palindrom_grammar.set_A(0, 3, 3, 0.2);
	palindrom_grammar.set_A(0, 5, 5, 0.2);
	//
	palindrom_grammar.set_A(2, 0, 1, 1.0);
	palindrom_grammar.set_A(4, 0, 3, 1.0);
	palindrom_grammar.set_A(6, 0, 5, 1.0);
	/*
	 * Terminal symbols
	 */
	palindrom_grammar.set_B(1, 0, 1.0);
	palindrom_grammar.set_B(3, 1, 1.0);
	palindrom_grammar.set_B(5, 2, 1.0);
	/*
	 * Normalize
	 */
	palindrom_grammar.normalize();

	int n_samples = 10000;

	int MAX_LENGTH = 256;

	int * sentence = new int[MAX_LENGTH];
	int length;

	int error_code;

	for(int i = 0; i < 100; ++i){
		error_code = palindrom_grammar.produce_sentence(sentence,
				length,
				MAX_LENGTH);
		if(error_code == 0){
			for(int i = 0; i < length; ++i){
				std::cout << sentence[i] << " ";
			}std::cout << std::endl;
		}else if(error_code == 1){
			std::cout << "ERROR (TOO LONG)" << std::endl;
		}else if(error_code == 2){
			std::cout << "ERROR (LOOPING)" << std::endl;
		}
	}

	delete [] sentence;

	return 0;
}
