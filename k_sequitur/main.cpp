#include <iostream>

#include "rule.h"
#include "k_sequitur.h"
#include "file_reader.h"

#include <list>

typedef int T;
typedef std::list<T>                                    T_list;
typedef typename T_list::iterator                       iter;
typedef std::pair<T, T>                                 T_pair;
typedef std::pair<iter, iter>                           iter_pair;
typedef std::list<iter>                                 iter_list;
typedef std::list<iter_pair>                            iter_pair_list;
typedef typename iter_pair_list::iterator               iter_pair_list_iter;



int main(){

    std::string file_path = "/Users/francois/Grammar_TS/Sequitur/data/achuSeq_8.csv";

    int * raw_input;
    int length;

    read_csv(file_path, raw_input, length);

    std::cout << "N elements: " << length << std::endl;

    /*
    for(int i =  0; i < length; ++i){
        std::cout << raw_input[i];
    }std::cout << std::endl;
    */

    int cut = length;
    int k = 3;
    int q = 6;

    K_sequitur<T> k_sequitur(raw_input, cut, k);

    for(int i = 0; i < cut - 1; ++i){
        k_sequitur.next();
    }

    //k_sequitur.print_input();
    //k_sequitur.reconstruct_input();
    //k_sequitur.print_grammar();
    //k_sequitur.compute_ref_counts();
    //k_sequitur.print_ref_counts();

    k_sequitur.collapse_grammar(q);

    T * input_string = new T[cut];
    int input_string_length;
    T * lhs_array = new T[cut];
    int n_lhs;
    T* rhs_array = new T[cut * cut];
    int * rhs_lengths = new int[cut];

    int * barcode_array = new int[cut * cut];
    int * barcode_lengths = new int[cut];

    int * ref_count_array = new int[cut];
    int * pop_count_array = new int[cut];

    k_sequitur.fill_with_info(lhs_array, n_lhs,
                              cut,
                              input_string, input_string_length,
                              rhs_array, rhs_lengths,
                              barcode_array, barcode_lengths,
                              ref_count_array, pop_count_array);

    //k_sequitur.print_input();
    //k_sequitur.reconstruct_input();

    /*
    std::cout << std::endl;
    for(int i = 0; i < n_lhs; ++i){
        std::cout << "Rule: " << lhs_array[i]
                  << " (refs: " << ref_count_array[i] << ")"
                  << " (pop: " << pop_count_array[i] << ")"
                  << " (barcode: ";
        for(int j = 0; j < barcode_lengths[i]; ++j){
            std::cout << barcode_array[i*cut + j];
        }
        std::cout << ") -> ";
        for(int j = 0; j < rhs_lengths[i]; ++j){
            std::cout << rhs_array[i*cut + j] << " ";
        }
        std::cout << std::endl;
    }
    */

    return 0;
}
