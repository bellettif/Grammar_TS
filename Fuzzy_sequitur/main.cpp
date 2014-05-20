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

    for(int i =  0; i < length; ++i){
        std::cout << raw_input[i];
    }std::cout << std::endl;

    return 0;
}
