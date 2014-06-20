#include <iostream>
#include <string>
#include "rule.h"
#include "sequitur.h"
#include "name_generator.h"
#include "file_reader.h"

typedef std::pair<int, Rule<int>*>                          int_pointing_c;
typedef std::list<int_pointing_c>                           int_d_linked_list;
typedef typename int_d_linked_list::iterator                int_it;
typedef std::unordered_map<int, std::list<int>>             hash_map_int_i;
typedef std::unordered_map<size_t,
        std::pair<int_it, int>>                             hash_map_size_t_pair_int_it_i;
typedef std::unordered_map<size_t, int>                     hash_map_size_t_int;
typedef std::unordered_map<int, Rule<int>>                  hash_map_int_rule_int;
typedef typename hash_map_int_rule_int::iterator            int_gram_it;


int main(){

    std::string file_path = "/Users/francois/Grammar_TS/Sequitur/data/achuSeq_1.csv";

    int * content;
    int n_elts;


    read_csv(file_path, content, n_elts);

    std::cout << "N elements: " << n_elts << std::endl;

    int root_symbol = 0;

    Sequitur<int> seq(content, n_elts, 0, Name_generator<int>(0));

    for(int i = 0; i < n_elts; ++i){
        std::cout << std::endl;
        std::cout << "Iteration" << i << std::endl;
        seq.append_next();
        seq.compute_next();
    }

    std::cout << std::endl;
    std::cout << "ENFORCING UTILITY RULE" << std::endl;
    std::cout << std::endl;

    seq.enforce_utility();
    seq.print();

    int stream_length = seq.get_stream().get_RHS().size();
    int * final_stream = new int[stream_length];
    const int_d_linked_list & stream_rhs = seq.get_stream().get_RHS();
    int current_index = 0;
    for(auto x : stream_rhs){
        final_stream[current_index++] = x.first;
    }

    const hash_map_int_rule_int & grammar = seq.get_grammar();

    int n_rules = grammar.size() - 1;
    int ** rhs_array;
    int * lhs = new int[n_rules];
    int * rhs_lengths = new int[n_rules];
    int * ref_counts = new int[n_rules];
    int max_length = 0;

    int current_length;
    for(auto xy : grammar){
        if(xy.first == 0) continue;
        current_length = xy.second.get_RHS().size();
        if(current_length > max_length) max_length = current_length;
    }

    rhs_array = new int*[n_rules];
    for(int i = 0; i < n_rules; ++i){
        rhs_array[i] = new int[max_length];
    }

    int line = 0;
    int j;
    for(auto xy : grammar){
        if(xy.first == 0) continue;
        current_length = xy.second.get_RHS().size();
        lhs[line] = xy.first;
        rhs_lengths[line] = xy.second.get_RHS().size();
        ref_counts[line] = xy.second.get_ref_count();
        j = 0;
        for(auto x : xy.second.get_RHS()){
            rhs_array[line][j++] = x.first;
        }
        for(; j < max_length; ++j){
            rhs_array[line][j] = root_symbol;
        }
        ++ line;
    }

    std::cout << std::endl;
    std::cout << "Number of rules: " << n_rules << std::endl;
    std::cout << "Max length: " << max_length << std::endl;
    std::cout << std::endl;


    std::cout << "Sentence: " << root_symbol << " -> ";
    for(int i = 0; i < stream_length; ++i){
        std::cout << final_stream[i] << " ";
    }std::cout << std::endl;

    std::cout << "Grammar: " << std::endl;
    for(int i = 0; i < n_rules; ++i){
        std::cout << "Rule: " << lhs[i] << ", refs: " << ref_counts[i] << ", len: " << rhs_lengths[i] << " -> ";
        for(int h = 0; h < max_length; ++h){
            std::cout << rhs_array[i][h] << " ";
        }std::cout << std::endl;
    }


    for(int i = 0; i < n_rules; ++i){
        delete[] rhs_array[i];
    }
    delete[] rhs_array;

    delete[] final_stream;
    delete[] lhs;
    delete[] rhs_lengths;
    delete[] ref_counts;

    delete [] content;

}
