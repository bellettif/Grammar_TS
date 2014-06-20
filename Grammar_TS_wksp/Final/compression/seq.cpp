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


static void exec_sequitur(int * content,
							int root_symbol,
							int n_elts,
							int * rhs_array,
							int max_length,
							int & length_overflow,
							int * final_stream,
							int & stream_length,
							int * lhs,
							int * rhs_lengths,
							int * ref_counts,
							int & n_rules){

	Sequitur<int> seq(content, n_elts, 0, Name_generator<int>(root_symbol));

    for(int i = 0; i < n_elts; ++i){
        seq.append_next();
        seq.compute_next();
    }
    seq.enforce_utility();

    stream_length = seq.get_stream().get_RHS().size();
    const int_d_linked_list & stream_rhs = seq.get_stream().get_RHS();
    int current_index = 0;
    for(auto x : stream_rhs){
        final_stream[current_index++] = x.first;
    }

    const hash_map_int_rule_int & grammar = seq.get_grammar();
    n_rules = grammar.size() - 1;

    for(auto xy : grammar){
        if(xy.first == 0) continue;
        length_overflow += (xy.second.get_RHS().size() > max_length);
    }

    int line = 0;
    int j;
    for(auto xy : grammar){
        if(xy.first == 0) continue;
        lhs[line] = xy.first;
        rhs_lengths[line] = xy.second.get_RHS().size();
        ref_counts[line] = xy.second.get_ref_count();
        j = 0;
        for(auto x : xy.second.get_RHS()){
            rhs_array[line * max_length + (j++)] = x.first;
        }
        for(; j < max_length; ++j){
            rhs_array[line * max_length + j] = root_symbol;
        }
        ++ line;
    }

}
