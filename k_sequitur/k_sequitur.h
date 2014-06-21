#ifndef K_SEQUITUR_H
#define K_SEQUITUR_H

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <boost/functional/hash.hpp>

#include "name_generator.h"
#include "rule.h"
#include "counter.h"

template<typename T>
class K_sequitur{

typedef Rule<T>                                         rule;
typedef std::list<T>                                    T_list;
typedef std::unordered_set<T>                           T_set;
typedef typename T_list::iterator                       iter;
typedef std::list<iter>                                 iter_list;
typedef std::pair<T_list *, iter>                       merge_coord;
typedef std::list<merge_coord>                          merge_coord_list;
typedef std::unordered_map<T, merge_coord_list>         T_merge_coord_list_map;
typedef std::list<Rule<T>*>                             rule_T_pt_list;
typedef std::unordered_map<T, Rule<T>*>                 T_rule_T_pt_map;
typedef std::unordered_map<T, iter_list>                T_rule_iter_list_map;
typedef std::unordered_map<T, int>                      T_int_map;
typedef std::unordered_map<T, std::string>              T_string_map;
typedef std::pair<T, T>                                 T_pair;
typedef std::list<T_pair>                               T_pair_list;
typedef std::pair<iter, iter>                           iter_pair;
typedef std::list<iter_pair>                            iter_pair_list;
typedef typename iter_list::iterator                    iter_list_iter;

typedef std::function<size_t(T_pair)>                   T_pair_hash;
public: T_pair_hash T_pair_hasher = [&](const T_pair & pair){
    boost::hash<std::pair<T, T>> hasher;
    return hasher(pair);
};

typedef std::unordered_map<T_pair, rule,
                           T_pair_hash>                 grammar_hashmap;


private:

    Name_generator<T>           _name_gen;
    grammar_hashmap             _grammar;
    Counter<T>                  _counter;
    const T *                   _raw_input;
    const int                   _input_length;
    const int                   _K;
    T_list                      _input;
    int                         current_index           = 0 ;
    iter                        current_position            ;
    iter                        previous_position;
    rule_T_pt_list              _order_of_creation;
    T_rule_T_pt_map             _grammar_accessor;
    T_int_map                   _pop_counts;
    T_int_map                   _ref_counts;
    T_string_map                _bar_codes;
    T_merge_coord_list_map      _ref_graph;
    T_set                       _non_terminals;

public:
    K_sequitur(const T * raw_input,
               const int & input_length,
               const int & K):
                _grammar(128, T_pair_hasher),
                _raw_input(raw_input),
                _input_length(input_length),
                _K(K),
                _input(_raw_input, _raw_input + _input_length),
                current_position(_input.begin()),
                previous_position(_input.begin()){
        ++ current_position;
    }

    void next(){
        T_pair current_pair (*(previous_position), *(current_position));
        //std::cout << "Prev: " << *(previous_position) << " , current:" << *(current_position) << std::endl;

        while(apply_rule_to_last(current_pair)){
            if(previous_position == _input.begin()){
                return;
            }else{
                -- previous_position;
                current_pair = T_pair(*(previous_position), *(current_position));
            }
        }

        // Update grammar count
        iter_pair current_iter_pair (previous_position, current_position);
        record_new_pair(current_iter_pair);

        // Step out of the modification zone
        ++current_position;

        while(create_rule(current_pair)){
            apply_rule(current_pair);
        }

        // Step into the normal position
        previous_position = current_position;
        -- previous_position;

    }


    void record_new_pair(const iter_pair & current_iter_pair){
        _counter.add_ref(current_iter_pair);
    }


    bool create_rule(const T_pair & current_pair){
        bool modified = false;
        std::list<T_pair> created;
        std::string first_barcode;
        std::string second_barcode;
        T current_lhs;
        if(_counter.get_ref_count(current_pair) >= _K){
            modified = true;
            _grammar.emplace(current_pair,
                             Rule<T>(_name_gen.next_name(), current_pair));
            _order_of_creation.push_back(&(_grammar.at(current_pair)));
            current_lhs = _grammar.at(current_pair).get_lhs();
            _grammar_accessor[current_lhs] = &(_grammar.at(current_pair));
            _ref_counts[current_lhs] = 0;
            _pop_counts[current_lhs] = 0;
            _non_terminals.insert(current_lhs);
            if(_bar_codes.count(current_pair.first) != 0){
                first_barcode = _bar_codes[current_pair.first];
            }else{
                first_barcode = std::to_string(current_pair.first);
            }
            if(_bar_codes.count(current_pair.second) != 0){
                second_barcode = _bar_codes[current_pair.second];
            }else{
                second_barcode = std::to_string(current_pair.second);
            }
            _bar_codes[current_lhs] = first_barcode + "_" + second_barcode; // n log n memory requirement
            created.push_back(current_pair);
            /*
            std::cout << "Created rule " << _grammar.at(current_pair).get_lhs()
                      << " " << current_pair.first
                      << ", " << current_pair.second << std::endl;
            */
        }
        return modified;
    }


    bool apply_rule_to_last(const T_pair & current_pair){
        if(_grammar.count(current_pair) == 0){
            return false;
        }else{
            /*
            std::cout << "Applying rule to appended "
                      << current_pair.first << " "
                      << current_pair.second << std::endl;
            */
            iter stop = current_position;
            ++ stop;
            _counter.delete_ref(iter_pair(previous_position, current_position),
                                _input, stop);
            _grammar.at(current_pair).apply(_input,
                                            previous_position,
                                            current_position);
            return true;
        }
    }


    bool apply_rule(T_pair & current_pair){
        bool begin = false;
        T_pair next_creation;

        // In current keep backward partner of first modified
        // There will be no rule to apply with new symbol as it is no in the grammar obviously

        iter_pair front = _counter.pop_front(current_pair);

        if(front.first != _input.begin()){
            iter_pair front_copy = front;
            next_creation = T_pair(*(--(front_copy.first)), _grammar.at(current_pair).get_lhs());
        }else{
            begin = true;
        }

        bool first = true;
        do{
            if(!first){
                front = _counter.pop_front(current_pair);
            }else{
                first = false;
            }
            _counter.delete_ref(front, _input, current_position);
            _grammar.at(current_pair).apply(_input,
                                            front.first,
                                            front.second);
            _counter.create_ref(iter_pair(front.first, front.second),
                                _input,
                                current_position);
        }while(_counter.get_ref_count(current_pair) > 0);

        if(!begin){
            if(_counter.get_ref_count(current_pair) >= _K){
                current_pair = next_creation;
                return true;
            }else{
                return false;
            }
        }else{
            return false;
        }
    }

    void compute_ref_counts(){
        _order_of_creation.reverse();
        T lhs;
        T_list * rhs;
        for(auto xy : _ref_counts){
            _ref_counts[xy.first] = 0;
            _pop_counts[xy.first] = 0;
        }
        _ref_graph.clear();

        // Initializing popularity and ref counts
        for(iter x = _input.begin(); x != _input.end(); ++ x){
            if(_ref_counts.count(*x) == 0) continue; // terminal symbol
            ++ _ref_counts[*x];
            ++ _pop_counts[*x];
            if(_ref_graph.count(*x) == 0){
                _ref_graph.emplace(*x, merge_coord_list());
            }
            _ref_graph.at(*x).push_back(merge_coord(&_input, x));
        }

        // Going through the map
        T_int_map local_refs;
        for(Rule<T> * current_rule : _order_of_creation){
            lhs = current_rule->get_lhs();
            rhs = current_rule->get_rhs_pt();
            local_refs.clear();
            for(iter x = current_rule->get_rhs().begin(); x != current_rule->get_rhs().end(); ++ x){
                if(_ref_counts.count(*x) != 0){
                    if(_ref_graph.count(*x) == 0){
                        _ref_graph.emplace(*x, merge_coord_list());
                    }
                    _ref_graph.at(*x).push_back(merge_coord(current_rule->get_rhs_pt(), x));
                    if(local_refs.count(*x) == 0){
                        local_refs[*x] = 0;
                    }
                    ++ local_refs[*x];
                }
            }
            for(auto xy : local_refs){
                _pop_counts[xy.first] += xy.second * _pop_counts[lhs];
                ++ _ref_counts[xy.first];
            }
        }

        _order_of_creation.reverse();
    }

    void collapse_grammar(int minimum){
        //std::cout << std::endl;
        //std::cout << "COLLAPSING GRAMMAR" << std::endl;
        compute_ref_counts();
        //_order_of_creation.reverse();
        T_list to_delete;
        for(Rule<T>* x : _order_of_creation){
            if(_ref_counts.at(x->get_lhs()) < minimum){
                //std::cout << "\rDeleting rule " << x->get_lhs() << std::endl;
                to_delete.push_back(x->get_lhs());
                for(auto rule : x->get_rhs()){
                    -- _ref_counts[rule];
                }
                for(auto z : _ref_graph.at(x->get_lhs())){
                    //if(z.first == &_input) std::cout << "BOOOM" << *(z.second) << std::endl;
                    x->merge(*(z.first), z.second);
                }
            }
        }
        for(auto x : to_delete){
            _ref_counts.erase(x);
            _pop_counts.erase(x);
            _bar_codes.erase(x);
            _grammar_accessor.erase(x);
        }
        //_order_of_creation.reverse();
        for(auto z : _input){
            if(z < 0 && _grammar_accessor.count(z) == 0) std::cout << "ERROR " << z << std::endl;
        }
        //std::cout << std::endl;
        recompute_bar_codes();
        compute_ref_counts();
    }

    void print_input(){
        for(iter x = _input.begin(); x != current_position; ++ x){
            std::cout << *x << " ";// << //"(" << &(*x) << "), ";
        }std::cout << std::endl;
    }

    void print_counts() const{
        _counter.print();
    }

    void print_grammar() const{
        for(auto xy : _grammar_accessor){
            std::cout << "Rule " << xy.first << " -> ";
            for(auto z : xy.second->get_rhs()){
                std::cout << z << "_";
            }std::cout << std::endl;
        }std::cout << std::endl;
    }

    void print_ref_counts() const{
        for(auto xy : _ref_counts){
            std::cout << xy.first << " refs: "
                      << xy.second << " used: "
                      << _pop_counts.at(xy.first) << " barcode: "
                      << _bar_codes.at(xy.first) << std::endl;
        }std::cout << std::endl;
    }

    void print(){
        print_input();
        print_grammar();
        print_counts();
    }

    void fill_with_info(T * lhs_array, int & n_lhs,
                        int stride,
                        T * input_string, int & input_string_length,
                        T * rhs_array, int * rhs_lengths,
                        int * barcode_array, int * barcode_lengths,
                        int * ref_count_array, int * pop_count_array){
        n_lhs = _grammar_accessor.size();
        int i = 0;
        int j = 0;
        input_string_length = _input.size();
        for(auto x : _input){
            input_string[j++] = x;
        }
        std::string barcode;
        for(auto xy : _grammar_accessor){
            lhs_array[i] = xy.first;
            rhs_lengths[i] = xy.second->get_rhs().size();
            j = 0;
            for(auto x : xy.second->get_rhs()){
                rhs_array[i*stride + j++] = x;
            }
            barcode = _bar_codes[xy.first];
            barcode_lengths[i] = barcode.length();
            for(j = 0; j < barcode.length(); ++j){
                barcode_array[i*stride + j] = barcode[j] - '0';
            }
            ref_count_array[i] = _ref_counts[xy.first];
            pop_count_array[i] = _pop_counts[xy.first];
            ++i;
        }
    }

    void recompute_bar_codes(){
        _bar_codes.clear();
        T lhs;
        T_list * rhs;
        for(Rule<T> * rule_ptr : _order_of_creation){
            lhs = rule_ptr->get_lhs();
            rhs = rule_ptr->get_rhs_pt();
            _bar_codes[lhs] = "";
            iter preend = rhs->end();
            std::advance(preend, -1);
            for(iter x = rhs->begin(); x != rhs->end(); ++x){
                if(_non_terminals.count(*x) == 0){
                    _bar_codes[lhs] += std::to_string(*x);
                }else{
                    _bar_codes[lhs] += _bar_codes[*x];
                }
            }
        }
    }

    void reconstruct_input(){
        std::string result;
        for(auto x : _input){
            if(_bar_codes.count(x) != 0){
                result += _bar_codes[x];
            }else{
                result += std::to_string(x);
            }
        }
        std::cout << result << std::endl;
    }


};



#endif // K_SEQUITUR_H
