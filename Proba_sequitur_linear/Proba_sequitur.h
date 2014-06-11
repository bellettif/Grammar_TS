#ifndef PROBA_SEQUITUR_H
#define PROBA_SEQUITUR_H

#include <vector>
#include <list>
#include <unordered_map>
#include <string>

#include "element.h"
#include "mem_sandwich.h"
#include "decision_making.h"

typedef std::vector<std::vector<int>>           int_vect_vect;
typedef std::unordered_map<int, double>         int_double_map;
typedef std::unordered_map<int, int>            int_int_map;
typedef std::pair<int, int>                     int_pair;
typedef std::pair<int, int_pair>                int_int_pair_pair;
typedef std::unordered_map<int, int_pair>       int_int_pair_map;
typedef std::unordered_map<int, std::string>    int_string_map;
typedef std::unordered_map<std::string, int>    string_int_map;
typedef Element                                 elt;
typedef std::list<elt>                          elt_list;
typedef std::vector<elt_list>                   elt_list_vect;
typedef elt_list::iterator                      elt_list_iter;
typedef std::vector<Mem_sandwich>               mem_vect;

class Proba_sequitur{

private:

    const int               _n_select;

    elt_list_vect           _inference_samples;
    elt_list_vect           _counting_samples;

    int_double_map          _relative_counts;
    int_int_map             _absolute_counts;
    int_int_map             _levels;

    int_int_pair_map        _rules;

    int_string_map          _hashcodes;
    int_double_map          _bare_lks;

    mem_vect                _sample_memory;
    mem_vect                _counting_memory;

    const string_int_map &  _to_index_map;
    const int_string_map &  _to_string_map;

    int_pair_double_map     _pattern_scores;


public:
    Proba_sequitur(const int & n_select,
                   const int_vect_vect & inference_samples,
                   const int_vect_vect & counting_samples,
                   const string_int_map & to_index_map,
                   const int_string_map & to_string_map):
        _n_select(n_select),
        _inference_samples(inference_samples.size()),
        _counting_samples(counting_samples.size()),
        _sample_memory(inference_samples.size()),
        _counting_memory(counting_samples.size()),
        _to_index_map(to_index_map),
        _to_string_map(to_string_map),
        _pattern_scores(10, pair_hasher)
    {
        // Initialize inference samples
        elt_list * current_list;
        for(int i = 0; i < inference_samples.size(); ++i){
            current_list = & _inference_samples.at(i);
            _sample_memory.at(i).set_target_list(current_list);
            for(int j = 0; j < inference_samples.at(i).size(); ++j){
                current_list->push_back(
                            elt(i, j,
                                inference_samples.at(i).at(j)));
                _bare_lks[inference_samples.at(i).at(j)] += 1.0;
            }
            for(elt_list_iter x = current_list->begin();
                              x != current_list->end();
                              ++x){
                x->_iter = x;
                if(x != current_list->begin()){
                    x->_has_prev = true;
                    x->_prev = std::prev(x);
                }
                if(x != std::prev(current_list->end())){
                    x->_has_next = true;
                    x->_next = std::next(x);
                    _sample_memory.at(i).add_pair({x->_iter, x->_next});
                }
            }
        }

        // Initialize counting samples
        for(int i = 0; i < counting_samples.size(); ++i){
            current_list = & _counting_samples.at(i);
            _counting_memory.at(i).set_target_list(current_list);
            for(int j = 0; j < counting_samples.at(i).size(); ++j){
                current_list->push_back(
                                elt(i, j,
                                counting_samples.at(i).at(j)));
            }
            for(elt_list_iter x = current_list->begin();
                              x != current_list->end();
                              ++x){
                x->_iter = x;
                if(x != current_list->begin()){
                    x->_has_prev = true;
                    x->_prev = std::prev(x);
                }
                if(x != std::prev(current_list->end())){
                    x->_has_next = true;
                    x->_next = std::next(x);
                    _counting_memory.at(i).add_pair({x->_iter, x->_next});
                }
            }
        }

        // Initializing bare lks
        double total_counts = 0.0;
        for(auto xy : _bare_lks){
            total_counts += xy.second;
        }
        for(auto xy : _bare_lks){
            _bare_lks[xy.first] /= total_counts;
        }
    }

    void print_inference_seq(int index){
        for(const elt & x : _inference_samples.at(index)){
            std::cout << x << " ";
        }std::cout << std::endl;
    }

    void print_counting_seq(int index){
        for(const elt & x : _counting_samples.at(index)){
            std::cout << x << " ";
        }std::cout << std::endl;
    }

    void print_bare_lks(){
        std::cout << "Printing bare lks: " << std::endl;
        for(auto xy : _bare_lks){
            std::cout << "Key: " << _to_string_map.at(xy.first) << " lk: " << xy.second << std::endl;
        }
    }

    void compute_pattern_scores(){
        std::cout << "Computing pattern scores" << std::endl;
        decision_making::compute_pattern_counts(_sample_memory,
                                                _pattern_scores);
        decision_making::delete_zeros(_pattern_scores);
        decision_making::compute_pattern_divergence(_bare_lks,
                                                    _rules,
                                                    _pattern_scores);
        std::cout << "Done" << std::endl;
    }

    void print_pattern_scores(){
        std::string left;
        std::string right;
        for(auto xy : _pattern_scores){
            if(xy.first.first >= 0){
                left = _to_string_map.at(xy.first.first);
            }else{
                left = std::to_string(xy.first.first);
            }
            if(xy.first.second >= 0){
                right = _to_string_map.at(xy.first.second);
            }else{
                right = std::to_string(xy.first.second);
            }
            std::cout << "Pattern "
                      << left
                      << " - "
                      << right
                      << ", score: " << xy.second << std::endl;
        }
    }

    void replace_best_patterns(){
        int_pair_vect best_pairs =
                decision_making::pick_best_patterns(_pattern_scores,
                                                    _n_select);
        int rule_index;
        /*
        std::cout << std::endl;
        for(Mem_sandwich & mem : _counting_memory){
            mem.print_center_lists(_to_string_map);
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */
        for(const int_pair & xy: best_pairs){
            rule_index = - (_rules.size() + 1);
            std::cout << "REPLACING PAIR "
                      << _to_string_map.at(xy.first)
                      << " "
                      << _to_string_map.at(xy.second)
                      << " by "
                      << rule_index
                      << std::endl;
            _rules[rule_index] = xy;
            for(Mem_sandwich & mem : _sample_memory){
                mem.remove_pair(xy, rule_index);
            }
            for(Mem_sandwich & mem : _counting_memory){
                mem.remove_pair(xy, rule_index);
            }
        }
        /*
        std::cout << std::endl;
        for(Mem_sandwich & mem : _counting_memory){
            mem.print_center_lists(_to_string_map);
            std::cout << std::endl;
        }
        */
    }


};


#endif // PROBA_SEQUITUR_H
