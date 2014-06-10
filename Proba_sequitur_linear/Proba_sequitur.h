#ifndef PROBA_SEQUITUR_H
#define PROBA_SEQUITUR_H

#include <vector>
#include <list>
#include <unordered_map>

#include "element.h"
#include "mem_sandwich.h"

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

    elt_list_vect           _inference_samples;
    elt_list_vect           _counting_samples;

    int_double_map          _relative_counts;
    int_int_map             _absolute_counts;
    int_int_map             _levels;

    int_int_pair_map        _rules;

    int_string_map          _hashcodes;
    int_double_map          _bare_lks;

    mem_vect                _sample_memory;
    mem_vect                _count_memory;

public:
    Proba_sequitur(const int_vect_vect & inference_samples,
                   const int_vect_vect & counting_samples,
                   int max_length = 10000):
        _inference_samples(inference_samples.size()),
        _counting_samples(counting_samples.size()),
        _sample_memory(inference_samples.size()),
        _count_memory(counting_samples.size())
    {
        // Initialize inference samples
        elt_list * current_list;
        for(int i = 0; i < inference_samples.size(); ++i){
            current_list = & _inference_samples.at(i);
            for(int j = 0; j < inference_samples.at(i).size(); ++j){
                current_list->push_back(
                            elt(i, j,
                                inference_samples.at(i).at(j)));
            }
            max_length = std::max<int>(max_length,
                                       current_list->size());
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
            std::cout << "Printing sample memory " << i << std::endl;
            _sample_memory.at(i).print();
            std::cout << "Done" << std::endl;
            std::cout << std::endl;
        }

        // Initialize counting samples
        for(int i = 0; i < counting_samples.size(); ++i){
            current_list = & _counting_samples.at(i);
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
                    _count_memory.at(i).add_pair({x->_iter, x->_next});
                }
            }
            std::cout << "Printing count memory " << i << std::endl;
            _count_memory.at(i).print();
            std::cout << "Done" << std::endl;
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



};


#endif // PROBA_SEQUITUR_H
