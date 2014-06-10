#ifndef PROBA_SEQUITUR_H
#define PROBA_SEQUITUR_H

#include <vector>
#include <list>
#include <unordered_map>

#include <element.h>

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


class Proba_sequitur{

private:

    std::list<elt>          _inference_samples;
    std::list<elt>          _counting_samples;

    int_double_map          _relative_counts;
    int_int_map             _absolute_counts;
    int_int_map             _levels;

    int_int_pair_map        _rules;

    int_string_map          _hashcodes;
    int_double_map          _bare_lks;


public:
    Proba_sequitur(const int_vect_vect & inference_samples,
                   const int_vect_vect & counting_samples,
                   ):
    {
        // Initialize inference samples
        for(int i = 0; i < inference_samples.size(); ++i){
            for(int j = 0; j < inference_samples.at(i).size(); ++j){
                _inference_samples.push_back(
                            elt(i, j,
                                inference_samples.at(i).at(j)));
            }
        }
        elt_list_iter begin_inf = _inference_samples.begin();
        elt_list_iter end_inf = std::prev(_inference_samples.end());
        for(elt_list_iter x = _inference_samples.begin();
                          x != _inference_samples.end();
                          ++x){
            x->_iter = x;
            if(x != begin_inf){
                x->_has_prev = true;
                x->_prev = std::prev(x);
            }
            if(x != end_inf){
                x->_has_next = true;
                x->_next = std::next(x);
            }
        }

        // Initialize counting samples
        for(int i = 0; i < counting_samples.size(); ++i){
            for(int j = 0; j < counting_samples.at(i).size(); ++j){
                _counting_samples.push_back(
                            elt(i, j,
                                couting_samples.at(i).at(j)));
            }
        }
        elt_list_iter begin_count = _counting_samples.begin();
        elt_list_iter end_count = std::prev(_counting_samples.end());
    }



};


#endif // PROBA_SEQUITUR_H
