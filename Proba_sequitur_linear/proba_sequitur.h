#ifndef PROBA_SEQUITUR_H
#define PROBA_SEQUITUR_H

#include <vector>
#include <list>
#include <unordered_map>
#include <string>
#include <iostream>

#include "element.h"
#include "mem_sandwich.h"
#include "decision_making.h"

typedef std::vector<double>                         double_vect;
typedef std::vector<double_vect>                    double_vect_vect;
typedef std::vector<int>                            int_vect;
typedef std::vector<int_vect>                       int_vect_vect;
typedef std::unordered_map<int, double>             int_double_map;
typedef std::unordered_map<int, int_double_map>     int_int_double_map;
typedef std::unordered_map<int, int>                int_int_map;
typedef std::unordered_map<int, int_int_map>        int_int_int_map;
typedef std::pair<int, int>                         int_pair;
typedef std::pair<int, int_pair>                    int_int_pair_pair;
typedef std::unordered_map<int, int_pair>           int_int_pair_map;
typedef std::unordered_map<int, std::string>        int_string_map;
typedef std::unordered_map<std::string, int>        string_int_map;
typedef Element                                     elt;
typedef std::list<elt>                              elt_list;
typedef std::vector<elt_list>                       elt_list_vect;
typedef elt_list::iterator                          elt_list_iter;
typedef std::vector<Mem_sandwich>                   mem_vect;
typedef std::vector<std::string>                    string_vect;
typedef std::vector<string_vect>                    string_vect_vect;
typedef std::pair<std::string, std::string>         string_pair;
typedef std::vector<string_pair>                    string_pair_vect;


class Proba_sequitur{

private:
    const int               _n_select;
    const int               _max_rules;
    const bool              _random;
    const double            _init_T;
    const double            _T_decay;

    double                  _T;

    elt_list_vect           _inference_samples;
    elt_list_vect           _counting_samples;

    int_int_double_map      _relative_counts;
    int_int_int_map         _absolute_counts;
    int_int_map             _levels;
    int_int_map             _depths;

    int_int_pair_map        _rules;

    int_string_map          _to_hash_map;
    string_int_map          _to_rule_index_map;
    int_double_map          _bare_lks;

    mem_vect                _sample_memory;
    mem_vect                _counting_memory;

    const string_int_map &  _to_index_map;
    const int_string_map &  _to_string_map;

    int_pair_double_map     _pattern_scores;
    double_vect             _divergences;


public:
    Proba_sequitur(const int & n_select,
                   const int & max_rules,
                   bool random,
                   const int_vect_vect & inference_samples,
                   const int_vect_vect & counting_samples,
                   const string_int_map & to_index_map,
                   const int_string_map & to_string_map,
                   const double & init_T = 0,
                   const double & T_decay = 0):
        _n_select(n_select),
        _max_rules(max_rules),
        _random(random),
        _init_T(init_T),
        _T(init_T),
        _T_decay(T_decay),
        _inference_samples(inference_samples.size()),
        _counting_samples(counting_samples.size()),
        _to_hash_map(to_string_map),
        _to_rule_index_map(to_index_map),
        _sample_memory(inference_samples.size()),
        _counting_memory(counting_samples.size()),
        _to_index_map(to_index_map),
        _to_string_map(to_string_map),
        _pattern_scores(10, pair_hasher)
    {
        // Initialize depth
        for(auto xy : to_string_map){
            _depths[xy.first] = 1;
        }
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
                if(x != std::prev(current_list->end())){
                    _sample_memory.at(i).add_pair({x, std::next(x)});
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
                if(x != std::prev(current_list->end())){
                    _counting_memory.at(i).add_pair({x, std::next(x)});
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
        decision_making::compute_pattern_counts(_sample_memory,
                                                _pattern_scores);
        decision_making::delete_zeros(_pattern_scores);
        decision_making::compute_pattern_divergence(_bare_lks,
                                                    _rules,
                                                    _pattern_scores);
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
            std::cout << left
                      << " "
                      << right
                      << ": " << xy.second << std::endl;
        }
    }

    void replace_best_patterns(int level){
        //std::cout << std::endl;
        int_pair_vect best_pairs;
        if(!_random){
            best_pairs = decision_making::pick_best_patterns(_pattern_scores,
                                                _n_select);
        }else{
            best_pairs = decision_making::pick_sto_patterns(_pattern_scores,
                                               _n_select,
                                               _T);
            _T *= (1.0 - _T_decay);
        }
        double tot_div = 0;
        for(auto xy : best_pairs){
            tot_div += _pattern_scores.at(xy);
        }
        _divergences.push_back(tot_div);
        int rule_index;
        int count;
        int length;
        for(const int_pair & xy: best_pairs){
            rule_index = - (_rules.size() + 1);
            std::string left;
            if(_to_string_map.count(xy.first) > 0){
                left = _to_string_map.at(xy.first);
            }else{
                left = std::to_string(xy.first);
            }
            std::string right;
            if(_to_string_map.count(xy.second) > 0){
                right = _to_string_map.at(xy.second);
            }else{
                right = std::to_string(xy.second);
            }
            _rules[rule_index] = xy;
            _levels[rule_index] = level;
            _to_hash_map[rule_index] = ">" + _to_hash_map[xy.first] + "-" +
                    _to_hash_map[xy.second] + "<";
            _depths[rule_index] = _depths[xy.first] + _depths[xy.second];
            _to_rule_index_map[_to_hash_map.at(rule_index)] = rule_index;
            _relative_counts.emplace(rule_index, int_double_map());
            _absolute_counts.emplace(rule_index, int_int_map());
            for(Mem_sandwich & mem : _sample_memory){
                mem.remove_pair(xy, rule_index);
            }
            for(int i = 0; i < _counting_memory.size(); ++i){
                Mem_sandwich & mem = _counting_memory.at(i);
                length = mem.get_n_symbols();
                count = mem.remove_pair(xy, rule_index);
                _relative_counts.at(rule_index)[i] = ((double) count) / ((double) length);
                _absolute_counts.at(rule_index)[i] = count;
            }
            _pattern_scores.erase(xy);
        }
        //std::cout << std::endl;
    }

    void run(){
        int current_iteration = 0;
        while(_rules.size() < _max_rules){
            next(++current_iteration);
        }
    }

    void next(int current_iteration){
        compute_pattern_scores();
        replace_best_patterns(current_iteration);
    }

    void print_counts() const{
        std::cout << std::endl;
        std::cout << "Printing counts" << std::endl;
        std::string left;
        std::string right;
        for(auto xy : _rules){
            if(xy.second.first >= 0){
                left = _to_string_map.at(xy.second.first);
            }else{
                left = std::to_string(xy.second.first);
            }
            if(xy.second.second >= 0){
                right = _to_string_map.at(xy.second.second);
            }else{
                right = std::to_string(xy.second.second);
            }
            std::cout << _to_hash_map.at(xy.first)
                      << " counts: " << std::endl;
            for(int i = 0; i < _counting_memory.size(); ++i){
                std::cout << "\t" << i << ": " << _absolute_counts.at(xy.first).at(i) << " "
                          << _relative_counts.at(xy.first).at(i) << std::endl;
            }std::cout << std::endl;
        }
    }

    string_vect_vect translate_inference_samples() const{
        string_vect_vect result(_inference_samples.size());
        for(int i = 0; i < _inference_samples.size(); ++i){
            for(elt x : _inference_samples.at(i)){
                result.at(i).push_back(_to_hash_map.at(x._content));
            }
        }
        return result;
    }

    string_vect_vect translate_counting_samples() const{
        string_vect_vect result(_counting_samples.size());
        for(int i = 0; i < _counting_samples.size(); ++i){
            for(elt x : _counting_samples.at(i)){
                result.at(i).push_back(_to_hash_map.at(x._content));
            }
        }
        return result;
    }

    void to_hashed_vectors(string_vect & hashcodes,
                           string_vect & rule_names,
                           string_pair_vect & hashed_rhs,
                           double_vect_vect & relative_counts,
                           int_vect_vect & absolute_counts,
                           int_vect & levels,
                           int_vect & depths,
                           double_vect & divergences) const{
        std::string left_hashcode;
        std::string right_hashcode;
        for(auto xy : _to_hash_map){
            if(xy.first >= 0) continue;
            hashcodes.push_back(xy.second);
            rule_names.push_back("r" + std::to_string(-xy.first) + "_");
            const int_pair & rhs = _rules.at(xy.first);
            left_hashcode = _to_hash_map.at(rhs.first);
            right_hashcode = _to_hash_map.at(rhs.second);
            hashed_rhs.push_back({left_hashcode, right_hashcode});
            relative_counts.emplace_back(double_vect());
            absolute_counts.emplace_back(int_vect());
            for(int i = 0; i < _counting_samples.size(); ++i){
                relative_counts.back().push_back(_relative_counts.at(xy.first).at(i));
                absolute_counts.back().push_back(_absolute_counts.at(xy.first).at(i));
            }
            levels.push_back(_levels.at(xy.first));
            depths.push_back(_depths.at(xy.first));
        }
        divergences = _divergences;
    }

};


#endif // PROBA_SEQUITUR_H
