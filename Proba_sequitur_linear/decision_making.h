#ifndef DECISION_MAKING_H
#define DECISION_MAKING_H

#include <vector>
#include <random>
#include <string>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "mem_sandwich.h"

typedef std::pair<int_pair, double>                     int_pair_double_pair;
typedef std::unordered_map<int_pair, double, pair_hash> int_pair_double_map;
typedef std::vector<int_pair>                           int_pair_vect;
typedef std::vector<int_pair_double_pair>               int_pair_double_pair_vect;
typedef std::unordered_set<int_pair, pair_hash>         int_pair_set;
typedef std::unordered_map<int, double>                 int_double_map;
typedef std::vector<Mem_sandwich>                       mem_vect;

typedef std::mt19937                                    RNG;

namespace decision_making{

inline static void compute_pattern_counts(const mem_vect & memory_vector,
                                          int_pair_double_map & counts){
    counts.clear();
    double total = 0;
    for(const Mem_sandwich & mem : memory_vector){
        for(auto xy : mem.get_first_maps()){
            counts[xy.first] += xy.second.size();
            total += xy.second.size();
        }
    }
    /*
    for(auto xy : counts){
        counts[xy.first] /= total;
    }
    */
}

inline static void compute_pattern_divergence(const int_double_map & bare_lks,
                                              int_pair_double_map & counts){
    int left;
    int right;
    double proba_pair;
    double proba_atoms;
    for(auto xy : counts){
        left = xy.first.first;
        right = xy.first.second;
        proba_pair = xy.second;
        proba_atoms = bare_lks.at(left) * bare_lks.at(right);
        counts[xy.first] = proba_pair * std::log(proba_pair / proba_atoms);
    }
}

inline static bool compare_scores(const int_pair_double_pair & x,
                                  const int_pair_double_pair & y){
    return (x.second > y.second);
}

inline static int_pair_vect pick_best_patterns(const int_pair_double_map & pattern_scores,
                                                int n_selected){
    int_pair_double_pair_vect items;
    for(auto xy : pattern_scores){
        items.emplace_back(int_pair_double_pair(xy.first, xy.second));
    }
    std::sort(items.begin(), items.end(), compare_scores);
    int n_to_take = std::min<int>(n_selected, items.size());
    int_pair_vect result(n_to_take);
    for(int i = 0; i < n_to_take; ++i){
        result[i] = items.at(i).first;
    }
    return result;
}

inline static int_pair_vect pick_sto_patterns(const int_pair_double_map & pattern_scores,
                                                int n_selected,
                                                const double & T,
                                                RNG & rng){

    if(pattern_scores.size() > n_selected){
        int_pair_set chosen_rules(0, pair_hasher);
        int_pair_double_pair_vect items;
        std::vector<double> scores;
        double total_score = 0;
        for(auto xy : pattern_scores){
            items.emplace_back(int_pair_double_pair(xy.first, xy.second));
            total_score += xy.second;
        }
        for(auto xy : pattern_scores){
            scores.push_back(std::exp(xy.second / (T*total_score)));
        }
        std::discrete_distribution<double> distrib(scores.begin(), scores.end());
        while(chosen_rules.size() < n_selected){
            chosen_rules.insert(items.at(distrib(rng)).first);
        }
        int_pair_vect result (chosen_rules.begin(), chosen_rules.end());
        return result;
    }else{
        int_pair_vect result;
        for(auto xy : pattern_scores){
            result.push_back(xy.first);
        }
        return result;
    }
}





}




#endif // DECISION_MAKING_H
