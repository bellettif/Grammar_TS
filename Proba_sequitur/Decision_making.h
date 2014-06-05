#ifndef DECISION_MAKING_H
#define DECISION_MAKING_H

#include <vector>
#include <random>
#include <string>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

typedef std::unordered_map<std::string, double>         string_double_map;
typedef std::vector<string_double_map>                  string_double_map_vect;
typedef std::vector<std::string>                        string_vect;
typedef std::pair<std::string, double>                  string_double_pair;
typedef std::vector<string_double_pair>                 string_double_pair_vect;
typedef std::unordered_set<std::string>                 string_set;

typedef std::mt19937                                    RNG;

namespace decision_making{

inline static bool compare_scores(const string_double_pair & x, const string_double_pair & y){
    return (x.second > y.second);
}

inline static string_vect pick_best_patterns(const string_double_map & pattern_scores,
                                             int n_selected){
    string_double_pair_vect items;
    for(auto xy : pattern_scores){
        items.emplace_back(string_double_pair(xy.first, xy.second));
    }
    std::sort(items.begin(), items.end(), compare_scores);
    int n_to_take = std::min<int>(n_selected, items.size());
    string_vect result(n_to_take);
    for(int i = 0; i < n_to_take; ++i){
        result[i] = items.at(i).first;
    }
    return result;
}


inline static string_vect pick_sto_patterns(const string_double_map & pattern_scores,
                                            int n_selected,
                                            const double & T,
                                            RNG & rng){

    if(pattern_scores.size() > n_selected){
        string_set chosen_rules;
        string_double_pair_vect items;
        std::vector<double> scores;
        double total_score = 0;
        for(auto xy : pattern_scores){
            items.emplace_back(string_double_pair(xy.first, xy.second));
            total_score += xy.second;
        }
        for(auto xy : pattern_scores){
            scores.push_back(std::exp(xy.second / (T*total_score)));
        }
        std::discrete_distribution<double> distrib(scores.begin(), scores.end());
        while(chosen_rules.size() < n_selected){
            chosen_rules.insert(items.at(distrib(rng)).first);
        }
        string_vect result (chosen_rules.begin(), chosen_rules.end());
        return result;
    }else{
        string_vect result;
        for(auto xy : pattern_scores){
            result.push_back(xy.first);
        }
        return result;
    }
}





}




#endif // DECISION_MAKING_H
