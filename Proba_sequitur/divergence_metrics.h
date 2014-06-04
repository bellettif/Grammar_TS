#ifndef DIVERGENCE_METRICS_H
#define DIVERGENCE_METRICS_H

#include <unordered_map>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

typedef std::unordered_map<std::string, double>         string_double_map;
typedef std::vector<string_double_map>                  string_double_map_vect;
typedef std::vector<std::string>                        string_vect;

namespace divergence_metrics{

static string_double_map inline to_probas(const string_double_map & counts){
    double total = 0;
    string_double_map result (counts.size());
    for(auto xy : counts){
        total += xy.second;
    }
    for(auto xy : counts){
        result[xy.first] = xy.second / total;
    }
    return result;
}

static string_double_map inline compute_divergence(const string_double_map & individual_probas,
                                                   const string_double_map & pair_probas,
                                                   bool atomic_bare_lk = false){
    string_double_map divergences (pair_probas.size());
    string_vect parse_result;
    for(auto xy : pair_probas){
        boost::algorithm::split(parse_result,
                                xy.first,
                                boost::is_any_of("-"));
        divergences[xy.first] = xy.second
                *
                std::log(xy.second /
                         (
                          individual_probas.at(parse_result.at(0))
                          *
                          individual_probas.at(parse_result.at(1))
                          )
                         );
    }
    return divergences;
}



}



#endif // DIVERGENCE_METRICS_H
