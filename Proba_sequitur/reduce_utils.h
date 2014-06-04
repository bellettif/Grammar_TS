#ifndef REDUCE_UTILS_H
#define REDUCE_UTILS_H

#include <unordered_map>
#include <string>
#include <vector>

typedef std::unordered_map<std::string, double>         string_double_map;
typedef std::vector<string_double_map>                  string_double_map_vect;

namespace reduce_utils{


static void inline reduce_scores(const string_double_map_vect & dicts,
                                 string_double_map & reduced){
    reduced.clear();
    for(const string_double_map & dict : dicts){
        for(auto xy : dict){
            reduced[xy.first] += xy.second;
        }
    }
}




}



#endif // REDUCE_UTILS_H
