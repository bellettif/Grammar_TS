#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <iostream>
#include <string>
#include <regex>
#include <iterator>
#include <unordered_map>

#include <boost/algorithm/string.hpp>

typedef std::pair<std::string, double>              string_double_pair;
typedef std::pair<std::string, std::string>         string_pair;
typedef std::vector<string_pair>                    string_pair_vect;
typedef std::vector<string_double_pair>             string_double_pair_vect;
typedef std::vector<std::string>                    string_vect;
typedef std::unordered_map<std::string, double>     string_double_map;

namespace string_utils{

inline static void split(const std::string & input,
                         const std::string & delimiter,
                         std::vector<std::string> & split_result){
    boost::algorithm::split(split_result,
                            input,
                            boost::is_any_of(delimiter));
}

}


#endif // STRING_UTILS_H
