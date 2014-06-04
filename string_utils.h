#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <iostream>
#include <string>
#include <regex>
#include <iterator>

#include <boost/algorithm/string.hpp>

namespace string_utils{

static void inline split_string(const std::string & input,
                                const std::string & delimiter,
                                std::vector<std::string> split_result){
    boost::algorithm::split(split_result,
                            line,
                            boost::is_any_of(delimiter));
}


}


#endif // STRING_UTILS_H
