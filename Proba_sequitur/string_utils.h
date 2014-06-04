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

inline static void split_rhs(const std::string & rhs,
                             std::string & left,
                             std::string & right){
    string_vect split_result;
    split(rhs,
          "-",
          split_result);
    left = split_result.at(0);
    right = split_result.at(1);
}

inline static int compute_n_words(const std::string & input,
                                   const std::string & delimiter = " "){
    string_vect split_result;
    boost::algorithm::split(split_result,
                            input,
                            boost::is_any_of(delimiter));
    return split_result.size();
}

inline static std::vector<int> compute_n_words(const string_vect & inputs,
                                   const std::string & delimiter = " "){
    std::vector<int> result;
    for(const std::string & input : inputs){
        result.push_back(compute_n_words(input,
                                         delimiter));
    }
    return result;
}

inline static int count(std::string & input,
                        const std::string & sub_string){
    std::regex e(sub_string);
    std::regex_iterator<std::string::iterator> rit (input.begin(),
                                                    input.end(),
                                                    e);
    std::regex_iterator<std::string::iterator> rend;
    int count = 0;
    while (rit!=rend) {
      ++count;
      ++rit;
    }
    return count;
}

inline static std::vector<double> count(string_vect & inputs,
                                        const std::string & sub_string){
    std::vector<double> result;
    for(std::string & input : inputs){
        result.push_back(count(input,
                               sub_string));
    }
    return result;
}

inline static std::vector<double> relative_count(string_vect & inputs,
                                                 const std::string & sub_string){
    std::vector<double> result;
    double current_count;
    double current_length;
    std::vector<std::string> split_result;
    for(std::string & input : inputs){
        current_count = count(input, sub_string);
        boost::algorithm::split(split_result,
                                input,
                                boost::is_any_of(" "));
        current_length = split_result.size();
        if (current_length > 0){
            result.push_back(current_count / current_length);
        }else{
            result.push_back(0);
        }
    }
    return result;
}

inline static std::vector<double> merged_relative_count(string_vect & inputs,
                                                        const std::string & sub_string){
    std::vector<double> result;
    double current_count;
    double total_length = 0;
    std::vector<std::string> split_result;
    for(std::string & input : inputs){
        boost::algorithm::split(split_result,
                                input,
                                boost::is_any_of(" "));
        total_length += split_result.size();
    }
    for(std::string & input : inputs){
        current_count = count(input, sub_string);
        if (total_length > 0){
            result.push_back(current_count / total_length);
        }else{
            result.push_back(0);
        }
    }
    return result;
}


inline static std::string replace(const std::string & input,
                                  const std::string & replaced,
                                  const std::string & replacement){
    std::regex e(replaced);
    return std::regex_replace(input, e, replacement);
}

inline static void replace(std::string & input,
                           const std::string & replaced,
                           const std::string & replacement){
    std::regex e(replaced);
    input = std::regex_replace(input, e, replacement);
}

inline static void replace(string_vect & inputs,
                           const std::string & replaced,
                           const std::string & replacement){
    for(std::string & input : inputs){
        replace(input,
                replaced,
                replacement);
    }
}

inline static void compute_individual_counts(const string_vect & candidates,
                                             string_vect & sequences,
                                             string_double_map & scores){
    double score;
    scores.clear();
    for(const std::string & x : candidates){
        score = 0;
        for(std::string & seq : sequences){
            score += count(seq, x);
        }
        if(score > 0) scores.emplace(x, score);
    }
}

inline static void compute_individual_counts(const string_vect & candidates,
                                             std::string & sequence,
                                             string_double_map & scores){
    double score;
    scores.clear();
    for(const std::string & x : candidates){
        score = count(sequence, x);
        if(score > 0) scores.emplace(x, score);
    }
}

inline static void compute_pair_counts(const string_vect & candidates,
                                       string_vect & sequences,
                                       string_double_map & scores){
    std::string current_pattern;
    double score;
    scores.clear();
    for(const std::string & x : candidates){
        for(const std::string & y : candidates){
            current_pattern = x + " " + y;
            score = 0;
            for(std::string & seq : sequences){
                score += count(seq, current_pattern);
            }
            if(score > 0) scores.emplace(x + "-" + y, score);
        }
    }
}

inline static void compute_pair_counts(const string_vect & candidates,
                                       std::string & sequence,
                                       string_double_map & scores){
    std::string current_pattern;
    double score;
    scores.clear();
    for(const std::string & x : candidates){
        for(const std::string & y : candidates){
            current_pattern = x + " " + y;
            score = count(sequence, current_pattern);
            if(score > 0) scores.emplace(x + "-" + y, score);
        }
    }
}

}


#endif // STRING_UTILS_H
