#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <iostream>
#include <string>
#include <regex>
#include <iterator>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <random>
#include <cstdlib>      // std::rand, std::srand

#include <boost/algorithm/string.hpp>

typedef std::mt19937        RNG;

namespace preprocessing{

static void inline r_shuffle(std::string & sentence,
                             RNG & rng){
    std::function<int(int)> my_rand = [&](int i){
        return rng() % i;
    };

    std::vector<std::string> split_sentence;
    boost::algorithm::split(split_sentence,
                            sentence,
                            boost::is_any_of(" "));
    std::random_shuffle(split_sentence.begin(), split_sentence.end(), my_rand);
    sentence = "";
    for(int i = 0; i < split_sentence.size(); ++i){
        sentence += split_sentence.at(i);
        if(i != split_sentence.size() - 1){
            sentence += " ";
        }
    }
}

static void inline r_mask(std::string & sentence,
                          RNG & rng,
                          double proba){
    std::vector<std::string> split_sentence;
    boost::algorithm::split(split_sentence,
                            sentence,
                            boost::is_any_of(" "));
    std::bernoulli_distribution distribution(1.0 - proba);
    sentence = "";
    for(int i = 0; i < split_sentence.size(); ++i){
        if(distribution(rng)){
            sentence += split_sentence.at(i);
            if(i != split_sentence.size() - 1){
                sentence += " ";
            }
        }
    }
}



}


#endif // PREPROCESSING_H
