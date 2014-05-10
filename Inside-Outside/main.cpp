#include <iostream>
#include <random>
#include <unordered_map>
#include <array>

#include "file_reader.h"
#include "inside_proba.h"
#include "stochastic_rule.h"

typedef std::string                             T;
typedef std::vector<T>                          T_vect;
typedef std::unordered_map<int, double>         int_double_hashmap;
typedef Stochastic_rule<T>                      SRule_T;
typedef std::vector<double>                     double_vect;
typedef std::vector<std::pair<int, int>>        pair_i_i_vect;
typedef std::mt19937                            RNG;

int main(){

    RNG                     my_rng;
    std::pair<int, int>     pair_1 (1, 1);
    std::pair<int, int>     pair_2 (1, 2);
    std::pair<int, int>     pair_3 (3, 4);
    double_vect             non_term_w ({0.3, 0.4, 0.5});
    pair_i_i_vect           non_term_s ({pair_1, pair_2, pair_3});

    T_vect                  term_s ({"bertrand", "jeannette", "henri"});
    double_vect             term_w ({0.6, 0.1, 0.1});

    SRule_T                 my_rule(0,
                                    non_term_w,
                                    non_term_s,
                                    term_w,
                                    term_s,
                                    my_rng);

    my_rule.print();

    /*
    std::string file_path = "/Users/francois/Grammar_TS/Sequitur/data/achuSeq_8.csv";

    int * raw_input;
    int length;

    read_csv(file_path, raw_input, length);

    std::cout << "N elements: " << length << std::endl;

    T_vect input(raw_input, raw_input + length);

    for(auto x : input){
        std::cout << x << " ";
    }std::cout << std::endl;

    return 0;
    */

    /*
    std::mt19937 rand_gen;
    const int n_sides = 6;
    std::array<double, n_sides> probas = {0.1, 0.1, 1.0, 0.1, 0.1, 0.1};
    double sum = 0;
    for(auto x : probas){
        sum += x;
    }

    int_double_hashmap exp_results;
    for(int i =0; i < n_sides; ++i){
        exp_results[i] = 0;
    }

    std::discrete_distribution<> distrib(probas.begin(), probas.end());

    for(int i = 0; i < 10000; ++i){
        ++ exp_results[distrib(rand_gen)];
    }

    for(auto xy : exp_results){
        std::cout << "Key: " << xy.first << ", count: " << xy.second << std::endl;
    }
    */
}
