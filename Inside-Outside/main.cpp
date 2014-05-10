#include <iostream>
#include <random>
#include <unordered_map>
#include <array>
#include <chrono>

#include "file_reader.h"
#include "inside_proba.h"
#include "stochastic_rule.h"
#include "scfg.h"

namespace std {
    std::string to_string(std::string x){
        return x;
    }
}

typedef std::string                             T;
typedef std::vector<T>                          T_vect;
typedef std::unordered_map<int, double>         int_double_hashmap;
typedef Stochastic_rule<T>                      SRule_T;
typedef SCFG<T>                                 SGrammar_T;
typedef std::vector<double>                     double_vect;
typedef std::pair<int, int>                     pair_i_i;
typedef std::vector<std::pair<int, int>>        pair_i_i_vect;
typedef std::mt19937                            RNG;
typedef std::pair<T, pair_i_i>                  derivation_result;

int main(){

    auto duration =  std::chrono::system_clock::now().time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    RNG                     my_rng(millis);

    std::pair<int, int>     A_pair_1 (1, 1);
    std::pair<int, int>     A_pair_2 (1, 2);
    std::pair<int, int>     A_pair_3 (3, 2);
    double_vect             A_non_term_w ({0.3, 0.4, 0.5});
    pair_i_i_vect           A_non_term_s ({A_pair_1,
                                           A_pair_2,
                                           A_pair_3});
    T_vect                  A_term_s ({"Bernadette", "Colin", "Michel"});
    double_vect             A_term_w ({0.6, 0.1, 0.1});
    SRule_T                 A_rule(1,
                                    A_non_term_w,
                                    A_non_term_s,
                                    A_term_w,
                                    A_term_s,
                                    my_rng);

    std::pair<int, int>     B_pair_1 (3, 1);
    std::pair<int, int>     B_pair_2 (2, 1);
    double_vect             B_non_term_w ({0.7, 0.8});
    pair_i_i_vect           B_non_term_s ({B_pair_1,
                                           B_pair_2});
    T_vect                  B_term_s({"Pierre", "Mathieu"});
    double_vect             B_term_w({0.3, 0.6});
    SRule_T                 B_rule(2,
                                   B_non_term_w,
                                   B_non_term_s,
                                   B_term_w,
                                   B_term_s,
                                   my_rng);

    std::pair<int, int>     C_pair_1 (3, 3);
    std::pair<int, int>     C_pair_2 (2, 2);
    std::pair<int, int>     C_pair_3 (2, 1);
    double_vect             C_non_term_w ({0.7, 0.8, 0.2});
    pair_i_i_vect           C_non_term_s ({C_pair_1,
                                           C_pair_2,
                                           C_pair_3});
    T_vect                  C_term_s({"Pierre", "Bernadette", "Jeanne"});
    double_vect             C_term_w({0.2, 0.2, 1.0});
    SRule_T                 C_rule(3,
                                   C_non_term_w,
                                   C_non_term_s,
                                   C_term_w,
                                   C_term_s,
                                   my_rng);


    std::vector<SRule_T>    rules ({A_rule, B_rule, C_rule});

    A_rule.print();
    B_rule.print();
    C_rule.print();

    SGrammar_T              grammar(rules);

    grammar.print_symbols();
    grammar.print_params();


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
