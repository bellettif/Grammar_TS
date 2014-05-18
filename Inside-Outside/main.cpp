#include <iostream>
#include <random>
#include <unordered_map>
#include <array>
#include <chrono>

#include "file_reader.h"
#include "in_out_proba.h"
#include "stochastic_rule.h"
#include "scfg.h"
#include "parse_tree.h"
#include "model_estimator.h"
#include "bare_estimator.h"

namespace std {
    std::string to_string(std::string x){
        return x;
    }
}

typedef std::string                             T;
typedef std::vector<T>                          T_vect;
typedef std::list<T>                            T_list;
typedef std::vector<T_vect>                     T_vect_vect;
typedef std::unordered_map<int, double>         int_double_hashmap;
typedef Stochastic_rule<T>                      SRule_T;
typedef SCFG<T>                                 SGrammar_T;
typedef std::vector<double>                     double_vect;
typedef std::pair<int, int>                     pair_i_i;
typedef std::vector<std::pair<int, int>>        pair_i_i_vect;
typedef std::mt19937                            RNG;
typedef std::pair<T, pair_i_i>                  derivation_result;
typedef In_out_proba<T>                         inside_T;
typedef Model_estimator<T>                      model_estim_T;
typedef Bare_estimator<T>                       bare_estim_T;

int main(){

    auto duration =  std::chrono::system_clock::now().time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    int first_option = 1100;
    int second_option = 11198;
    RNG                     my_rng(1100);

    std::pair<int, int>     A_pair_1 (1, 1);
    std::pair<int, int>     A_pair_2 (1, 2);
    std::pair<int, int>     A_pair_3 (2, 1);
    std::pair<int, int>     A_pair_4 (2, 2);
    double_vect             A_non_term_w ({0.2, 0.3, 0.1, 0.4});
    pair_i_i_vect           A_non_term_s ({A_pair_1,
                                           A_pair_2,
                                           A_pair_3,
                                           A_pair_4});
    T_vect                  A_term_s ({"Bernadette", "Colin", "Michel"});
    double_vect             A_term_w ({2.4, 0.1, 0.1});
    SRule_T                 A_rule(1,
                                    A_non_term_w,
                                    A_non_term_s,
                                    my_rng,
                                    A_term_w,
                                    A_term_s);
    //A_rule.print();

    std::pair<int, int>     B_pair_1 (2, 2);
    std::pair<int, int>     B_pair_2 (2, 1);
    std::pair<int, int>     B_pair_3 (1, 2);
    std::pair<int, int>     B_pair_4 (1, 1);
    double_vect             B_non_term_w ({0.2, 0.3, 0.4, 0.1});
    pair_i_i_vect           B_non_term_s ({B_pair_1,
                                           B_pair_2,
                                           B_pair_3,
                                           B_pair_4});
    T_vect                  B_term_s({"Pierre", "Mathieu"});
    double_vect             B_term_w({1.0, 0.2});
    SRule_T                 B_rule(2,
                                   B_non_term_w,
                                   B_non_term_s,
                                   my_rng,
                                   B_term_w,
                                   B_term_s);
    //B_rule.print();

    /*
    std::pair<int, int>     C_pair_1 (3, 3);
    std::pair<int, int>     C_pair_2 (2, 2);
    std::pair<int, int>     C_pair_3 (2, 1);
    double_vect             C_non_term_w ({0.3, 0.4, 0.4});
    pair_i_i_vect           C_non_term_s ({C_pair_1,
                                           C_pair_2,
                                           C_pair_3});
    T_vect                  C_term_s({"Pierre", "Bernadette", "Jeanne"});
    double_vect             C_term_w({0.1, 3.1, 0.2});
    SRule_T                 C_rule(3,
                                   C_non_term_w,
                                   C_non_term_s,
                                   my_rng,
                                   C_term_w,
                                   C_term_s);
    */

    //C_rule.print();

    std::pair<int, int>     S_pair_1 (1, 1);
    std::pair<int, int>     S_pair_2 (1, 2);
    std::pair<int, int>     S_pair_3 (2, 1);
    std::pair<int, int>     S_pair_4 (2, 2);
    double_vect             S_non_term_w ({0.6, 0.5, 0.4, 0.8});
    pair_i_i_vect           S_non_term_s ({S_pair_1,
                                           S_pair_2,
                                           S_pair_3,
                                           S_pair_4});

    SRule_T                 S_rule(3,
                                   S_non_term_w,
                                   S_non_term_s,
                                   my_rng);

    //S_rule.print();

    std::vector<SRule_T>    rules ({A_rule,
                                    B_rule,
                                    S_rule});

    A_rule.print();
    B_rule.print();
    //C_rule.print();
    S_rule.print();

    SGrammar_T              grammar(rules, 3);

    T_vect_vect inputs;
    T_list temp;
    for(int i = 0; i < 100000; ++i){
        temp = S_rule.complete_derivation(grammar);
        inputs.emplace_back(T_vect(temp.begin(),
                                   temp.end()));
    }

    model_estim_T model_estimator(grammar,
                                  inputs);

    model_estimator.estimate_from_inputs();

    model_estimator.print_estimates();

}
