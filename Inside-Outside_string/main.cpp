#include <iostream>
#include <random>
#include <unordered_map>
#include <array>
#include <chrono>

#include "file_reader.h"
#include "in_out_proba.h"
#include "raw_in_out.h"
#include "stochastic_rule.h"
#include "scfg.h"
#include "parse_tree.h"
#include "model_estimator.h"
#include "bare_estimator.h"
#include "array_utils.h"

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
typedef Stochastic_rule                         SRule_T;
typedef SCFG                                    SGrammar_T;
typedef std::vector<double>                     double_vect;
typedef std::pair<int, int>                     pair_i_i;
typedef std::vector<std::pair<int, int>>        pair_i_i_vect;
typedef std::mt19937                            RNG;
typedef std::pair<T, pair_i_i>                  derivation_result;
typedef In_out_proba                            inside_T;
typedef Model_estimator                         model_estim_T;
typedef Bare_estimator                          bare_estim_T;






int main(){

    int N = 5;
    int M = 10;
    double *** A;
    double **B;

    std::vector<std::string> terminal_chars = {"Bernard",
                                               "Jacques",
                                               "Jean",
                                               "Mathieu",
                                               "George",
                                               "Michel",
                                               "Bernadette",
                                               "Henriette",
                                               "Jeanne",
                                               "Elise"};


    auto duration =  std::chrono::system_clock::now().time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    RNG my_rng(millis);

    array_utils::allocate_arrays(A, B, N, M);

    std::cout << "Arrays allocated" << std::endl;

    array_utils::fill_arrays_with_random(A, B, N, M, my_rng);

    std::cout << "Arrays filled" << std::endl;

    array_utils::print_array_content(A, B, N, M);

    array_utils::deallocate_arrays(A, B, N, M);





    /*
    auto duration =  std::chrono::system_clock::now().time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    int first_option = 1100;
    int second_option = 11198;
    RNG                     my_rng(millis);

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

    std::vector<SRule_T>    rules ({A_rule,
                                    B_rule,
                                    S_rule});

    A_rule.print_rule();
    B_rule.print_rule();
    S_rule.print_rule();

    SGrammar_T              grammar(rules, 3);

    T_vect_vect inputs;
    T_list temp;
    for(int i = 0; i < 1000; ++i){
        temp = S_rule.complete_derivation(grammar);
        inputs.emplace_back(T_vect(temp.begin(),
                                   temp.end()));
    }

    int N = grammar.get_n_non_terms();
    int M = grammar.get_n_terms();

    double*** A = grammar.get_A();
    double** B = grammar.get_B();

    std::cout << "Perturbated parameters" << std::endl;
    grammar.print_params();

    model_estim_T model_estimator(grammar,
                                  inputs);

    std::vector<std::string> sentence = {"Bernadette",
                                         "Bernadette",
                                         "Mathieu",
                                         "Mathieu",
                                         "Bernadette",
                                         "Mathieu",
                                         "Mathieu"};

    In_out_proba in_out_proba(grammar,
                              sentence,
                              A,
                              B);

    double*** E;
    double*** F;
    int M_;
    int N_;
    int length;

    in_out_proba.get_inside_outside(E,
                                    F,
                                    N_,
                                    M_,
                                    length);

    for(int i = 0; i < N_; ++i){
        std::cout << "Non terminal character " << i << std::endl;
        for(int s = 0; s < length; ++s){
            std::cout << "\t";
            for(int r = 0; r < length; ++r){
                if(E[i][s][r] == 0){
                    std::cout << "0.0000000 ";
                }
                std::cout << E[i][s][r] << " ";
            }std::cout << std::endl;
        }
    }
    */

}
