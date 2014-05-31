#include <iostream>
#include <random>
#include <unordered_map>
#include <array>
#include <chrono>

#include "file_reader.h"
#include "in_out_proba.h"
#include "flat_in_out.h"
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

    SRule_T                 S_rule(0,
                                   S_non_term_w,
                                   S_non_term_s,
                                   my_rng);

    std::vector<SRule_T>    rules ({A_rule,
                                    B_rule,
                                    S_rule});

    A_rule.print_rule();
    B_rule.print_rule();
    S_rule.print_rule();

    SGrammar_T              grammar(rules, 0);

    int n_samples = 100;

    T_vect_vect inputs;
    T_list temp;
    for(int i = 0; i < n_samples; ++i){
        temp = S_rule.complete_derivation(grammar);
        inputs.emplace_back(T_vect(temp.begin(),
                                   temp.end()));
    }

    int N = grammar.get_n_non_terms();
    int M = grammar.get_n_terms();

    double*** A = grammar.get_A();
    double** B = grammar.get_B();

    double* A_flat = new double[N*N*N];
    double* B_flat = new double[N*M];

    double*** A_converted = new double**[N];
    double** B_converted = new double*[N];

    std::unordered_map<int, int> index_to_non_term = grammar.get_index_to_non_term();

    int ii;
    int jj;
    int kk;
    for(int i = 0; i < N; ++i){
        A_converted[i] = new double*[N];
        ii = index_to_non_term[i];
        for(int j = 0; j < N; ++j){
            jj = index_to_non_term[j];
            A_converted[i][j] = new double[N];
            for(int k = 0; k < N; ++k){
                kk = index_to_non_term[k];
                A_converted[i][j][k] = A[ii][jj][kk];
            }
        }
        B_converted[i] = new double[M];
        for(int j = 0; j < N; ++j){
            B_converted[i][j] = B[ii][j];
        }
    }


    array_utils::flatten_params(A_converted, B_converted, N, M, A_flat, B_flat);

    T_vect terminals = grammar.get_index_to_term_vect();

    Model_estimator model_estim (grammar,
                                 inputs);

    Flat_in_out fao (A_flat, B_flat, N, M, terminals);

    std::cout << "Fao created" << std::endl;

    model_estim.estimate_from_inputs();

    double * sample_probas = new double[inputs.size()];
    double * new_A = new double[N*N*N];
    double * new_B = new double[N*M];

    std::cout << "Estimating" << std::endl;

    fao.estimate_A_B(inputs,
                     sample_probas,
                     new_A, new_B);

    std::cout << "Estimation done" << std::endl;

    std::cout << "A estimates" << std::endl;
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k){
                std::cout << "i = " << i << ", j = " << j << ", k = " << k
                          << ", A = " << A[i][j][k]
                          << ", A flat = " << new_A[i*N*N + j*N + k] << std::endl;
            }
        }
    }
    std::cout << "B estimates" << std::endl;
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            std::cout << "i = " << i << ", j = " << j
                      << ", B = " << B[i][j]
                      << ", B flat = " << new_B[i*N + j] << std::endl;
        }
    }

    delete[] new_B;
    delete[] new_A;
    delete[] sample_probas;
    delete[] B;
    delete[] A;
}
