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
typedef In_out_proba<T>                         inside_T;

int main(){

    auto duration =  std::chrono::system_clock::now().time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    RNG                     my_rng(1100);

    std::pair<int, int>     A_pair_1 (1, 1);
    std::pair<int, int>     A_pair_2 (1, 2);
    std::pair<int, int>     A_pair_3 (3, 2);
    double_vect             A_non_term_w ({0.5, 0.3, 0.1});
    pair_i_i_vect           A_non_term_s ({A_pair_1,
                                           A_pair_2,
                                           A_pair_3});
    T_vect                  A_term_s ({"Bernadette", "Colin", "Michel"});
    double_vect             A_term_w ({0.6, 0.1, 0.1});
    SRule_T                 A_rule(1,
                                    A_non_term_w,
                                    A_non_term_s,
                                    my_rng,
                                    A_term_w,
                                    A_term_s);

    std::pair<int, int>     B_pair_1 (3, 1);
    std::pair<int, int>     B_pair_2 (2, 1);
    double_vect             B_non_term_w ({0.1, 0.7});
    pair_i_i_vect           B_non_term_s ({B_pair_1,
                                           B_pair_2});
    T_vect                  B_term_s({"Pierre", "Mathieu"});
    double_vect             B_term_w({0.3, 0.6});
    SRule_T                 B_rule(2,
                                   B_non_term_w,
                                   B_non_term_s,
                                   my_rng,
                                   B_term_w,
                                   B_term_s);

    std::pair<int, int>     C_pair_1 (3, 3);
    std::pair<int, int>     C_pair_2 (2, 2);
    std::pair<int, int>     C_pair_3 (2, 1);
    double_vect             C_non_term_w ({0.3, 0.4, 0.4});
    pair_i_i_vect           C_non_term_s ({C_pair_1,
                                           C_pair_2,
                                           C_pair_3});
    T_vect                  C_term_s({"Pierre", "Bernadette", "Jeanne"});
    double_vect             C_term_w({0.8, 0.4, 1.0});
    SRule_T                 C_rule(3,
                                   C_non_term_w,
                                   C_non_term_s,
                                   my_rng,
                                   C_term_w,
                                   C_term_s);

    std::pair<int, int>     S_pair_1 (1, 3);
    std::pair<int, int>     S_pair_2 (2, 3);
    double_vect             S_non_term_w ({0.5, 0.5});
    pair_i_i_vect           S_non_term_s ({S_pair_1,
                                           S_pair_2});

    SRule_T                 S_rule(4,
                                   S_non_term_w,
                                   S_non_term_s,
                                   my_rng);

    std::vector<SRule_T>    rules ({A_rule,
                                    B_rule,
                                    C_rule,
                                    S_rule});

    A_rule.print();
    B_rule.print();
    C_rule.print();
    S_rule.print();

    SGrammar_T              grammar(rules, 4);

    std::list<T> result = S_rule.complete_derivation(grammar);

    for(auto x : result){
        std::cout << x << " ";
    }std::cout << std::endl;

    T_vect input (result.begin(),
                  result.end());

    std::cout << "Length of input " << input.size() << std::endl;

    inside_T probaCmpter(grammar, input);

    probaCmpter.compute_inside_probas();
    probaCmpter.compute_outside_probas();

    probaCmpter.print_probas();

    std::cout << std::endl;

    double parse_proba;
    Parse_tree<T> parse_tree = probaCmpter.run_CYK(parse_proba);

    std::cout << parse_proba << std::endl;

}
