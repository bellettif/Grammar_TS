#ifndef RULE_H
#define RULE_H


#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <string>
#include <random>
#include <deque>

typedef std::unordered_map<std::string, int>              string_int_map;
typedef std::vector<std::string>                          string_vect;
typedef std::vector<string_vect>                          string_vect_vect;
typedef std::discrete_distribution<>                      choice_distrib;
typedef std::mt19937                                      RNG;
typedef std::vector<int>::iterator                        vect_it;
typedef std::list<int>::iterator                          list_it;
typedef std::vector<double>                               double_vect;
typedef std::vector<int>                                  int_vect;


/*
 *  This class formalizes the notion of stochastic grammar rule in order
 *      to be able to generate sentences from a grammar defined by its
 *      non-terminal symbol and terminal symbol probability arrays.
 */
class Rule{

    /*
     *  Dimensions of the grammar
     */
    const int                       _N;
    const int                       _M;
    /*
     *  Weighted choice distributions used for several choices:
     *      - whether a terminal or a non terminal is going to be created
     *      - in case of non terminal, which one is going to be created
     *      - in case of terminal, which one is going to be created
     */
    choice_distrib                  _emission_choice;
    choice_distrib                  _non_term_choice;
    choice_distrib                  _term_choice;
    /*
     *  List of terminal symbols
     */
    const string_vect &             _terminals;


public:

    Rule(const int i,                       // Index of the rule
         const double * const A,            // Grammar parameters
         const double * const B,
         const int N,
         const int M,
         const string_vect & terminals):    // List of terminal symbols
        _N(N),
        _M(M),
        _terminals(terminals)
    {
        double total_emission_weight = 0;
        double total_non_emission_weight = 0;
        for(int j = 0; j < _M; ++j){
            total_emission_weight += B[i*_M + j];
        }
        total_non_emission_weight = 1.0 - total_emission_weight;
        _emission_choice = choice_distrib({total_non_emission_weight,
                                           total_emission_weight});
        _non_term_choice = choice_distrib(A + i*N*N, A + (i+1)*N*N);
        _term_choice = choice_distrib(B + i*M, B+ (i+1)*M);
    }

    /*
     *  Compule a derivation sample of the rule
     *      Emission return a bool = true if a terminal character
     *          is emitted, false is two non terminal characters are generated.
     *      If a terminal character is emited, the result is stored in term.
     *      If two non terminal characters are created, the results are stored in
     *          left and right.
     */
    void inline derivation(RNG & rng,
                           bool & emission,
                           int & term,
                           int & left,
                           int & right){
        emission = (_emission_choice(rng) == 1);
        if(emission){
            term = _term_choice(rng);
        }else{
            int non_term = _non_term_choice(rng);
            left = non_term / _N;
            right = non_term % _N;
        }
    }
};


#endif // RULE_H
