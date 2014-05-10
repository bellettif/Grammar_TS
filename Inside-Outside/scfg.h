#ifndef SCFG_H
#define SCFG_H

#include<vector>
#include<list>
#include<unordered_map>
#include<random>

#include "stochastic_rule.h"

template<typename T>
class SCFG{

typedef std::vector<double>                                 double_vect;
typedef std::discrete_distribution                          choice_dist;
typedef std::vector<int>                                    int_vect;
typedef std::pair<int_vect, choice_dist>                    choice_gen;
typedef std::pair<choice_gen, choice_gen>                   complete_choice_gen;
typedef std::unordered_map<int, complete_choice_gen>        int_choice_gen_hashmap;
typedef std::unordered_map<int,
                int_choice_gen_hashmap>                     int_int_choice_gen_hashmap;
typedef std::vector<T>                                      T_vect;

private:
    const   int_int_choice_gen_hashmap          A; // Non term generation table
    const   int_choice_gen_hashmap              B;



};


#endif // SCFG_H
