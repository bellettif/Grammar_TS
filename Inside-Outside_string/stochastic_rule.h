#ifndef STOCHASTIC_RULE_H
#define STOCHASTIC_RULE_H

#include <vector>
#include <list>
#include <random>
#include <array>
#include <iostream>
#include <string>

typedef std::string T;
typedef std::vector<float>                  float_vect;
typedef std::pair<int, int>                 pair_i_i;
typedef std::vector<std::pair<int, int>>    pair_i_i_vect;
typedef std::vector<T>                      T_vect;
typedef std::mt19937                        RNG;
typedef std::discrete_distribution<>        choice_distrib;
typedef std::pair<T, pair_i_i>              derivation_result;

class SCFG;

class Stochastic_rule{



private:
    const int                           _rule_name;

    float_vect                         _non_term_w;
    pair_i_i_vect                       _non_term_s;
    float                              _non_term_totw;

    float_vect                         _term_w;
    T_vect                              _term_s;
    float                              _term_totw;

    RNG &                               _rng;

    choice_distrib                      _non_term_term_choice;
    choice_distrib                      _non_term_choice;
    choice_distrib                      _term_choice;


public:
    Stochastic_rule(const int &              rule_name,
                    const float_vect &      non_term_w,
                    const pair_i_i_vect &    non_term_s,
                    RNG &                    rng,
                    const float_vect &      term_w = float_vect(),
                    const T_vect &           term_s = T_vect()):
        _rule_name(rule_name),
        _non_term_w(non_term_w),
        _non_term_s(non_term_s),
        _non_term_totw(0),
        _term_w(term_w),
        _term_s(term_s),
        _term_totw(0),
        _rng(rng),
        _non_term_choice(_non_term_w.begin(), _non_term_w.end()),
        _term_choice(_term_w.begin(), _term_w.end()){
        for(const float & x : _non_term_w){
            _non_term_totw += x;
        }
        for(const float & x : _term_w){
            _term_totw += x;
        }
        float total_weight = _non_term_totw + _term_totw;
        _non_term_term_choice = choice_distrib({_non_term_totw, _term_totw});
        _non_term_totw /= total_weight;
        _term_totw /= total_weight;
        for(int i = 0; i < _non_term_w.size(); ++i){
            _non_term_w[i] /= total_weight;
        }
        for(int i = 0; i < _term_w.size(); ++i){
            _term_w[i] /= total_weight;
        }
    }

    void print_rule() const{
        std::cout << "Characteristics of rule: " << _rule_name << std::endl;
        std::cout << "Non terminal weights: (total = " << _non_term_totw << ")" << std::endl;
        for(int i = 0; i < _non_term_w.size(); ++i){
            std::cout << "\t" << _rule_name << " (" << _non_term_w[i] << ") -> "
                      << _non_term_s[i].first
                      << " " << _non_term_s[i].second << std::endl;
        }std::cout << std::endl;
        std::cout << "Terminal weights: (total = " << _term_totw << ")" << std::endl;
        for(int i = 0; i < _term_w.size(); ++i){
            std::cout << "\t" << _rule_name << " (" << _term_w[i] << ") -> "
                      << _term_s[i] << std::endl;
        }std::cout << std::endl;
    }

    derivation_result derive(bool & terminal_emission){
        int term_or_non_term = _non_term_term_choice(_rng);
        derivation_result result;
        if(term_or_non_term == 0){
            terminal_emission = false;
            pair_i_i temp = _non_term_s[_non_term_choice(_rng)];
            result.second.first = temp.first;
            result.second.second = temp.second;
            return result;
        }else{
            terminal_emission = true;
            result.first = _term_s[_term_choice(_rng)];
            return result;
        }
    }

    std::list<T> complete_derivation(SCFG & grammar);

    int get_name() const{
        return _rule_name;
    }

    const float_vect & get_non_term_w() const{
        return _non_term_w;
    }

    const pair_i_i_vect & get_non_term_s() const{
        return _non_term_s;
    }

    const float_vect & get_term_w() const{
        return _term_w;
    }

    const T_vect & get_term_s() const{
        return _term_s;
    }

};



#endif // STOCHASTIC_RULE_H
