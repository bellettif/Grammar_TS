#ifndef STOCHASTIC_RULE_H
#define STOCHASTIC_RULE_H

#include <vector>
#include <random>
#include <array>

template<typename T>
class Stochastic_grammar;

template<typename T>
class Stochastic_rule{

typedef std::vector<double>                 double_vect;
typedef std::vector<std::pair<int, int>>    pair_i_i_vect;
typedef std::vector<T>                      T_vect;
typedef std::mt19937                        RNG;
typedef std::discrete_distribution<>        choice_distrib;

private:
    const int                           _rule_name;

    double_vect                         _non_term_w;
    pair_i_i_vect                       _non_term_s;
    double                              _non_term_totw;

    double_vect                         _term_w;
    T_vect                              _term_s;
    double                              _term_totw;

    RNG &                               _rng;

    choice_distrib                      _term_non_term_choice;
    choice_distrib                      _non_term_choice;
    choice_distrib                      _term_choice;


public:
    Stochastic_rule(const int &              rule_name,
                    const double_vect &      non_term_w,
                    const pair_i_i_vect &    non_term_s,
                    const double_vect &      term_w,
                    const T_vect &           term_s,
                    RNG &                    rng):
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
        for(const double & x : _non_term_w){
            _non_term_totw += x;
        }
        for(const double & x : _term_w){
            _term_totw += x;
        }
        double total_weight = _non_term_totw + _term_totw;
        _non_term_totw /= total_weight;
        _term_totw /= total_weight;
        for(int i = 0; i < _non_term_w.size(); ++i){
            _non_term_w[i] /= total_weight;
        }

        for(int i = 0; i < _term_w.size(); ++i){
            _term_w[i] /= total_weight;
        }
    }

    void print() const{
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


};



#endif // STOCHASTIC_RULE_H
