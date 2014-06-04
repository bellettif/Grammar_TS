#ifndef PROBA_SEQUITUR_H
#define PROBA_SEQUITUR_H

#include <unordered_map>
#include <string>
#include <vector>
#include <random>

#include <boost/algorithm/string.hpp>

#include "string_utils.h"
#include "reduce_utils.h"
#include "divergence_metrics.h"
#include "preprocessing.h"
#include "decision_making.h"

typedef std::mt19937                                            RNG;
typedef std::unordered_map<std::string, std::string>            string_string_map;
typedef std::unordered_map<std::string, int>                    string_int_map;
typedef std::unordered_map<std::string, double>                 string_double_map;
typedef std::vector<std::string>                                string_vect;
typedef std::unordered_map<std::string, std::vector<double>>    string_double_vect_map;
typedef std::vector<double>                                     double_vect;
typedef std::vector<std::vector<int>>                           int_vect_vect;
typedef std::unordered_set<std::string>                         string_set;

class Proba_sequitur{

private:
    const string_vect &         _sequences;
    const string_vect &         _sequences_for_counts;
    string_vect                 _terminal_parses;
    string_vect                 _terminal_parses_for_counts;

    string_string_map           _rules;
    string_string_map           _rule_to_hashcode;

    string_int_map              _rule_to_level;
    string_int_map              _counts;
    string_double_vect_map      _relative_counts;
    string_double_map           _cum_relative_counts;
    string_double_map           _rule_divs;

    double_vect                 _div_levels;
    int_vect_vect               _lengths;

    const int                   _degree;
    const int                   _max_rules;
    const bool                  _atomic_bare_lk;
    const bool                  _stochastic;
    RNG *                       _rng;
    const double                _init_T                                 = 0;
    const double                _T                                      = 0;
    const double                _T_decrease_rate                        = 0;

    string_double_map           _bare_lk_table;

    string_vect                 _term_chars;


public:
    Proba_sequitur(const string_vect & sequences,
                   const string_vect & sequences_for_counts,
                   int degree,
                   int max_rules,
                   bool atomic_bare_lk,
                   bool stochastic,
                   RNG * rng = 0,
                   const double & init_T = 0,
                   const double & T_decrease_rate = 0):
        _sequences(sequences),
        _sequences_for_counts(sequences_for_counts),
        _degree(degree),
        _max_rules(max_rules),
        _atomic_bare_lk(atomic_bare_lk),
        _stochastic(stochastic),
        _rng(rng),
        _init_T(init_T),
        _T_decrease_rate(T_decrease_rate)
    {
        string_vect split_seq;
        string_set temp_terms;
        for(const std::string & sequence : _sequences){
            boost::algorithm::split(split_seq,
                                    sequence,
                                    boost::is_any_of(" "));
            temp_terms.insert()
        }

    }

    void reset(){
        _terminal_parses.clear();
        _terminal_parses_for_counts.clear();
        _rules.clear();
        _rule_to_hashcode.clear();
        _rule_to_level.clear();
        _counts.clear();
        _relative_counts.clear();
        _cum_relative_counts.clear();
        _rule_divs.clear();
        _div_levels.clear();
        _lengths.clear();
        _bare_lk_table.clear();
        _T = _init_T;
    }

    void run(){
        _terminal


        string_vect inference_sequences (sequences);
        string_vect count_sequences (sequences_for_counts);
    }


};


#endif // PROBA_SEQUITUR_H
