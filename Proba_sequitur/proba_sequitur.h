#ifndef PROBA_SEQUITUR_H
#define PROBA_SEQUITUR_H

#include <unordered_map>
#include <string>
#include <vector>
#include <random>
#include <iostream>

#include <boost/algorithm/string.hpp>

#include "string_utils.h"
#include "reduce_utils.h"
#include "divergence_metrics.h"
#include "preprocessing.h"
#include "decision_making.h"
#include "print_tools.h"

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
    string_vect                 _sequences;
    string_vect                 _sequences_for_counts;
    string_vect                 _terminal_parses;
    string_vect                 _terminal_parses_for_counts;

    string_string_map           _rules;
    string_string_map           _rule_to_hashcode;

    string_int_map              _rule_to_level;
    string_int_map              _counts;
    string_double_vect_map      _relative_counts;
    string_double_map           _merged_relative_counts;
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
    double                      _T                                      = 0;
    const double                _T_decrease_rate                        = 0;

    int                         _current_iter                           = 0;

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
        _sequences(sequences.begin(), sequences.end()),
        _sequences_for_counts(sequences_for_counts.begin(), sequences_for_counts.end()),
        _degree(degree),
        _max_rules(max_rules),
        _atomic_bare_lk(atomic_bare_lk),
        _stochastic(stochastic),
        _rng(rng),
        _init_T(init_T),
        _T_decrease_rate(T_decrease_rate)
    {}

    inline void init_bare_lk(){
        string_vect split_seq;
        string_set temp_terms;
        int n_symbols = 0;
        for(const std::string & sequence : _sequences){
            boost::algorithm::split(split_seq,
                                    sequence,
                                    boost::is_any_of(" "));
            for(const std::string & x : split_seq){
                temp_terms.insert(x);
            }
            n_symbols += split_seq.size();
        }
        for(const std::string & x : temp_terms){
            if (x == "") continue;
            _term_chars.push_back(x);
        }
        string_utils::compute_individual_counts(_term_chars,
                                                _sequences,
                                                _bare_lk_table);
        divergence_metrics::to_probas_inplace(_bare_lk_table);
    }

    inline void extract_candidates(string_vect & candidates){
        string_vect split_seq;
        string_set temp_terms;
        candidates.clear();
        for(const std::string & sequence : _sequences){
            boost::algorithm::split(split_seq,
                                    sequence,
                                    boost::is_any_of(" "));
            for(const std::string & x : split_seq){
                temp_terms.insert(x);
            }
        }
        for(const std::string & x : temp_terms){
            if (x == "") continue;
            candidates.push_back(x);
        }
    }

    inline std::string compute_hash_code(const std::string & rhs){
        std::string left;
        std::string right;
        string_utils::split_rhs(rhs, left, right);
        std::string left_part;
        if (_rule_to_hashcode.count(left) != 0){
            left_part = _rule_to_hashcode[left];
            string_utils::replace(left_part,
                              "\-",
                              "_");
        }else{
            left_part = left; // terminal
        }
        std::string right_part;
        if (_rule_to_hashcode.count(right) != 0){
            right_part = _rule_to_hashcode[right];
            string_utils::replace(right_part,
                              "\-",
                              "_");
        }else{
            right_part = right;
        }
        return ">" + left_part + "-" + right_part + "<";
    }

    inline void reset(){
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
        _current_iter = 0;
    }

    inline void run(){
        reset();
        init_bare_lk();
        string_double_map       current_divs;
        string_double_map       pair_counts;
        string_double_map       pair_probas;
        string_vect             candidates;
        string_vect             best_patterns;
        bool no_more_parsing = false;
        int next_rule_index;
        std::string lhs;
        std::string rhs_pattern;
        double tot_div;
        std::string left;
        std::string right;
        while((! no_more_parsing) && (_rules.size() < _max_rules)){
            _current_iter ++;
            if(_current_iter > 10) return;
            std::cout << "Iteration " << _current_iter << " " <<
                         _rules.size() << " rules" << std::endl;
            extract_candidates(candidates);
            string_utils::compute_pair_counts(candidates,
                                              _sequences,
                                              pair_counts);
            pair_probas = divergence_metrics::to_probas(pair_counts);
            current_divs = divergence_metrics::compute_divergence(_bare_lk_table,
                                                                  pair_probas);
            if(!_stochastic){
                best_patterns = decision_making::pick_best_patterns(current_divs,
                                                                    _degree);
            }else{
                best_patterns = decision_making::pick_sto_patterns(current_divs,
                                                                   _degree,
                                                                   _T,
                                                                   *_rng);
                _T *= (1.0 - _T_decrease_rate);
            }
            tot_div = 0;
            std::cout << "\tBest patterns" << std::endl;
            for(const std::string & rhs : best_patterns){
                rhs_pattern = string_utils::replace(rhs, "-", " ");
                next_rule_index = _rules.size() + 1;
                lhs = "r" + std::to_string(next_rule_index) + "_";
                _rules[lhs] = rhs;
                _rule_divs[lhs] = current_divs.at(rhs);
                tot_div += current_divs.at(rhs);
                _rule_to_hashcode[lhs] = compute_hash_code(rhs);
                _rule_to_level[lhs] = _current_iter;
                _counts[lhs] = pair_counts.at(rhs);
                _relative_counts[lhs] = string_utils::relative_count(_sequences_for_counts,
                                                                     rhs_pattern);
                string_utils::split_rhs(rhs,
                                        left,
                                        right);
                if(_atomic_bare_lk){
                    _bare_lk_table[lhs] = _bare_lk_table[left] * _bare_lk_table[right];
                }else{
                    _merged_relative_counts[left] = string_utils::merged_relative_count(_sequences_for_counts,
                                                                                        rhs_pattern);
                    _merged_relative_counts[right] = string_utils::merged_relative_count(_sequences_for_counts,
                                                                                         rhs_pattern);
                    _bare_lk_table[lhs] = _merged_relative_counts[left] * _merged_relative_counts[right];
                }
                std::cout << "\t\t" << rhs << " " << current_divs.at(rhs)
                          << " bare_lk " << _bare_lk_table.at(lhs) << std::endl;
                string_utils::replace(_sequences,
                                      rhs_pattern,
                                      lhs);
                string_utils::replace(_sequences_for_counts,
                                      rhs_pattern,
                                      lhs);
            }
            std::cout << tot_div << std::endl;
            _lengths.push_back(string_utils::compute_n_words(_sequences_for_counts));
            _div_levels.push_back(tot_div);
            std::cout << "Done" << std::endl;
        }
    }

    inline void print_state(){
        std::cout << std::endl;
        std::cout << "Printing state of proba sequitur" << std::endl;
        print_sequences();
        print_sequences_for_counts();
        print_terminal_parses();
        print_terminal_parses_for_counts();
        print_rules();
        print_term_chars();
        print_bare_lk();
        print_rule_to_hashcode();
        print_rule_to_level();
        print_counts();
        print_relative_counts();
        print_cumulated_relative_counts();
        print_rule_divs();
        print_div_levels();
        std::cout << std::endl;
    }

    inline void print_sequences(){
        std::cout << "\tSequences:" << std::endl;
        print_tools::print_vect<std::string>(_sequences, "\t\t");
    }

    inline void print_sequences_for_counts(){
        std::cout << "\tSequences for counts:" << std::endl;
        print_tools::print_vect<std::string>(_sequences_for_counts, "\t\t");
    }

    inline void print_terminal_parses(){
        std::cout << "\tTerminal parses:" << std::endl;
        print_tools::print_vect<std::string>(_terminal_parses, "\t\t");
    }

    inline void print_terminal_parses_for_counts(){
        std::cout << "\tTerminal parses for counts:" << std::endl;
        print_tools::print_vect<std::string>(_terminal_parses_for_counts, "\t\t");
    }

    inline void print_rules(){
        std::cout << "\tRules:" << std::endl;
        print_tools::print_map<std::string, std::string>(_rules, "\t\t");
    }

    inline void print_term_chars(){
        std::cout << "\tTerm chars:" << std::endl;
        print_tools::print_vect<std::string>(_term_chars, "\t\t");
    }

    inline void print_rule_to_hashcode(){
        std::cout << "\tRule to hashcode:" << std::endl;
        print_tools::print_map<std::string, std::string>(_rule_to_hashcode, "\t\t");
    }

    inline void print_rule_to_level(){
        std::cout << "\tRule to level:" << std::endl;
        print_tools::print_map<std::string, int>(_rule_to_level, "\t\t");
    }

    inline void print_counts(){
        std::cout << "\tCounts:" << std::endl;
        print_tools::print_map<std::string, int>(_counts, "\t\t");
    }

    inline void print_relative_counts(){
        std::cout << "\tRelative counts:" << std::endl;
        print_tools::print_map<std::string, double>(_relative_counts, "\t\t");
    }

    inline void print_cumulated_relative_counts(){
        std::cout << "\tCumulated relative counts:" << std::endl;
        print_tools::print_map<std::string, double>(_cum_relative_counts, "\t\t");
    }

    inline void print_rule_divs(){
        std::cout << "\tRule divs:" << std::endl;
        print_tools::print_map<std::string, double>(_rule_divs, "\t\t");
    }

    inline void print_div_levels(){
        std::cout << "\tDiv levels:" << std::endl;
        print_tools::print_vect<double>(_div_levels, "\t\t");
    }

    inline void print_bare_lk(){
        std::cout << "\tBare likelihood table:" << std::endl;
        print_tools::print_map<std::string, double>(_bare_lk_table, "\t\t");
    }

};


#endif // PROBA_SEQUITUR_H
