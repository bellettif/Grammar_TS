#ifndef SCFG_H
#define SCFG_H

#include<vector>
#include<list>
#include<unordered_map>
#include<unordered_set>
#include<random>
#include<tuple>

#include "stochastic_rule.h"

template<typename T>
class SCFG{

typedef Stochastic_rule<T>                                  rule_T;
typedef std::vector<double>                                 double_vect;
typedef std::unordered_map<int, rule_T>                     int_rule_hashmap;
typedef std::tuple<int, int, int>                           non_term_tuple;
typedef std::unordered_set<non_term_tuple>                  non_term_tuple_set;
typedef std::pair<int, int>                                 pair_i_i;
typedef std::unordered_set<T>                               T_set;
typedef std::unordered_set<int>                             int_set;
typedef std::unordered_map<int, int>                        int_int_map;
typedef std::unordered_map<int, T>                          int_T_map;
typedef std::unordered_map<T, int>                          T_int_map;
typedef std::vector<T>                                      T_vect;
typedef std::pair<T, pair_i_i>                              derivation_result;

private:
    int_rule_hashmap                    _grammar;
    int                                 _root_symbol;
    int                                 _n_non_terms;
    int                                 _n_terms;
    int_T_map                           _index_to_term;
    T_int_map                           _term_to_index;
    int_int_map                         _index_to_non_term;
    int_int_map                         _non_term_to_index;
    T_set                               _all_terms;
    int_set                             _all_non_terms;
    double***                           _A;
    double**                            _B;


public:
    SCFG(const std::vector<rule_T> & rules,
         const int & root_symbol) :
        _root_symbol(root_symbol)
    {
        _n_non_terms = 0;
        _n_terms = 0;

        //std::cout << "Coucou" << std::endl;

        // First run to setup indices
        for(const rule_T & rule : rules){
            if(_all_non_terms.count(rule.get_name()) == 0){
                _grammar.emplace(rule.get_name(), rule);
                _index_to_non_term.emplace(_n_non_terms,
                                           rule.get_name());
                _non_term_to_index.emplace(rule.get_name(),
                                           _n_non_terms);
                ++ _n_non_terms;
                _all_non_terms.insert(rule.get_name());
            }
            for(const T & x : rule.get_term_s()){
                if(_all_terms.count(x) == 0){
                    _index_to_term.emplace(_n_terms,
                                           x);
                    _term_to_index.emplace(x,
                                           _n_terms);
                    ++ _n_terms;
                    _all_terms.insert(x);
                }
            }
        }

        //std::cout << "Coucou 2" << std::endl;

        const int N = _n_non_terms;
        const int M = _n_terms;

        int i;
        int j;
        int k;

        //Second run to setup A and B
        _A = new double**[N];
        for(i = 0; i < N; ++i){
          _A[i] = new double*[N];
            for(j = 0; j < N; ++j){
                _A[i][j] = new double[N];
                for(k = 0; k < N; ++k){
                    _A[i][j][k] = 0;
                }
            }
        }

        _B = new double*[N];
        for(i = 0; i < N; ++i){
            _B[i] = new double[M];
            for(k = 0; k < M; ++k){
                _B[i][k] = 0;
            }
        }

        int l;
        for(const rule_T & rule: rules){
            i = _non_term_to_index[rule.get_name()];
            l = 0;
            const double_vect & weights = rule.get_non_term_w();
            for(const pair_i_i & derivation: rule.get_non_term_s()){
                j = _non_term_to_index[derivation.first];
                k = _non_term_to_index[derivation.second];
                _A[i][j][k] = weights[l++];
            }
            l = 0;
            const double_vect & new_weights = rule.get_term_w();
            for(const T & term: rule.get_term_s()){
                k = _term_to_index[term];
                _B[i][k] = new_weights[l++];
            }
        }
    }

    ~SCFG(){
        for(int i = 0; i < _n_non_terms; ++i){
            for(int j = 0; j < _n_non_terms; ++j){
                delete[] _A[i][j];
            }
            delete[] _A[i];
        }
        delete[] _A;
        for(int i = 0; i < _n_non_terms; ++i){
            delete[] _B[i];
        }
        delete[] _B;
    }

    rule_T & get_rule(int i){
        return _grammar.at(i);
    }

    void print_symbols() const{
        std::cout << "Non terminal symbols: " << std::endl;
        for(auto x : _all_non_terms){
            std::cout << x << " ";
        }std::cout << std::endl;
        std::cout << "Index to non terminal:" << std::endl;
        for(auto xy : _index_to_non_term){
            std::cout << "\tIndex: " << xy.first << ", Symbol: " << xy.second << std::endl;
        }std::cout << std::endl;
        std::cout << "Non terminal to index:" << std::endl;
        for(auto xy : _non_term_to_index){
            std::cout << "\tSymbol: " << xy.first << ", Index: " << xy.second << std::endl;
        }std::cout << std::endl;
        std::cout << "Terminal symbols: " << std::endl;
        for(auto x : _all_terms){
            std::cout << x << " ";
        }std::cout << std::endl;
        std::cout << "Index to terminal: " << std::endl;
        for(auto xy : _index_to_term){
            std::cout << "\tIndex: " << xy.first << ", Symbol: " << xy.second << std::endl;
        }std::cout << std::endl;
        std::cout << "Terminal to index: " << std::endl;
        for(auto xy : _term_to_index){
            std::cout << "\tSymbol: " << xy.first << ", Index: " << xy.second << std::endl;
        }std::cout << std::endl;
    }

    void print_params() const{
        for(int i = 0; i < _n_non_terms; ++i){
            std::cout << "Params A of non term " << i << std::endl;
            for(int j = 0; j < _n_non_terms; ++j){
                for(int k = 0; k < _n_non_terms; ++k){
                    std::cout << _A[i][j][k] << " ";
                }std::cout << std::endl;
            }
            std::cout << "Params B of non term " << i << std::endl;
            for(int k = 0; k < _n_terms; ++k){
                std::cout << _B[i][k] << " ";
            }std::cout << std::endl;
        }
    }

    double*** get_A() const{
        return _A;
    }

    int get_n_non_terms() const{
        return _n_non_terms;
    }

    double** get_B() const{
        return _B;
    }

    int get_n_terms() const{
        return _n_terms;
    }

    std::list<T> generate_sequence(const std::vector<int> & S){
        std::list<T> result;
        std::list<T> temp;
        for(int i = S.size() - 1; i > -1; --i){
            temp = _grammar.at(S[i]).complete_derivation(*this);
            result.insert(result.begin(),
                          temp.begin(),
                          temp.end());
        }
        return result;
    }

    const T_int_map & get_term_to_index() const{
        return _term_to_index;
    }

    const int_T_map & get_index_to_term() const{
        return _index_to_term;
    }

    const int_int_map & get_non_term_to_index() const{
        return _non_term_to_index;
    }

    const int_int_map & get_index_to_non_term() const{
        return _index_to_non_term;
    }

    const int & get_root_symbol() const{
        return _root_symbol;
    }

    const int & get_root_index() const{
        return _non_term_to_index.at(_root_symbol);
    }


};


#endif // SCFG_H
