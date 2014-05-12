#ifndef MODEL_ESTIMATOR_H
#define MODEL_ESTIMATOR_H

#include "in_out_proba.h"


template<typename T>
class Model_estimator{

typedef std::vector<T>                                  T_vect;
typedef std::vector<T_vect>                             T_vect_vect;
typedef std::unordered_map<int, int>                    int_int_map;
typedef std::unordered_map<int, T>                      int_T_map;
typedef std::unordered_map<T, int>                      T_int_map;
typedef SCFG<T>                                         SGrammar_T;
typedef In_out_proba<T>                                 in_out_T;


private:
    const SGrammar_T &      _init_grammar;
    double***               _A;
    double**                _B;
    int                     _N;
    int                     _M;
    const T_int_map &       _term_to_index;
    const int_T_map &       _index_to_term;
    const int_int_map &     _non_term_to_index;
    const int_int_map &     _index_to_non_term;
    int                     _root_symbol;
    int                     _root_index;

    in_out_T*               _in_out_cpter;

    T_vect_vect             _inputs;
    double***               _A_estim;
    double                  _A_tot_weight;
    double**                _B_estim;
    double                  _B_tot_weight;


public:
    Model_estimator(const SGrammar_T & init_grammar,
                    const T_vect_vect & inputs):
        _A(init_grammar.get_A()),
        _N(init_grammar.get_n_non_terms()),
        _B(init_grammar.get_B()),
        _M(init_grammar.get_n_terms()),
        _term_to_index(init_grammar.get_term_to_index()),
        _index_to_term(init_grammar.get_index_to_term()),
        _non_term_to_index(init_grammar.get_non_term_to_index()),
        _index_to_non_term(init_grammar.get_index_to_non_term()),
        _root_symbol(init_grammar.get_root_symbol()),
        _root_index(init_grammar.get_non_term_to_index().at(_root_symbol)),
        _inputs(inputs),
        _in_out_cpter(0)
    {
        _A_estim = new double**[_N];
        for(int i = 0; i < _N; ++i){
            _A_estim[i] = new double*[_N];
            for(int j = 0; j < _N; ++j){
                _A_estim[i][j] = new double[_N];
                for(int k = 0; k < _N; ++k){
                    _A_estim[i][j][k] = 0;
                }
            }
        }
        _B_estim = new double*[_N];
        for(int i = 0; i < _N; ++i){
            _B_estim[i] = new double[_M];
            for(int k = 0; k < _M; ++k){
                _B_estim[i][k] = 0;
            }
        }
    }

    ~Model_estimator(){
        delete _in_out_cpter;
        for(int i = 0; i < _N; ++i){
            for(int j = 0; j < _N; ++j){
                delete[] _A_estim[i][j];
            }
            delete[] _A_estim[i];
        }
        delete[] _A_estim;
        for(int i = 0; i < _N; ++i){
            delete[] _B_estim[i];
        }
        delete[] _B_estim;
    }


};



#endif // MODEL_ESTIMATOR_H
