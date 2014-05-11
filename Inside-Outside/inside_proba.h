#ifndef INSIDE_PROBA_H
#define INSIDE_PROBA_H

#include <vector>

#include "scfg.h"

template<typename T>
class Inside_proba{

typedef std::vector<T>                          T_vect;
typedef std::unordered_map<int, int>            int_int_map;
typedef std::unordered_map<int, T>              int_T_map;
typedef std::unordered_map<T, int>              T_int_map;
typedef SCFG<T>                                 SGrammar_T;

private:
    double***               _A;
    double**                _B;
    int                     _N;
    int                     _M;
    const T_int_map &       _term_to_index;
    const int_T_map &       _index_to_term;
    const int_int_map &     _non_term_to_index;
    const int_int_map &     _index_to_non_term;
    const T_vect &          _input;
    int                     _length;
    double***               _E;

public:
    Inside_proba(const SGrammar_T & grammar,
                 const T_vect & input):
        _A(grammar.get_A()),
        _N(grammar.get_n_non_terms()),
        _B(grammar.get_B()),
        _M(grammar.get_n_terms()),
        _term_to_index(grammar.get_term_to_index()),
        _index_to_term(grammar.get_index_to_term()),
        _non_term_to_index(grammar.get_non_term_to_index()),
        _index_to_non_term(grammar.get_index_to_non_term()),
        _input(input),
        _length(input.size()){
        _E = new double**[_N];
        for(int i = 0; i < _N; ++i){
            _E[i] = new double*[_length];
            for(int j = 0; j < _length; ++j){
                _E[i][j] = new double[_length];
                for(int k = 0; k < _length; ++k){
                    _E[i][j][k] = 0;
                }
            }
        }
    }

    void compute_level(int current_length){
        if(current_length == 0){
            for(int i = 0; i < _N; ++i){
                for(int s = 0; s < _length; ++s){
                    _E[i][s][s] = _B[i][_term_to_index.at(_input[s])];
                }
            }
        }else{
            int s;
            int t;
            for(int i = 0; i < _N; ++i){
                for(int s = 0; s < _length - current_length; ++s){
                    t = s + current_length;
                    for(int r = s; r < t; ++r){
                        for(int j = 0; j < _N; ++j){
                            for(int k = 0; k < _N; ++k){
                                _E[i][s][t] += _A[i][j][k] * _E[j][s][r] * _E[k][r+1][t];
                            }
                        }
                    }
                }
            }
        }
    }

    void compute_inside_probas(){
        for(int l = 0; l < _length; ++ l){
            compute_level(l);
        }
    }

    void print_E(){
        for(int i = 0; i < _N; ++i){
            std::cout << "E matrix for non term "
                      << _index_to_non_term.at(i) << std::endl;
            for(int s = 0; s < _length; ++s){
                for(int t = 0; t < _length; ++t){
                    if(_E[i][s][t] == 0){
                        std::cout << "0.00000000 ";
                    }else{
                        std::cout << _E[i][s][t] << " ";
                    }
                }std::cout << std::endl;
            }
        }
    }




};


#endif // INSIDE_PROBA_H
