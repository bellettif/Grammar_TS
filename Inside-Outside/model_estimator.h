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
    double**                _B_estim;

public:
    Model_estimator(const SGrammar_T & init_grammar,
                    const T_vect_vect & inputs):
        _init_grammar(init_grammar),
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

    double estimate_from_inputs(){
        double*** E;
        double*** F;
        int N;
        int M;
        int length;
        double proba;
        for(auto current_input : _inputs){
            for(auto x : current_input){
                std::cout << x << " ";
            }std::cout << std::endl;
            _in_out_cpter = new in_out_T(_A, _B, _N, _M,
                                             _term_to_index,
                                             _index_to_term,
                                             _non_term_to_index,
                                             _index_to_non_term,
                                             _root_symbol,
                                             _root_index,
                                             current_input);
            _in_out_cpter->get_inside_outside(E, F, N, M, length);
            proba = E[_root_index][0][length-1];
            if(proba == 0){
                continue;
            }else{
                // Estim A
                for(int i = 0; i < N; ++i){
                    for(int j = 0; j < N; ++j){
                        for(int k = 0; k < N; ++k){
                            for(int s = 0; s < length; ++s){
                                for(int t = s + 1; t < length; ++t){
                                    for(int r = s; r < t; ++r){
                                        _A_estim[i][j][k] +=
                                            1 / proba * _A[i][j][k] * E[j][s][r] * E[k][r+1][t] * F[i][s][t];
                                    }
                                }
                            }
                        }
                    }
                }
                // Estim B
                T current_obs;
                int current_obs_index;
                for(int i = 0; i < N; ++i){
                    for(int s = 0; s < length; ++s){
                        current_obs = current_input[s];
                        current_obs_index = _term_to_index.at(current_obs);
                        _B_estim[i][current_obs_index] +=
                                1 / proba * E[i][s][s]
                                    * F[i][s][s];
                    }
                }
            }
            delete _in_out_cpter;
        }
        // Normalization of A
        double current_sum;
        for(int i = 0; i < _N; ++i){
            current_sum = 0;
            for(int j = 0; j < _N; ++j){
                for(int k = 0; k < _N; ++k){
                    current_sum += _A_estim[i][j][k];
                }
            }
            if(current_sum == 0) continue;
            for(int j = 0; j < _N; ++j){
                for(int k = 0; k < _N; ++k){
                    _A_estim[i][j][k] /= current_sum;
                }
            }
        }
        // Normalization of B
        for(int i = 0; i < _N; ++i){
            current_sum = 0;
            for(int m = 0; m < _M; ++m){
                current_sum += _B_estim[i][m];
            }
            if(current_sum == 0) continue;
            for(int m = 0; m < _M; ++m){
                _B_estim[i][m] /= current_sum;
            }
        }
    }

    void print_estimates(){
        std::cout << "Actual A matrix" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "A matrix of index " << i << std::endl;
            for(int j = 0; j < _N; ++j){
                std::cout << "\t";
                for(int k = 0; k < _N; ++k){
                    if(_A[i][j][k] == 0){
                        std::cout << "0.00000000 ";
                    }else{
                        std::cout << _A[i][j][k] << " ";
                    }
                }std::cout << std::endl;
            }
        }
        std::cout << "Estimated A matrix" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "Estimated A matrix of index " << i << std::endl;
            for(int j = 0; j < _N; ++j){
                std::cout << "\t";
                for(int k = 0; k < _N; ++k){
                    if(_A_estim[i][j][k] == 0){
                        std::cout << "0.00000000 ";
                    }else{
                        std::cout << _A_estim[i][j][k] << " ";
                    }
                }std::cout << std::endl;
            }
        }
        std::cout << "Actual B matrix" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "B matrix of index " << i << std::endl;
            std::cout << "\t";
            for(int j = 0; j < _M; ++j){
                if(_B[i][j] == 0){
                    std::cout << "0.00000000 ";
                }else{
                    std::cout << _B[i][j] << " ";
                }
            }std::cout << std::endl;
        }
        std::cout << "Estimated B matrix" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "Estimated B matrix of index " << i << std::endl;
            std::cout << "\t";
            for(int j = 0; j < _M; ++j){
                if(_B_estim[i][j] == 0){
                    std::cout << "0.00000000 ";
                }else{
                    std::cout << _B_estim[i][j] << " ";
                }
            }std::cout << std::endl;
        }
    }

};



#endif // MODEL_ESTIMATOR_H
