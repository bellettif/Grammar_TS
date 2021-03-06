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

    std::vector<in_out_T*>  _in_out_cpters;

    T_vect_vect             _inputs;
    double***               _A_estim;
    double**                _B_estim;

public:
    Model_estimator(const SGrammar_T & init_grammar,
                    const T_vect_vect & inputs,
                    double*** initial_A_guess = 0,
                    double** initial_B_guess = 0):
        _init_grammar(init_grammar),
        _A(initial_A_guess),
        _B(initial_B_guess),
        _N(init_grammar.get_n_non_terms()),
        _M(init_grammar.get_n_terms()),
        _term_to_index(init_grammar.get_term_to_index()),
        _index_to_term(init_grammar.get_index_to_term()),
        _non_term_to_index(init_grammar.get_non_term_to_index()),
        _index_to_non_term(init_grammar.get_index_to_non_term()),
        _root_symbol(init_grammar.get_root_symbol()),
        _root_index(init_grammar.get_non_term_to_index().at(_root_symbol)),
        _inputs(inputs)
    {
        if(_A == 0){
            _A = init_grammar.get_A();
        }
        if(_B == 0){
            _B = init_grammar.get_B();
        }
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

    void estimate_from_inputs(){
        int n_inputs = _inputs.size();
        std::vector<double> weights;
        _in_out_cpters = std::vector<in_out_T*>();
        double *** E;
        double *** F;
        int N;
        int M;
        int length;
        double proba;
        in_out_T * current_cpter;
        std::vector<int> input_indices;
        for(int current_it = 0; current_it < _inputs.size() ; ++current_it){
            current_cpter = new in_out_T(_A, _B, _N, _M,
                                             _term_to_index,
                                             _index_to_term,
                                             _non_term_to_index,
                                             _index_to_non_term,
                                             _root_symbol,
                                             _root_index,
                                             _inputs[current_it]);
            current_cpter->get_inside_outside(E, F, N, M, length);
            proba = E[_root_index][0][length-1];
            weights.push_back(1.0 / proba);
            _in_out_cpters.push_back(current_cpter);
            input_indices.push_back(current_it);
        }
        double den;
        double temp;
        double num;
        int current_term_index;
        // Estimation of A
        for(int i = 0; i < _N; ++i){
            den = 0;
            for(int current_it = 0; current_it < _in_out_cpters.size(); ++current_it){
                _in_out_cpters[current_it]->get_inside_outside(E, F, N, M, length);
                temp = 0;
                for(int s = 0; s < length; ++s){
                    for(int t = s; t < length; ++t){
                        temp += E[i][s][t] * F[i][s][t];
                    }
                }
                den += temp * weights[current_it];
            }
            for(int j = 0; j < _N; ++j){
                for(int k = 0; k < _N; ++k){
                    num = 0;
                    for(int current_it = 0; current_it < _in_out_cpters.size(); ++current_it){
                        _in_out_cpters[current_it]->get_inside_outside(E, F, N, M, length);
                        temp = 0;
                        for(int s = 0; s < length - 1; ++s){
                            for(int t = s + 1; t < length; ++t){
                                for(int r = s; r < t; ++r){
                                    temp += _A[i][j][k] * E[j][s][r] * E[k][r+1][t] * F[i][s][t];
                                }
                            }
                        }
                        num += temp * weights[current_it];
                    }
                    _A_estim[i][j][k] = num / den;
                }
            }
        }
        // Estimation of B
        for(int i = 0; i < _N; ++i){
            den = 0;
            for(int current_it = 0; current_it < _in_out_cpters.size(); ++current_it){
                _in_out_cpters[current_it]->get_inside_outside(E, F, N, M, length);
                temp = 0;
                for(int s = 0; s < length; ++s){
                    for(int t = s; t < length; ++t){
                        temp += E[i][s][t] * F[i][s][t];
                    }
                }
                den += temp * weights[current_it];
            }
            for(int m = 0; m < _M; ++m){
                num = 0;
                for(int current_it = 0; current_it < _in_out_cpters.size(); ++current_it){
                    _in_out_cpters[current_it]->get_inside_outside(E, F, N, M, length);
                    for(int t = 0; t < length; ++t){
                        current_term_index = _term_to_index.at(_inputs[input_indices[current_it]][t]);
                        if(current_term_index == m){
                            _B_estim[i][m] += E[i][t][t] * F[i][t][t] * weights[current_it] / den;
                        }
                    }
                }
            }
        }
        for(int current_it = 0; current_it < _in_out_cpters.size(); ++current_it){
            delete _in_out_cpters[current_it];
        }
    }

    void print_estimates(){
        /*
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
        */
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
        /*
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
        */
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

    double*** get_A_estim(){
        return _A_estim;
    }

    double*** get_B_estim(){
        return _B_estim;
    }

    double*** get_A(){
        return _A;
    }

    double*** get_B(){
        return _B;
    }

    void set_A(double*** A){
        _A = A;
    }

    void set_B(double*** B){
        _B = B;
    }

    void swap_model_estim(){
        _A = _A_estim;
        _B = _B_estim;
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

};



#endif // MODEL_ESTIMATOR_H
