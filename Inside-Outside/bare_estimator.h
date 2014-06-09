#ifndef BARE_ESTIMATOR_H
#define BARE_ESTIMATOR_H

#include<vector>
#include<unordered_map>

template<typename T>
class Bare_estimator{

typedef std::vector<T>                                      T_vect;
typedef std::vector<int>                                    int_vect;
typedef std::unordered_map<int, int>                        int_int_map;
typedef std::unordered_map<int, T>                          int_T_map;
typedef std::unordered_map<T, int>                          T_int_map;

private:
    double***               _A;
    double**                _B;
    double***               _E;
    double***               _F;
    int                     _N;
    int                     _M;
    int                     _root_index;
    int                     _length;
    std::vector<int>        _input;

public:
    Bare_estimator(double***    A,
                   double**     B,
                   int          N,
                   int          M,
                   int          root_index,
                   const std::vector<T> & input,
                   const T_int_map & term_to_index):
        _A(A),
        _B(B),
        _N(N),
        _M(M),
        _root_index(root_index),
        _length(input.size()),
        _input(input.size())
    {
        _E = new double**[_N];
        _F = new double**[_N];
        for(int i = 0; i < _N; ++i){
            _E[i] = new double*[_length];
            _F[i] = new double*[_length];
            for(int j = 0; j < _length; ++j){
                _E[i][j] = new double[_length];
                _F[i][j] = new double[_length];
                for(int k = 0; k < _length; ++k){
                    _E[i][j][k] = 0;
                    _F[i][j][k] = 0;
                }
            }
        }
        for(int i = 0; i < input.size(); ++i){
            _input[i] = term_to_index.at(input[i]);
        }
    }

    void compute_inside_probas(){
        int i;
        int j;
        int k;
        int s;
        int t;
        int r;
        for(int l = 0; l < _length; ++l){
            if(l == 0){
                for(i = 0; i < _N; ++i){
                    for(s = 0; s < _length; ++s){
                        _E[i][s][s] = _B[i][_input[s]];
                    }
                }
            }else{
                for(i = 0; i < _N; ++i){
                    for(s = 0; s < _length - l; ++s){
                        t = s + l;
                        for(r = s; r < t; ++r){
                            for(j = 0; j < _N; ++j){
                                for(k = 0; k < _N; ++k){
                                    _E[i][s][t] += _A[i][j][k] * _E[j][s][r] * _E[k][r+1][t];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void compute_outside_probas(){
        int i;
        int j;
        int k;
        int s;
        int t;
        int r;
        for(int l = _length; l >= 0; --l){
            if(l == _length){
                _F[_root_index][0][_length - 1] = 1;
            }else{
                for(i = 0; i < _N; ++i){
                    for(s = 0; s < _length - l; ++s){
                        t = s + l;
                        for(r = 0; r < s; ++r){
                            for(j = 0; j < _N; ++j){
                                for(k = 0; k < _N; ++k){
                                    _F[i][s][t] += _F[j][r][t] * _A[j][k][i] * _E[k][r][s-1];
                                }
                            }
                        }
                        for(r = t + 1; r < _length; ++r){
                            for(j = 0; j < _N; ++j){
                                for(k = 0; k < _N; ++k){
                                    _F[i][s][t] += _F[j][s][r] * _A[j][i][k] * _E[k][t+1][r];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void print_A(){
        std::cout << "Printing actual A" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "A matrix of " << i << std::endl;
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
        std::cout << std::endl;
    }

    void print_B(){
        std::cout << "Printing actual B" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "B matrix of " << i << std::endl;
            std::cout << "\t";
            for(int k = 0; k < _M; ++k){
                if(_B[i][k] == 0){
                    std::cout << "0.00000000 ";
                }else{
                    std::cout << _B[i][k] << " ";
                }
            }std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void print_E(){
        std::cout << "Printing E" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "E matrix of " << i << std::endl;
            for(int s = 0; s < _length; ++s){
                std::cout << "\t";
                for(int t = 0; t < _length; ++t){
                    if(_E[i][s][t] == 0){
                        std::cout << "0.00000000 ";
                    }else{
                        std::cout << _E[i][s][t] << " ";
                    }
                }std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    void print_F(){
        std::cout << "Printing F" << std::endl;
        for(int i = 0; i < _N; ++i){
            std::cout << "F matrix of " << i << std::endl;
            for(int s = 0; s < _length; ++s){
                std::cout << "\t";
                for(int t = 0; t < _length; ++t){
                    if(_F[i][s][t] == 0){
                        std::cout << "0.00000000 ";
                    }else{
                        std::cout << _F[i][s][t] << " ";
                    }
                }std::cout << std::endl;
            }
        }
    }

    void validity_check(){
        double current_sum;
        for(int s = 0; s < _length; ++s){
            for(int t = 0; t < s; ++t){
                std::cout << "0.00000000 ";
            }
            for(int t = s; t < _length; ++t){
                current_sum = 0;
                for(int i = 0; i < _N; ++i){
                    current_sum += _E[i][s][t] * _F[i][s][t];
                }
                if(current_sum == 0){
                    std::cout << "0.00000000 ";
                }else{
                    std::cout << current_sum;
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};




#endif // BARE_ESTIMATOR_H
