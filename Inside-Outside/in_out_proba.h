#ifndef INSIDE_PROBA_H
#define INSIDE_PROBA_H

#include <vector>
#include <tuple>
#include <queue>

#include "scfg.h"
#include "parse_tree.h"

template<typename T>
class In_out_proba{

typedef std::vector<T>                                  T_vect;
typedef std::unordered_map<int, int>                    int_int_map;
typedef std::unordered_map<int, T>                      int_T_map;
typedef std::unordered_map<T, int>                      T_int_map;
typedef SCFG<T>                                         SGrammar_T;
typedef std::pair<int, int>                             non_term_der;

typedef std::unordered_map<int, double>                 int_double_hashmap;
typedef std::unordered_map<int,
                int_double_hashmap>                     int_int_double_hashmap;
typedef std::unordered_map<int,
                int_int_double_hashmap>                 int_int_int_double_hashmap;


typedef std::tuple<int, int, int>                       trip;
typedef std::tuple<int, int, int, Parse_tree<T>*>       stack_quad;
typedef std::queue<stack_quad>                          stack_quad_queue;
typedef std::unordered_map<int, trip>                   int_trip_hashmap;
typedef std::unordered_map<int,
                int_trip_hashmap>                       int_int_trip_hashmap;
typedef std::unordered_map<int,
                int_int_trip_hashmap>                   int_int_int_trip_hashmap;


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
    double***               _F;
    int                     _root_symbol;
    int                     _root_index;
    bool                    _inside_computed        = false;
    bool                    _outside_computed       = false;

public:

    In_out_proba(const SGrammar_T & grammar,
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
        _length(input.size()),
        _root_symbol(grammar.get_root_symbol()),
        _root_index(grammar.get_non_term_to_index().at(_root_symbol))
    {
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
        _F = new double**[_N];
        for(int i = 0; i < _N; ++i){
            _F[i] = new double*[_length];
            for(int j = 0; j  < _length; ++j){
                _F[i][j] = new double[_length];
                for(int k = 0; k < _length; ++k){
                    _F[i][j][k] = 0;
                }
            }
        }
    }

    In_out_proba(double*** A, double** B,
                 int N, int M,
                 const T_int_map & term_to_index,
                 const int_T_map & index_to_term,
                 const int_int_map & non_term_to_index,
                 const int_int_map & index_to_non_term,
                 int root_symbol,
                 int root_index,
                 const T_vect & input):
        _A(A), _B(B), _N(N), _M(M),
        _term_to_index(term_to_index),
        _index_to_term(index_to_term),
        _non_term_to_index(non_term_to_index),
        _index_to_non_term(index_to_non_term),
        _input(input),
        _length(input.size()),
        _root_symbol(root_symbol),
        _root_index(root_index)
    {
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
        _F = new double**[_N];
        for(int i = 0; i < _N; ++i){
            _F[i] = new double*[_length];
            for(int j = 0; j  < _length; ++j){
                _F[i][j] = new double[_length];
                for(int k = 0; k < _length; ++k){
                    _F[i][j][k] = 0;
                }
            }
        }
    }

    ~In_out_proba(){
        for(int i = 0; i < _N; ++i){
            for(int j = 0;j < _length; ++j){
                delete[] _E[i][j];
            }
            delete[] _E[i];
        }
        delete[] _E;
        for(int i = 0; i < _N; ++i){
            for(int j = 0;j < _length; ++j){
                delete[] _F[i][j];
            }
            delete[] _F[i];
        }
        delete[] _F;
    }

    void get_inside_outside(double*** & E,
                            double*** & F,
                            int & N,
                            int & M,
                            int & length){
        if(!_inside_computed){
            compute_inside_probas();
        }
        if(!_outside_computed){
            compute_outside_probas();
        }
        E = _E;
        F = _F;
        N = _N;
        M = _M;
        length = _length;
    }

    void compute_inside_level(int current_length){
        if(current_length == 0){
            for(int i = 0; i < _N; ++i){
                for(int s = 0; s < _length; ++s){
                    _E[i][s][s] = _B[i][_term_to_index.at(_input[s])];
                }
            }
        }else{
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
            compute_inside_level(l);
        }
        _inside_computed = true;
    }

    void compute_outside_level(int current_length){
        if(! _inside_computed){
            compute_inside_probas();
        }
        if(current_length == _length){
            _F[_root_index][0][_length - 1] = 1;
        }else{
            int s;
            int t;
            for(int i = 0; i < _N; ++i){
                for(s = 0; s < _length - current_length; ++s){
                    t = s + current_length;
                    for(int r = 0; r < s; ++r){
                        for(int j = 0; j < _N; ++j){
                            for(int k = 0; k < _N; ++k){
                                _F[i][s][t] += _F[j][r][t] * _A[j][k][i] * _E[k][r][s-1];
                            }
                        }
                    }
                    for(int r = t + 1; r < _length; ++r){
                        for(int j = 0; j < _N; ++j){
                            for(int k = 0; k < _N; ++k){
                                _F[i][s][t] += _F[j][s][r] * _A[j][i][k] * _E[k][t+1][r];
                            }
                        }
                    }
                }
            }
        }
    }

    void compute_outside_probas(){
        for(int l = _length; l >= 0; --l){
            compute_outside_level(l);
        }
        _outside_computed = true;
    }

    void compute_all_probas(){
        if(!_inside_computed){
            compute_inside_probas();
        }
        if(!_outside_computed){
            compute_outside_probas();
        }
    }

    void print_probas(){
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
            std::cout << "F matrix for non term "
                      << _index_to_non_term.at(i) << std::endl;
            for(int s = 0; s < _length; ++s){
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


    double run_CYK(){
        int_int_int_double_hashmap      gammas;
        int_int_int_trip_hashmap        taus;
        int                             i;
        int                             j;
        int                             k;
        int                             s;
        int                             t;
        int                             r;
        trip                            current_arg_max;
        double                          current_max;
        double                          current_value;
        for(int l = 0; l < _length; ++l){
            if(l == 0){
                for(i = 0; i < _N; ++i){
                    for(s = 0; s < _length; ++s){
                        gammas[i][s][s] = std::log(_B[i][_term_to_index.at(_input[s])]);
                        taus[i][s][s] = trip(i, i, -1);
                    }
                }
            }else{
                for(i = 0; i < _N; ++i){
                    for(s = 0; s < _length - l; ++s){
                        t = s + l;
                        current_arg_max = trip(0, 0, s);
                        current_max = gammas[0][s][s] + gammas[0][s+1][t]
                                + std::log(_A[i][0][0]);
                        for(j = 0; j < _N; ++j){
                            for(k = 0; k < _N; ++k){
                                for(r = s; r < t; ++r){
                                    current_value = gammas[j][s][r] + gammas[k][r+1][t]
                                            + std::log(_A[i][j][k]);
                                    if(current_value > current_max){
                                        current_arg_max = trip(j, k, r);
                                        current_max = current_value;
                                    }
                                }
                            }
                        }
                        gammas[i][s][t] = current_max;
                        taus[i][s][t] = current_arg_max;
                    }
                }
            }
        }
        return std::exp(gammas[_root_index][0][_length - 1]);
    }

    Parse_tree<T> run_CYK(double & parse_proba){
        int_int_int_double_hashmap      gammas;
        int_int_int_trip_hashmap        taus;
        int                             i;
        int                             j;
        int                             k;
        int                             s;
        int                             t;
        int                             r;
        trip                            current_arg_max;
        double                          current_max;
        double                          current_value;
        for(int l = 0; l < _length; ++l){
            if(l == 0){
                for(i = 0; i < _N; ++i){
                    for(s = 0; s < _length; ++s){
                        gammas[i][s][s] = std::log(_B[i][_term_to_index.at(_input[s])]);
                        taus[i][s][s] = trip(i, i, -1);
                    }
                }
            }else{
                for(i = 0; i < _N; ++i){
                    for(s = 0; s < _length - l; ++s){
                        t = s + l;
                        current_arg_max = trip(0, 0, s);
                        current_max = gammas[0][s][s] + gammas[0][s+1][t]
                                + std::log(_A[i][0][0]);
                        for(j = 0; j < _N; ++j){
                            for(k = 0; k < _N; ++k){
                                for(r = s; r < t; ++r){
                                    current_value = gammas[j][s][r] + gammas[k][r+1][t]
                                            + std::log(_A[i][j][k]);
                                    if(current_value > current_max){
                                        current_arg_max = trip(j, k, r);
                                        current_max = current_value;
                                    }
                                }
                            }
                        }
                        gammas[i][s][t] = current_max;
                        taus[i][s][t] = current_arg_max;
                    }
                }
            }
        }
        parse_proba = std::exp(gammas[_root_index][0][_length - 1]);

        trip                current_trip;
        stack_quad          current_quad;
        Parse_tree<T>       result(_root_symbol);
        stack_quad_queue    stack;
        stack.push(stack_quad(_root_index, 0, _length - 1, &result));
        int left_symbol_index;
        int right_symbol_index;
        int parent_index;
        int left_index;
        int right_index;
        int cut;
        Parse_tree<T>*      current_parent;
        Parse_tree<T>*      current_left;
        Parse_tree<T>*      current_right;
        while(! stack.empty()){
            current_quad = stack.front();
            stack.pop();
            parent_index = std::get<0>(current_quad);
            left_index = std::get<1>(current_quad);
            right_index = std::get<2>(current_quad);
            current_parent = std::get<3>(current_quad);
            current_trip = taus[parent_index][left_index][right_index];
            left_symbol_index = std::get<0>(current_trip);
            right_symbol_index = std::get<1>(current_trip);
            cut = std::get<2>(current_trip);
            if(cut == -1){
                current_parent->set_terminal(_input[left_index]);
            }else{
                current_left = new Parse_tree<T>(_index_to_non_term.at(left_symbol_index));
                current_right = new Parse_tree<T>(_index_to_non_term.at(right_symbol_index));
                current_parent->set_non_terminal(current_left, current_right);
                stack.push(stack_quad(right_symbol_index, cut + 1, right_index, current_right));
                stack.push(stack_quad(left_symbol_index, left_index, cut, current_left));
            }
        }
        return result;
    }

    void check_integrity(){
        double temp_sum;
        int i;
        int s;
        int t;
        std::cout << "Inside outside sums: " << std::endl;
        for(s = 0; s < _length; ++s){
            for(int q = 0; q < s; ++q){
                std::cout << "0.00000000 ";
            }
            for(t = s; t < _length; ++t){
                temp_sum = 0;
                for(i = 0; i < _N; ++i){
                    temp_sum += _E[i][s][t] * _F[i][s][t];
                }
                if(temp_sum == 0){
                    std::cout << "0.00000000 ";
                }else{
                    std::cout << temp_sum << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    void print_A_and_B(){
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
    }

};


#endif // INSIDE_PROBA_H
