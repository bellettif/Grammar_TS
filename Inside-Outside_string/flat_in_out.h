#ifndef FLAT_IN_OUT_H
#define FLAT_IN_OUT_H

#include <vector>
#include <unordered_map>
#include <string>

typedef std::unordered_map<std::string, int>              string_int_map;
typedef std::vector<std::string>                          string_vect;

class Flat_in_out
{

private:
    double*                                     _A;     //Flatenned 3D array
    double*                                     _B;     //Flatenned 3D array
    int                                         _N;
    int                                         _M;
    string_vect                                 _terminals;
    string_int_map                              _terminal_to_index;

public:
    Flat_in_out(double* A, double* B,
               int N, int M,
               const string_vect & terminals):
        _A(A),
        _B(B),
        _N(N),
        _M(M),
        _terminals(terminals)
    {
        for(int i = 0; i < _terminals.size(); ++i){
            _terminal_to_index.emplace(_terminals[i], i);
        }
    }

    void inline compute_probas_flat(const std::vector<string_vect> & samples,
                                    double * probas){
        int n_samples = samples.size();
        for(int i = 0; i < n_samples; ++i){
            probas[i] = compute_proba_flat(samples[i]);
        }
    }

    double inline compute_proba_flat(const string_vect & sample){
        int length = sample.size();
        double* E = new double[_N * length * length];
        compute_inside_probas_flat(E, sample);
        double result = E[0*length*length + 0*length + length - 1];
        delete[] E;
        return result;
    }

    void inline compute_inside_outside_flat(double* E, double* F,
                                       const string_vect & sample){
        compute_inside_probas_flat(E, sample);
        compute_outside_probas_flat(E, F, sample);
    }

    void inline compute_inside_probas_flat(double* E, const string_vect & sample){
        int length = sample.size();
        for(int i = 0; i < _N; ++i){
            for(int s = 0; s < length; ++s){
                for(int t = 0; t < length; ++t){
                    E[i*length*length + s*length + t] = 0.0;
                }
            }
        }
        for(int level = 0; level < length; ++ level){
            compute_inside_level(E, sample, level);
        }
    }

    void inline compute_inside_level(double* E, const string_vect & sample,
                                     int level){
        int length = sample.size();
        if(level == 0){
            for(int i = 0; i < _N; ++i){
                for(int s = 0; s < length; ++s){
                    E[i*length*length + s*length + s] = _B[i*_M + _terminal_to_index.at(sample[s])];
                }
            }
        }else{
            int t;
            int s;
            int r;
            int j;
            int k;
            for(int i = 0; i < _N; ++i){
                for(s = 0; s < length - level; ++s){
                    t = s + level;
                    for(r = s; r < t; ++r){
                        for(j = 0; j < _N; ++j){
                            for(k = 0; k < _N; ++k){
                                E[i*length*length + s*length + t] +=
                                        _A[i*_N*_N + j*_N + k]
                                        *
                                        E[j*length*length + s*length + r]
                                        *
                                        E[k*length*length + (r+1)*length + t];
                            }
                        }
                    }
                }
            }
        }
    }

    // Inside must have been computed already
    void inline compute_outside_probas_flat(double* E, double * F,
                                       const string_vect & sample){
        int length = sample.size();
        for(int i = 0; i < _N; ++i){
            for(int s = 0; s < length; ++s){
                for(int t = 0; t < length; ++t){
                    F[i*length*length + s*length + t] = 0.0;
                }
            }
        }
        for(int level = length; level >= 0; --level){
            compute_outside_level(E, F, sample, level);
        }
    }

    void inline compute_outside_level(double * E, double * F,
                                      const string_vect & sample,
                                      int level){
        int length = sample.size();
        if(level == length){
            F[0*length*length + 0*length + (length - 1)] = 1.0;
        }else{
            int t;
            int s;
            int r;
            int j;
            int k;
            for(int i = 0; i < _N; ++i){
                for(s = 0; s < length - level; ++s){
                    t = s + level;
                    for(r = 0; r < s; ++r){
                        for(j = 0; j < _N; ++j){
                            for(k = 0; k < _N; ++k){
                                F[i*length*length + s*length + t] +=
                                        F[j*length*length + r*length + t]
                                        *
                                        _A[j*_N*_N + k*_N + i]
                                        *
                                        E[k*length*length + r*length + (s-1)];
                            }
                        }
                    }
                    for(r = t + 1; r < length; ++r){
                        for(j = 0; j < _N; ++j){
                            for(k = 0; k < _N; ++k){
                                F[i*length*length + s*length + t] +=
                                        F[j*length*length + s*length + r]
                                        *
                                        _A[j*_N*_N + i*_N + k]
                                        *
                                        E[k*length*length + (t+1)*length + r];
                            }
                        }
                    }
                }
            }
        }
    }

    void inline estimate_A_B(const std::vector<string_vect> & samples,
                             double* sample_probas,
                             double* new_A,
                             double* new_B){
        int n_samples = samples.size();
        int * sample_lengths = new int[n_samples];
        int length;
        int id;
        for(id = 0; id < samples.size(); ++id){
            sample_lengths[id] = samples[id].size();
        }
        double** Es = new double*[n_samples];
        double** Fs = new double*[n_samples];
        for(id = 0; id < n_samples; ++id){
            length = sample_lengths[id];
            Es[id] = new double[_N*length*length];
            Fs[id] = new double[_N*length*length];
        }

        for(id = 0; id < n_samples; ++id){
            compute_inside_outside_flat(Es[id], Fs[id], samples[id]);
            length = sample_lengths[id];
            sample_probas[id] = Es[id][0*length*length + 0*length + length - 1];
            std::cout << "Id: " << id << " proba " << sample_probas[id] << std::endl;
        }

        double den;
        double num;
        double temp;
        int i;
        int j;
        int k;
        int t;
        int s;
        int r;
        for(i = 0; i < _N; ++i){
            den = 0;
            for(id = 0; id < n_samples; ++id){
                length = sample_lengths[id];
                if(sample_probas[id] == 0){
                    continue;
                }
                temp = 0;
                for(s = 0; s < length; ++s){
                    for(t = s; t < length; ++t){
                        temp += Es[id][i*length*length + s*length + t]
                                *
                                Fs[id][i*length*length + s*length + t];
                    }
                }
                den += temp / sample_probas[id];
            }
            for(j = 0; j < _N; ++j){
                for(k = 0; k < _N; ++k){
                    num = 0;
                    for(id = 0; id < n_samples; ++id){
                        if(sample_probas[id] == 0){
                            continue;
                        }
                        length = sample_lengths[id];
                        temp = 0;
                        for(s = 0; s < length - 1; ++s){
                            for(t = s + 1; t < length; ++t){
                                for(r = s; r < t; ++r){
                                    temp += _A[i*_N*_N + j*_N + k]
                                            *
                                            Es[id][j*length*length + s*length + r]
                                            *
                                            Es[id][k*length*length + (r+1)*length + t]
                                            *
                                            Fs[id][i*length*length + s*length + t];
                                }
                            }
                        }
                        num += temp / sample_probas[id];
                    }
                    new_A[i*_N*_N + j*_N + k] = num / den;
                }
            }
            for(j = 0; j < _M; ++j){
                num = 0;
                for(id = 0; id < n_samples; ++id){
                    temp = 0;
                    if(sample_probas[id] == 0){
                        continue;
                    }
                    length = sample_lengths[id];
                    for(s = 0; s < length; ++s){
                        if(_terminal_to_index[samples[id][s]] == j){
                            temp += Es[id][i*length*length + s*length + t]
                                    *
                                    Fs[id][i*length*length + s*length + t];
                        }
                    }
                    num += temp / sample_probas[id];
                }
                new_B[i*_N + j] = num / den;
            }
        }

        for(id = 0; id < n_samples; ++id){
            delete[] Es[id];
            delete[] Fs[id];
        }
        delete[] Es;
        delete[] Fs;
        delete[] sample_lengths;
    }

};

#endif // FLAT_IN_OUT_H
