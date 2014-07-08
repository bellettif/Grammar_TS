#ifndef FLAT_IN_OUT_H
#define FLAT_IN_OUT_H


#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <string>
#include <random>
#include <deque>

#include "rule.h"

typedef std::unordered_map<std::string, int>              string_int_map;
typedef std::vector<std::string>                          string_vect;
typedef std::vector<string_vect>                          string_vect_vect;
typedef std::discrete_distribution<>                      choice_distrib;
typedef std::mt19937                                      RNG;
typedef std::vector<int>::iterator                        vect_it;
typedef std::list<int>::iterator                          list_it;
typedef std::vector<double>                               double_vect;
typedef std::vector<int>                                  int_vect;

static auto duration =  std::chrono::system_clock::now().time_since_epoch();
static auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
static RNG core_rng (millis);

static const int MAX_ITER_IN_DERIVATION = 10000;


/*
 *      Handles Inside Outside algorithm with 1D arrays.
 *          1D arrays improve Python compatibility.
 */
class Flat_in_out
{

private:
    /*
     *  Parameters of the grammar
     *      _A is the non terminal transformation matrix (N * N * N)
     *      _B is the terminal emission matrix (N * M)
     *      _N is the number of non terminal symbols
     *      _M is the number of terminal symbols
     *      _terminals in the list of terminal symbols
     *      _terminal to index is its reversed dictionary counterpart
     *
     */
    const double * const                         _A;                 // Flatenned 3D array (N * N * N)
    const double * const                         _B;                 // Flatenned 3D array (N * N * N)
    const int                                   _N;
    const int                                   _M;
    const string_vect                           _terminals;
    const string_int_map                        _terminal_to_index;
    /*
     *  The rule map will be built in case sentences need to be generated
     *      and rules need to be used in a rule map format
     */
    bool                                        _built_rule_map         = false;
    std::unordered_map<int, Rule>               _rule_map;
    /*
     *  Index of the root symbol
     */
    const int                                   _root_index;

public:
    /*
     *  A and B are given by Python (already allocated)
     */
    Flat_in_out(double* A, double* B,
                   int N, int M,
                   const string_vect & terminals,
                   int root_index):
        _A(A),
        _B(B),
        _N(N),
        _M(M),
        _terminals(terminals),
        _root_index(root_index)
    {
        const string_int_map * const term_to_indx_pt = & _terminal_to_index;
        for(int i = 0; i < _terminals.size(); ++i){
            const_cast<string_int_map *>(term_to_indx_pt)->emplace(_terminals[i], i);
        }
    }

    /*
     *  Compute the probability that the grammar generates sample
     *      for each sample in grammar. The result in stored in probas
     *      which is assumed to be of size size_of(double) * n_samples
     */
    void inline compute_probas_flat(const std::vector<string_vect> & samples,
                                    double * probas){
        int n_samples = samples.size();
        for(int i = 0; i < n_samples; ++i){
            probas[i] = compute_proba_flat(samples[i]);
        }
    }

    /*
     *  Compute the probability that the grammar generates sample
     *      The inside part of the inside-outside algorithm is used
     *      for that.
     *
     */
    double inline compute_proba_flat(const string_vect & sample){
        int length = sample.size();
        double* E = new double[_N * length * length];
        compute_inside_probas_flat(E, sample);
        /*
         *  Return the probability that root index generates the whole sequence
         *      from index 0 to index length - 1
         */
        double result = E[_root_index*length*length + 0*length + length - 1];
        delete[] E;
        return result;
    }

    /*
     *  Run the inside outside algorithm
     */
    void inline compute_inside_outside_flat(double* E, double* F,
                                       const string_vect & sample){
        compute_inside_probas_flat(E, sample);
        compute_outside_probas_flat(E, F, sample);
    }

    /*
     *  Run the inside part of the algorithm on all subsequences of sample
     *
     *  The result will be stored in E whose dimension is N * length * length
     *      (flattenned version of the 3D array)
     *
     */
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

    /*
     *  Run the inside part of the algorithm on subsequences on length == level
     *
     *  E is the linear flattened version of a 3D array of dimension
     *      N * length * length
     *  Level in the length of the subsequence that is currently being examined
     *      in sample.
     *  The results will be stored in E
     *
     */
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
            double temp_A;
            double temp_sum;
            for(int i = 0; i < _N; ++i){
                for(s = 0; s < length - level; ++s){
                    t = s + level;
                    E[i*length*length + s*length + t] = 0;
                }
                for(j = 0; j < _N; ++j){
                    for(k = 0; k < _N; ++k){
                        temp_A = _A[i*_N*_N + j*_N + k];
                        if(temp_A == 0.0){
                            continue;
                        }
                        for(s = 0; s < length - level; ++s){
                            t = s + level;
                            temp_sum = 0;
                            for(r = s; r < t; ++r){
                                /*
                                 *  i is the index of parent the non term symbol
                                 *  s is the index of the starting position in the parsed sequence
                                 *  t is the index of the ending position in the parsed sequence
                                 *      Here we have t - s = level as per the definition of the function
                                 *  j is the index of the left child non term symbol
                                 *  k is the index of the right child non term symbol
                                 *
                                 */
                                 temp_sum +=
                                        temp_A
                                        *
                                        E[j*length*length + s*length + r]
                                        *
                                        E[k*length*length + (r+1)*length + t];
                            }
                            E[i*length*length + s*length + t] += temp_sum;
                        }
                    }
                }
            }
        }
    }

    /*
     *  Run the outside part of the algorithm
     *      Inside results must have been computed already (result stored in E)
     *      Outside results will be computed and stored in F (dimension N * length * length)
     */
    void inline compute_outside_probas_flat(double* E, double * F,
                                       const string_vect & sample){
        int length = sample.size();
        /*
         *  Initialize outside probas with zeros
         */
        for(int i = 0; i < _N; ++i){
            for(int s = 0; s < length; ++s){
                for(int t = 0; t < length; ++t){
                    /*
                     *  i is the index of the parent element
                     *  s the index of the start position of the subsequence
                     *  t the index of the end position of the subsequence
                     */
                    F[i*length*length + s*length + t] = 0.0;
                }
            }
        }
        /*
         *  Compute the outside probabilities for each level, i.e. for each
         *      length of subscripts in the sample
         */
        for(int level = length; level >= 0; --level){
            compute_outside_level(E, F, sample, level);
        }
    }

    /*
     *  Run the oustide part of the algorithm for all subsequences of length == level
     *      Results of the inside part of the algorithm have already been computed
     *      and are assumed to be available in E.
     *      Results of the oustide part will be stored in F.
     */
    void inline compute_outside_level(double * E, double * F,
                                      const string_vect & sample,
                                      int level){
        int length = sample.size();
        if(level == length){
            F[_root_index*length*length + _root_index*length + (length - 1)] = 1.0;
        }else{
            int t;
            int s;
            int r;
            int j;
            int k;
            double temp_A_left;
            double temp_A_right;
            for(int i = 0; i < _N; ++i){
                for(j = 0; j < _N; ++j){
                    for(k = 0; k < _N; ++k){
                        temp_A_left = _A[j*_N*_N + k*_N + i];
                        temp_A_right = _A[j*_N*_N + i*_N + k];
                        for(s = 0; s < length - level; ++s){
                            t = s + level;
                            if(temp_A_left != 0){
                                for(r = 0; r < s; ++r){
                                    F[i*length*length + s*length + t] +=
                                            F[j*length*length + r*length + t]
                                            *
                                            temp_A_left
                                            *
                                            E[k*length*length + r*length + (s-1)];
                                }
                            }
                            if(temp_A_right != 0){
                                for(r = t + 1; r < length; ++r){
                                    F[i*length*length + s*length + t] +=
                                            F[j*length*length + s*length + r]
                                            *
                                            temp_A_right
                                            *
                                            E[k*length*length + (t+1)*length + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /*
     *  Estimate matrices A and B with Baum-Welch algorithm (EM algo)
     *      This runs one iteration of the EM algo.
     *
     *  Input: list of sentences in samples
     *  Output: probability that each sample has been generated by the grammar
     *      in sample_probas (pre-allocated by Python)
     *          new estimation of matrix A in new_A (dimension N * N * N) (pre-allocated by Python)
     *          new estimation of matrix B in new_B (dimension N * M) (pre-allocated by Python)
     *
     */
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
        /*
         *  Prepare memory for as many runs of the inside outside
         *      algorithm as there are sample
         */
        double** Es = new double*[n_samples];
        double** Fs = new double*[n_samples];
        for(id = 0; id < n_samples; ++id){
            length = sample_lengths[id];
            Es[id] = new double[_N*length*length];
            Fs[id] = new double[_N*length*length];
        }

        /*
         *  Run the inside outside algorithm for each sample and
         *      get the proba that the corresponding sample has been generated
         */
        for(id = 0; id < n_samples; ++id){
            compute_inside_outside_flat(Es[id], Fs[id], samples[id]);
            length = sample_lengths[id];
            sample_probas[id] = Es[id][_root_index*length*length + _root_index*length + length - 1];
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
        double temp_A;
        double temp_F;
        for(i = 0; i < _N; ++i){
            /*
             *  Estimation of A[i, :, :]
             */
            /*
             *  Compute denominator in estimation of A[i, : ,:]
             */
            den = 0;
            for(id = 0; id < n_samples; ++id){
                length = sample_lengths[id];
                if(sample_probas[id] == 0){
                    continue;
                }
                temp = 0;
                for(s = 0; s < length; ++s){
                    for(t = s; t < length; ++t){
                        /*
                         *  s is the beginning of the subsequence
                         *  t is the end of the subsequence
                         */
                        temp += Es[id][i*length*length + s*length + t]
                                *
                                Fs[id][i*length*length + s*length + t];
                    }
                }
                den += temp / sample_probas[id];
            }
            for(j = 0; j < _N; ++j){
                for(k = 0; k < _N; ++k){
                    temp_A = _A[i*_N*_N + j*_N + k];
                    /*
                     *  Estimate element A[i, j, k]
                     */
                    num = 0;
                    for(id = 0; id < n_samples; ++id){
                        if(sample_probas[id] == 0){
                            continue;
                        }
                        length = sample_lengths[id];
                        temp = 0;
                        for(s = 0; s < length - 1; ++s){
                            for(t = s + 1; t < length; ++t){
                                temp_F = Fs[id][i*length*length + s*length + t];
                                for(r = s; r < t; ++r){
                                    /*
                                     *  s is the beginning of the subsequence
                                     *  t is the end of the subsequence
                                     *  r is the current index at which the subsequence is
                                     *      split into a left generated element and a right
                                     *      generated element
                                     */
                                    temp += temp_A
                                            *
                                            Es[id][j*length*length + s*length + r]
                                            *
                                            Es[id][k*length*length + (r+1)*length + t]
                                            *
                                            temp_F;
                                }
                            }
                        }
                        num += temp / sample_probas[id];
                    }
                    new_A[i*_N*_N + j*_N + k] = num / den;
                }
            }
            /*
             *  Estimation of B[i, :]
             */
            for(j = 0; j < _M; ++j){
                /*
                 *  Estimate element B[i, j]
                 */
                num = 0;
                for(id = 0; id < n_samples; ++id){
                    temp = 0;
                    if(sample_probas[id] == 0){
                        continue;
                    }
                    length = sample_lengths[id];
                    for(s = 0; s < length; ++s){
                        /*
                         *  s is the position of the current terminal
                         *      symbol being examined in the sequence
                         */
                        if(_terminal_to_index.at(samples[id][s]) == j){
                            temp += Es[id][i*length*length + s*length + s]
                                    *
                                    Fs[id][i*length*length + s*length + s];
                        }
                    }
                    num += temp / sample_probas[id];
                }
                new_B[i*_M + j] = num / den;
            }
        }

        /*
         *  Clear memory of inside-outside algorithm
         */
        for(id = 0; id < n_samples; ++id){
            delete[] Es[id];
            delete[] Fs[id];
        }
        delete[] Es;
        delete[] Fs;
        delete[] sample_lengths;
    }

    /*
     *  Generate n_sentences samples of the grammar.
     *  If the grammar is unstable, the function will not terminate.
     *      In that case, does_not_terminate will be true
     *
     */
    string_vect_vect inline produce_sentences(int n_sentences,
                                              bool & does_not_terminate){
        return produce_sentences(n_sentences,
                                 does_not_terminate,
                                 core_rng);
    }

    /*
     *  Same as before by rng can is not necessarily the core one
     */
    string_vect_vect inline produce_sentences(int n_sentences,
                                              bool & does_not_terminate,
                                              RNG & rng){
        string_vect_vect result (n_sentences);
        for(int i = 0; i < n_sentences; ++i){
            if(!produce_sentence(result.at(i), rng)){
                result.clear();
                does_not_terminate = true;
                return result;
            }
        }
        does_not_terminate = false;
        return result;
    }

    /*
     *  Compute the MC signature of the grammar.
     *  Produce n_sentences.
     *  For each type of sentence generate, compute the count among
     *      all the sequences that have been generated.
     *
     *  Input: n_sentences, number of samples that will be used
     *         max_length, if not zero, all sequences of length > max_length
     *              will be discarded (avoid pathological cases when comparing signatures)
     *  Output: freqs, the count of each type of sentence that has been generated
     *          doest_not_terminate, in case the grammar is unstable and the algorithm
     *          loops, this will be true.
     */
    void inline compute_frequences(int n_sentences,
                                    int_vect & freqs,
                                    string_vect & strings,
                                    bool & does_not_terminate,
                                    int max_length = 0){
        string_vect_vect samples = produce_sentences(n_sentences,
                                                     does_not_terminate);
        if(does_not_terminate){
            return;
        }
        string_int_map counts;
        std::string temp;
        int length;
        for(const string_vect & sample : samples){
            if ((max_length != 0) && (sample.size() > max_length)) {
                continue;
            }
            temp = "";
            length = sample.size();
            for(int i = 0; i < length; ++i){
                if (i == length - 1){
                    temp += sample.at(i);
                }else{
                    temp += sample.at(i) + " ";
                }
            }
            counts[temp] ++;
        }
        for(auto xy : counts){
            freqs.push_back(xy.second);
            strings.push_back(xy.first);
        }
    }

    /*
     *  Produce one sentence with the grammar. (Rule class will be used for that).
     *  If it has not been built yet, rule_map will be built.
     */
    inline bool produce_sentence(string_vect & result, RNG & rng){
        if(!_built_rule_map){
            build_rule_map();
        }
        /*
         *  The temporary sequence of symbols (that will be both non terminals
         *      and terminals) will be stored as integers (corresponding to the index
         *      of the rule or the index of the terminal)
         */
        std::list<int>                                  temp_result = {_root_index};
        /*
         *  List of instruction (two for BFS swap), contain the list of indices
         *      that need processing in temp_result, i.e., non terminal symbols
         *      that correpsonding to rules that need further derivation
         *      prior to producing terminal symbols
         */
        std::vector<list_it>                            index_to_do = {temp_result.begin()};
        std::vector<list_it>                            next_index_to_do;
        /*
         *  Preparing output that is going to be passed by reference
         */
        bool emission;
        int terminal;
        int left;
        int right;
        /*
         *  Preparing iterators for BFS
         */
        Rule * current_rule;
        list_it second_it;
        int current_iter = 0;
        /*
         *  BFS construction of the sentence
         */
        while(!index_to_do.empty()){
            for(list_it & it : index_to_do){
                current_rule = &(_rule_map.at(*it));
                current_rule->derivation(rng, emission, terminal, left, right);
                if(emission){
                    *it = terminal;
                }else{
                    *it = right;
                    it = temp_result.insert(it, left);
                    next_index_to_do.push_back(it);
                    second_it = it;
                    std::advance(second_it, 1);
                    next_index_to_do.push_back(second_it);
                }
            }
            index_to_do.swap(next_index_to_do);
            next_index_to_do.clear();
            ++ current_iter;
            /*
             *  Prevent infinite loop
             */
            if(current_iter == MAX_ITER_IN_DERIVATION){
                return false;
            }
        }
        for(auto x : temp_result){
            result.push_back(_terminals.at(x));
        }
        return true;
    }

    /*
     *  Build a map of rules that are ready for derivation and production
     *      of a sentence.
     */
    void inline build_rule_map(){
        for(int i = 0; i < _N; ++i){
            _rule_map.emplace(i, Rule(i, _A, _B, _N, _M, _terminals));
        }
        _built_rule_map = true;
    }

};

#endif // FLAT_IN_OUT_H
