#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

#include <iostream>

namespace array_utils{

static void allocate_arrays(double*** & A_3d,
                            double** & B_2d,
                            int N,
                            int M){

    double*** A = new double**[N];
    for(int i = 0; i < N; ++i){
        A[i] = new double*[N];
        for(int j = 0; j < N; ++j){
            A[i][j] = new double[N];
        }
    }
    double** B = new double*[N];
    for(int i = 0; j < N; ++j){
        B[i] = new double[M];
    }
}

static void deallocate_arrays(double*** A,
                              double** B,
                              int N,
                              int M){
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            delete[] A[i][j];
        }
        delete[] A[i];
    }
    delete A;
    for(int i = 0; i < N; ++i){
        delete[] B[i];
    }
    delete B;
}

static void print_array_content(double*** A,
                                double** B,
                                int N,
                                int M){
    std::cout << "A array:" << std::endl;
    for(int i = 0; i < N; ++i){
        std::cout << "\tSlice number " << i << std::endl;
        for(int j = 0; j < N; ++j){
            std::cout << "\t\t";
            for(int k = 0; k < N; ++k){
                if(A[i][j][k] == 0){
                    std::cout << "0.00000000 ";
                }else{
                    std::cout << A[i][j][k] << " ";
                }
            }std::cout << std::endl;
        }
    }
    std::cout << "B array:" << std::endl;
    for(int i = 0; i < N; ++i){
        std::cout << "\t\t";
        for(int j = 0; j < M; ++j){
            if(B[i][j] == 0){
                std::cout << "0.00000000 ";
            }else{
                std::cout << B[i][j] << " ";
            }
        }std::cout << std::endl;
    }
    std::cout << std::endl;
}

static void flatten_params(double*** A_3d, double** B_2d,
                           int N, int M,
                           double* & A_1d, double* & B_1d){
    A_1d = new double[N*N*N]; // stride is N, N
    B_1d = new double[N*M]; // stride is M
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k){
                A_1d[i*N*N + j*N + k] = A_3d[i][j][k];
            }
        }
    }
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            B_1d[i*M + j] = B_2d[i][j];
        }
    }
}


static void fill_arrays_with_random(double*** A,
                                    double** B,
                                    int N, int M,
                                    RNG rng){
    std::uniform_real_distribution<double> noise(0.01, 1.0);
    double temp_sum;
    for(int i = 0; i < N; ++i){
        temp_sum = 0;
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k){
                temp_value = noise(rng);
                temp_sum += temp_value;
                A[i][j][k] = temp_value;
            }
        }
        for(int j = 0; j < M; ++j){
            temp_value = noise(rng);
            temp_sum += temp_value;
            B[i][j] = temp_value;
        }
        // Normalizing
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k){
                A[i][j][k] /= temp_sum;
            }
        }
        for(int j = 0; j < M; ++j){
            B[i][j] /= temp_sum;
        }
    }
}


}
#endif // ARRAY_UTILS_H
