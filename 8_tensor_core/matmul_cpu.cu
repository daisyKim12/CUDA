#include "cmath"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <iostream>

#define NUM 10
#define TILE_WIDTH 16

// initializing array size N with random float between 0 and 100
void init_array(float *arr, long long int N, int seed) {
    
    srand(seed);

    for(long long int i = 0; i<N; i++){
        arr[i] = ((float)rand() / RAND_MAX) * 100.0;
    }
}

// display partial array size m
void print_array(float *arr, int m) {
    for(int i = 0; i < m; i++) {
        std::cout << arr[i]<< " ";
    }
    std::cout << "...\n" << std::endl;
}

void save_result(float* arr, int width, int height, const char* file_name) {
    
    std::ofstream dst(file_name);
    if(!dst.is_open()) {
        std::cerr << "Error: can not open the file";
    }

    for(int i = 0; i<height; i++) {
        for(int j = 0; j<width; j++) {
            dst << arr[i * width + j] << ",";
        }
        dst << "\n";
    }
    dst.close();

}

void save_run_time(double* run_time, double* bandwidth, int n, const char* file_name) {
    
    std::ofstream dst(file_name);
    if(!dst.is_open()) {
        std::cerr << "Error: can not open the file";
    }

    for(int ver = 1; ver <= n; ver++) {
        dst << "ver " << ver <<" time: " << run_time[ver-1] <<"seconds\n";
        dst << "ver " << ver <<" BW: " << bandwidth[ver-1] <<"GB/s\n";
        dst << "\n";
    }
    dst.close();
}

template <int rows, int columns, int inners, int tileSize>
inline void matmulImplTiling(const float *left, const float *right,
                             float *result) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];
} } } } }

double matmul_cpu(float* A, float* B, float* C) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    matmulImplTiling<4096, 4096, 4096, 16>(A, B, C);
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    return sec.count();
}

int main(int argc, char *argv[]) {

    if(argc < 2) {
        std::cout << "please enter execution size\n";
        return 1;
    }

    long long int nx = atoi(argv[1]);
    long long int nxy = nx * nx;

    float *M_h = new float[nxy];
    float *N_h = new float[nxy];
    float *result_h = new float[nxy];
    // float *M_d, *N_d, *result_d;

    init_array(M_h, nxy, 8811);
    init_array(N_h, nxy, 9700);

    // // device memory allocation
    // cudaMalloc((void**)&M_d, nxy * sizeof(float));
    // cudaMalloc((void**)&N_d, nxy * sizeof(float));
    // cudaMalloc((void**)&result_d, nxy * sizeof(float));
    // // copy host memory to device
    // cudaMemcpy(M_d, M_h, nxy * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(N_d, N_h, nxy * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "running on cpu... \n";
    // cudaError_t cudaErr = cudaGetLastError();
    // std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;    
    double cpu_time = matmul_cpu(M_h, N_h, result_h);
    std::cout << "execution time: " << cpu_time << std::endl;

    // // copy result to host memory
    // cudaMemcpy(result_h, result_d, nxy * sizeof(float), cudaMemcpyDeviceToHost);

    print_array(result_h, NUM);

    // cudaFree(M_d); cudaFree(N_d); cudaFree(result_d);
    delete[] M_h; delete[] N_h; delete[] result_h;

    return 0;
}