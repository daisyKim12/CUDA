#include "_util.h"
#include "trans.h"
    
#include <chrono>
#include <stdlib.h>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <fstream>


/* 
64 threads per block using 32 warps total 2048 threads on fly 100% utilization
copying two size 64 float matrix into shared memory total 512B
*/

int main(int argc, char *argv[]) {

    // double *run_time = new double[10];

    // long width = WIDTH;
    // long height = HEIGHT;

    long width = atoi(argv[1]);
    long height = atoi(argv[1]);
    long total_size = width * height;

    const char* file = "trans.txt";
    std::ofstream dst(file, std::ios::app); // Open file in append mode
    if(!dst.is_open()) {
        std::cerr << "Error: can not open the file";
    }
    dst << "----------------" << width << "----------------" << "\n";

    float *A_h = new float[total_size];
    float *B_h = new float[total_size];
    float *A_d, *B_d;

    init_array(A_h, total_size, 1248);
    
    // memory allocation
    cudaMalloc((void**)&A_d, total_size * sizeof(float));
    cudaMalloc((void**)&B_d, total_size * sizeof(float));
    // copy host memory to device
    cudaMemcpy(A_d, A_h, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // for(int ver = 1; ver <=6; ver++)
    // {
    //     std::cout << "< ver " << ver << " running ... >\n";
    //     run_time[ver-1] = time_transpose(A_d, B_d, width, TILE_WIDTH, ver);
    //     cudaError_t cudaErr = cudaGetLastError();
    //     std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;
    // }
    for(int i = 0; i< 1000; i++) {
        int ver = 6;
        std::cout << "< ver " << ver << " running ... >\n";
        double run_time = time_transpose(A_d, B_d, width, TILE_WIDTH, ver);
        std::cout << "run time: " << run_time << "sec" << std::endl;
        cudaError_t cudaErr = cudaGetLastError();
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;
        
        // run_time[0] = time_transpose(A_d, B_d, width, TILE_WIDTH, 6);

        // copy result to host memory
        cudaMemcpy(B_h, B_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "host memory: A" << std::endl;
        print_array(A_h, NUM);

        std::cout << "host memory: B" << std::endl;
        print_array(B_h, NUM);

        // write result to avg.txt file
        std::cout << "saving..." << std::endl;
        const char* file = "trans.txt";
        std::ofstream dst(file, std::ios::app); // Open file in append mode
        if(!dst.is_open()) {
            std::cerr << "Error: can not open the file";
        }
        dst << run_time << "\n";

        dst.close();
    }
    

    cudaFree(A_d); cudaFree(B_d);
    delete[] A_h; delete[] B_h;

    return 0;
}