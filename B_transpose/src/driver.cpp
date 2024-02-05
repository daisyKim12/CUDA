#include "_util.h"
#include "trans.h"
    
#include <chrono>
#include <stdlib.h>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>


/* 
64 threads per block using 32 warps total 2048 threads on fly 100% utilization
copying two size 64 float matrix into shared memory total 512B
*/

int main(void) {

    const char* A_file = "../result/large_2048x2048_32/A.txt";
    const char* B_file = "../result/large_2048x2048_32/B.txt";
    const char* time_file = "../result/large_2048x2048_32/time.txt";
    double *run_time = new double[10];

    long width = WIDTH;
    long height = HEIGHT;
    long total_size = width * height;

    float *A_h = new float[total_size];
    float *B_h = new float[total_size];
    float *A_d, *B_d;

    init_array(A_h, total_size, 1248);
    
    // memory allocation
    cudaMalloc((void**)&A_d, total_size * sizeof(float));
    cudaMalloc((void**)&B_d, total_size * sizeof(float));
    // copy host memory to device
    cudaMemcpy(A_d, A_h, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    for(int ver = 1; ver <=6; ver++)
    {
        std::cout << "< ver " << ver << " running ... >\n";
        run_time[ver-1] = time_transpose(A_d, B_d, width, TILE_WIDTH, ver);
        cudaError_t cudaErr = cudaGetLastError();
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;
    }

    // run_time[0] = time_transpose(A_d, B_d, width, TILE_WIDTH, 6);

    // copy result to host memory
    cudaMemcpy(B_h, B_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: A" << std::endl;
    print_array(A_h, NUM);
    save_result(A_h, width, height, A_file);

    std::cout << "host memory: B" << std::endl;
    print_array(B_h, NUM);
    save_result(B_h, width, height, B_file);

    save_run_time(run_time, 6, time_file);

    cudaFree(A_d); cudaFree(B_d);
    delete[] A_h; delete[] B_h;

    return 0;
}