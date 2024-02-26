#include <stdio.h>
#include <cuda_runtime.h>
#include "1_util/common.h"

__global__ void mb1(const float* A, float* C, int stride)
{
    /* 
        stride = 1, each thread will access a different cache line
        stride = 8, each consecutive group of eight threads will access the same cache line
    */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    C[((idx / stride) * 32) + (idx % stride)] = A[((idx / stride) * 32) + (idx % stride)];
}

int main(int argc, char** argv)
{

    // calculate execution time
    double start, finish, duration;
    double avg_execution_time = 0;

    int long long nx;
    int power = 22;
    nx = 1 << power;
    size_t nBytes = nx * sizeof(int);
    int stride;
    if(argc > 1)
        stride = atoi(argv[1]);

    float *A_d, *C_d;
    CUDA_CHECK(cudaMallocManaged((void**)&A_d, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&C_d, nBytes));

    if(argc > 1) {
        std::cout << "stride : " << stride << std::endl;
        GET_TIME(start);
        dim3 grid(128,1,1);
        dim3 block(64,1,1);
        mb1<<<grid, block>>>(A_d, C_d, stride);
        CUDA_CHECK(cudaDeviceSynchronize());
        GET_TIME(finish);
        duration = finish - start;
        std::cout << "execution time: " << duration << std::endl;
    }
    else {
        for(stride = 1; stride <= 32; stride = stride *2) {
            for(int iter = 0; iter < 1000; iter++) {  
                GET_TIME(start);
                dim3 grid(128,1,1);
                dim3 block(64,1,1);
                mb1<<<grid, block>>>(A_d, C_d, stride);
                CUDA_CHECK(cudaDeviceSynchronize());
                GET_TIME(finish);
                duration = finish - start;
                avg_execution_time += duration;
            }
            std::cout << "stride " << stride << " : " << avg_execution_time << std::endl;
            avg_execution_time = 0;
        }
    }


    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}