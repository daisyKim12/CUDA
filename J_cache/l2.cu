#include <stdio.h>
#include <cuda_runtime.h>
#include "1_util/common.h"

__global__ void mb2(float* A, float* C)
{
   
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx == 0) {
        C[idx] = A[idx];
        C[idx+1] = A[idx];
        C[idx] = C[idx] + A[idx];
        A[idx] = C[idx] + C[idx+1];
    }
}

int main(int argc, char** argv)
{

    // check device properties
    int nDevice = 0;

    cudaGetDeviceCount(&nDevice);
    std::cout << "Number of device: " << nDevice <<std::endl;

    if(nDevice != 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        std::cout << "   L1 cache size: " << prop.l2CacheSize;
    }


    // calculate execution time
    double start, finish, duration;
    double avg_execution_time = 0;

    int long long nx;
    int power = 22;
    nx = 1 << power;
    size_t nBytes = nx * sizeof(int);

    float *A_d, *C_d;
    CUDA_CHECK(cudaMallocManaged((void**)&A_d, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&C_d, nBytes));

    GET_TIME(start);
    dim3 grid(128,1,1);
    dim3 block(64,1,1);
    mb2<<<grid, block>>>(A_d, C_d);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    duration = finish - start;
    std::cout << "execution time: " << duration << std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}