#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <iostream>

const int N = 1 << 20;

__global__ void kernel_A(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i)) * 2;
    }
}

__global__ void kernel_B(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i)) / 3;
    }
}

int main()
{
    const int num_streams = 8;
    const int num_arr = 8*2;

    cudaStream_t streams[num_streams];
    float *data[num_arr];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[2*i], N * sizeof(float));
        cudaMalloc(&data[2*i+1], N * sizeof(float));


        kernel_A <<<1, 1, 0, streams[i]>>>(data[2 * i], N);
        kernel_B <<<1, 1, 0, streams[i]>>>(data[2 * i + 1], N);
    }

    cudaDeviceReset();

    return 0;
}

