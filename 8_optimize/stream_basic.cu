#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <iostream>

const int N = 1 << 20;

__device__ 
uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__global__ void kernel(float *x, int n, int* sm)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    int* sm;
    cudaMallocManaged((void**)&sm, 64*8*sizeof(int));

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N, sm);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0, sm);
    }

    cudaDeviceSynchronize();
    for(int i = 0; i<64*8; i++) {
        printf("threadblock %d: %d\n", i, sm[i]);
    }
    cudaDeviceReset();

    return 0;
}