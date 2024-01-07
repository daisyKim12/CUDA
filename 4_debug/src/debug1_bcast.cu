#include "debug1_bcast.h"
#include <iostream>

__global__ void bcast(int* d_mem) {
    __shared__ int a;
    
    if(threadIdx.x == 0) {
        a = 123;
        // d_mem[100000] = a;
    }
    __syncthreads();
    d_mem[threadIdx.x] = a;
}

void launch_bcast(int* d_mem) {
    bcast<<<1, 1024>>>(d_mem);
}