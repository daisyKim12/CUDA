#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "debug1_bcast.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define THREADS 1024

int main(int argc, char* argv[]) {
    int* d_mem;
    cudaMalloc((void**)&d_mem, THREADS*sizeof(int));
    launch_bcast(d_mem);
    int* h_mem = new int[THREADS];
    cudaMemcpy(h_mem, d_mem, THREADS*sizeof(int), cudaMemcpyDeviceToHost);
    int cnt = 0;
    for(int i = 0; i<THREADS; i++) {
        if(h_mem[i] != 123) cnt++;
    }
    printf("cnt: %d\n", cnt);

    return 0;
}