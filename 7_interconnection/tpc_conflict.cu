/* Reverse Engineering TPC Organization */

#include <iostream>
#include <cuda_runtime.h>
#include "9_util/common.h"
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>

__device__ 
uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}


__global__ 
void memory_write_test(int array_size, uint config_sm_id)
{
    /* 
    sm_id: current sm
    config_sm_id: the sm number that i want to activate to see the influence to sm0 execution time
    */

    int i;
    uint sm_id = get_smid();
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    int amount = array_size / block_size;       // size of data for a single thread
    int base = amount * thread_idx;             // base index for a single thread

    // if current sm is sm0
    if (sm_id == 0) {
        // all thread in sm0 will write to arr_A-> sequential write 
        for(i = 0; i<amount; i++) {
            arr_A[base + i] = thread_idx;
        }
    }
    // if current sm is the sm that i want to check
    else if(sm_id == config_sm_id) {
        // all thread in config sm will write to arr_B-> sequential write
        for(i = 0; i<amount; i++) {
            arr_B[base + i] = thread_idx;
        }
    }

}


int main(int argc, char** argv) {

    //setup device
    int nDevice = 0;

    cudaGetDeviceCount(&nDevice);
    std::cout << "Number of device: " << nDevice << "\n";

    for(int i = 0; i<nDevice; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        std::cout << "Device Number: " << i;
        std::cout << "  Device name: " << prop.name;
        std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate;
        std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;
    }

    double start, finish;
    
    int long long nx;
    int power = 14;
    if (argc > 1)
        power = atoi(argv[1]);
    nx = 1 << power;
    size_t = nBytes = nx * sizeof(int);



    return 0;
}