#include <stdio.h>
#include <cuda_runtime.h>
#include "1_util/common.h"

__global__ void mb1(const float* A, float* C, int evict_offset_byte)
{
    /* 
        evict_offset = 32 : 32/sizeof(float) = 32/4 = 8 float
    */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cache_line_size = 128;
    int data_size = sizeof(float);

    // all thread read cache line
    C[idx * cache_line_size] = A[idx * cache_line_size];

    // sync all threads in a block
    __syncthreads();

    // if idx == 0 read one more after thread sync
    if(idx == 0) 
    {
        C[0] = A[evict_offset_byte/data_size];
    }

}

int main(int argc, char** argv)
{   

    // cofigure L1 cache
    /* 
        https://codeyarns.com/tech/2011-06-27-how-to-set-cache-configuration-in-cuda.html#gsc.tab=0
        There is 64 KB of memory for each multiprocessor.
        This per-multiprocessor on-chip memory is split and used for both shared memory and L1 cache.
        By default, 48 KB is used as shared memory and 16 KB as L1 cache.

        The cudaDeviceSetCacheConfig function can be used to set preference for shared memory or L1 cache globally
        for all CUDA kernels in your code and even those used by Thrust. 
        The option cudaFuncCachePreferShared prefers shared memory, 
        that is, it sets 48 KB for shared memory and 16 KB for L1 cache. 
        cudaFuncCachePreferL1 prefers L1, that is, it sets 16 KB for shared memory and 48 KB for L1 cache. 
        cudaFuncCachePreferNone uses the preference set for the device or thread.
    */

    // L1 48KB, Shared Memory 16KB
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

// checking number of cache line in L1 cache
    int n_cache_line = 384;

    /* 
        // if direct mapped 
        L1 cache size = 48 * 1024 B
        Cache line size = 128 B
        number of cache line = 384
        
        // if not need to check the cache line numeber by executing
    */

    


// checking L1 cache evict granularity
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

    for(int evict_size = 32; evict_size <= 512; evict_size = evict_size * 2) {
        // single block with number of thread matching the number of cache line
        dim3 grid(1,1,1);
        dim3 block(n_cache_line,1,1);
        for(int iter = 0; iter < 1000000; iter++) {
            GET_TIME(start);
            mb1<<<grid, block>>>(A_d, C_d, evict_size);
            CUDA_CHECK(cudaDeviceSynchronize());
            GET_TIME(finish);
            duration = finish - start;
            avg_execution_time = avg_execution_time + duration;
        }
        avg_execution_time = avg_execution_time / 1000;
        std::cout << "evict size " << evict_size << ": " << duration << std::endl;
        avg_execution_time = 0;
    }

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}