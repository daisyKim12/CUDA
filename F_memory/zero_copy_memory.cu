#include <stdio.h>
#include <cuda_runtime.h>
#include "9_util/common.h"

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0e-8;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
}

void initialData(float* arr, const int N)
{
    for (int i = 0; i < N; i++)
        arr[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

__global__
void sumArrays(float* A, float* B, float* C, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

__global__
void sumArraysZeroCopy(float* A, float* B, float* C, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char** argv)
{
    // setup device
    int nDevice = 0;

    cudaGetDeviceCount(&nDevice);
    printf("Number of device: %d\n", nDevice);

    for(int i = 0; i<nDevice; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device Number: %d\n", 0);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    // setup data size of arrays
    int power = 10;
    if (argc > 1)
        power = atoi(argv[1]);
    
    int nElem = 1 << power;
    size_t nBytes = nElem * sizeof(float);

    if (power < 18) {
        printf("Array size %d power %d  nBytes %3.0f KB\n", nElem, power, nBytes/1024.f);
    }
    else {
        printf("Array size %d power %d  nBytes %3.0f MB\n", nElem, power, nBytes/(1024.f * 1024.f));
    }

    // part 1: using device memory
    printf("part 1: using device memory\n");
    printf("Array size: %d\n", nElem);
    float* h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, nBytes));

    // transfer data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // setup execution configuration
    int threads = 512;
    dim3 blocks(threads);
    dim3 grids((blocks.x + nElem - 1) / blocks.x);

    double start, finish;
    GET_TIME(start);
    sumArrays<<<grids, blocks>>>(d_A, d_B, d_C, nElem);
    GET_TIME(finish);
    printf("kernel time: \t %f sec\n", finish - start);

    // copy kernel result back to host side
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    
    // free host memory
    free(h_A);
    free(h_B);

    // part 2: using zero-copy memory for array A and B
    printf("part 2: using zero-copy memory for array A and B\n");
    printf("Array size: %d\n", nElem);

    // allocate zero-copy memory
    CUDA_CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // pass the pointer to device
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0));
    
    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // execute kernel with zero copy memory
    GET_TIME(start);
    sumArraysZeroCopy<<<grids, blocks>>>(d_A, d_B, d_C, nElem);    
    GET_TIME(finish);
    printf("kernel time: \t %f sec\n", finish - start);

    // copy kernel result back to host side
    CUDA_CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    
    // free host memory
    free(hostRef);
    free(gpuRef);

    // reset device
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}