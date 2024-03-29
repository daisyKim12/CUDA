/* 
    Managed Memory를 사용하면 초기화에 소요되는 시간이 훨씬 더 크다는 것을 볼 수 있습니다.
    할당되는 메모리는 처음에는 GPU에서 할당되지만, 초기값을 설정하는 것은 CPU에서 이루어지기 때문에 CPU에서 먼저 참조됩니다.
    이를 위해서 초기화를 수행하기 전에 시스템이 할당된 메모리 내용을 device에서 host로 전달해주어야 하는데, 
    이는 manual 코드에서는 수행되지 않는 것이며 이러한 동작 때문에 조금 더 시간이 더 소요됩니다.

    Host에서 수행되는 matrix sum 함수가 실행될 때, 이미 전체 matrix가 이미 CPU에 상주하고 있기 때문에 실행 시간은 유사합니다.
    그리고, 워밍업으로 커널이 한 번 수행되는데, 이때 사용되는 matrix가 device로 다시 마이그레이션합니다.
    따라서 실제 수행 시간을 측정하는데 사용되는 커널이 실행될 때에는 해당 matrix 데이터가 GPU에 존재하는 상태입니다.
    만약 워밍업 커널이 실행되지 않는다면 managed memory를 사용하는 커널의 실행 속도는 훨씬 더 느려질 것입니다.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include "1_util/common.h"
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>

void initialData(float* in, const int size)
{
    for (int i = 0; i < size; i++)
        in[i] = (rand() & 0xFF) / 10.f;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float* ia = A;
    float* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int size)
{
    double epsilon = 1.0e-8;

    for (int i = 0; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
  
}


__global__
void sumMatrixOnGPU(float* A, float* B, float* C, const int width, const int height)
{
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = row * width + col;

    if (col < width && row < height)
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

    // setup size of matrix
    int nx, ny;
    int power = 12;
    if (argc > 1)
        power = atoi(argv[1]);
    nx = ny = 1 << power;

    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);

// using cudaMemcpy
    printf("part 1: using cudaMemcpy\n");
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float *M_d, *N_d,  *S_d;
    float *M_h = new float[nBytes];
    float *N_h = new float[nBytes];
    float *S_h = new float[nBytes];
    CUDA_CHECK(cudaMalloc((void**)&M_d, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&N_d, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&S_d, nBytes));
    
    double start, finish;
    GET_TIME(start);
    initialData(M_h, nxy);
    initialData(N_h, nxy);
    GET_TIME(finish);
    printf("initialization: \t %f sec\n", finish - start);

    GET_TIME(start);
    cudaMemcpy(M_d, M_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, nBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    printf("memcpy host -> dev: \t %f sec\n", finish - start);

    GET_TIME(start);
    sumMatrixOnHost(M_h, N_h, S_h, nx, ny);
    GET_TIME(finish);
    printf("sumMatrix on host:\t %f sec\n", finish - start);

    // invode kernel at host side
    int dimX = 32;
    int dimY = 32;
    dim3 blocks(dimX, dimY);
    dim3 grids((nx + blocks.x - 1) / blocks.x, (ny + blocks.y - 1) / blocks.y);
    
    // warm-up kernel
    sumMatrixOnGPU<<<grids, blocks>>>(M_d, N_d, S_d, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    GET_TIME(start);
    sumMatrixOnGPU<<<grids, blocks>>>(M_d, N_d, S_d, nx, ny);
    GET_TIME(finish);
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>>\n", finish-start, grids.x, grids.y, blocks.x, blocks.y);

    GET_TIME(start);
    CUDA_CHECK(cudaMemcpy(S_h, S_d, nBytes, cudaMemcpyDeviceToHost))
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    printf("memcpy host -> dev: \t %f sec\n", finish - start);

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(S_d));
    delete[] M_h;
    delete[] N_h;
    delete[] S_h;

// using Unified memory
    printf("part 2: using unified memory\n");
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    // cudaMallocManaged will allocate on unified memory
    float *A, *B, *hostRef, *gpuRef;
    CUDA_CHECK(cudaMallocManaged((void**)&A, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&B, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&hostRef, nBytes));
    CUDA_CHECK(cudaMallocManaged((void**)&gpuRef, nBytes));

    // initialize time
    GET_TIME(start);
    initialData(A, nxy);
    initialData(B, nxy);
    GET_TIME(finish);
    printf("initialization: \t %f sec\n", finish - start);

    //memset함수는 어떤 메모리의 시작점부터 연속된 범위를 어떤 값으로(바이트 단위) 모두 지정하고 싶을 때 사용하는 함수이다.
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result check
    GET_TIME(start);
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    GET_TIME(finish);
    printf("sumMatrix on host:\t %f sec\n", finish - start);

    // warm-up kernel
    sumMatrixOnGPU<<<grids, blocks>>>(A, B, gpuRef, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    GET_TIME(start);
    sumMatrixOnGPU<<<grids, blocks>>>(A, B, gpuRef, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>>\n", finish-start, grids.x, grids.y, blocks.x, blocks.y);

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(hostRef));
    CUDA_CHECK(cudaFree(gpuRef));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

