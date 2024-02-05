#include <stdio.h>
#include <cuda_runtime.h>
#include "common.h"
#include "_util.h"
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>

#define PRINT 10
#define SKEW 1

#define TILE_WIDTH 16
#define WIDTH 8192

__global__ void matmul(float* M, float* N, float* R, const long long int width)
{   
    __shared__ float sub_tile_M[TILE_WIDTH * 4][TILE_WIDTH];
    __shared__ float sub_tile_N[TILE_WIDTH][TILE_WIDTH];

    long tx = threadIdx.x;  long ty = threadIdx.y;
    long bx = blockIdx.x;   long by = blockIdx.y;
    
    long col = tx + TILE_WIDTH * bx;
    long row = ty + (TILE_WIDTH * 4) * by;

    float acc_1 = 0;
    float acc_2 = 0;
    float acc_3 = 0;
    float acc_4 = 0;


    for(int idx = 0 ; idx < width/TILE_WIDTH; idx ++) {
        //load top sqaure of M
        sub_tile_M[ty][tx] = M[row * width + idx * TILE_WIDTH + tx];
        sub_tile_M[ty + TILE_WIDTH][tx] = M[(row + TILE_WIDTH)* width + idx * TILE_WIDTH + tx];
        sub_tile_M[ty + TILE_WIDTH*2][tx] = M[(row + TILE_WIDTH*2)* width + idx * TILE_WIDTH + tx];
        sub_tile_M[ty + TILE_WIDTH*3][tx] = M[(row + TILE_WIDTH*3)* width + idx * TILE_WIDTH + tx];


        // load single square of N
        sub_tile_N[ty][tx] = N[(idx * TILE_WIDTH + ty) * width + col];

        __syncthreads();

        // compute
        for(int k = 0; k < TILE_WIDTH; k++) {
            acc_1 += sub_tile_M[ty][k] * sub_tile_N[k][tx];
            acc_2 += sub_tile_M[ty + TILE_WIDTH][k] * sub_tile_N[k][tx];
            acc_3 += sub_tile_M[ty + TILE_WIDTH*2][k] * sub_tile_N[k][tx];
            acc_4 += sub_tile_M[ty + TILE_WIDTH*3][k] * sub_tile_N[k][tx];

        }
        

        __syncthreads();

    }

    R[row * width + col] = acc_1;
    R[(row + TILE_WIDTH) * width + col] = acc_2;
    R[(row + TILE_WIDTH * 2) * width + col] = acc_3;
    R[(row + TILE_WIDTH * 3) * width + col] = acc_4;


}

int main(int argc, char** argv) {

    long long int width = WIDTH;
    long long int total_size = width * width;

// part 1: using memcpy
    printf("part 1: using memcpy\n");

    double start, finish;

    GET_TIME(start);
    float *M_h = new float[total_size];
    float *N_h = new float[total_size];
    float *result_h = new float[total_size];
    float *M_d, *N_d, *result_d;
    cudaMalloc((void**)&M_d, total_size * sizeof(float));
    cudaMalloc((void**)&N_d, total_size * sizeof(float));
    cudaMalloc((void**)&result_d, total_size * sizeof(float));
    GET_TIME(finish);
    printf("allocating host memory:\t\t %f sec\n", finish - start);

    GET_TIME(start);
    init_array(M_h, total_size, 8811);
    init_array(N_h, total_size, 9700);
    GET_TIME(finish);
    printf("initializing on host:\t\t %f sec\n", finish - start);


    GET_TIME(start);
    cudaMemcpy(M_d, M_h, total_size *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, total_size *sizeof(float), cudaMemcpyHostToDevice);
    GET_TIME(finish);
    //printf("copying memory:\t\t %f sec\n", finish - start);

    dim3 dimGrid(width/TILE_WIDTH, (width/TILE_WIDTH)/4, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    
    GET_TIME(start);
    matmul<<<dimGrid, dimBlock>>>(M_d, N_d, result_d, width);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("kernel run time:\t\t %f sec\n", finish - start);

    cudaMemcpy(result_h, result_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(result_h, PRINT);

    cudaFree(M_d); cudaFree(N_d); cudaFree(result_d);
    delete[] M_h; delete[] N_h; delete[] result_h;

//part 2: using unified memory
    printf("part 2: using unified memory\n");

    float *p, *q, *r;

    //Attach buffers 'p' and 'q' to CPU and buffer 'r' to GPU
    GET_TIME(start);
    cudaMallocManaged(&p, total_size * sizeof(float), cudaMemAttachHost);
    cudaMallocManaged(&q, total_size * sizeof(float), cudaMemAttachHost);
    cudaMallocManaged(&r, total_size * sizeof(float));
    GET_TIME(finish);
    printf("allocating memory:\t\t %f sec\n", finish - start);

    GET_TIME(start);
    init_array(p, total_size, 8811);
    init_array(q, total_size, 9700);
    GET_TIME(finish);
    printf("initializing on host:\t\t %f sec\n", finish - start);

    cudaStreamAttachMemAsync(NULL, p, 0, cudaMemAttachGlobal);
    cudaStreamAttachMemAsync(NULL, q, 0, cudaMemAttachGlobal);
    
    GET_TIME(start);
    matmul<<<dimGrid, dimBlock>>>(p, q, r, width);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("kernel run time:\t\t %f sec\n", finish - start);

    cudaStreamAttachMemAsync(NULL, r, 0, cudaMemAttachHost);
    cudaStreamSynchronize(NULL);

    print_array(r, PRINT);

    cudaFree(p); cudaFree(q); cudaFree(r);

//part 3: using pinned memory
    printf("part 3: using pinned memory\n");

    float *x, *y, *gpuRef;
    GET_TIME(start);
    CUDA_CHECK(cudaMallocHost((void**)&x, total_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&y, total_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged((void**)&gpuRef, total_size * sizeof(float)));
    GET_TIME(finish);
    printf("allocating memory:\t\t %f sec\n", finish - start);

    GET_TIME(start);
    init_array(x, total_size, 8811);
    init_array(y, total_size, 9700);
    GET_TIME(finish);
    printf("initializing on host:\t\t %f sec\n", finish - start);

    memset(gpuRef, 0, total_size * sizeof(float));
    
    GET_TIME(start);
    matmul<<<dimGrid, dimBlock>>>(x, y, gpuRef, width);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    printf("kernel run time:\t\t %f sec\n", finish - start);

    print_array(gpuRef, PRINT);

    cudaFree(x); cudaFree(y); cudaFree(gpuRef);
    return 0;

}