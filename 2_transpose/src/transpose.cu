#include "trans.h"
#include <chrono>
#include <iostream>

__global__ void transpose_ver1(float *A, float *B, long width) {
    /* width size thread group reads a single row vecter 
    and accesses B in col wise way saving the value */
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int col = tid % width;
    int row = tid / width;

    if(col < width && row < width)
        B[col* width + row] = A[row * width + col];
}

__global__ void transpose_ver2(float *A, float *B, long width) {
    /* reading in a col wise direction will be faster than
    reading in a row wise direction */
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int col = tid % width;
    int row = tid / width;

    if(col < width && row < width)
        B[row * width + col] = A[col * width + row];
}

__global__ void transpose_ver3(float *A, float *B, long width) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < width && row < width)
        B[col * width + row] = A[row * width + col];
}

__global__ void transpose_ver4(float *A, float *B, long width) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col < width && row < width)
        B[row * width + col] = A[col * width + row];
}

__global__ void transpose_ver5(float *A, float *B, long width) {
    
    __shared__ float buffer[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;   int ty = threadIdx.y;
    int bx = blockIdx.x;    int by = blockIdx.y;
    int bd = blockDim.x;

    int col = tx + bd * bx;
    int row = ty + bd * by;

    if(col < width && row < width)
        buffer[ty][tx] = A[row * width + col];

    __syncthreads();

    if(col < width && row < width)
        B[col * width + row] = buffer[ty][tx];

    __syncthreads();
    
}

__global__ void transpose_ver6(float *A, float *B, long width) {
    /* with version 5 there is a 32 stride pattern memory access
    causing a bank conflict, so padding is need for performance optimization */
    
    __shared__ float buffer[1024 + SKEW * 32];
    
    int tx = threadIdx.x;   int ty = threadIdx.y;
    int bx = blockIdx.x;    int by = blockIdx.y;
    int bd = blockDim.x;

    int col = tx + bd * bx;
    int row = ty + bd * by;

    if(col < width && row < width)
        buffer[ty * (TILE_WIDTH + SKEW) + tx] = A[row * width + col];

    __syncthreads();

    if(col < width && row < width)
        B[col * width + row] = buffer[ty * (TILE_WIDTH + SKEW) + tx ];

    __syncthreads();

}

double time_transpose(float *A, float *B, long width, long tile_width, int ver) {
    
    // define 1D block, grid dimension
    dim3 dimGrid_1D(width*width/tile_width, 1, 1);
    dim3 dimBlock_1D(tile_width, 1, 1);

    // define 2D block, grid dimension
    dim3 dimGrid_2D(width/tile_width, width/tile_width, 1);
    dim3 dimBlock_2D(tile_width, tile_width, 1); 

    // check kernel run time
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    //kernel invocation    
    switch(ver){
        case 1:
            transpose_ver1<<<dimGrid_1D, dimBlock_1D>>>(A, B, width);
            break;
        case 2:
            transpose_ver2<<<dimGrid_1D, dimBlock_1D>>>(A, B, width);
            break;
        case 3:
            transpose_ver3<<<dimGrid_2D, dimBlock_2D>>>(A, B, width);
            break;
        case 4:
            transpose_ver4<<<dimGrid_2D, dimBlock_2D>>>(A, B, width);
            break; 
        case 5:
            transpose_ver5<<<dimGrid_2D, dimBlock_2D>>>(A, B, width);
            break;
        case 6:
            transpose_ver6<<<dimGrid_2D, dimBlock_2D>>>(A, B, width);
            break;
        default:
            break;
    }
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    std::cout << "ver " << ver <<" : " << sec.count() <<"seconds"<< std::endl;

    return sec.count();
}