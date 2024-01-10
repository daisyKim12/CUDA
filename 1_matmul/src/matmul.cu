#include "matmul.h"


__global__ void ver1(float* M, float* N, float* R, const long long int width)
{

    __shared__ float sub_tile_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_tile_N[TILE_WIDTH][TILE_WIDTH];

    long tx = threadIdx.x;  long ty = threadIdx.y;
    long bx = blockIdx.x;   long by = blockIdx.y;
    
    long col = tx + TILE_WIDTH * bx;
    long row = ty + TILE_WIDTH * by;
    float acc = 0;

    // idx is a stride in block
    // M must move in block group row wise dir, N must move in block group row wise dir  
    for(int idx = 0; idx < width/TILE_WIDTH; idx++) 
    {
        // load data to shared memory
        // row and col is fixed. M is row fixed, N is col fixed.
        sub_tile_M[ty][tx] = M[ row * width + idx * TILE_WIDTH + tx];
        sub_tile_N[ty][tx] = N[ (idx * TILE_WIDTH + ty) * width + col];

        // sync threads before compute
        __syncthreads();

        // compute
        for(int k = 0; k < TILE_WIDTH; k++)
            acc += sub_tile_M[ty][k] * sub_tile_N[k][tx];
        
        // sync threads before end of compute
        __syncthreads();

    }
    
    R[row * width + col] = acc;

}


__global__ void ver2(float* M, float* N, float* R, const long long int width)
{

    __shared__ float sub_tile_M[TILE_WIDTH][TILE_WIDTH * 2];
    __shared__ float sub_tile_N[TILE_WIDTH * 2][TILE_WIDTH];

    long tx = threadIdx.x;  long ty = threadIdx.y;
    long bx = blockIdx.x;   long by = blockIdx.y;
    
    long col = tx + TILE_WIDTH * bx;
    long row = ty + TILE_WIDTH * by;
    float acc = 0;

    // idx is a stride in block
    // M must move in block group row wise dir, N must move in block group row wise dir  
    for(long idx = 0; idx < width/TILE_WIDTH/2; idx++) 
    {
        // load data to shared memory
        // row and col is fixed. M is row fixed, N is col fixed.
        sub_tile_M[ty][tx] = M[ row * width + (2 * idx) * TILE_WIDTH + tx];
        sub_tile_M[ty][tx + TILE_WIDTH] = M[ row * width + (2 * idx + 1) * TILE_WIDTH + tx];
        
        sub_tile_N[ty][tx] = N[ ((2 * idx) * TILE_WIDTH + ty) * width + col];
        sub_tile_N[ty + TILE_WIDTH][tx] = N[ ((2 * idx + 1) * TILE_WIDTH + ty) * width + col];

        // sync threads before compute
        __syncthreads();

        // compute
        for(int k = 0; k < TILE_WIDTH*2; k++)
        {
            acc += sub_tile_M[ty][k] * sub_tile_N[k][tx];
        }
        
        // sync threads before end of compute
        __syncthreads();

    }

    
    R[row * width + col] = acc;
    
}


double run_matmul(float *M, float *N,  float *out, long long int width, int tile_width, int ver) {

    //define block, grid dimension
    //need to make threads to cover the overall matrix
    dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    switch(ver){
        case 1:
            ver1<<<dimGrid, dimBlock>>>(M, N, out, width);
            break;
        case 2:
            ver2<<<dimGrid, dimBlock>>>(M, N, out, width);
            break;
        case 3:

            break;
        case 4:

            break;
        case 5:

            break;
        default:
            break;
    }
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;

    return sec.count();
}