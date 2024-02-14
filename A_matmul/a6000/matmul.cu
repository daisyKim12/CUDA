#include "matmul.h"
#include <algorithm>

void ver0(float* M, float* N, float* r, const long long int width) {
    // Assuming TILE_WIDTH is defined somewhere

    // Allocate 2D arrays for submatrices
    float subtile_M[TILE_WIDTH][TILE_WIDTH];
    float subtile_N[TILE_WIDTH][TILE_WIDTH];
    
    int i, j, k, x, y, z;
    int incr = TILE_WIDTH;

    for (i = 0; i < width; i += incr) {
        for (j = 0; j < width; j += incr) {
            for (k = 0; k < width; k += incr) {
                // Load submatrices into shared memory
                for (x = 0; x < incr; x++) {
                    for (y = 0; y < incr; y++) {
                        subtile_M[x][y] = M[(i + x) * width + k + y];
                        subtile_N[x][y] = N[(k + x) * width + j + y];
                    }
                }

                // Compute submatrix multiplication
                for (x = 0; x < incr; x++) {
                    for (y = 0; y < incr; y++) {
                        float acc = 0;
                        for (z = 0; z < incr; z++) {
                            acc += subtile_M[x][z] * subtile_N[z][y];
                        }
                        // Update result matrix
                        r[(i + x) * width + j + y] += acc;
                    }
                }
            }
        }
    }
}

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
    __shared__ float sub_tile_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_tile_N[TILE_WIDTH][TILE_WIDTH * 2];

    long tx = threadIdx.x;  long ty = threadIdx.y;
    long bx = blockIdx.x;   long by = blockIdx.y;
    
    long col = tx + (TILE_WIDTH * 2) * bx;
    long row = ty + TILE_WIDTH * by;

    float acc_left = 0;
    float acc_right = 0;

    for(int idx = 0 ; idx < width/TILE_WIDTH; idx ++) {
        //load single square of M
        sub_tile_M[ty][tx] = M[row * width + idx * TILE_WIDTH + tx];

        //load left square of N
        sub_tile_N[ty][tx] = N[(idx * TILE_WIDTH + ty) * width + col];
        //load right square of N
        sub_tile_N[ty][tx + TILE_WIDTH] = N[(idx * TILE_WIDTH + ty) * width + col + TILE_WIDTH];

        __syncthreads();

        // compute
        for(int k = 0; k < TILE_WIDTH; k++) {
            acc_left += sub_tile_M[ty][k] * sub_tile_N[k][tx];
            acc_right += sub_tile_M[ty][k] * sub_tile_N[k][tx + TILE_WIDTH];

        }
      

        __syncthreads();

    }

    R[row * width + col] = acc_left;
    R[row * width + col + TILE_WIDTH] = acc_right;
}

__global__ void ver3(float* M, float* N, float* R, const long long int width)
{   
    __shared__ float sub_tile_M[TILE_WIDTH * 2][TILE_WIDTH];
    __shared__ float sub_tile_N[TILE_WIDTH][TILE_WIDTH];

    long tx = threadIdx.x;  long ty = threadIdx.y;
    long bx = blockIdx.x;   long by = blockIdx.y;
    
    long col = tx + TILE_WIDTH * bx;
    long row = ty + (TILE_WIDTH * 2) * by;

    float acc_top = 0;
    float acc_bottom = 0;

    for(int idx = 0 ; idx < width/TILE_WIDTH; idx ++) {
        //load top sqaure of M
        sub_tile_M[ty][tx] = M[row * width + idx * TILE_WIDTH + tx];
        //load bottom sqaure of M
        sub_tile_M[ty + TILE_WIDTH][tx] = M[(row + TILE_WIDTH)* width + idx * TILE_WIDTH + tx];

        // load single square of N
        sub_tile_N[ty][tx] = N[(idx * TILE_WIDTH + ty) * width + col];

        __syncthreads();

        // compute
        for(int k = 0; k < TILE_WIDTH; k++) {
            acc_top += sub_tile_M[ty][k] * sub_tile_N[k][tx];
            acc_bottom += sub_tile_M[ty + TILE_WIDTH][k] * sub_tile_N[k][tx];
        }

        __syncthreads();

    }

    R[row * width + col] = acc_top;
    R[(row + TILE_WIDTH) * width + col] = acc_bottom;

}

__global__ void ver4(float* M, float* N, float* R, const long long int width)
{   

    __shared__ float sub_tile_M[TILE_WIDTH * 4][TILE_WIDTH];
    __shared__ float sub_tile_N[TILE_WIDTH * 1][TILE_WIDTH];

    long long int tx = threadIdx.x;  long long int ty = threadIdx.y;
    long long int bx = blockIdx.x;   long long int by = blockIdx.y;
    
    long long int col = tx + TILE_WIDTH * bx;
    long long int row = ty + (TILE_WIDTH * 4) * by;

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

double run_matmul(float *M, float *N,  float *out, long long int width, int tile_width, int ver) {

    long long int height = width;
    //define block, grid dimension
    //need to make threads to cover the overall matrix
    dim3 dimGrid1(width/TILE_WIDTH, height/TILE_WIDTH, 1);
    dim3 dimGrid2(width/TILE_WIDTH/2, height/TILE_WIDTH, 1);
    dim3 dimGrid3(width/TILE_WIDTH, (height/TILE_WIDTH)/2, 1);
    dim3 dimGrid4(width/TILE_WIDTH, (height/TILE_WIDTH)/4, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    switch(ver){
        case 0:
            //ver0(M, N, out, width);
            break;
        case 1:
            ver1<<<dimGrid1, dimBlock>>>(M, N, out, width);
            break;
        case 2:
            //horizontal enlarge
            ver2<<<dimGrid2, dimBlock>>>(M, N, out, width);
            break;
        case 3:
            //vertical enlarge
            ver3<<<dimGrid3, dimBlock>>>(M, N, out, width);
            break;
        case 4:
            //vertical enlarge with 4 element per thread
            ver4<<<dimGrid4, dimBlock>>>(M, N, out, width);
            break;
        case 5: 
            break;
        default:
            break;
    }
    
    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
    }

    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;

    return sec.count();
}