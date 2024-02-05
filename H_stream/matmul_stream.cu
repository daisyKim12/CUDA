/* 
stream order does matter
*/
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <iostream>

#define NUM 10
#define SKEW 1

#define TILE_WIDTH 16
#define WIDTH 2048     //65536 main memory shortage

__device__ 
uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__device__
uint get_nsmid(void){

    uint ret;

    asm("mov.u32 %0, %%nsmid;" : "=r"(ret));

    return ret;

}

__global__ void matmul(float* M, float* N, float* R, const long long int width)
{   
    // check sm id
    if(threadIdx.x == 0){
        
        //printf("number of sm allocated for kernel: %d \n", get_nsmid());
        //printf("%d ", get_smid());
    }

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

// initializing array size N with random float between 0 and 100
void init_array(float *arr, long long int N, int seed) {
    
    srand(seed);

    for(long long int i = 0; i<N; i++){
        arr[i] = ((float)rand() / RAND_MAX) * 100.0;
    }
}

// display partial array size m
void print_array(float *arr, int m) {
    for(int i = 0; i < m; i++) {
        std::cout << arr[i]<< " ";
    }
    std::cout << "...\n" << std::endl;
}

int main(int argc, char *argv[]) {

    long long int width = WIDTH;
    // if(argc > 1) {
    //     width = (1 << atoi(argv[1]));
    // }
    long long int tile_width = TILE_WIDTH;
    long long int total_size = width * width;

    float *M_h = new float[total_size];
    float *N_h = new float[total_size];
    float *result_1_h = new float[total_size];
    float *result_2_h = new float[total_size];
    float *result_3_h = new float[total_size];
    float *result_4_h = new float[total_size];


    float *M_1_d, *N_1_d, *result_1_d;
    float *M_2_d, *N_2_d, *result_2_d;
    float *M_3_d, *N_3_d, *result_3_d;
    float *M_4_d, *N_4_d, *result_4_d;

    init_array(M_h, total_size, 8811);
    init_array(N_h, total_size, 9700);

    std::cout << "------------------stream-------------------" << std::endl;

    // make stream
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // memory allocation
    cudaMallocHost((void**)&M_1_d, total_size * sizeof(float));
    cudaMallocHost((void**)&N_1_d, total_size * sizeof(float));
    cudaMallocHost((void**)&result_1_d, total_size * sizeof(float));

    cudaMallocHost((void**)&M_2_d, total_size * sizeof(float));
    cudaMallocHost((void**)&N_2_d, total_size * sizeof(float));
    cudaMallocHost((void**)&result_2_d, total_size * sizeof(float));

    cudaMallocHost((void**)&M_3_d, total_size * sizeof(float));
    cudaMallocHost((void**)&N_3_d, total_size * sizeof(float));
    cudaMallocHost((void**)&result_3_d, total_size * sizeof(float));

    cudaMallocHost((void**)&M_4_d, total_size * sizeof(float));
    cudaMallocHost((void**)&N_4_d, total_size * sizeof(float));
    cudaMallocHost((void**)&result_4_d, total_size * sizeof(float));

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    dim3 dimGrid(width/tile_width, (width/tile_width)/4, 1);
    dim3 dimBlock(tile_width, tile_width, 1);

    // memory copy
    cudaMemcpyAsync(M_1_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_1_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    matmul<<<dimGrid, dimBlock, 0, stream1>>>(M_1_d, N_1_d, result_1_d, width);
    cudaMemcpyAsync(result_1_h, result_1_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpyAsync(M_2_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_2_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    matmul<<<dimGrid, dimBlock, 0, stream2>>>(M_2_d, N_2_d, result_2_d, width);
    cudaMemcpyAsync(result_2_h, result_2_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpyAsync(M_3_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_3_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    matmul<<<dimGrid, dimBlock, 0, stream3>>>(M_3_d, N_3_d, result_3_d, width);
    cudaMemcpyAsync(result_3_h, result_3_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpyAsync(M_4_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_4_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    matmul<<<dimGrid, dimBlock, 0, stream4>>>(M_4_d, N_4_d, result_4_d, width);
    cudaMemcpyAsync(result_4_h, result_4_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // kernel launch
    // cudaMemcpyAsync(M_1_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(N_1_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(M_2_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(N_2_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(M_3_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(N_3_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(M_4_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(N_4_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    
    // matmul<<<dimGrid, dimBlock, 0, stream1>>>(M_1_d, N_1_d, result_1_d, width);
    // matmul<<<dimGrid, dimBlock, 0, stream2>>>(M_2_d, N_2_d, result_2_d, width);
    // matmul<<<dimGrid, dimBlock, 0, stream3>>>(M_3_d, N_3_d, result_3_d, width);
    // matmul<<<dimGrid, dimBlock, 0, stream4>>>(M_4_d, N_4_d, result_4_d, width);

    // cudaMemcpyAsync(result_1_h, result_1_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpyAsync(result_2_h, result_2_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpyAsync(result_3_h, result_3_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpyAsync(result_4_h, result_4_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);





    // memory copy





    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    
    // total time
    std::cout << "\ntotal time: " << sec.count() << "seconds" << std::endl;

    cudaError cudaErr = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl; 
    std::cout << "check result" << std::endl;
    print_array(result_1_h, NUM);

    
    // cudaFree(M_1_d); cudaFree(N_1_d); cudaFree(result_1_d);
    // delete[] M_h; delete[] N_h; delete[] result_1_h;

    return 0;
}