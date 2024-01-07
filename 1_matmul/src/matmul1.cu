#include "_util.h"


__global__ void mat_mul(float* M, float* N, float* R, const long width)
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


__global__ void optimal_mat_mul(float* M, float* N, float* R, const long width)
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
    for(int idx = 0; idx < width/TILE_WIDTH/2; idx++) 
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



int main(void) {

    const char* M_file = "../result/large_1024_32/M.txt";
    const char* N_file = "../result/large_1024_32/N.txt";
    const char* result_file = "../result/large_1024_32/result.txt";
    const char* optimal_file = "../result/large_1024_32/optimal_result.txt";

    long width = WIDTH;
    long height = HEIGHT;
    long total_size = width * height;

    float *M_h = new float[total_size];
    float *N_h = new float[total_size];
    float *result_h = new float[total_size];
    float *result_optimal_h = new float[total_size];
    float *M_d, *N_d, *result_d, *result_optimal_d;

    init_array(M_h, total_size, 8811);
    init_array(N_h, total_size, 9700);

    // memory allocation
    cudaMalloc((void**)&M_d, total_size * sizeof(float));
    cudaMalloc((void**)&N_d, total_size * sizeof(float));
    cudaMalloc((void**)&result_d, total_size * sizeof(float));
    cudaMalloc((void**)&result_optimal_d, total_size * sizeof(float));
    // copy host memory to device
    cudaMemcpy(M_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    //define block, grid dimension
    //need to make threads to cover the overall matrix
    dim3 dimGrid(width/TILE_WIDTH, height/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    //kernel invocation
    
        /* 
        64 threads per block using maximum 32 warps total 2048 threads on fly 100% utilization
        copying two size 64 float matrix into shared memory total 512B
        */
    
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    mat_mul<<<dimGrid, dimBlock>>>(M_d, N_d, result_d, width);
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    
    std::chrono::system_clock::time_point optimize_start = std::chrono::system_clock::now();
    optimal_mat_mul<<<dimGrid, dimBlock>>>(M_d, N_d, result_optimal_d, width);
    std::chrono::duration<double>optimize_sec = std::chrono::system_clock::now() - optimize_start;

    //cudaError_t cudaErr = cudaGetLastError();
    //std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;
    
    // copy result to host memory
    cudaMemcpy(result_h, result_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_optimal_h, result_optimal_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: M" << std::endl;
    print_array(M_h, NUM);
    save_result(M_h, width, height, M_file);

    std::cout << "host memory: M" << std::endl;
    print_array(N_h, NUM);
    save_result(N_h, width, height, N_file);

    std::cout << "host memory: result_h" << std::endl;
    print_array(result_h, NUM);
    save_result(result_h, width, height, result_file);

    std::cout << "host memory: result_optimal_h" << std::endl;
    print_array(result_optimal_h, NUM);
    save_result(result_optimal_h, width, height, optimal_file);

    std::cout << "-----------------------result-----------------------" << std::endl;
    std::cout << "original mat mul : " << sec.count() <<"seconds"<< std::endl;
    std::cout << "optimize mat mul : " << optimize_sec.count() <<"seconds"<< std::endl;
    
    cudaFree(M_d); cudaFree(N_d); cudaFree(result_d);
    delete[] M_h; delete[] N_h; delete[] result_h;

    return 0;
}