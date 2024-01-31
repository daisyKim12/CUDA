/* Reverse Engineering TPC Organization */

#include "9_util/common.h"

#define MAX_SM 68
#define PRINT_NUM 16384

void initialData(int* in, const int size)
{
    for (int i = 0; i < size; i++)
        in[i] = rand() & 0xFF;
}

void checkArr(int* in, const int num)
{
    for(int i = 0; i< num; i++) {
        std::cout << in[i] << " ";
    }
    std::cout << "\n";
}


__device__ 
uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__device__ void kern(int *sm){

   if (threadIdx.x==0)
      sm[blockIdx.x]=get_smid();

}

/* 
The goal here is to evaluate a memory-intensive program that continuously accesses the L2 cache 
(and bypass L1 cache such that the interconnect is accessed). 
The code does sequential memory write access and ensures that all memory partitions 
-(and corresponding L2 cache) are accessed by the SM.

We execute this synthetic code concurrently on SM0 and one other SM in the GPU, 
i.e., only two SMs are active.
*/

__device__ int flag_A  = 0;
__device__ int flag_B  = 0;


__global__ 
void memory_write_test(int* A_h, int* B_h, int array_size, uint fixed_sm_id, uint config_sm_id, int* sm_d)
{

    kern(sm_d);

    /* 
    sm_id: current sm
    config_sm_id: the sm number that i want to activate to see the influence to sm0 execution time
    */

    uint sm_id = get_smid();
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    int amount = array_size / block_size;       // size of data for a single thread
    int base = amount * thread_idx;             // base index for a single thread

    // if current sm is sm0
    if (sm_id == fixed_sm_id) {
        // all thread in sm0 will write to A_h-> sequential write 
        for(int i = 0; i<amount; i++) {
            A_h[base + i] = thread_idx;
            //A_h[base + i] = sm_id;
        }
        flag_A = 1;
    }
    // if current sm is the sm that i want to check
    else if(sm_id == config_sm_id) {
        // all thread in config sm will write to B_h-> sequential write
        for(int i = 0; i<amount; i++) {
            B_h[base + i] = thread_idx;
            //B_h[base + i] = sm_id;
        }
        flag_B = 1;
    }
    else {
        for (int a = 0; a < 13684; a++)
            for (int b = 0; b < 13684; b++);
                for (int b = 0; b < 13684; b++);
    }
    // else {
    //     while (1) {
    //         if(flag_A == 1)
    //             break;  
    //         for (int temp = 0; temp < 13684; temp++);
    //     }
    //     flag_A = 0;
    //     flag_B = 0;
    // }

}


int main(int argc, char** argv) {

    if( argc <= 1)
        return 0;

    int fixed_sm_id = atoi(argv[1]);

    double start, finish;
    
    int long long nx;
    int power = 22;
    nx = 1 << power;
    size_t nBytes = nx * sizeof(int);

    int *A_h = new int[nx];
    int *B_h = new int[nx];
    int *A_d, *B_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&B_d, nBytes));
    
    // initialData(A_h, nx);
    // initialData(B_h, nx);

    // checkArr(A_h, PRINT_NUM);
    // checkArr(B_h, PRINT_NUM);

    cudaMemcpy(A_d, A_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, nBytes, cudaMemcpyHostToDevice);

    double *time = new double[MAX_SM];

    double max = -1;
    double avg = 0;
    int max_idx = -1;

    //check sm
    int *sm_h, *sm_d;
    sm_h= (int *)malloc(MAX_SM*sizeof(int));
    cudaMalloc((void**)&sm_d,MAX_SM*sizeof(*sm_d));

    // warmup kernel
    // memory_write_test<<<nx/8, 8>>>(A_d, B_d, nx, fixed_sm_id, 0);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // memory_write_test<<<nx/8, 8>>>(A_d, B_d, nx, fixed_sm_id, 0);
    // CUDA_CHECK(cudaDeviceSynchronize());

    for(int i = 0; i<MAX_SM; i++) {
        GET_TIME(start);
        memory_write_test<<<nx/8, 8>>>(A_d, B_d, nx, fixed_sm_id, i, sm_d);
        CUDA_CHECK(cudaDeviceSynchronize());
        GET_TIME(finish);
        double duration = finish - start;
        time[i] = duration;
        if(max < duration) {
            max = duration;
            max_idx = i;
        }
        avg = avg + duration;
    }

    for(int i = 0; i<MAX_SM; i++) {
        std::cout << i << ", " << time[i] << "\n";
    }
        std::cout << "\n";

    avg = avg / MAX_SM;
    std::cout << "Maximum: (" << max_idx << ")" << max << "\n";
    std::cout << "Average: " << avg << "\n";
    
    CUDA_CHECK(cudaMemcpy(A_h, A_d, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_h, B_d, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sm_h, sm_d, MAX_SM*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());                                //must add cuda device sync to get updated array
    
    // checkArr(A_h, PRINT_NUM);
    // checkArr(B_h, PRINT_NUM);

    for (int i=0;i<MAX_SM;i++)
        printf("thread block %d -> %d\n",i,sm_h[i]);

    delete[] A_h;
    delete[] B_h;
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));

    return 0;
}