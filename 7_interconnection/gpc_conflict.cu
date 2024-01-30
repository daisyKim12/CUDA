/* Reverse Engineering GPC Organization */

#include "9_util/common.h"

#define TPC_PER_GPC 6
#define MAX_SM 68               // SM   index 0~67
#define MAX_TPC 34              // TPC  index 0_33
#define PRINT_NUM 10

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

__device__
bool is_in(uint a, uint * arr) {
    for(int i = 0; i < TPC_PER_GPC; i++)
        if(a == arr[i] * 2)
            return true;
    return false;
}

/* 
The goal here is to evaluate a memory-intensive program that continuously accesses the L2 cache 
(and bypass L1 cache such that the interconnect is accessed). 
The code does sequential memory write access and ensures that all memory partitions 
-(and corresponding L2 cache) are accessed by the SM.

To identify which SMs are co-located within each GPC, we activate one SM in each of the 6 TPCs
- i.e., 6 SMs in total. 
Using a similar approach as before, we always activate TPC0 and then vary the TPC that is selected to run concurrently with TPC0. 
We use only one SM from each TPC. 5 other TPCs are randomly selected or made active and we run the evaluation 200 times. 
Unlike the TPC channel evaluation where we only selected 2 SMs, 6 SMs are needed for this evaluation because of the bandwidth
*/


__global__ 
void memory_write_test(int* A_h, int* B_h, int array_size, uint fixed_sm_id, uint *config_sm_id)
{
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
    if (sm_id == fixed_sm_id * 2) {
        // all thread in sm0 will write to A_h-> sequential write 
        for(int i = 0; i<amount; i++) {
            A_h[base + i] = thread_idx;
        }
    }
    // if current sm is the sm that i want to check
    else if(is_in(sm_id, config_sm_id)) {
        // all thread in config sm will write to B_h-> sequential write
        for(int i = 0; i<amount; i++) {
            B_h[base + i] = thread_idx;
        }
    }

}


int main(int argc, char** argv) {

    if( argc <= 1)
        return 0;

    int fixed_sm_id = atoi(argv[1]);

    double start, finish;
    
    int long long nx;
    int power = 14;
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

    // warmup kernel
    // memory_write_test<<<nx/8, 8>>>(A_d, B_d, nx, fixed_sm_id, 0);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // memory_write_test<<<nx/8, 8>>>(A_d, B_d, nx, fixed_sm_id, 0);
    // CUDA_CHECK(cudaDeviceSynchronize());

    uint check_sm_list[5] = {3,6,8,12,33};
    GET_TIME(start);
    memory_write_test<<<nx/8, 8>>>(A_d, B_d, nx, fixed_sm_id, check_sm_list);
    CUDA_CHECK(cudaDeviceSynchronize());
    GET_TIME(finish);
    double duration = finish - start;
    std::cout << duration << "\n";

    // time[i] = duration;
    // if(max < duration) {
    //     max = duration;
    //     max_idx = i;
    // }
    // avg = avg + duration;
    

    // for(int i = 0; i<MAX_SM; i++) {
    //     std::cout << i << ": " << time[i] << "\n";
    // }
    //     std::cout << "\n";

    avg = avg / MAX_SM;
    std::cout << "Maximum: (" << max_idx << ")" << max << "\n";
    std::cout << "Average: " << avg << "\n";
    
    CUDA_CHECK(cudaMemcpy(A_h, A_d, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_h, B_d, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());                                //must add cuda device sync to get updated array
    
    // checkArr(A_h, PRINT_NUM);
    // checkArr(B_h, PRINT_NUM);

    delete[] A_h;
    delete[] B_h;
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));

    return 0;
}