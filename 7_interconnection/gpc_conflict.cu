/* Reverse Engineering GPC Organization */

#include "9_util/common.h"
#include <iostream>
#include <fstream>

#define TPC_PER_GPC 6
#define MAX_SM 68               // SM   index 0~67
#define MAX_TPC 34             // TPC  index 0_33
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
void memory_write_test(int* C_d, int long long array_size, int* tpc_list)
{    
    uint sm_id = get_smid();

    int long long row_size = array_size / 6;

    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    int amount = array_size / block_size;       // size of data for a single thread
    int base = amount * thread_idx;             // base index for a single thread

    for(int i = 0; i<TPC_PER_GPC; i++) {
        if(sm_id == tpc_list[i] * 2) {
            for(int j = 0 ; j < amount; j++) {
                C_d[i * row_size + base + j] = thread_idx;
            }
        }
    }
}


int main(int argc, char** argv) {

    // if( argc < 6){
    //     std::cout << "please enter 6 number ranging 0 to 33" << std::endl;
    //     return 0;
    // }

    // uint tpc_list_h[6];
    // for(int i = 0; i<TPC_PER_GPC; i++) {
    //     tpc_list_h[i] = atoi(argv[i+1]);
    // }
    //int tpc_list_h[TPC_PER_GPC] = {10,3,6,8,12,11};

    uint tpc_list_h[6];
    double start, finish;
    
    int long long nx;
    int power = 14;
    nx = (1 << power) * TPC_PER_GPC;
    size_t nBytes = nx * sizeof(int);


    int *C_h = new int[nx];
    int *C_d;
    int *tpc_list_d;
    int *tpc_list_unified;
    CUDA_CHECK(cudaMalloc((void**)&C_d, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&tpc_list_d, TPC_PER_GPC * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged((void**)&tpc_list_unified, TPC_PER_GPC * sizeof(int)));

    // initialData(A_h, nx);
    // initialData(B_h, nx);

    //checkArr(C_h, PRINT_NUM);

    cudaMemcpy(C_d, C_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(tpc_list_d, tpc_list_h, TPC_PER_GPC * sizeof(int), cudaMemcpyHostToDevice);

    //fix tcp id
    for(int i = 0; i < TPC_PER_GPC ; i++)
        tpc_list_unified[i] = atoi(argv[1]);
    // warmup kernel
    memory_write_test<<<nx/TPC_PER_GPC/8, 8>>>(C_d, nx, tpc_list_unified);
    CUDA_CHECK(cudaDeviceSynchronize());
    memory_write_test<<<nx/TPC_PER_GPC/8, 8>>>(C_d, nx, tpc_list_unified);
    CUDA_CHECK(cudaDeviceSynchronize());

    //double* max = new double[MAX_TPC];
    double* avg = new double[MAX_TPC];

    for(int i = 0 ; i < MAX_TPC; i++) {
        tpc_list_unified[1] = i;

        for(int a = 0 ; a < MAX_TPC; a++) {
            for(int b = 0 ; b < MAX_TPC; b++) {
                for(int c = 0 ; c < MAX_TPC; c++) {
                    for(int d = 0 ; d < MAX_TPC; d++) {

                        // set tcp_list
                        tpc_list_unified[2] = a;
                        tpc_list_unified[3] = b;
                        tpc_list_unified[4] = c;
                        tpc_list_unified[5] = d;
                        

                        GET_TIME(start);
                        memory_write_test<<<nx/TPC_PER_GPC/8, 8>>>(C_d, nx, tpc_list_unified);
                        CUDA_CHECK(cudaDeviceSynchronize());
                        GET_TIME(finish);
                        double duration = finish - start;
                        //std::cout << duration << "\n";
                        avg[i] = avg[i] + duration;
                    }
                }
            }
        }
    }

    for(int i = 0; i<MAX_TPC; i++) {
        std::cout << i << ": " << avg[i] << "\n";
    }
    std::cout << "\n";

    const char* file = "avg.txt";
    std::ofstream dst(file);
    if(!dst.is_open()) {
        std::cerr << "Error: can not open the file";
    }

    for(int i = 0; i<MAX_TPC; i++) {
        dst << i << ": " << avg[i] << "\n";
    }
    dst.close();
    


    CUDA_CHECK(cudaMemcpy(C_h, C_d, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());                                //must add cuda device sync to get updated array
    
    //checkArr(C_h, PRINT_NUM);

    delete[] C_h;
    CUDA_CHECK(cudaFree(C_d));

    return 0;
}


    // for(int i = 0; i<TPC_PER_GPC; i++){
    //     if(sm_id == tpc_list[i] * 2){
    //         switch(i)
    //         {
    //             case 0: 
    //                 for(int j =  0; j < amount; j++)
    //                     A_d[0] = 5;
    //                 break;
    //             // case 1:                            
    //             //     for(int j =  0; j < amount; j++)
    //             //         B_d[base + j] = sm_id;
    //             //     break;
    //             // case 2:
    //             //     for(int j =  0; j < amount; j++)
    //             //         C_d[base + j] = sm_id;
    //             //     break;
    //             // case 3:
    //             //     for(int j =  0; j < amount; j++)
    //             //         D_d[base + j] = sm_id;
    //             //     break;
    //             // case 4:
    //             //     for(int j =  0; j < amount; j++)
    //             //         E_d[base + j] = sm_id;
    //             //     break;
    //             // case 5:
    //             //     for(int j =  0; j < amount; j++)
    //             //         F_d[base + j] = sm_id;
    //             //     break;
    //             default:
    //                 break;
    //         }
    //     }
    // }