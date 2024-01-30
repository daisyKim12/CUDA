/* Reverse Engineering TPC Organization */

#include "9_util/common.h"

#define MAX_SM 68

void initialData(int* in, const int size)
{
    for (int i = 0; i < size; i++)
        in[i] = rand() & 0xFF;
}


__device__ 
uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}


__global__ 
void memory_write_test(int* arr_A, int* arr_B, int array_size, uint config_sm_id)
{
    /* 
    sm_id: current sm
    config_sm_id: the sm number that i want to activate to see the influence to sm0 execution time
    */

    int i;
    uint sm_id = get_smid();
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;

    // int amount = array_size / block_size;       // size of data for a single thread
    // int base = amount * thread_idx;             // base index for a single thread
    int amount = 200;
    int base = 0;

    // if current sm is sm0
    if (sm_id == 10) {
        // all thread in sm0 will write to arr_A-> sequential write 
        for(i = 0; i<amount; i++) {
            arr_A[base + i] = thread_idx;
        }
    }
    // if current sm is the sm that i want to check
    else if(sm_id == config_sm_id) {
        // all thread in config sm will write to arr_B-> sequential write
        for(i = 0; i<amount; i++) {
            arr_B[base + i] = thread_idx;
        }
    }

}

__global__ void kern(int *sm){

   if (threadIdx.x==0)

      sm[blockIdx.x]=get_smid();

}

int main(int argc, char** argv) {

    //setup device
    int nDevice = 0;

    cudaGetDeviceCount(&nDevice);
    std::cout << "Number of device: " << nDevice << "\n";

    for(int i = 0; i<1; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        std::cout << "Device Number: " << i << "\n";
        std::cout << "  Device name: " << prop.name << "\n";
        std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n";
        std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << "\n";
        std::cout << "Turing Arch\n";
        std::cout << "  #.SM: 68, TPC: 2SM, GPC: 6TPC" << "\n\n";
    }

    double start, finish;
    
    int long long nx;
    int power = 14;
    // if (argc > 1)
    //     power = atoi(argv[1]);
    nx = 1 << power;
    size_t nBytes = nx * sizeof(int);

    int *arr_A = new int[nx];
    int *arr_B = new int[nx];
    int *arr_A_d, *arr_B_d;
    // CUDA_CHECK(cudaMallocHost((void**)&arr_A, nBytes));
    // CUDA_CHECK(cudaMallocHost((void**)&arr_B, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&arr_A_d, nBytes));
    CUDA_CHECK(cudaMalloc((void**)&arr_B_d, nBytes));
    
    //initialData(arr_A, nx);
    //initialData(arr_B, nx);

    cudaMemcpy(arr_A_d, arr_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(arr_B_d, arr_B, nBytes, cudaMemcpyHostToDevice);

    for(int i = 0; i< 5; i++) {
        std::cout << arr_A[i] << " ";
    }
    std::cout << "\n";
    for(int i = 0; i< 5; i++) {
        std::cout << arr_B[i] << " ";
    }
    std::cout << "\n";

    int threads = 32;
    dim3 blocks(threads,0);
    dim3 grids(nx/threads, 0);
    
    // std::cout << "--------------NO Contention-------------\n";
    // GET_TIME(start);
    // memory_write_test<<<grids, blocks>>>(arr_A_d, arr_B_d, nx, 100);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // GET_TIME(finish);
    // std::cout << "sm0 write exe time: " << finish-start << "\n\n";

    int compare = 0;
    double *time = new double[MAX_SM];
    if (argc > 1)
    {   
        compare = atoi(argv[1]);
        std::cout << "--------------Contention With sm" << compare << "-------------\n";
        GET_TIME(start);
        memory_write_test<<<grids, blocks>>>(arr_A_d, arr_B_d, nx, compare);
        CUDA_CHECK(cudaDeviceSynchronize());
        GET_TIME(finish);
        std::cout << "sm0 write exe time: " << finish-start << "\n\n";
    }
    else
    {   
        //warmup kernel
        memory_write_test<<<grids, blocks>>>(arr_A_d, arr_B_d, nx, 0);

        double max = -1;
        double avg = 0;
        int max_idx = -1;
        for(int i = 0; i<MAX_SM; i++) {
            GET_TIME(start);
            memory_write_test<<<grids, blocks>>>(arr_A_d, arr_B_d, nx, i);
            CUDA_CHECK(cudaDeviceSynchronize());
            GET_TIME(finish);
            double duration = finish - start;
            time[i] = duration;
            if( max < duration) {
                max = duration;
                max_idx = i;
            }
            avg = avg + duration;
        }
        for(int i = 0; i<MAX_SM; i++) {
            std::cout << i << ": " << time[i] << "\n";
        }
        avg = avg / MAX_SM;
        std::cout << "\nMaximum: (" << max_idx << ")" << max << "\n";
        std::cout << "Average: " << avg << "\n";
    }
    
    CUDA_CHECK(cudaMemcpy(arr_A, arr_A_d, nBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(arr_B, arr_B_d, nBytes, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i< 5; i++) {
        std::cout << arr_A[i] << " ";
    }
    std::cout << "\n";
    for(int i = 0; i< 5; i++) {
        std::cout << arr_B[i] << " ";
    }
    std::cout << "\n";

    CUDA_CHECK(cudaFreeHost(arr_A));
    CUDA_CHECK(cudaFreeHost(arr_B));
    CUDA_CHECK(cudaFree(arr_A_d));
    CUDA_CHECK(cudaFree(arr_B_d));

    return 0;
}

// int main(int argc, char *argv[]){

//     int N = atoi(argv[1]);

//     int *sm, *sm_d;
//     sm = (int *) malloc(N*sizeof(*sm));
//     cudaMalloc((void**)&sm_d,N*sizeof(*sm_d));

//     kern<<<N,N>>>( sm_d);

//     cudaMemcpy(sm, sm_d, N*sizeof(int), cudaMemcpyDeviceToHost);

//     for (int i=0;i<N;i++)
//         printf("%d %d\n",i,sm[i]);

//     return 0;
// }