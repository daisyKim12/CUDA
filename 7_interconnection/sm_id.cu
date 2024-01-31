#include "9_util/common.h"

__device__ 
uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}
__global__ void kern(int *sm){

   if (threadIdx.x==0)
      sm[blockIdx.x]=get_smid();

}

int main(int argc, char *argv[]){

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
    }

    int N = atoi(argv[1]);

    // sm_h for host memory pointer
    // sm_d for device memory pointer
    int *sm_h, *sm_d;
    sm_h= (int *)malloc(N*sizeof(int));
    cudaMalloc((void**)&sm_d,N*sizeof(*sm_d));

    kern<<<N,32>>>(sm_d);

    cudaMemcpy(sm_h, sm_d, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0;i<N;i++)
        printf("thread block %d -> sm%d\n",i,sm_h[i]);

    return 0;
}