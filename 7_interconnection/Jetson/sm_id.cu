#include "9_util/common.h"

#define SHARED_MEM 48000
#define SHARED_SIZE (SHARED_MEM / 4)
#define TILE_WIDTH 4
__device__ 
uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__device__ int cnt[32];

__global__ void kern(int *sm, int * A_d){

    cnt[get_smid()] = 0 ;
    __syncwarp();


    int volatile arr[32];
    for(int i = 0; i < blockDim.x; i++) {
        arr[i] = i;
    }
    
    __shared__ int fill[SHARED_SIZE];
    
    for(int i = 0; i < SHARED_SIZE ; i++) {
        fill[i] = A_d[threadIdx.x];
        fill[i] += i;
    }

    if (threadIdx.x==0)
      sm[blockIdx.x]=get_smid();

    for(int i = 0; i < SHARED_SIZE ; i++) {
        A_d[threadIdx.x] = fill[i];
    }
    
    cnt[get_smid()] = cnt[get_smid()] + 1;

    while(1){
        if(cnt[get_smid()] == 32)
            break;
    }

}

__global__ void matmul(float* M, float* N, float* R, const long long int width, int *sm)
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
        std::cout << "  Shared Memory Per Block (KB): " << prop.sharedMemPerBlock/1.0e3 << "\n";
    }

    // sm_h for host memory pointer
    // sm_d for device memory pointer
    int *sm_h, *sm_d;
    int *A_h, *A_d, *M, *N, *R;

    long int int nx = atoi(arg[1]);
    long long int nxy = nx * nx;
    size_t Bytes= nxy * sizeof(int)

    sm_h = (int *)malloc(nx * sizeof(int));
    A_h = (int *)malloc(nx * sizeof(int));
    cudaMalloc((void**)&sm_d,nx * sizeof(int));
    cudaMalloc((void**)&A_d, nx * sizeof(int));
    cudaMallocManaged((void**)&M, Bytes);
    cudaMallocManaged((void**)&N, Bytes);
    cudaMallocManaged((void**)&R, Bytes);

    cudaMemcpy(A_d, A_h, N*sizeof(*A_d), cudaMemcpyHostToDevice);
    kern<<<N,32>>>(sm_d, A_d);
    kern<<<N,32>>>(sm_d, A_d);
    kern<<<N,32>>>(sm_d, A_d);
    cudaMemcpy(sm_h, sm_d, nxy, cudaMemcpyDeviceToHost);

    for (int i=0;i<N;i++)
        printf("thread block %d : %d\n",i,sm_h[i]);

    return 0;
}