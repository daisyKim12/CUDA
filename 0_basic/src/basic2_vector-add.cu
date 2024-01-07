#include <iostream>
#include <cstdlib>
#include <cmath>
#define M 10

// kernel for vector addtion
__global__ void vectorAdd(float* A_d, float* B_d, float* C_d, int N) {
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
        C_d[i] = A_d[i] + B_d[i];

}


// initializing array size N with random float between 0 and 100
void init_array(float *arr, int N, int seed) {
    
    srand(seed);

    for(int i = 0; i<N; i++){
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


// driver function
int main() {
    // setting array size to 2^20
    long N = 1<< 20;

    float *A_h = new float[N];
    float *B_h = new float[N];
    float *C_h = new float[N];
    float *A_d, *B_d, *C_d;

    init_array(A_h, N, 9900);
    init_array(B_h, N, 1234);
    
    std::cout << "host memory: A" << std::endl;
    print_array(A_h, M);
    std::cout << "host memory: B" << std::endl;
    print_array(B_h, M);

    // allocate device memory
    cudaMalloc((void**)&A_d, N*sizeof(float));
    cudaMalloc((void**)&B_d, N*sizeof(float));
    cudaMalloc((void**)&C_d, N*sizeof(float));

    // copy host memory to device
    cudaMemcpy(A_d, A_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N*sizeof(float), cudaMemcpyHostToDevice);
    
    // kernel invocation
    vectorAdd<<<(N+255)/256, 256>>>(A_d, B_d, C_d, N);

    cudaError_t cudaErr = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;

    // copy result to host memory
    cudaMemcpy(C_h, C_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: C" << std::endl;
    print_array(C_h, M);
    
    // free allocation
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
}
