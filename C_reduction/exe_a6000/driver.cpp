#include "_util.h"
#include "reduce.h"

int main(int argc, char *argv[]) {

    int ver = 4;

    // long long int size = SIZE;
    long long int size = atoi(argv[1]);
    int blocksize = BLOCK_SIZE;

    float *A_h = new float[size];
    float *B_h = new float[size];
    float *A_d, *B_d;

    init_array(A_h, size, 1248);

    // memory allocation
    cudaMalloc((void**)&A_d, size * sizeof(float));
    cudaMalloc((void**)&B_d, size * sizeof(float));
    // copy host memory to device
    cudaMemcpy(A_d, A_h, size * sizeof(float), cudaMemcpyHostToDevice);

    double run_time;
    std::cout << "< ver " << ver << " running ... >\n";
    run_time = run_reduction(A_d, B_d, size, blocksize, ver);
    cudaError_t cudaErr = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;

    // copy result to host memory
    cudaMemcpy(B_h, B_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: A" << std::endl;
    print_array(A_h, PRINT_NUM);

    std::cout << "host memory: B" << std::endl;
    print_array(B_h, PRINT_NUM);

    std::cout << "run time: " << run_time << "sec" << std::endl;
    cudaFree(A_d); cudaFree(B_d);
    delete[] A_h; delete[] B_h;

    return 0;
}