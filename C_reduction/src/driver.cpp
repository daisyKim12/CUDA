#include "_util.h"
#include "reduce.h"

int main(int argc, char *argv[]) {

    if(argc < 2) {
        std::cout << "please enter version number";
        return 1;
    }

    int ver = 4;

    const char* A_file = "../result/large_1048576_128/A.txt";
    const char* B_file = "../result/large_1048576_128/B.txt";
    const char* time_file = "../result/large_1048576_128/time.txt";
    
    double *run_time = new double[10];
    double *bandwidth = new double[10];

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

    std::cout << "< ver " << ver << " running ... >\n";
    run_time[ver-1] = run_reduction(A_d, B_d, size, blocksize, ver);
    bandwidth[ver-1] = size * 4 / run_time[ver-1] / 1e9;
    cudaError_t cudaErr = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;

    // copy result to host memory
    cudaMemcpy(B_h, B_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: A" << std::endl;
    print_array(A_h, PRINT_NUM);
    save_result(A_h, size/blocksize, blocksize, A_file);

    std::cout << "host memory: B" << std::endl;
    print_array(B_h, PRINT_NUM);
    save_result(B_h, size/blocksize, blocksize, B_file);

    std::cout << "ver " << ver <<" time: " << run_time[ver-1] <<"seconds"<< std::endl;
    std::cout << "ver " << ver <<" BW: " << bandwidth[ver-1] <<"GB/s"<< std::endl;

    save_run_time(run_time, bandwidth, 6, time_file);

    cudaFree(A_d); cudaFree(B_d);
    delete[] A_h; delete[] B_h;

    return 0;
}