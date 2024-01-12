#include "_util.h"
#include "matmul.h"

int main(int argc, char *argv[]) {

    if(argc < 2) {
        std::cout << "please enter version number\n";
        return 1;
    }

    int ver = atoi(argv[1]);

    const char* M_file = "../result/large_1024_32/M.txt";
    const char* N_file = "../result/large_1024_32/N.txt";
    const char* result_file = "../result/large_1024_32/result.txt";
    const char* optimal_file = "../result/large_1024_32/optimal_result.txt";
    const char* time_file = "../result/large_1024_32/time.txt";

    double *run_time = new double[10];

    long long int width = WIDTH;
    int tile_width = TILE_WIDTH;
    long long int total_size = width * width;

    float *M_h = new float[total_size];
    float *N_h = new float[total_size];
    float *result_h = new float[total_size];
    float *M_d, *N_d, *result_d;

    init_array(M_h, total_size, 8811);
    init_array(N_h, total_size, 9700);

    // memory allocation
    cudaMalloc((void**)&M_d, total_size * sizeof(float));
    cudaMalloc((void**)&N_d, total_size * sizeof(float));
    cudaMalloc((void**)&result_d, total_size * sizeof(float));
    // copy host memory to device
    cudaMemcpy(M_d, M_h, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "< ver " << ver << " running ... >\n";
    cudaError_t cudaErr = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << "\n" << std::endl;    
    run_time[ver-1] = run_matmul(M_d, N_d, result_d, width, tile_width, ver);

    // copy result to host memory
    cudaMemcpy(result_h, result_d, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: result_h" << std::endl;
    print_array(result_h, NUM);
    //save_result(result_h, width, width, M_file);

    std::cout << "\n--------------------result------------------\n" << std::endl;
    std::cout << "ver " << ver <<" time: " << run_time[ver-1] <<"seconds"<< std::endl;
    
    cudaFree(M_d); cudaFree(N_d); cudaFree(result_d);
    delete[] M_h; delete[] N_h; delete[] result_h;

    return 0;
}