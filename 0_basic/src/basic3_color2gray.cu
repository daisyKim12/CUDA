#include <fstream>
#include <iostream>
#define M 10
#define CHANNELS 3


__global__ void rgb2gray(float* rgb, float* gray, long width, long height) {

    long col = threadIdx.x + blockDim.x * blockIdx.x;
    long row = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(col < width && row < height) {

        long gray_index = row * width + col;
        long rgb_index = gray_index * CHANNELS;

        float r = rgb[rgb_index];
        float g = rgb[rgb_index +1];
        float b = rgb[rgb_index +2];

        gray[gray_index] = 0.21*r + 0.71*g + 0.07*b;

    }
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

void save_result(float* arr, int width, int height, const char* file_name) {
    
    std::ofstream dst(file_name);
    if(!dst.is_open()) {
        std::cerr << "Error: can not open the file";
    }

    for(int i = 0; i<height; i++) {
        for(int j = 0; j<width; j++) {
            dst << arr[i * width + j] << " ";
        }
        dst << "\n\n";
    }
    dst.close();

}


int main(void) {

    const char* in_file_name = "basic3_input.txt";
    const char* out_file_name = "basic3_result.txt";
    
    // setting array size to 2^20
    long N = 128*128*3;
    long width = 128*3;
    long height = 128;

    float *rgb_h = new float[N];
    float *gray_h = new float[N];
    float *rgb_d, *gray_d;

    init_array(rgb_h, N, 9900);

    // allocate device memory
    cudaMalloc((void**)&rgb_d, N*sizeof(float));
    cudaMalloc((void**)&gray_d, N*sizeof(float));
    // copy host memory to device
    cudaMemcpy(rgb_d, rgb_h, N*sizeof(float), cudaMemcpyHostToDevice);

    // define block, grid dimension
    dim3 dimGrid((width + 7) / 8, (height + 7) / 8, 1);
    dim3 dimBlock(8, 8, 1);

    // kernel invocation
    rgb2gray<<<dimGrid, dimBlock>>>(rgb_d, gray_d, width ,height);
    cudaError_t cudaErr = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;

    // copy result to host memory
    cudaMemcpy(gray_h, gray_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: rgb2gray" << std::endl;
    print_array(rgb_h, M);
    save_result(rgb_h, width, height, in_file_name);

    std::cout << "host memory: grayImage" << std::endl;
    print_array(gray_h, M);
    save_result(gray_h, width, height, out_file_name);

    // free allocation
    cudaFree(rgb_d); cudaFree(gray_d);
    delete[] rgb_h; delete[] gray_h;

    return 0;
}