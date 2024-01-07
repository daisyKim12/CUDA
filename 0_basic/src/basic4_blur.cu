#include <iostream>
#define PRINT_SIZE 20
#define BLUR_SIZE 1
#define BLUR_KERNEL_SIZE (((BLUR_SIZE)+1) * ((BLUR_SIZE)+1))

__global__ void blur(float * in, float * out, int width , int height) {

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;


    if( col < width && row < height) {
        float temp = 0;
        
        for(int i = -BLUR_SIZE; i < BLUR_SIZE + 1; i++) {
            for(int j = -BLUR_SIZE; j < BLUR_SIZE + 1; j++) {

                int col_added = col + i;
                int row_added = row + i;
                
                if(col_added > -1 && col_added < width && row_added > -1 && row_added <height)
                    temp += in[row_added * width + col_added];
            }
        }

        out[row * width + col] = temp / BLUR_KERNEL_SIZE;
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

int main(void) {

    int width = 128*3;
    int height = 128*3;
    int N = width * height;

    float *A_h = new float[N];
    float *B_h = new float[N];
    float *A_d, *B_d;
    
    // initialize host memory
    init_array(A_h, N, 1234);

    // allocate device memory
    cudaMalloc((void**) &A_d, N * sizeof(float));
    cudaMalloc((void**) &B_d, N * sizeof(float));

    // copy host memory to device
    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);

    // kernel invocation
    dim3 dimGrid(width/8, height/8, 1);
    dim3 dimBlock(8,8,1);
    blur<<<dimGrid, dimBlock>>>(A_d, B_d, width, height);

    cudaError_t cudaErr = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;

    // copy result to host memory
    cudaMemcpy(B_h, B_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "host memory: rgb2gray" << std::endl;
    print_array(A_h, PRINT_SIZE);

    std::cout << "host memory: grayImage" << std::endl;
    print_array(B_h, PRINT_SIZE);

    // free allocated memory
    cudaFree(A_d);
    cudaFree(B_d);
    delete[] A_h; delete[] B_h;

    return 0;

}