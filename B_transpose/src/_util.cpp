#include "_util.h"

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
            dst << arr[i * width + j] << ",";
        }
        dst << "\n";
    }
    dst.close();

}

void save_run_time(double* arr, int n, const char* file_name) {
    
    std::ofstream dst(file_name);
    if(!dst.is_open()) {
        std::cerr << "Error: can not open the file";
    }

    for(int i = 0; i < n; i++) {
        dst << "ver " << (i+1) << ": " << arr[i] << "sec \n";
    }
    dst.close();
}