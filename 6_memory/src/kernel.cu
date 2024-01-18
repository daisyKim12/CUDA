#include "kernel.h"

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

void gray_image(float *in, float *out, int width, int height) {

}

void blur_image(float *in, float *out, int width, int height) {

}