#include "reduce.h"

__global__ void ver1(float *in, float *out) {

    extern __shared__ float sdata[];

    int tid = threadIdx.x; 
    long i = threadIdx.x + blockDim.x * blockIdx.x;

    // load data into shared memory
    sdata[tid] = in[i];

    __syncthreads();

    // do reduction in shared memory
/* branch divergence and % operation is slow */
    for(int s = 1; s < blockDim.x; s *= 2) {
        if(tid % (2*s) == 0) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    // write the reduction result of this block into gobal memory
    if(tid == 0) 
        out[blockIdx.x] = sdata[0]; 

}

/* branch divergence and % operation is slow */
__global__ void ver2(float* in, float* out) {

    extern __shared__ float sdata[];

    int tid = threadIdx.x; 
    long i = threadIdx.x + blockDim.x * blockIdx.x;

    // load data into shared memory
    sdata[tid] = in[i];

    __syncthreads();

    // do reduction in shared memory
/* shared memory bank conflict  */
    for(int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;

        if(index < blockDim.x) {
            sdata[index] = sdata[index] + sdata[index + s];
        }

        __syncthreads();
    }

    // write the reduction result of this block into gobal memory
    if(tid == 0) 
        out[blockIdx.x] = sdata[0]; 

}

/* shared memory bank conflict  */
__global__ void ver3(float* in, float* out) {

    extern __shared__ float sdata[];

    int tid = threadIdx.x; 
    long i = threadIdx.x + blockDim.x * blockIdx.x;

    // load data into shared memory
    sdata[tid] = in[i];

    __syncthreads();

    // do reduction in shared memory
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
/* half of the thread is ideal starting from the first iteration */
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write the reduction result of this block into gobal memory
    if(tid == 0) 
        out[blockIdx.x] = sdata[0]; 

}

/* half of the thread is ideal starting from the first iteration */
__global__ void ver4(float* in, float* out) {

    extern __shared__ float sdata[];

    int tid = threadIdx.x; 
    long i = threadIdx.x + (blockDim.x*2) * blockIdx.x;

    // load data into shared memory
    sdata[tid] = in[i] + in[i + blockDim.x];

    __syncthreads();

    // do reduction in shared memory
    for(int s = blockDim.x; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write the reduction result of this block into gobal memory
    if(tid == 0) 
        out[blockIdx.x] = sdata[0]; 
}

/* last wrap unrolling */
__global__ void ver5(float* in, float* out) {

    extern __shared__ float sdata[];

    int tid = threadIdx.x; 
    long i = threadIdx.x + (blockDim.x*2) * blockIdx.x;

    // load data into shared memory
    sdata[tid] = in[i] + in[i + blockDim.x];

    __syncthreads();

    // do reduction in shared memory
    // blockdim.x = 64, shared memory size = 128
    for(int s = blockDim.x; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

/* last wrap unrolling */
    if(tid < 32)
    {
        sdata[tid] += sdata[tid + 32];
        __syncwarp();
        sdata[tid] += sdata[tid + 16];
        __syncwarp();
        sdata[tid] += sdata[tid + 8];
        __syncwarp();
        sdata[tid] += sdata[tid + 4];
        __syncwarp();
        sdata[tid] += sdata[tid + 2];
        __syncwarp();
        sdata[tid] += sdata[tid + 1];       
    }
    
    // write the reduction result of this block into gobal memory
    if(tid == 0) 
        out[blockIdx.x] = sdata[0]; 

}


/* complete unrolling */
template <unsigned int _blocksize>
__global__ void ver6(float* in, float* out) {
    
    extern __shared__ float sdata[];

    int tid = threadIdx.x; 
    long i = threadIdx.x + (blockDim.x*2) * blockIdx.x;

    // load data into shared memory
    sdata[tid] = in[i] + in[i + blockDim.x];

    __syncthreads();

    // do reduction in shared memory
/* complete unroll, only code that satisfies if statements
dynamically compiles in compile time*/


    if(_blocksize >= 512) {
        if(tid < 256)
            sdata[tid] = sdata[tid + 256];
        __syncthreads();
    }

    if(_blocksize >= 256) {
        if(tid < 128)
            sdata[tid] = sdata[tid + 128];
        __syncthreads();
    }

    if(_blocksize >= 128) {
        if(tid < 64)
            sdata[tid] = sdata[tid + 64];
        __syncthreads();
    }

/* last wrap unrolling */
    if(tid < 32)
    {
        if(_blocksize >= 64){ sdata[tid] += sdata[tid + 32]; __syncwarp(); }
        if(_blocksize >= 32){ sdata[tid] += sdata[tid + 16];__syncwarp(); }
        if(_blocksize >= 16){ sdata[tid] += sdata[tid + 8];__syncwarp(); }
        if(_blocksize >= 8){ sdata[tid] += sdata[tid + 4];__syncwarp(); }
        if(_blocksize >= 4){ sdata[tid] += sdata[tid + 2];__syncwarp(); }
        if(_blocksize >= 2){ sdata[tid] += sdata[tid + 1]; }
    }
    
    // write the reduction result of this block into gobal memory
    if(tid == 0) 
        out[blockIdx.x] = sdata[0]; 

}

void run_ver6(float* in, float* out, long size, long blocksize) {

    blocksize = blocksize * 2;
    switch(blocksize) {
        case 512:
            ver6<512><<< size / (blocksize * 2), blocksize, blocksize * 2 * sizeof(float)>>>(in, out); break;
        case 256:
            ver6<256><<< size / (blocksize * 2), blocksize, blocksize * 2 * sizeof(float)>>>(in, out); break;
        case 128:
            ver6<128><<< size / (blocksize * 2), blocksize, blocksize * 2 * sizeof(float)>>>(in, out); break;
        case 64:
            // ver6<64><<<dimGrid, dimBlock, smem_size>>>(in, out);  break;
        case 32:
            // ver6<32><<<dimGrid, dimBlock, smem_size>>>(in, out); break;
        case 16:    
            // ver6<16><<<dimGrid, dimBlock, smem_size>>>(in, out); break;
        case 8:
            // ver6<8><<<dimGrid, dimBlock, smem_size>>>(in, out); break;
        case 4:
            // ver6<4><<<dimGrid, dimBlock, smem_size>>>(in, out); break;
        case 2:
            // ver6<2><<<dimGrid, dimBlock, smem_size>>>(in, out); break;
        case 1:
            // ver6<1><<<dimGrid, dimBlock, smem_size>>>(in, out); break;
        default:
            break;
    }
}

int calcuate_iter(long size, int blocksize) {
    int result = 0;
    for(int iter = size / blocksize; iter > 1; iter /= blocksize) {
        result += 1;
    }
    return result;
}

double run_reduction(float *in, float *out, long size, int blocksize, int ver) {
    
    int local_size = size;
    int iter = calcuate_iter(size,blocksize);
    int cnt = 0;


    // check kernel run time
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    //kernel invocation    
    switch(ver){
        case 1:
            ver1<<<local_size/blocksize, blocksize, blocksize * sizeof(float)>>>(in, out);
            for(; iter > 0; iter--) {
                local_size /= blocksize;
                ver1<<<local_size, blocksize, blocksize * sizeof(float)>>>(out, out);
            }
            break;
        case 2:
            ver2<<<local_size/blocksize, blocksize, blocksize * sizeof(float)>>>(in, out);
            for(; iter > 0; iter--) {
                local_size /= blocksize;
                ver2<<<local_size, blocksize, blocksize * sizeof(float)>>>(out, out);
            }  
            break;
        case 3:
            ver3<<<local_size/blocksize, blocksize, blocksize * sizeof(float)>>>(in, out);
            for(; iter > 0; iter--) {
                local_size /= blocksize;
                ver3<<<local_size, blocksize, blocksize * sizeof(float)>>>(out, out);
            }
            break;
        case 4:
            ver4<<<local_size/blocksize, blocksize/2, blocksize * sizeof(float)>>>(in, out);
            cnt ++;
            for(; iter > 0; iter--) {
                local_size /= blocksize;
                ver4<<<local_size, blocksize/2, blocksize * sizeof(float)>>>(out, out);
                cnt++;
            }
            printf("cnt: %d\n", cnt);
            break; 
        case 5:
            ver5<<<local_size/blocksize, blocksize/2, blocksize * sizeof(float)>>>(in, out);
            for(; iter > 0; iter--) {
                local_size /= blocksize;
                ver5<<<local_size, blocksize/2, blocksize * sizeof(float)>>>(out, out);
            }
            break;
        case 6:
            //run_ver6(in, out, local_size, blocksize);
            ver6<64><<< size / (blocksize), blocksize/2, blocksize * sizeof(float)>>>(in, out);

            for(; iter > 0; iter--) {
                local_size /= blocksize;
                //run_ver6(out, out,local_size, blocksize);
                ver6<64><<< local_size, blocksize/2, blocksize * sizeof(float)>>>(out, out);
            }
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    // std::cout << "ver " << ver <<" : " << sec.count() <<"seconds"<< std::endl;

    return sec.count();
}