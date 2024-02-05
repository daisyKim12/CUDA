#define TILE_WIDTH 32

__global__ void ver2(float* M, float* N, float* R, const long long int width)
{

    __shared__ float sub_tile_M[TILE_WIDTH][TILE_WIDTH * 2];
    __shared__ float sub_tile_N[TILE_WIDTH * 2][TILE_WIDTH];

    long tx = threadIdx.x;  long ty = threadIdx.y;
    long bx = blockIdx.x;   long by = blockIdx.y;
    
    long col = tx + TILE_WIDTH * bx;
    long row = ty + TILE_WIDTH * by;
    float acc = 0;

    // idx is a stride in block
    // M must move in block group row wise dir, N must move in block group row wise dir  
    for(long idx = 0; idx < width/TILE_WIDTH/2; idx++) 
    {
        // load data to shared memory
        // row and col is fixed. M is row fixed, N is col fixed.
        sub_tile_M[ty][tx] = M[ row * width + (2 * idx) * TILE_WIDTH + tx];
        sub_tile_M[ty][tx + TILE_WIDTH] = M[ row * width + (2 * idx + 1) * TILE_WIDTH + tx];
        
        sub_tile_N[ty][tx] = N[ ((2 * idx) * TILE_WIDTH + ty) * width + col];
        sub_tile_N[ty + TILE_WIDTH][tx] = N[ ((2 * idx + 1) * TILE_WIDTH + ty) * width + col];

        // sync threads before compute
        __syncthreads();

        // compute
        for(int k = 0; k < TILE_WIDTH*2; k++)
        {
            acc += sub_tile_M[ty][k] * sub_tile_N[k][tx];
        }
        
        // sync threads before end of compute
        __syncthreads();

    }

    
    R[row * width + col] = acc;
    
}

__global__ void ver5(float* M, float* N, float* R, const long long int width)
{   
    __shared__ float sub_tile_M[TILE_WIDTH * TILE_WIDTH + SKEW * 32];
    __shared__ float sub_tile_N[TILE_WIDTH][TILE_WIDTH * 2];

    long tx = threadIdx.x;  long ty = threadIdx.y;
    long bx = blockIdx.x;   long by = blockIdx.y;
    
    long col = tx + (TILE_WIDTH * 2) * bx;
    long row = ty + TILE_WIDTH * by;

    float acc_left = 0;
    float acc_right = 0;

    for(int idx = 0 ; idx < width/TILE_WIDTH; idx ++) {
        //load single square of M
        sub_tile_M[ty * (TILE_WIDTH + SKEW) + tx] = M[row * width + idx * TILE_WIDTH + tx];

        //load left square of N
        sub_tile_N[ty][tx] = N[(idx * TILE_WIDTH + ty) * width + col];
        //load right square of N
        sub_tile_N[ty][tx + TILE_WIDTH] = N[(idx * TILE_WIDTH + ty) * width + col + TILE_WIDTH];

        __syncthreads();

        // compute
        for(int k = 0; k < TILE_WIDTH; k++) {
            acc_left += sub_tile_M[ty * (TILE_WIDTH + SKEW) + k] * sub_tile_N[k][tx];
            acc_right += sub_tile_M[ty * (TILE_WIDTH + SKEW) + k] * sub_tile_N[k][tx + TILE_WIDTH];
        }

        __syncthreads();

    }

    R[row * width + col] = acc_left;
    R[row * width + col + TILE_WIDTH] = acc_right;
}