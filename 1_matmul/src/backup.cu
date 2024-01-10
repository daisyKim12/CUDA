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