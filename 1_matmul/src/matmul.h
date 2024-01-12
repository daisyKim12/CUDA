#ifndef _MATMUL_H
#define _MATMUL_H


#define NUM 10

// #define TILE_WIDTH 32
// #define WIDTH 1024
// #define WIDTH 4096

#define TILE_WIDTH 32
#define WIDTH 16384

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <iostream>

double run_matmul(float *M, float *N, float *out, long long int size, int blocksize, int ver);


#endif