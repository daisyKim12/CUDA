#ifndef _REDUCE_H
#define _REDUCE_H

#define PRINT_NUM 10

//small
// #define BLOCK_SIZE 8
// #define SIZE 1024

//mid
// #define BLOCK_SIZE 16
// #define SIZE 4096

//large
#define BLOCK_SIZE 128
//#define SIZE 4194304
//#define SIZE 1048576
#define SIZE 4096

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include <iostream>

double run_reduction(float *in, float *out, long size, int blocksize, int ver);

#endif