#ifndef _UTIL_H
#define _UTIL_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>


#define NUM 10

// mid
// #define TILE_WIDTH 8
// #define WIDTH 128
// #define HEIGHT 128

// large
#define TILE_WIDTH 32
#define WIDTH 1024
#define HEIGHT 1024

// a_large
// #define TILE_WIDTH 32
// #define WIDTH 4096
// #define HEIGHT 4096


void init_array(float *arr, int N, int seed);

void print_array(float *arr, int m);

void save_result(float* arr, int width, int height, const char* file_name);

#endif
