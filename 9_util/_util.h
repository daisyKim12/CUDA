#ifndef _UTIL_H
#define _UTIL_H

#include "cmath"
#include <iostream>
#include <fstream>

void init_array(float *arr, long long int N, int seed);

void print_array(float *arr, int m);

void save_result(float* arr, int width, int height, const char* file_name);

void save_run_time(double* run_time, double* bandwidth, int n, const char* file_name);

#endif
