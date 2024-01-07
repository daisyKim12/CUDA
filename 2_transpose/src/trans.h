#ifndef TRANS_H
#define TRANS_H

#define NUM 10
#define TILE_WIDTH 32
#define WIDTH 2048
#define HEIGHT 2048
#define SKEW 1

double time_transpose(float *A, float *B, long width, long tile_width, int ver);
#endif