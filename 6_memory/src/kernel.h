#ifndef _KERNEL_H
#define _KERNEL_H

#define CHANNELS 3
#define BLUR_SIZE 1
#define BLUR_KERNEL_SIZE (((BLUR_SIZE)+1) * ((BLUR_SIZE)+1))

void gray_image(float *in, float *out, int width, int height);

void blur_image(float *in, float *out, int width, int height);

#endif