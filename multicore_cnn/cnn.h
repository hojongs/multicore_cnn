#ifndef _CNN_H
#define _CNN_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <CL/cl.h>
//#define PROFILE_ENABLE

using namespace std::chrono;

void cnn_init();
void cnn(float *images, float **network, int *labels, float *confidences, int num_images);

void print_usage_and_exit(char **argv);
void* read_bytes(const char *fn, size_t n);
float* read_images(size_t n);
int* read_labels(size_t n);
float* read_network();
float** slice_network(float *p);
float* alloc_layer(size_t n);
void convolution_layer(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N);
void pooling_layer(float *inputs, float *outputs, int D, int N);

#endif 
