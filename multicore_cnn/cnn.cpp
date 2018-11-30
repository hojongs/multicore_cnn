#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "cnn.h"

void initOpenCL(int platform_idx, int gpu_idx);

extern const char* CLASS_NAME[];

clock_t pooling_clock = 0;
clock_t conv_clock = 0;
clock_t fc_clock = 0;
clock_t softmax_clock = 0;
clock_t find_max_clock = 0;

void pooling2x2(float *input, float *output, int N) {
    int i, j, k, l;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float max = 0;
            for (k = 0; k < 2; k++) {
                for (l = 0; l < 2; l++) {
                    float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
                    max = (max > pixel) ? max : pixel;
                }
            }
            output[i * N + j] = max;
        }
    }
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
void pooling_layer(float *inputs, float *outputs, int D, int N) {
	int i;
    for (i = 0; i < D; i++) {
        float * input = inputs + i * N * N * 4;
        float * output = outputs + i * N * N;
        pooling2x2(input, output, N);
    }
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
#define ReLU(x) (((x)>0)?(x):0)
void convolution_layer(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N, int batch_size)
{
    memset(outputs, 0, sizeof(float) * N * N * D2 * batch_size);
	clConv(inputs, outputs, filters, D2, D1, N, batch_size);

	for (int batch = 0; batch < batch_size; batch++)
	{
		for (int out_channel = 0; out_channel < D2; out_channel++) {
			float * output = outputs + (N*N*D2*batch) + (N*N*out_channel);
			float bias = biases[out_channel];
			for (int i = 0; i < N * N; i++) {
				output[i] = ReLU(output[i] + bias);
			}
		}
	}
}

/*
 * M = output size
 * N = input size
 */
void fc_layer(float *input_neuron, float *output_neuron, float *weights, float *biases, int M, int N) {
	int i, j;
    for (j = 0; j < M; j++) {
        float sum = 0;
        for (i = 0; i < N; i++) {
            sum += input_neuron[i] * weights[j * N + i];
        }
        sum += biases[j];
        output_neuron[j] = ReLU(sum);
    }
}

void softmax(float *output, int N) {
    int i;
    float max = output[0];
    for (i = 1; i < N; i++) {
        max = (output[i] > max)?output[i]:max;
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += expf(output[i] - max);
    }
    for (i = 0; i < N; i++) {
        output[i] = expf(output[i] - max) / sum;
    }
}

int find_max(float *fc, int N) {
    int i;
    int maxid = 0;
    float maxval = 0;
    for (i = 0; i < N; i++) {
        if (maxval < fc[i]) {
            maxval = fc[i];
            maxid = i;
        }
    }
    return maxid;
}

float* alloc_layer(size_t n)
{
    return (float*)malloc(n * sizeof(float));
}

void cnn_init() {
	int platform_idx = 0;
	int gpu_idx = 1;
	initOpenCL(platform_idx, gpu_idx);
}

void cnn(float *images, float **network, int *labels, float *confidences, int num_images) {
	int batch_size = 1;

    // slice the network into weights and biases
    float *w1_1, *b1_1, *w1_2, *b1_2;
    float *w2_1, *b2_1, *w2_2, *b2_2;
    float *w3_1, *b3_1, *w3_2, *b3_2, *w3_3, *b3_3;
    float *w4_1, *b4_1, *w4_2, *b4_2, *w4_3, *b4_3;
    float *w5_1, *b5_1, *w5_2, *b5_2, *w5_3, *b5_3;
    float *w1, *b1, *w2, *b2, *w3, *b3;
    w1_1 = network[0]; b1_1 = network[1];
    w1_2 = network[2]; b1_2 = network[3];
    w2_1 = network[4]; b2_1 = network[5];
    w2_2 = network[6]; b2_2 = network[7];
    w3_1 = network[8]; b3_1 = network[9];
    w3_2 = network[10]; b3_2 = network[11];
    w3_3 = network[12]; b3_3 = network[13];
    w4_1 = network[14]; b4_1 = network[15];
    w4_2 = network[16]; b4_2 = network[17];
    w4_3 = network[18]; b4_3 = network[19];
    w5_1 = network[20]; b5_1 = network[21];
    w5_2 = network[22]; b5_2 = network[23];
    w5_3 = network[24]; b5_3 = network[25];
    w1 = network[26]; b1 = network[27];
    w2 = network[28]; b2 = network[29];
    w3 = network[30]; b3 = network[31];

    // allocate memory for output of each layer
    float *c1_1, *c1_2, *p1;
    float *c2_1, *c2_2, *p2;
    float *c3_1, *c3_2, *c3_3, *p3;
    float *c4_1, *c4_2, *c4_3, *p4;
    float *c5_1, *c5_2, *c5_3, *p5;
    float *fc1, *fc2, *fc3;
    c1_1 = alloc_layer(64 * 32 * 32 * batch_size);
    c1_2 = alloc_layer(64 * 32 * 32 * batch_size);
    p1   = alloc_layer(64 * 16 * 16 * batch_size);
    c2_1 = alloc_layer(128 * 16 * 16 * batch_size);
    c2_2 = alloc_layer(128 * 16 * 16 * batch_size);
    p2   = alloc_layer(128 * 8 * 8 * batch_size);
    c3_1 = alloc_layer(256 * 8 * 8 * batch_size);
    c3_2 = alloc_layer(256 * 8 * 8 * batch_size);
    c3_3 = alloc_layer(256 * 8 * 8 * batch_size);
    p3   = alloc_layer(256 * 4 * 4 * batch_size);
    c4_1 = alloc_layer(512 * 4 * 4 * batch_size);
    c4_2 = alloc_layer(512 * 4 * 4 * batch_size);
    c4_3 = alloc_layer(512 * 4 * 4 * batch_size);
    p4   = alloc_layer(512 * 2 * 2 * batch_size);
    c5_1 = alloc_layer(512 * 2 * 2 * batch_size);
    c5_2 = alloc_layer(512 * 2 * 2 * batch_size);
    c5_3 = alloc_layer(512 * 2 * 2 * batch_size);
    p5   = alloc_layer(512 * 1 * 1 * batch_size);
    fc1  = alloc_layer(512 * batch_size);
    fc2  = alloc_layer(512 * batch_size);
    fc3  = alloc_layer(10 * batch_size);

	clock_t start;

    // run network
    for(int i = 0; i < num_images; i+= batch_size)
    {
        float *image = images + i * 3 * 32 * 32;

		start = clock();
        convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32, batch_size);
        convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32, batch_size);
		conv_clock += clock() - start;
		start = clock();
		pooling_layer(c1_2, p1, 64, 16);
		pooling_clock += clock() - start;

		start = clock();
		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16, batch_size);
        convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16, batch_size);
		conv_clock += clock() - start;
		start = clock();
		pooling_layer(c2_2, p2, 128, 8);
		pooling_clock += clock() - start;

		start = clock();
		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8, batch_size);
        convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8, batch_size);
        convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8, batch_size);
		conv_clock += clock() - start;
		start = clock();
		pooling_layer(c3_3, p3, 256, 4);
		pooling_clock += clock() - start;

		start = clock();
		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4, batch_size);
        convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4, batch_size);
        convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4, batch_size);
		conv_clock += clock() - start;
		start = clock();
		pooling_layer(c4_3, p4, 512, 2);
		pooling_clock += clock() - start;

		start = clock();
		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2, batch_size);
        convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2, batch_size);
        convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2, batch_size);
		conv_clock += clock() - start;
		start = clock();
		pooling_layer(c5_3, p5, 512, 1);
		pooling_clock += clock() - start;

		start = clock();
		fc_layer(p5, fc1, w1, b1, 512, 512);
        fc_layer(fc1, fc2, w2, b2, 512, 512);
        fc_layer(fc2, fc3, w3, b3, 10, 512);
		fc_clock += clock() - start;

		start = clock();
		softmax(fc3, 10);
		softmax_clock += clock() - start;

		start = clock();
        labels[i] = find_max(fc3, 10);
        confidences[i] = fc3[labels[i]];
		find_max_clock += clock() - start;

		fprintf(stdout, "Image %04d/%04d: %s %f\n", i+1, num_images, CLASS_NAME[labels[i]], confidences[i]);
    }

    free(c1_1); free(c1_2); free(p1);
    free(c2_1); free(c2_2); free(p2);
    free(c3_1); free(c3_2); free(c3_3); free(p3);
    free(c4_1); free(c4_2); free(c4_3); free(p4);
    free(c5_1); free(c5_2); free(c5_3); free(p5);
    free(fc1); free(fc2); free(fc3);
}
