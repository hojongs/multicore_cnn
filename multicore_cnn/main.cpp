#include "cnn.h"
#pragma warning(disable:4996)

int compare_result(int argc, char **argv);

extern clock_t pooling_clock, conv_clock, fc_clock, softmax_clock, find_max_clock, RELU_clock;
extern long long write_nsec, kernel_nsec;
extern const char *CLASS_NAME[];

int main(int argc, char **argv)
{
    if (argc != 3) {
        print_usage_and_exit(argv);
    }

    int num_images = atoi(argv[1]);
    float *images = read_images(num_images);
    float *network = read_network();
    float **network_sliced = slice_network(network);
    int *labels = (int*)calloc(num_images, sizeof(int));
    float *confidences = (float*)calloc(num_images, sizeof(float));

    cnn_init();
    clock_t start = clock();
    cnn(images, network_sliced, labels, confidences, num_images);
	clock_t end = clock();
    printf("Elapsed time: %f sec\n", (double)(end - start) / CLK_TCK);

    FILE *of = fopen(argv[2], "w");
    int *labels_ans = read_labels(num_images);
    double acc = 0;
    for (int i = 0; i < num_images; ++i) {
        fprintf(of, "Image %04d: %s %f\n", i, CLASS_NAME[labels[i]], confidences[i]);
        if (labels[i] == labels_ans[i]) ++acc;
    }
    fprintf(of, "Accuracy: %f\n", acc / num_images);
    fclose(of);

    free(images);
    free(network);
    free(network_sliced);
    free(labels);
    free(confidences);
    free(labels_ans);

	printf("pooling  : %lf sec \n", (double)pooling_clock / CLOCKS_PER_SEC);
	printf("conv     : %lf sec \n", (double)conv_clock / CLOCKS_PER_SEC);
	printf("- find_max : %lf sec \n", (double)find_max_clock / CLOCKS_PER_SEC);
	printf("- write    : %lf sec \n", write_nsec / 1000000000.0);
	printf("- kernel   : %lf sec \n", kernel_nsec / 1000000000.0);
	printf("- RELU     : %lf sec \n", (double)RELU_clock / CLOCKS_PER_SEC);
	printf("fc       : %lf sec \n", (double)fc_clock / CLOCKS_PER_SEC);
	printf("softmax  : %lf sec \n", (double)softmax_clock / CLOCKS_PER_SEC);

	char* params[] = { "", "result.out", "seq.out", NULL };
	compare_result(3, params);

    return 0;
}
