
#define ReLU(x) (((x)>0)?(x):0)

__kernel void conv(
		__global const float* inputs,
		__global const float* filters,
		__global	   float* outputs,
		__global const float* biases,
		const int D1,
		const int D2,
		const int N
	) 
{
	const int out_channel = get_global_id(0);
	const int gid_1 = get_global_id(1);
	const int lid_1 = get_local_id(1);

	const int batch = gid_1 / (N * N);
	const int remain = gid_1 % (N * N);
	const int i = remain / N;
	const int j = remain % N;

	__global const float* input;
	__global const float* filter;
    __global float* output = outputs + N * N * (D2*batch + out_channel);

	__local const float* l_filter;

	
	float sum = 0;
	for (int in_channel = 0; in_channel < D1; in_channel++) {
		input = inputs + N * N * (D1*batch + in_channel);
		filter = filters + 3 * 3 * (out_channel * D1 + in_channel);
		for (int k = 0; k < 3; k++) {
			for (int l = 0; l < 3; l++) {
				int x = i + k - 1;
				int y = j + l - 1;
				if (x >= 0 && x < N && y >= 0 && y < N)
					sum += input[x * N + y] * filter[k * 3 + l];
			}
		}
	}
	const float bias = biases[out_channel];
	output[i * N + j] = ReLU(sum + bias);
}
