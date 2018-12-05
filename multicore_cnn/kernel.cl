
#define ReLU(x) (((x)>0)?(x):0)

__kernel void conv(
		__global float* inputs,
		__global float* filters,
		__global float* outputs,
		__global float* biases,
		int D1,
		int D2,
		int N
	) 
{
	int gid_0 = get_global_id(0);
	int batch = get_global_id(1);

	int out_channel = gid_0 / (N*N);
	int remain = gid_0 % (N*N);
	int i = remain / N;
	int j = remain % N;

	float bias = biases[out_channel];
	
	__global float* input;
	__global float* filter;
    __global float* output = outputs + N * N * (D2*batch + out_channel);
	
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
	output[i * N + j] = ReLU(sum + bias);
}
