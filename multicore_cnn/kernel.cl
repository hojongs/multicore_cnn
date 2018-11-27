__kernel void conv(
		__global float* inputs,
		__global float* filters,
		__global float* outputs,
		int D1,
		int N,
		int out_channel,
		int in_channel
	) 
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	
    __global float* input = inputs + N * N * in_channel;
    __global float* filter = filters + 3 * 3 * (out_channel * D1 + in_channel);
    __global float* output = outputs + N * N * out_channel;

    float sum = 0;
    for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
            int x = i + k - 1;
            int y = j + l - 1;
            if (x >= 0 && x < N && y >= 0 && y < N)
                sum += input[x * N + y] * filter[k * 3 + l];
        }
    }
    output[i * N + j] += sum;
}
