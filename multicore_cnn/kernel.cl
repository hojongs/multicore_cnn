#define ReLU(x) (((x)>0)?(x):0)

__kernel void conv(
		__global const float* inputs,
		__global const float* filters,
		__global float* outputs,
		__constant float* biases,
		const int D1,
		const int D2,
		const int N,
		const int imageCnt,
		__local float* l_filter,
		__local float* l_input,
		const int in_channel
	) 
{
	const int out_channel = get_global_id(0);
	const int batch = get_global_id(1) / (N*N);
	const int remain = get_global_id(1) % (N*N);
	const int i = remain / N;
	const int j = remain % N;
	const int lid = get_local_id(1);
	const int lsize = get_local_size(1);

	__global const float* input = inputs + N * N * (D1*batch + in_channel);
    __global float* output = outputs + N * N * (D2*batch + out_channel);
	__global const float* filter = filters + out_channel * D1 * 3 * 3 + in_channel*3*3;

	if (lid < 9)
		l_filter[lid] = filter[lid];
	if (N <= 16)
		l_input[lid] = input[lid % (N*N)];
	__local int batch_start;
	if (lid == 0)
		batch_start = batch;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (batch >= imageCnt)
		return;

	float sum = 0;
	//for (int in_channel = 0; in_channel < D1; in_channel++)
    {
		//__global const float* input = inputs + N * N * (D1*batch + in_channel);
		//__global const float* filter = filters + out_channel * D1 * 3 * 3 + in_channel*3*3;

		if (N <= 16)
		{
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N)
						//sum += input[x * N + y] * l_filter[(in_channel*3*3) + (k*3) + l];
						//sum += input[x * N + y] * filter[(k*3) + l];
						sum += l_input[(batch-batch_start)*N*N + x * N + y] * l_filter[(k*3) + l];
				}
			}
		}
		else
		{
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N)
						//sum += input[x * N + y] * l_filter[(in_channel*3*3) + (k*3) + l];
						//sum += input[x * N + y] * filter[(k*3) + l];
						sum += input[x * N + y] * l_filter[(k*3) + l];
				}
			}
		}
	}
	//const float bias = biases[out_channel];
	//output[i * N + j] = ReLU(sum + bias);
	output[i * N + j] += sum;
}

__kernel void fc(
		__global const float* input_neuron,
		__global const float* weights,
		__global float* output_neuron,
		__global const float* biases,
		const int inN,
		const int outM,
		const int batch_size,
		const int imageCnt,
		__local float* l_weights
	)
{
	const int out = get_global_id(0);
	const int batch = get_global_id(1);
	const int lid = get_local_id(1);
	const int lsize = get_local_size(1);

	if (lid < inN)
	{
		for(int l=0;l<inN;l+=lsize)
			l_weights[l+lid] = weights[out * inN + l+lid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (batch >= imageCnt)
		return;

	float sum = 0;
	for (int in = 0; in < inN; in++)
		//sum += input_neuron[batch * inN + in] * weights[out * inN + in];
		sum += input_neuron[batch * inN + in] * l_weights[in];

	output_neuron[batch * outM + out] = ReLU(sum + biases[out]);
}
