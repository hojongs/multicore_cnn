inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void conv(
		__global float* inputs,
		__global float* filters,
		__global float* outputs,
		int D1,
		int D2,
		int N
	) 
{
	int out_channel = get_global_id(0);
	int batch = get_global_id(1);
	
    __global float* output = outputs + N * N * (D2*batch + out_channel);

	for (int in_channel = 0; in_channel < D1; in_channel++)
    {
		__global float* input = inputs + N * N * (D1*batch + in_channel);
		__global float* filter = filters + 3 * 3 * (out_channel * D1 + in_channel);
		for(int i=0; i<N; i++) { 
			for(int j=0; j<N; j++) {
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
		}
	}
}
