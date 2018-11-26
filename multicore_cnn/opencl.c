#pragma warning(disable:4996)
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); \
    exit(EXIT_FAILURE); \
  }
#define AND &&
#define OR ||
#define NOT !
#define STR_LEN 65536


const char *getErrorString(cl_int error)
{
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

char* getSourceCode(_In_ const char *file_name, _Out_ size_t *len) {
	char* source_code;
	size_t length;
	FILE* file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file); // in text mode, incorrect size
	rewind(file);

	source_code = (char *)calloc(sizeof(char), length + 1);
	//length = fread(source_code, sizeof(char), length, file);
	fread(source_code, length, 1, file);
	fclose(file);

	source_code[length] = '\0';
	*len = length;
	return source_code;
}

cl_kernel getKernel(cl_context context, cl_device_id device, const char* source_file_name, const char* kernel_name)
{
	char str[STR_LEN] = { 0 };
	cl_int err;
	cl_kernel kernel;

	cl_uint src_cnt = 1;
	size_t source_size;
	char* source_code = getSourceCode(source_file_name, &source_size);

	cl_program program = clCreateProgramWithSource(context, src_cnt, (const char**)&source_code, &source_size, &err);
	CHECK_ERROR(err);

	free(source_code);

	char option[1024] = { 0 };
	sprintf(option, "");
	err = clBuildProgram(program, 1, &device, option, NULL, NULL);
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, STR_LEN, str, NULL);
	printf("%s \n", str);
	CHECK_ERROR(err);

	kernel = clCreateKernel(program, kernel_name, &err);
	CHECK_ERROR(err);

	return kernel;
}

cl_device_id getDevice()
{
	char str[STR_LEN] = { 0 };
	cl_int err;
	cl_device_id device;

	const int GPU_PLATFORM_IDX = 0;
	const int GPU_DEV_IDX = 0;

	cl_uint num_platforms;
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);
	printf("Number of platforms: %u \n\n", num_platforms);
	for (cl_uint p = 0; p < num_platforms; p++)
	{
		printf("platform: %u\n", p);
		cl_platform_id platform = platforms[p];

		// platform info
		err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, STR_LEN, str, NULL);
		CHECK_ERROR(err);
		printf("- CL_PLATFORM_NAME\t:%s\n", str);

		err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STR_LEN, str, NULL);
		CHECK_ERROR(err);
		printf("- CL_PLATFORM_VENDOR\t:%s\n\n", str);

		// get device
		cl_uint num_devices;
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		CHECK_ERROR(err);
		printf("Number of devices:\t%u\n\n", num_devices);

		cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		CHECK_ERROR(err);

		// devices info
		for (cl_uint d = 0; d < num_devices; d++)
		{
			printf("device: %u\n", d);

			if (p == GPU_PLATFORM_IDX AND d == GPU_DEV_IDX)
				device = devices[d];

			cl_device_type device_type;
			err = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_TYPE\t:");
			if (device_type & CL_DEVICE_TYPE_CPU) printf("  CL_DEVICE_TYPE_CPU");
			else if (device_type & CL_DEVICE_TYPE_GPU) printf("  CL_DEVICE_TYPE_GPU");
			else if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf("  CL_DEVICE_TYPE_ACCELERATOR");
			else if (device_type & CL_DEVICE_TYPE_DEFAULT) printf("  CL_DEVICE_TYPE_DEFAULT");
			else if (device_type & CL_DEVICE_TYPE_CUSTOM) printf("  CL_DEVICE_TYPE_CUSTOM");
			printf("\n");

			err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, STR_LEN, str, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_NAME\t: %s\n", str);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, STR_LEN, str, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_VERSION\t: %s\n", str);

			cl_command_queue_properties props = 0;
			// 2.0
			//err = clGetDeviceInfo(devices[d], CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, sizeof(cl_command_queue_properties), &props, NULL);
			err = clGetDeviceInfo(devices[d], CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &props, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES\t:");
			if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) printf(" CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |");
			if (props & CL_QUEUE_PROFILING_ENABLE) printf(" CL_QUEUE_PROFILING_ENABLE |");
			printf("\n");

			size_t max_work_group_size;
			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE : %lu\n", max_work_group_size);

			cl_ulong global_mem_size;
			err = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_GLOBAL_MEM_SIZE : %llu\n", global_mem_size);

			cl_uint max_constant_args;
			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), &max_constant_args, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_CONSTANT_ARGS : %u\n", max_constant_args);

			cl_ulong max_constant_buffer_size;
			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &max_constant_buffer_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE : %llu\n", max_constant_buffer_size);

			cl_ulong local_mem_size;
			err = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_LOCAL_MEM_SIZE : %llu\n", local_mem_size);

			cl_ulong max_mem_alloc_size;
			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_MEM_ALLOC_SIZE : %llu\n\n", local_mem_size);
		}
		free(devices);
		devices = NULL;
	}
	free(platforms);
	platforms = NULL;

	return device;
}

void initOpenCL(int gpu_idx)
{
	cl_int err = 0;
	char str[STR_LEN] = { 0 };

	float range_end = 1000.0f;
	const int work_cnt = 538870912 / (int)pow(2, 10);

	cl_device_id device = getDevice();

	// get max work group size
	size_t max_work_group_size = 0;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_ERROR(err);

	// init work group size
	size_t work_group_size = max_work_group_size;
	while (work_cnt % work_group_size != 0)
		work_group_size /= 2;

	// init work group cnt
	const int workGroupCnt = work_cnt / work_group_size;
	printf("work_group_size=%u \n", work_group_size);

	cl_ulong local_mem_size = 0;
	err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	CHECK_ERROR(err);

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	// 2.0
	//cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT, 0 };
	//queue = clCreateCommandQueueWithProperties(context, devices[gpu_idx], NULL, &err);
	cl_command_queue data_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	CHECK_ERROR(err);
	cl_command_queue kernel_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	CHECK_ERROR(err);

	cl_kernel kernel = getKernel(context, device, "integral.cl", "integral");

	//cl_mem bufOut = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Item) * workGroupCnt, NULL, &err);
	//CHECK_ERROR(err);

	//err = clSetKernelArg(kernel, 0, sizeof(cl_float), &range_end);
	//CHECK_ERROR(err);
	//err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &work_cnt);
	//CHECK_ERROR(err);
	//err = clSetKernelArg(kernel, 2, sizeof(cl_float) * work_group_size, NULL);
	//CHECK_ERROR(err);
	//err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufOut);
	//CHECK_ERROR(err);

	//const int work_dim = 1;
	//size_t* global_work_size = (size_t*)calloc(sizeof(size_t), work_dim);
	//global_work_size[0] = work_group_size;

	//size_t* local_work_size = (size_t*)calloc(sizeof(size_t), work_dim);
	//local_work_size[0] = work_group_size;

	//Vector out = initVector(workGroupCnt);
	//cl_event kernel_event;

	//for (int offset = 0; offset < workGroupCnt; offset++)
	//{
	//	err = clEnqueueNDRangeKernel(
	//		kernel_queue, kernel, work_dim, NULL,
	//		global_work_size, local_work_size,
	//		0, NULL, &kernel_event[offset]);
	//	CHECK_ERROR(err);
	//}

	//double gpu_sum = 0;
	//cl_event* read_event = (cl_event*)calloc(workGroupCnt, sizeof(cl_event));
	//for (int offset = 0; offset < workGroupCnt; offset++)
	//{
	//	// stuck
	//	//err = clEnqueueReadBuffer(data_queue, bufOut, CL_TRUE, sizeof(Item) * offset, sizeof(Item), &out.arr[offset],
	//	//    1, &kernel_event[offset], &read_event[offset]);

	//	clWaitForEvents(1, &kernel_event[offset]);
	//	err = clEnqueueReadBuffer(data_queue, bufOut, CL_TRUE, sizeof(Item) * offset, sizeof(Item), &out.arr[offset],
	//		0, NULL, &read_event[offset]);
	//	CHECK_ERROR(err);
	//	//printf("out.arr[%d]=%f \n", offset, out.arr[offset]);
	//	gpu_sum += out.arr[offset];
	//}

	//cl_ulong total_start = 0, total_end = 0;
	//err = clGetEventProfilingInfo(kernel_event[0], CL_PROFILING_COMMAND_START,
	//	sizeof(cl_ulong), &total_start, NULL);
	//CHECK_ERROR(err);
	//err = clGetEventProfilingInfo(read_event[workGroupCnt - 1], CL_PROFILING_COMMAND_END,
	//	sizeof(cl_ulong), &total_end, NULL);
	//CHECK_ERROR(err);
	//printf("elapsed gpu total time = %llu ns \n", total_end - total_start);
	//printf("gpu sum : %f \n", gpu_sum);

	//free(kernel_event);
	//free(read_event);
	//free(id_offset);
	//free(group_offset);
	//free(global_work_size);
	//free(local_work_size);
	//freeVector(out);

	//return 0;
}