#include "../common//common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 300000
#define NSTREAM 4

__global__ void kernel_1()
{
	double sum = 0.0;
	for(int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}
__global__ void kernel_2()
{
	double sum = 0.0;
	for(int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}
__global__ void kernel_3()
{
	double sum = 0.0;
	for(int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}
__global__ void kernel_4()
{
	double sum = 0.0;
	for(int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}

int main(int argc, char **argv)
{
	int n_streams = NSTREAM;
	int isize = 1;
	int iblock = 1;
	int bigcase = 0;

	if(argc > 1) n_streams = atoi(argv[1]);
	if(argc > 2) bigcase = atoi(argv[2]);

	float elapsed_time;

	char *iname = "CUDA_DEVICE_MAX_CONNECTIONS";
	setenv(iname, "32", 1);
	char *ivalue = getenv(iname);
	printf("%s = %s\n", iname, ivalue);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name, n_streams);
	CHECK(cudaSetDevice(dev));

	if(deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
	{
		if(deviceProp.concurrentKernels == 0)
		{
			printf("> GPU does not support concurrent kernel execution (SM3.5 or higher required)\n");
			printf("> CUDA kernel runs will be serialized\n");
		}
		else
		{
			printf("> GPU does not support Hyper Q\n");
			printf("> GPU kernel runs will have limited concurrency\n");
		}
	}
	printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

	cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(cudaStream_t));
	for (int i = 0; i < n_streams; i++)
		CHECK(cudaStreamCreate(&streams[i]));

	if (bigcase == 1)
	{
		iblock = 512;
		isize = 1 << 12;
	}

	dim3 block(iblock);
	dim3 grid(isize / iblock);
	printf("> grid %d block %d\n", grid.x, block.x);

	cudaEvent_t start, stop;
	CHECK(cudaEventRecord(start, 0));

	for(int i = 0; i < n_streams; i++)
	{
		kernel_1 <<<grid, block, 0, streams[i]>>>();
		kernel_2 <<<grid, block, 0, streams[i]>>>();
		kernel_3 <<<grid, block, 0, streams[i]>>>();
		kernel_4 <<<grid, block, 0, streams[i]>>>();
	}

	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchonize(stop));

	CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Measure time for parallel execution = %.3f s\n", elapsed_time / 1000.0f);

	for(int i = 0; i < n_streams; i++)
		CHECK(cudaStreamDestroy(streams[i]));
	free(streams);

	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));
	CHECK(cudaDeviceReset());

	return 0;
}
