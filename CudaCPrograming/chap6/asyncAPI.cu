#include "../common//common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void kernel(float *g_data, float value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	g_data[idx] = g_data[idx] + value;
}

void checkResults(float *data, const int n, const float x)
{
	for(int i = 0; i < n; i++)
	{
		if (data[i] != x)
		{
			printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
			return 0;
		}
	}
	return 1;
}

int main(int argc, char **argv)
{
	printf("> %s Starting...\n", argv[0]);

	int devID = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, devID));
	printf("> Using Device %d: %s\n", devID, deviceProp.name);
	CHECK(cudaSetDevice(devID));

	int nElem = 1 << 24;
	printf("> vector size = %d\n", nElem);
	size_t nBytes = nElem * sizeof(int);
	float value = 10.0f;

	float *h_a = 0;
	CHECK(cudaMallocHost((void **)&h_a, nBytes));
	memset(h_a, 0, nBytes);

	float *d_a = 0;
	CHECK(cudaMalloc((void **)&d_a, nBytes));
	CHECK(cudaMemset(d_a, 255, nBytes));

	dim3 block(512);
	dim3 grid((num + block.x - 1) / block.x);
	printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x, block.y);

	cudaEvent_t stop;
	CHECK(cudaEventCreate(&stop));
    CHECK(cudaMemcpyAsync(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
	kernel<<<grid, block>>>(d_a, value);
	CHECK(cudaMemcpyAsync(h_a, d_a, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaEventRecord(&stop));

	unsigned long int counter = 0;
	while(cudaEventQuery(stop) == cudaErrorNotReady)
		counter++;
	printf("CPU executed &lu iterations while waiting for GPU to finish\n", counter);

	bool bFinalResults = (bool) checkResults(h_a, num, value);

	CHECK(cudaEventDestroy(stop));
	CHECK(cudaFreeHost(h_a));
	CHECK(cudaFree(d_a));
	CHECK(cudaDeviceReset());

	exit(bFinalResults ? EXIT_SUCCESS: EXIT_FAILURE);
}
