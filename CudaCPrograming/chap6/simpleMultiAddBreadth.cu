#include "../common//common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define NSTREAM 4
#define BDIM 128

void initialData(float *ip, int size)
{
	int i;
	for(int i = 0; i < size; i++)
		ip[i] = (float) std::rand() & 0xFF / 10.0f;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	for(int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		for(int i = 0; i < N; i++)
			C[idx] = A[idx] + B[idx];
	}
}

void checkResults(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for(int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef) > epsilon)
		{
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
}

int main(int argc, char **argv)
{
	printf("> %s Starting...\n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("> Using Device %d: %s\n", dev, deviceProp.name);
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


	char *iname = "CUDA_DEVICE_MAX_CONNECTIONS";
	setenv(iname, "1", 1);
	char *ivalue = getenv(iname);
	printf("> %s=%s\n", iname, ivalue);
	printf("> with streams=%d\n", NSTREAM);

	int nElem = 1 << 18;
	printf("> vector size = %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
	CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault));
	CHECK(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));
	CHECK(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));

	initialData(h_A, nElem);
	initialData(h_B, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	float *d_A, *d_B, *d_C;
	CHECK(cudaMalloc((float **)&d_A, nBytes));
	CHECK(cudaMalloc((float **)&d_B, nBytes));
	CHECK(cudaMalloc((float **)&d_C, nBytes));

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));

	dim3 block(BDIM);
	dim3 grid((nElem + block.x - 1) / block.x);
	printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x, block.y);

	CHECK(cudaEventRecord(start, 0));
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchonize(stop));
	float memcpy_h2d_time;
	CHECK(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));

	CHECK(cudaEventRecord(start, 0));
	sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchonize(stop));
	float kernel_time;
	CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

	CHECK(cudaEventRecord(start, 0));
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchonize(stop));
	float memcpy_d2h_time;
	CHECK(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
	float itotal = memcpy_h2d_time + kernel_time + memcpy_d2h_time;

	printf("\n");
	printf("Measured timings (throughput): \n");
	printf("Memcpy host to device\t: %f ms (%f GB/s)\n", memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
	printf("Memcpy device to host\t: %f ms (%f GB/s)\n", memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
	printf("Kernel\t\t\t: %f ms (%f GB/s)\n", kernel_time, (nBytes * 2e-6) / kernel_time);
	printf("Total\t\t\t: %f ms (%f GB/s)\n", itotal, (nBytes * 2e-6) / itotal);

	// grid parallel 
	int iElem = nElem / NSTREAM;
	size_t iBytes = iElem * sizeof(float);
	grid.x = (iElem + block.x - 1) / block.x;

	cudaStream_t stream[NSTREAM];
	for(int i = 0; i < NSTREAM; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	CHECK(cudaEventRecord(start, 0));
	for(int i = 0; i < NSTREAM; i++)
	{
		int ioffset = i * iElem;
		CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
		CHECK(cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
	}

	for(int i = 0; i < NSTREAM; i++)
	{
		int ioffset = i * iElem;
		sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset], *d_C[ioffset], iElem);
	}


	for(int i = 0; i < NSTREAM; i++)
	{
		int ioffset = i * iElem;
		CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]));
	}

	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchonize(stop));
	float execution_time;
	CHECK(cudaEventElapsedTime(&execution_time, start, stop));

	printf("\n");
	printf("Actual results from overlapped data transfers:\n");
	printf(" overlap with %d streams: %f ms (%f GB/s)\n", NSTREAM, execution_time, (nBytes * 2e-6)/execution_time);
	printf("Speedup			:%f\n", ((itotal -execution_time) * 100.0f) / itotal);

	CHECK(cudaGetLastError());

	checkResults(hostRef, gpuRef, nElem);

	//free memory
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));
	CHECK(cudaFreeHost(h_A));
	CHECK(cudaFreeHost(h_B));
	CHECK(cudaFreeHost(hostRef));
	CHECK(cudaFreeHost(gpuRef));

	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	for(int i = 0; i < NSTREAM; i++)
		CHECK(cudaStreamDestroy(streams[i]));

	CHECK(cudaDeviceReset());

	return 0;
}
