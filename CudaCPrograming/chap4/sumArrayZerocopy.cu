#include <cuda_runtime.h>
#include <stdio.h>

void checkresult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;

	for(int i = 0; i < N; i++)
	{
		if(abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			printf("Arrays do not match");
			printf("Host %5.2f gpu %5.2f at current %d/n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	return;
}

void initialData(float *ip, int size)
{
	int i;
	for(i = 0; i < size; i++)
	{
		ip[i] = (float)(randn() & 0xFF) / 10.0f;
	}
	return;
}

void sumArrayHost(float *A, float *B, float *C, const int N)
{
	for(int i = 0; i < N; i++)
		C[i] = A[i] + B[i];
}

__global__ void sumArrays(float* A, float *B, float *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char**argv)
{
	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("device %dL %s memory size %d nbytes %5.2fMB\n", dev, deviceProp.name, isize, nbytes/(1024.0f * 1024.0f));

	int ipower = 10;
	if(argc > 1) ipower = atoi(argv[i]);
	int nElem = 1 << ipower;
	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	initialData(h_A, nElem);
	initialData(h_B, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	sumArrayHost(h_A, h_B, hostRef, nElem);

	float *d_A, *d_B, *d_C;
	cudaMalloc((float **)&d_A, nBytes);
	cudaMalloc((float **)&d_B, nBytes);
	cudaMalloc((float **)&d_C, nBytes);
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	int iLen = 512;
	dim3 block(iLen);
	dim3 grid ((nElem + block.x - 1) / block.x);

	sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);

	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	checkresult(hostRef, gpuRef, nElem);

	free(h_A);
	free(h_B);
	cudaFree(d_A);
	cudaFree(d_B);

	//zero copy
	cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped);

	initialData(h_A, nElem);
	initialData(h_B, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	cudaHostGetDevicePointer((void **)&d_A, (void **)&h_A, 0);
	cudaHostGetDevicePointer((void **)&d_B, (void **)&h_B, 0);

	sumArrayHost(h_A, h_B, hostRef, nElem);
	sumArraysZeroCopy(d_A, d_B, gpuRef, nElem);

	cudaMemcpyDeviceToHost(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	checkresult(hostRef, gpuRef, nElem);

	cudaFree(d_C);
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	free(gpuRef);
	free(hostRef);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
