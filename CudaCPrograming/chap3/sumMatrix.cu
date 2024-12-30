#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *ip, const int size)
{
	for(int i = 0; i < size; i++)
		ip[i] = (float) (std::rand() & 0xFF) / 10.0f;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;

	for(int iy = 0; iy < ny; iy++)
	{
		for(int ix = 0; ix < nx; ix++)
		{
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}
	return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double episilon = 1.0E-8;
	for(int i = 0; i < N; i++)
	{
		if(abs(hostRef[i] - gpuRef[i]) > episilon)
		{
			printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
			printf("Array do not match.\n\n");
			break;
		}
	}
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
	unsigned int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
	unsigned int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	unsigned int idx = thread_id + blockDim.x * blockDim.y * block_id;

	if(idx < NX * NY)
		C[idx] = A[idx] + B[idx];
}

int main(int argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	CHECK(cudaSetDevice(dev));

	int nx = 1 << 14;
	int ny = 1 << 14;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	size_t iStart = seconds();
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	size_t iElaps = seconds() - iStart;

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	iStart = seconds();
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
	iElaps = seconds() - iStart;

	float *d_A, *d_B, *d_C;
	CHECK(cudaMalloc((void **)&d_A, nBytes));
	CHECK(cudaMalloc((void **)&d_B, nBytes));
	CHECK(cudaMalloc((void **)&d_C, nBytes));

	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

	int dimx = 32;
	int dimy = 32;

	if(argc > 2)
	{
		dimx = atoi(argv[1]);
		dimy = atoi(argv[2]);
	}

	dim3 block(dimx, dimy);
	dim3 grid((nx + dimx - 1) / dimx, (ny + dimy - 1) / dimy);

	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	printf("sumMatrixOnGPU2D <<<(%d, %d), (%d, %d)>>> elapsed %d ms\n", grid.x, grid.y, block.x, block.y, iElaps);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkResult(hostRef, gpuRef, nxy);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}
