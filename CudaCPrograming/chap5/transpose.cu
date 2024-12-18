#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <string>
#include "../common/common.h"

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
		ip[i] = (float)(std::rand() & 0xFF) / 10.0f;
	}
	return;
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
	for(int iy = 0; iy < ny; iy++)
		for(int ix = 0; ix < nx; ix++)
			out[ix * ny + iy] = in[iy * nx + ix];
}

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}

// copy data in rows
__global__ void copyRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}

// copy data in cols
__global__ void copyCol(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[ix * ny + iy] = in[ix * ny + iy];
}

//transpose: read in rows and write in cols
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[ix * ny + iy] = in[iy * nx + ix];
}

// transpose: read in cols and write in rows;
__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[iy * nx + ix] = in[ix * ny + iy];
}

//transpose: read in rows and write in cols + unroll 4 blocks
__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int ti = iy * ny + ix;
	unsigned int to = ix * ny + iy;

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[to] = in[ti];
		out[to + ny * blockDim.x] = in[ti + blockDim.x];
		out[to + 2 * ny * blockDim.x] = in[ti + 2 * blockDim.x];
		out[to + 3 * ny * blockDim.x] = in[ti + 3 * blockDim.x];
	}
}

//transpose: read in cols and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y * 4 + threadIdx.y;
	unsigned int ti = iy * nx + ix;
	unsigned int to = ix * ny + iy;

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[ti] = in[to];
		out[ti + blockDim.x] = in[to + blockDim.x * ny];
		out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
		out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
	}
}

//transpose: read in rows and write in cols + diagonal coordinate transpose
__global__ void transposeDiagonalRow(float *out, float *in, const int nx, const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix = blockDim.x * blk_x + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[iy * nx + ix];
	}
}

//transpose: read in cols and write in cols + diagonal coordinate transpose
__global__ void transposeDiagonalCol(float *out, float *in, const int nx, const int ny)
{
	unsigned int blk_y = blockIdx.x;
	unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

	unsigned int ix = blockDim.x * blk_x + threadIdx.x;
	unsigned int iy = blockDim.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[iy * nx + ix] = in[ix * ny + iy];
}

int main(argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("device %d: %s", dev, deviceProp.name);
	cudaSetDevice(dev);

	int nx = 1 << 11;
	int ny = 1 << 11;
	int iKernel = 0;
	int blockx = 16;
	int blocky = 16;

	if (argc > 1) iKernel = atoi(argv[1]);
	if (argc > 2) blockx = atoi(argv[2]);
	if (argc > 3) blocky = atoi(argv[3]);
	if (argc > 4) nx = atoi(argv[4]);
	if (argc > 5) ny = atoi(argv[5]);

	size_t nBytes = nx * ny * sizeof(float);

	dim3 block(blockx, blocky);
	dim3 grid((nx + blockx - 1) / blockx, (ny + blocky - 1) / blocky);

	float *h_A = (float *)malloc(nBytes);
	float *hostRef = (float *)malloc(nBytes);
	float *gpuRef = (float *)malloc(nBytes);

	initialData(h_A, nx * ny);
	transposeHost(hostRef, h_A, nx, ny);

	float *d_A, *d_C;
	cudaMalloc((float **)&d_A, nBytes);
	cudaMalloc((float **)&d_C, nBytes);
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

	double iStart = seconds();
	warmup<<<grid, block>>>(d_C, d_A, nx, ny);
	cudaDeviceSynchronize();
	double iElaps = seconds() - iStart;
	printf("warmup elapsed %f sec\n", iElaps);
	cudaGetLastError();

	// kernel pointer and descriptor
	void(*kernel)(float *, float *, int, int);
	std::string kernelName;

	switch(iKernel)
	{
		case 0:
			kernel = &copyRow;
			kernelName = "CopyRow    ";
			break;
		case 1:
			kernel = &copyCol;
			kernelName = "CopyCol    ";
			break;
		case 2:
			kernel = &transposeNaiveRow;
			kernelName = "NaiveRow    ";
			break;
		case 3:
			kernel = &transposeNaiveCol;
			kernelName = "NaiveCol    ";
			break;
		case 4:
			kernel = &transposeUnroll4Row;
			kernelName = "Unroll4Row    ";
			grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
			break;
		case 5:
			kernel = &transposeUnroll4Col;
			kernelName = "Unroll4Col    ";
			grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
			break;
		case 6:
			kernel = &transposeDiagonalRow;
			kernelName = "DiagonalRow    ";
			break;
		case 7:
			kernel = &transposeDiagonalCol;
			kernelName = "DiagonalCol    ";
			break;
	}

	iStart = seconds();
	kernel<<<grid, block>>>(d_C, d_A, nx, ny);
	cudaDeviceSynchronize();
	iElaps = seconds() - iStart;

	float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
	printf("%s elapsed %d sec <<<grid (%d, %d), block (%d, %d)>>> effective bandwidth %f GB\n", kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);
	if(iKernel > 1)
	{
		cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
		cudaGetLastError();
	}

	cudaFree(d_A);
	cudaFree(d_C);
	free(h_A);
	free(hostRef);
	free(gpuRef);
	cudaDeviceReset();

	return EXIT_SUCCESS;
}
