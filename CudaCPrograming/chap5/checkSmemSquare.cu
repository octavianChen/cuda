#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIMX 32
#define BDIMY 32
#define IPAD 1

void printData(char *msg, int *in, const int size)
{
	printf("%s: ", msg);

	for(int i = 0; i < size; i++)
	{
		printf("%4d", in[i]);
		fflush(stdout);
	}
	printf("\n\n");
}

__global__ void setRowReadRow(int *out)
{
	__shared__ int tile[BDIMY][BDIMX];

	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
	__synthreads();
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out)
{
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.x][threadIdx.y] = idx;
	__synthreads();
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out)
{
	__shared__ int tile[BDIMY][BDIMX];

	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
	__synthreads();

	out[idx] = tile[threadIdx.x][threadIdx.y];
}


__global__ void setRowReadColPad(int *out)
{
	__shared__ int tile[BDIMY][BDIMX + IPAD];

	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
	__synthreads();
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int *out)
{
	extern __shared__ int tile[];

	unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

	tile[row_idx] = row_idx;
	__synthreads();
	out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColDynPad(int *out)
{
	extern __shared__ int tile[];
	unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
	unsigned int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;
	unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[row_idx] = g_idx;
	__synthreads();
	out[g_idx] = tile[col_idx];
}

int main(int argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	cudaShareMemConfig pConfig;
	CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
	printf("with Bank Mode:%s", pConfig == 1? "4-Byte": "8-Byte");

	int nx = BDIMX;
	int ny = BDIMY;

	bool iprintf = 0;

	if(argc > 1) iprintf = atoi(argv[1]);

	size_t nBytes = nx * ny * sizeof(int);

	dim3 block(BDIMX, BDIMY);
	dim3 grid (1, 1);
	prinf("<<<grid (%d, %d), block (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	int *d_c;
	CHECK(cudaMalloc((int *)&d_C, nBytes));
	int *gpuRef = (int *)malloc(nBytes);

	CHECK(cudaMemset(d_C, 0, nBytes));
	setRowReadRow<<<grid, block>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if(iprintf) printData("setRowReadRow   ", gpuRef, nx * ny);

	CHECK(cudaMemset(d_C, 0, nBytes));
	setColReadCol<<<grid, block>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if(iprintf) printData("setColReadCol   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
	setRowReadCol<<<grid, block>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if(iprintf) printData("setRowReadCol   ", gpuRef, nx * ny);

	CHECK(cudaMemset(d_C, 0, nBytes));
	setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizof(int)>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if(iprintf) printData("setRowReadColDyn   ", gpuRef, nx * ny);


    CHECK(cudaMemset(d_C, 0, nBytes));
	setRowReadColPad<<<grid, block>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if(iprintf) printData("setRowReadColPad   ", gpuRef, nx * ny);


    CHECK(cudaMemset(d_C, 0, nBytes));
	setRowReadColDynPad<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(int)>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if(iprintf) printData("setRowReadColDynPad   ", gpuRef, nx * ny);

	CHECK(cudaFree(d_C));
	free(gpuRef);

	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}
