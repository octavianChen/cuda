#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

int recursiveReduce(int *data, int const size)
{
	if(size == 1) return data[0];

	int const stride = size / 2;
	for(int i = 0; i < stride; i++)
		data[i] += data[i + stride];
	return recursiveReduce(data, stride);
}

__global__ void reduceNeighboared(int *g_idata, int *g_odata, unsigned int n)
{
	// block and thread are 1D
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	int *idata = g_idata + blockIdx.x * blockDim.x;

	if(idx >= n) return;

	for(int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if((tid % (2 * stride)) == 0)
			idata[tid] += idata[tid + stride];

		__syncthreads();
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboaredLess(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx + blockDim.x * blockIdx.x;
	int *idata = g_idata + blockIdx.x * blockDim.x;

	if(idx >= n) return;

	for(int stride=1; stride < blockDim.x; stride *= 2)
	{
		int index = 2 * stride * tid;
		if(index < blockDim.x)
			idata[index] += idata[index + stride];

		__syncthreads();
	}
	if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int *idata = g_idata + blockIdx.x * blockDim.x;
	if(idx >= n) return;

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];

		__syncthreads();
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;

	int *idata = g_idata + blockIdx * blockDim.x * 2;

	if(idx + blockDim.x < n)
		g_idata[idx] += g_idata[idx + blockDim.x];
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];
		__syncthreads();
	}
	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = tid + blockIdx.x * blockDim.x * 4;
	int *idata = g_idata + blockIdx.x * blockDim.x * 4;

	if(idx + 3 * blockDim.x < n)
		g_idata[idx] = g_idata[idx] + g_idata[idx + blockDim.x] + g_idata[idx + blockDim.x * 2] + g_idata[idx + blockDim.x * 3];
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];
		__syncthreads();
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
	int *idata = g_idata + blockDim.x * blockIdx.x * 8;

	if(idx + 7 * blockDim.x < n)
	{
		int a0 = g_idata[idx];
		int a1 = g_idata[idx + blockDim.x];
		int a2 = g_idata[idx + blockDim.x * 2];
		int a3 = g_idata[idx + blockDim.x * 3];
		int a4 = g_idata[idx + blockDim.x * 4];
		int a5 = g_idata[idx + blockDim.x * 5];
		int a6 = g_idata[idx + blockDim.x * 6];
		int a7 = g_idata[idx + blockDim.x * 7];
		g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
	}
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];
		__syncthreads();
	}
	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
	int *idata = g_idata + blockDim.x * blockIdx.x * 8;

	if(idx + 7 * blockDim.x < n)
	{
		int a0 = g_idata[idx];
		int a1 = g_idata[idx + blockDim.x];
		int a2 = g_idata[idx + blockDim.x * 2];
		int a3 = g_idata[idx + blockDim.x * 3];
		int a4 = g_idata[idx + blockDim.x * 4];
		int a5 = g_idata[idx + blockDim.x * 5];
		int a6 = g_idata[idx + blockDim.x * 6];
		int a7 = g_idata[idx + blockDim.x * 7];
		g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
	}
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];
		__syncthreads();
	}

	if(tid < 32)
	{
		volatile int *vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
	int *idata = g_idata + blockIdx.x * blockDim.x * 8;

	if(idx + 7 * blockDim.x < n)
	{
		int a0 = g_idata[idx];
		int a1 = g_idata[idx + blockDim.x];
		int a2 = g_idata[idx + blockDim.x * 2];
		int a3 = g_idata[idx + blockDim.x * 3];
		int a4 = g_idata[idx + blockDim.x * 4];
		int a5 = g_idata[idx + blockDim.x * 5];
		int a6 = g_idata[idx + blockDim.x * 6];
		int a7 = g_idata[idx + blockDim.x * 7];
		g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
	}
	__syncthreads();

	if(blockDim.x >= 1024 && tid < 512)
		idata[tid] += idata[tid + 512];
	__syncthreads();

	if(blockDim.x >= 512 && tid < 256)
		idata[tid] += idata[tid + 256];
	__syncthreads();

	if(blockDim.x >= 256 && tid < 128)
		idata[tid] += idata[tid + 128];
	__syncthreads();

	if(blockDim.x >= 128 && tid < 64)
		idata[tid] += idata[tid + 64];
	__syncthreads();

	if(tid < 32)
	{
		volatile int *vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
	int *idata = g_idata + blockDim.x * blockIdx.x * 8;

	if(idx + 7 * blockDim.x < n)
	{
		int a0 = g_idata[idx];
		int a1 = g_idata[idx + blockDim.x];
		int a2 = g_idata[idx + blockDim.x * 2];
		int a3 = g_idata[idx + blockDim.x * 3];
		int a4 = g_idata[idx + blockDim.x * 4];
		int a5 = g_idata[idx + blockDim.x * 5];
		int a6 = g_idata[idx + blockDim.x * 6];
		int a7 = g_idata[idx + blockDim.x * 7];
		g_idata[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
	}
	__syncthreads();

	if(blockDim.x >= 1024 && tid < 512)
		idata[tid] += idata[tid + 512];
	__syncthreads();

	if(blockDim.x >= 512 && tid < 256)
		idata[tid] += idata[tid + 256];
	__syncthreads();

	if(blockDim.x >= 256 && tid < 128)
		idata[tid] += idata[tid + 128];
	__syncthreads();

	if(blockDim.x >= 128 && tid < 64)
		idata[tid] += idata[tid + 64];
	__syncthreads();

	if(tid < 32)
	{
		volatile int *vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps(int *g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 2;
	int *idata = g_idata + blockDim.x * blockIdx.x * 2;

	if(idx + blockDim.x < n)
		g_idata[idx] += g_idata[idx + blockDim.x];
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];
		__syncthreads();
	}

	if(tid < 32)
	{
		volatile int *vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **agrv)
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting reduction at ", agrv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	bool bResult = false;
	int size = 1 << 24;
	int blocksize = 512;
	if(argc > 1)
		blocksize = atoi(argv[i]);

	dim3 block(blocksize, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);
	printf("grid %d block %d\n", grid.x, block.x);

	size_t bytes = size * sizeof(int);
	int *h_idata = (int *)malloc(bytes);
	int *h_odata = (int *)malloc(grid.x * sizeof(int));
	int *tmp = (int *)malloc(bytes);

	for(int i = 0; i < size; i++)
		h_idata[i] = (int)(std::rand() & 0xFF);
	memcpy(tmp, h_idata, bytes);

	double iStart, iElaps;
	int gpu_sum = 0;

	int *d_idata = NULL;
	int *d_odata = NULL;
	CHECK(cudaMalloc((void **) &d_idata, bytes));
	CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

	iStart = seconds();
	int cpu_sum = recursiveReduce(tmp, size);
	iElaps = seconds() - iStart;
	printf("cpu reduce elapsed %f sec, cpu_sum: %d\n", iElaps, cpu_sum);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceNeighboared<<<grid, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x; i++)
		gpu_sum += h_odata[i];
	printf("gpu Neighboured elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceNeighboaredLess<<<grid, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x; i++)
		gpu_sum += h_odata[i];
	printf("gpu NeighbouredLess elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);


	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x; i++)
		gpu_sum += h_odata[i];
	printf("gpu reduceInterleaved elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x/2 * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x/2; i++)
		gpu_sum += h_odata[i];
	printf("gpu reduceUnrolling2 elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x/2, block.x);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceInterleaved<<<grid.x/4, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x/4 * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x/4; i++)
		gpu_sum += h_odata[i];
	printf("gpu reduceUnrolling4 elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x/4, block.x);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceUnrolling8<<<grid.x/8, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x/8; i++)
		gpu_sum += h_odata[i];
	printf("gpu reduceUnrolling8 elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x/8; i++)
		gpu_sum += h_odata[i];
	printf("gpu reduceUnrollWarps8 elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceCompleteUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x/8; i++)
		gpu_sum += h_odata[i];
	printf("gpu reduceCompleteUnrollWarps8 elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();

	switch (blocksize)
	{
	case 1024:
		reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
		break;
	case 512:
		reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
		break;
	case 256:
		reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
		break;
	case 128:
		reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
		break;
	case 64:
		reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
		break;
	}
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i = 0; i < grid.x/8; i++)
		gpu_sum += h_odata[i];
	printf("gpu reduceCompleteUnroll elapsed %f sec, gpu_sum: %d, <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

	free(h_idata);
	free(h_odata);
	CHECK(cudaFree(d_idata));
	CHECK(cudaFree(d_odata));

	CHECK(cudaDeviceReset());

	bResult = (gpu_sum == cpu_sum);
	if(!bResult)
		printf("Test Failed\n");
	return EXIT_SUCCESS;
}
