#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char**argv)
{
	int dev = 0;
	cudaSetDevice(dev);

	unsigned int isize = 1<<22;
	unsigned int nbytes = isize * sizeof(float);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("device %dL %s memory size %d nbytes %5.2fMB\n", dev, deviceProp.name, isize, nbytes/(1024.0f * 1024.0f));

	float *h_a = (float*) malloc(nbytes);
	cudaMallocHost((float**)&h_a, nbytes);
	for (unsigned int i = 0; i < nbytes; i++)
		h_a[i] = 0.5f;

	float *d_a;
	cudaMalloc((float**)&d_a, nbytes);

	cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDevideToHost);

	cudaFree(d_a);
	cudaFreeHost(h_a);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
