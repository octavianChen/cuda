#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable()
{
	printf("Device: The value of the global variable is %f\n", devData);
    devData += 2.0f;
}

int main(void)
{
	float value = 3.14;
	cudaMemcpyToSymbol(devData, &value, sizeof(float));
	printf("Host: copy %f to the global variable\n", value);

	checkGlobalVariable();

	cudaMemcpyFromSymbol(&value, devData, sizeof(float));
	printf("Host: The value changed by kernel to %f\n", value);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
