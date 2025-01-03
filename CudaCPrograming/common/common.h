/*************************************************************************
	> File Name: common.h
	> Created Time: 2024-12-18 21:39
 ************************************************************************/

#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H
#endif

inline double seconds()
{
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double)tp.tv_sec + (double)tp.tv_usec);
}

#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if(error != cudaSuccess) \
	{ \
		fprintf(stderr, "Error: %s: %d", __FILE__, __LINE__); \
		fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
	} \
}

#define CHECK_CUBLAS(call) \
{ \
	cudablasStatus_t error = call; \
	if (error != CUBLAS_STATUS_SUCCESS) \
	{ \
		fprintf(stderr, "Got CUBLAS error %d at %s: %d\n", err, __FILE__, __LINE__); \
		exit(1); \
	} \
}
