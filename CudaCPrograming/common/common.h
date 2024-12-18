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
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
