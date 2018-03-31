#pragma once

#include <stdio.h>  
#include <stdlib.h>
#include <time.h> 

#include <assert.h>
#include <cuda.h>  

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <texture_types.h>
#include <texture_fetch_functions.h>

#define CUDA_SAFE_CALL(err) _CUDA_SAFE_CALL(err, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR()    __cudaCheckError( __FILE__, __LINE__ )

struct Bin2D {
	int x;
	int y;

	int W;
	int H;
};

struct TreeNode {

	int BinId;

	int left_child;
	int right_child;

	int x;
	int y;

	int Wr;
	int Hr;
};

inline void _CUDA_SAFE_CALL(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err) {
		printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
			file, line, cudaGetErrorString(err));
	}
};

inline void __cudaCheckError(const char *file, const int line)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		//exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		//exit(-1);
	}
};
