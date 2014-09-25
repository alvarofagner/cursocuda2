/**
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "util.h"


#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)

__global__ void kernel(int *a, int *b, int *c) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}


int main(void) {
	cudaDeviceProp  prop;
	int whichDevice;
	CUDA_CHECK_RETURN(cudaGetDevice(&whichDevice));
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	cudaEvent_t     start, stop;
	float           elapsedTime;

	cudaStream_t    stream0, stream1;
	int *host_a, *host_b, *host_c;
	int *dev_a0, *dev_b0, *dev_c0;
	int *dev_a1, *dev_b1, *dev_c1;

	// start the timers
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	// initialize the streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream0));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));

	// allocate the memory on the GPU
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a0,
		N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_b0,
		N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_c0,
		N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a1,
		N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_b1,
		N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_c1,
		N * sizeof(int)));

	// allocate host locked memory, used to stream
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&host_a,
		FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&host_b,
		FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&host_c,
		FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));

	for (int i = 0; i<FULL_DATA_SIZE; i++) {
		host_a[i] = rand();
		host_b[i] = rand();
	}

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	// now loop over full data, in bite-sized chunks
	for (int i = 0; i<FULL_DATA_SIZE; i += N * 2) {
		// enqueue copies


		// enqueue kernels in stream0 and stream1

		// enqueue copies of c from device to locked memory

	}
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream0));
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time taken:  %3.1f ms\n", elapsedTime);

	// cleanup the streams and memory
	CUDA_CHECK_RETURN(cudaFreeHost(host_a));
	CUDA_CHECK_RETURN(cudaFreeHost(host_b));
	CUDA_CHECK_RETURN(cudaFreeHost(host_c));
	CUDA_CHECK_RETURN(cudaFree(dev_a0));
	CUDA_CHECK_RETURN(cudaFree(dev_b0));
	CUDA_CHECK_RETURN(cudaFree(dev_c0));
	CUDA_CHECK_RETURN(cudaFree(dev_a1));
	CUDA_CHECK_RETURN(cudaFree(dev_b1));
	CUDA_CHECK_RETURN(cudaFree(dev_c1));
	CUDA_CHECK_RETURN(cudaStreamDestroy(stream0));
	CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));

	return 0;
}


