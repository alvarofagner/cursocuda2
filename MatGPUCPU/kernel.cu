#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"
#include "util.h"

#define BLOCK_X 8
#define BLOCK_Y 8

/*matrix multiplication kernels*/
__global__ void
MatrixMulDevice(float *Md, float *Nd, float *Pd, const int width)
{
	// calculate thread id
	unsigned int col = (blockDim.x * blockIdx.x) + threadIdx.x;
	unsigned int row = (blockDim.y * blockIdx.y) + threadIdx.y;
	unsigned int offset = row * width + col;

	if (offset < width * width)
	{

		float sum = 0;
		for (int k = 0; k < width; k++)
		{
			sum += Md[row * width + k] * Nd[k * width + col];
		}

		Pd[row*width + col] = sum;
	}
}


// main routine
int main()
{
	const int WIDTH = 128;
	float array1_h[WIDTH*WIDTH], array2_h[WIDTH*WIDTH],
		result_array_h[WIDTH*WIDTH], M_result_array_h[WIDTH*WIDTH];
	float *array1_d, *array2_d, *M_result_array_d; // device array

	float gpu_processing = 0;

	//input in host array
	for (int i = 0; i<WIDTH; i++)
	{
		for (int j = 0; j<WIDTH; j++)
		{
			array1_h[i * WIDTH + j] = 1;
			array2_h[i * WIDTH + j] = 2;
		}
	}

	//Reset no device
	CUDA_CHECK_RETURN(cudaDeviceReset());


	CUDA_CHECK_RETURN(cudaMalloc((void **)&array1_d, WIDTH*WIDTH*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&array2_d, WIDTH*WIDTH*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&M_result_array_d, WIDTH*WIDTH*sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(array1_d, array1_h, WIDTH*WIDTH*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(array2_d, array2_h, WIDTH*WIDTH*sizeof(float), cudaMemcpyHostToDevice));


	//calling kernal
	dim3 dimGrid(WIDTH / BLOCK_X, WIDTH / BLOCK_Y, 1);
	dim3 dimBlock(BLOCK_X, BLOCK_Y, 1);

	cudaEvent_t beginEvent;
	cudaEvent_t endEvent;

	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(beginEvent, 0);

	MatrixMulDevice << <dimGrid, dimBlock >> > (array1_d, array2_d, M_result_array_d, WIDTH);

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);

	cudaEventElapsedTime(&gpu_processing, beginEvent, endEvent);

	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	// Processing in CPU
	StopWatch watch;
	watch.start();
	MatrixMulHost(array1_h, array2_h, result_array_h, WIDTH);
	watch.stop();


	CUDA_CHECK_RETURN(cudaMemcpy(M_result_array_h, M_result_array_d, WIDTH*WIDTH*sizeof(int),
		cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	printf("Tempo gasto [CPU]: %lf (ms) \n", watch.elapsedTime());
	printf("Tempo gasto [GPU]: %lf (ms) \n", gpu_processing);

	TestMatrixResult(M_result_array_h, result_array_h, WIDTH) ? printf("Result: Success.") : printf("Result: Failure.");

	CUDA_CHECK_RETURN(cudaFree(array1_d));
	CUDA_CHECK_RETURN(cudaFree(array2_d));
	CUDA_CHECK_RETURN(cudaFree(M_result_array_d));

	return 0;
}

