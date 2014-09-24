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

#include <math.h>
#include "book.h"
#include "cpu_anim.h"
#include "cuda.h"
#define cimg_display 0
#include "CImg.h"


#define BLOCK_SIZE 8

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

/**
* This macro checks return value of the CUDA runtime call and exits
* the application if the call failed.
*/
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
		} }



// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.tga";
const char *outputFilename = "lena_filtered.tga";



//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void box_filter_kernel(float* output, int width, int height, int fWidth, int fHeight)
{

}

using namespace cimg_library;

int main(void) {

	cudaEvent_t e_Start,
		e_Stop;

	CImg<float> image(imageFilename);

	if (image.is_empty()){
		printf("Unable to source image file: %s\n", imageFilename);
		exit(-1);
	}

	//Reset no device
	CUDA_CHECK_RETURN(cudaDeviceReset());

	//Criando eventos
	CUDA_CHECK_RETURN(cudaEventCreate(&e_Start));
	CUDA_CHECK_RETURN(cudaEventCreate(&e_Stop));

	unsigned int size = image.width() * image.height() * sizeof(float);
	printf("Loaded '%s', %d x %d pixels\n", imageFilename, image.width(), image.height());


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid(image.width() / dimBlock.x, image.height() / dimBlock.y, 1);
	float *dData;

	// Execute the kernel
	CUDA_CHECK_RETURN(cudaEventRecord(e_Start, cudaEventDefault));
	box_filter_kernel << <dimGrid, dimBlock >> >(dData, image.width(), image.height(), image.width(), image.height());
	CUDA_CHECK_RETURN(cudaEventRecord(e_Stop, cudaEventDefault));

	float gpu_processing = 0;
	CUDA_CHECK_RETURN(cudaEventSynchronize(e_Stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&gpu_processing, e_Start, e_Stop));

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	printf("Processing time: %lf (ms)\n", gpu_processing);
	printf("%.2f Mpixels/sec\n",
		(image.width() * image.height() / (gpu_processing / 1000.0f)) / 1e6);


	// Allocate mem for the result on host side
	float *hOutputData = (float *)malloc(size);
	// copy result from device to host
	CUDA_CHECK_RETURN(cudaMemcpy(hOutputData,
		dData,
		size,
		cudaMemcpyDeviceToHost));

	CImg<float> imgTransformed;

	// Write result to file
	imgTransformed.save(outputFilename);

	printf("Wrote '%s'\n", outputFilename);


	printf("Success!\n");
}




