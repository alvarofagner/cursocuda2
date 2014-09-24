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
#include "util.h"
#include "cpu_anim.h"
#include "cuda.h"
#define cimg_display 0
#include "CImg.h"

#define BLOCK_SIZE 16


// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.tga";
const char *outputFilename = "lena_rot.tga";

////////////////////////////////////////////////////////////////////////////////
// Constants
const float angle = 0.5f;        // angle to rotate image by (in radians)

// Texture reference for 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> tex;


__global__ void transformKernel(float *outputData,
	int width,
	int height,
	float theta)
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float u = (float)x - (float)width / 2;
	float v = (float)y - (float)height / 2;
	float tu = u*cosf(theta) - v*sinf(theta);
	float tv = v*cosf(theta) + u*sinf(theta);

	tu /= (float)width;
	tv /= (float)height;

	// read from texture and write to global memory
	outputData[y*width + x] = tex2D(tex, tu + 0.5f, tv + 0.5f);
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

	// Allocate device memory for result
	float *dData = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&dData, size));

	// Allocate array and copy image data
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArray;
	CUDA_CHECK_RETURN(cudaMallocArray(&cuArray,
		&channelDesc,
		image.width(),
		image.height()));

	CUDA_CHECK_RETURN(cudaMemcpyToArray(cuArray,
		0,
		0,
		image.data(),
		size,
		cudaMemcpyHostToDevice));

	// Set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = true;    // access with normalized texture coordinates

	// Bind the array to the texture
	CUDA_CHECK_RETURN(cudaBindTextureToArray(tex, cuArray, channelDesc));

	dim3 dimBlock(8, 8, 1);
	dim3 dimGrid(image.width() / dimBlock.x, image.height() / dimBlock.y, 1);

	// Execute the kernel
	CUDA_CHECK_RETURN(cudaEventRecord(e_Start, cudaEventDefault));
	transformKernel << <dimGrid, dimBlock, 0 >> >(dData, image.width(), image.height(), angle);
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
	imgTransformed.assign(hOutputData, image.width(), image.height(), 1, 1);

	imgTransformed.save(outputFilename);

	printf("Wrote '%s'\n", outputFilename);

	CUDA_CHECK_RETURN(cudaFree(dData));
	CUDA_CHECK_RETURN(cudaFreeArray(cuArray));

	printf("Success!\n");
}