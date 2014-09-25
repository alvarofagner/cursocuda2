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

class Managed
{
public:
	void *operator new(size_t len){
		void *ptr;
		cudaMallocManaged(&ptr, len);
		return ptr;
	}

		void operator delete(void *ptr) {
		cudaFree(ptr);
	}
};

struct DataElement : public Managed
{
	char *name;
	int value;
};

__global__
void Kernel(DataElement *elem) {
	printf("On device: name=%s, value=%d\n", elem->name, elem->value);

	elem->name[0] = 'd';
	elem->value++;
}

void launch(DataElement *elem) {
	Kernel << < 1, 1 >> >(elem);
	cudaDeviceSynchronize();
}

int main(void)
{
	DataElement *e = new DataElement;

	e->value = 10;
	cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1));
	strcpy(e->name, "hello");

	launch(e);

	printf("On host: name=%s, value=%d\n", e->name, e->value);

	cudaFree(e->name);
	delete e;

	cudaDeviceReset();

	getchar();
}
