#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include "util.h"
#include "cuda.h"

#define BLOCK_SIZE 8
#define MAT_WIDTH 128


__device__ float * getSubMatrix(float *A, int row, int col, int stride)
{
	float *Asub;
	Asub = &A[stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

// Get a matrix element
__device__ float getElement(const float *A, int row, int col, int stride){
	return A[row *  stride + col];
}

// Set a matrix element
__device__ void setElement(float *A, int row, int col, int stride, float value)
{
	A[row * stride + col] = value;
}


__global__ void multMatrixS(float *C, float *A, float *B, const int width)
{

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	float *Csub = getSubMatrix(C, blockRow, blockCol, width);

	float Cvalue = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (width / BLOCK_SIZE); ++m) {

		float *Asub = getSubMatrix(A, blockRow, m, width);
		float *Bsub = getSubMatrix(B, m, blockCol, width);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = getElement(Asub, row, col, width);
		Bs[row][col] = getElement(Bsub, row, col, width);
		__syncthreads();

		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		__syncthreads();
	}

	setElement(Csub, row, col, width, Cvalue);
}


int main(int argc, char **argv){

	float *h_A, *h_B, *h_C, *h_C_CPU;
	int iC, jC;

	float   *d_C = NULL,
		*d_A = NULL,
		*d_B = NULL;

	printf("\nMultiplicando matriz - GPU\n");
	printf("Tamanho da matriz: %d x %d - memoria: [compartilhada]\n", MAT_WIDTH, MAT_WIDTH);

	h_A = (float*)malloc(MAT_WIDTH * MAT_WIDTH * sizeof(float));
	h_B = (float*)malloc(MAT_WIDTH * MAT_WIDTH * sizeof(float));
	h_C = (float*)malloc(MAT_WIDTH * MAT_WIDTH * sizeof(float));
	h_C_CPU = (float*)malloc(MAT_WIDTH * MAT_WIDTH * sizeof(float));


	for (jC = 0; jC < MAT_WIDTH; jC++){
		for (iC = 0; iC < MAT_WIDTH; iC++){
			int kC = jC * MAT_WIDTH + iC;
			h_A[kC] = (float)kC + 1;
			if (jC == iC)
				h_B[kC] = 1.0f;
			else
				h_B[kC] = 0.0f;

		}
	}

	//Reset no device
	CUDA_CHECK_RETURN(cudaDeviceReset());

	//GPU Memory
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_A, MAT_WIDTH * MAT_WIDTH * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_B, MAT_WIDTH * MAT_WIDTH * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_C, MAT_WIDTH * MAT_WIDTH * sizeof(float)));


	//Copy CPU --> GPU
	CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, MAT_WIDTH * MAT_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_B, h_B, MAT_WIDTH * MAT_WIDTH * sizeof(float), cudaMemcpyHostToDevice));


	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 numBlocks(MAT_WIDTH / threadsPerBlock.x, MAT_WIDTH / threadsPerBlock.y, 1);

	multMatrixS << <numBlocks, threadsPerBlock >> > (d_C, d_A, d_B, MAT_WIDTH);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	//Copy GPU --> CPU
	CUDA_CHECK_RETURN(cudaMemcpy(h_C, d_C, MAT_WIDTH*MAT_WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

	// CPU Mult
	MatrixMulHost(h_A, h_B, h_C_CPU, MAT_WIDTH);

	// Test result
	TestMatrixResult(h_C, h_C_CPU, MAT_WIDTH) ? printf("Result: Success.\n") : printf("Result: Failure.\n");


	// Free GPU Memory
	CUDA_CHECK_RETURN(cudaFree(d_A));
	CUDA_CHECK_RETURN(cudaFree(d_B));
	CUDA_CHECK_RETURN(cudaFree(d_C));

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_CPU);

	printf("END\n");

	return 0;
}