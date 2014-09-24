#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cpu_bitmap.h"
#include "util.h"
#include <math.h>

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {
	float   r, b, g;
	float   radius;
	float   x, y, z;
	__device__ float hit(float ox, float oy, float *n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};
#define SPHERES 20

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - DIM / 2);
	float   oy = (y - DIM / 2);

	float   r = 0, g = 0, b = 0;
	float   maxz = -INF;
	for (int i = 0; i<SPHERES; i++) {
		float   n;
		float   t = s[i].hit(ox, oy, &n);

		if (t > maxz) {
			float fscale = n;

			maxz = t;
		}
	}

	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
	unsigned char   *dev_bitmap;
};

int main(void) {
	DataBlock   data;

	CPUBitmap bitmap(DIM, DIM, &data);
	unsigned char   *dev_bitmap;

	// allocate memory on the GPU for the output bitmap
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_bitmap,
		bitmap.image_size()));

	// allocate temp memory, initialize it, copy to constant
	// memory on the GPU, then free our temp memory
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i<SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(s, temp_s, SPHERES * sizeof(Sphere)));
	free(temp_s);


	// generate a bitmap from our sphere data
	dim3    grids(DIM / 16, DIM / 16);
	dim3    threads(16, 16);
	kernel << <grids, threads >> >(dev_bitmap);

	// copy our bitmap back from the GPU for display
	CUDA_CHECK_RETURN(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
		bitmap.image_size(),
		cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(dev_bitmap));

	// display
	bitmap.display_and_exit();
}
