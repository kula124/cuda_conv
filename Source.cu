#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LOADBMP_IMPLEMENTATION

#include <stdio.h>
#include <math.h>
#include "bmpLoader.h"
#include "helper.h"

#define FILTER_SIZE 3
enum Filter
{
	BoxBlur = 0,
	Sharpen = 1
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
typedef unsigned char byte_t;
__global__ void convolution(byte_t* pixelMap, int* filter, double coef, byte_t* resultMap, int width, int height) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

	char r[FILTER_SIZE][FILTER_SIZE]; // temp result block
	for (int x = -1; x <= 1; x++)
		for (int y = -1; y <= 1; y++) {
			if (i == 1 && j == 1 && z == 1) {
				byte_t f = filter[FILTER_SIZE * (x + 1) + (j + 1)];
				byte_t m = *(&pixelMap[(i + x) * height + (j + y)] + z);
				printf("F:%d P:%d *:%lf\n", f, m, coef);
			}
			r[x + 1][y + 1] = (i + x > width || i + x < 0 || y + j > height || y + j < 0)
				? 150
				: *(&pixelMap[(i + x) * height + (j + y)] + z) * filter[FILTER_SIZE * (x + 1) + (j + 1)];
		}
	double sum = 0;
	for (int x = 0; x < 3; x++)
		for (int y = 0; y < 3; y++)
			sum += r[x][y];
	if (i == 1 && j == 1 && z == 1)
		printf("sum:%lf coef:%lf __ %lf %u\n", sum, coef, sum * coef, (byte_t)(int)ceil(sum*coef));
	sum *= coef;
	*(&resultMap[i * height + j] + z) = (byte_t)((int)ceil(sum));//*(&pixelMap[i * height + j] + z);
}

double coef[2] = { 1.0 / 9.0, 1.0 };
int filters[2][3][3] = {
	{
		{1,1,1},
		{1,1,1},
		{1,1,1}
	},
	{
		{0,-1,0},
		{-1,5,-1},
		{0,-1,0}
	}
};

int main(char** argv, int argc) {
	byte_t* pixels = NULL;
	imgsize_t width = 0, height = 0, size = 0;
	if (loadbmp_decode_file("lena.bmp", &pixels, &width, &height, LOADBMP_RGB))
	{
		puts("Error!");
		return 1;
	}
	size = width * height * 3;
	//int* flatFilter = (int*)flattenArray((void**)filters[BoxBlur], 3, 3, sizeof(int));
	int flatFilter[] = { 1,1,1,1,1,1,1,1,1 };
	byte_t* d_pixelMap, *d_resultMap, *h_resultMap;
	int* d_filter;
	gpuErrchk(cudaMalloc((void**)&d_filter, sizeof(int) * 3 * 3));
	gpuErrchk(cudaMalloc((void**)&d_pixelMap, sizeof(byte_t) * size));
	gpuErrchk(cudaMalloc((void**)&d_resultMap, sizeof(byte_t) * size));
	//---cpy

	gpuErrchk(cudaMemcpy(d_pixelMap, pixels, size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_filter, flatFilter, sizeof(int) * 3 * 3, cudaMemcpyHostToDevice));
	//DO STUFF
	/*
	Declare and allocate host and device memory. <
	Initialize host data. <
	Transfer data from the host to the device. <
	Execute one or more kernels. <
	Transfer results from the device to the host. <
	*/
	dim3 numberOfBlocks(width / 5, height / 5);
	dim3 threadsPerBlock(16, 16, 3);
	convolution <<<numberOfBlocks, threadsPerBlock >>> (d_pixelMap, d_filter, coef[BoxBlur], d_resultMap, width, height);

	gpuErrchk(cudaPeekAtLastError());
	h_resultMap = (byte_t*)malloc(sizeof(byte_t) * size);

	gpuErrchk(cudaMemcpy(h_resultMap, d_resultMap, size, cudaMemcpyDeviceToHost));

	// cudaDeviceSynchronize();
	loadbmp_encode_file("lena2.bmp", h_resultMap, width, height, LOADBMP_RGB);
	
	free(pixels);
	free(h_resultMap);
	// free(flatFilter);
	cudaFree(d_filter);
	cudaFree(d_resultMap);
	cudaFree(d_pixelMap);
	return 0;
}